from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .config import ModelConfig, load_config
from .metrics import aggregate_routing_metrics
from .model import GPT, ModelOutput
from .router import Top1Router, deterministic_capacity_mask, expert_capacity
from .runtime import environment_metadata, seed_everything, select_device, synchronize
from .train import _corpus_batch, _load_corpus


@dataclass(frozen=True)
class TopKRoutingMetrics:
    top_k: int
    capacity: int
    tokens: int
    routed_token_slots: int
    accepted_token_slots: int
    dropped_token_slots: int
    expert_counts: torch.Tensor
    raw_expert_counts: torch.Tensor
    coefficient_of_variation: float
    max_to_mean_load: float
    zero_expert_fraction: float
    expert_utilization: float
    load_balancing_loss: float
    mean_selected_probability: float

    @property
    def dropped_fraction(self) -> float:
        return self.dropped_token_slots / max(self.routed_token_slots, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "capacity": self.capacity,
            "tokens": self.tokens,
            "routed_token_slots": self.routed_token_slots,
            "accepted_token_slots": self.accepted_token_slots,
            "dropped_token_slots": self.dropped_token_slots,
            "dropped_fraction": self.dropped_fraction,
            "expert_counts": self.expert_counts.detach().cpu().tolist(),
            "raw_expert_counts": self.raw_expert_counts.detach().cpu().tolist(),
            "coefficient_of_variation": self.coefficient_of_variation,
            "max_to_mean_load": self.max_to_mean_load,
            "zero_expert_fraction": self.zero_expert_fraction,
            "expert_utilization": self.expert_utilization,
            "load_balancing_loss": self.load_balancing_loss,
            "mean_selected_probability": self.mean_selected_probability,
        }


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utilization_metrics(expert_counts: torch.Tensor) -> dict[str, float]:
    if expert_counts.ndim != 1 or expert_counts.numel() == 0:
        raise ValueError("expert_counts must have shape [experts]")
    counts = expert_counts.detach().float().cpu()
    mean = counts.mean()
    if float(mean) == 0.0:
        return {
            "coefficient_of_variation": 0.0,
            "max_to_mean_load": 0.0,
            "zero_expert_fraction": 1.0,
            "expert_utilization": 0.0,
        }
    zero_fraction = counts.eq(0).float().mean()
    return {
        "coefficient_of_variation": float(counts.std(unbiased=False) / mean),
        "max_to_mean_load": float(counts.max() / mean),
        "zero_expert_fraction": float(zero_fraction),
        "expert_utilization": float(1.0 - zero_fraction),
    }


def routing_metrics_for_top_k(
    probabilities: torch.Tensor, top_k: int, capacity: int
) -> TopKRoutingMetrics:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [tokens, experts]")
    token_count, expert_count = probabilities.shape
    if token_count <= 0:
        raise ValueError("probabilities must contain at least one token")
    if top_k <= 0 or top_k > expert_count:
        raise ValueError("top_k must be in [1, expert_count]")
    if capacity <= 0:
        raise ValueError("capacity must be positive")

    selected_probabilities, expert_indices = torch.topk(
        probabilities.float(), top_k, dim=-1
    )
    accepted = deterministic_capacity_mask(expert_indices, expert_count, capacity)
    routed_slots = token_count * top_k
    accepted_slots = int(accepted.sum().item())
    expert_counts = torch.bincount(
        expert_indices[accepted], minlength=expert_count
    ).detach()
    raw_expert_counts = torch.bincount(
        expert_indices.reshape(-1), minlength=expert_count
    ).detach()
    slot_fraction = raw_expert_counts.float() / max(routed_slots, 1)
    probability_fraction = probabilities.float().mean(dim=0)
    load_balancing_loss = expert_count * torch.sum(
        slot_fraction.to(probability_fraction.device) * probability_fraction
    )
    utilization = utilization_metrics(expert_counts)
    return TopKRoutingMetrics(
        top_k=top_k,
        capacity=capacity,
        tokens=token_count,
        routed_token_slots=routed_slots,
        accepted_token_slots=accepted_slots,
        dropped_token_slots=routed_slots - accepted_slots,
        expert_counts=expert_counts.cpu(),
        raw_expert_counts=raw_expert_counts.cpu(),
        coefficient_of_variation=utilization["coefficient_of_variation"],
        max_to_mean_load=utilization["max_to_mean_load"],
        zero_expert_fraction=utilization["zero_expert_fraction"],
        expert_utilization=utilization["expert_utilization"],
        load_balancing_loss=float(load_balancing_loss.detach().cpu()),
        mean_selected_probability=float(selected_probabilities.mean().detach().cpu()),
    )


def aggregate_top_k_metrics(
    metrics: Sequence[TopKRoutingMetrics],
) -> dict[str, Any]:
    if not metrics:
        raise ValueError("metrics must not be empty")
    top_k = metrics[0].top_k
    if any(item.top_k != top_k for item in metrics):
        raise ValueError("all metrics must use the same top_k")
    expert_counts = torch.stack([item.expert_counts for item in metrics]).sum(dim=0)
    raw_expert_counts = torch.stack(
        [item.raw_expert_counts for item in metrics]
    ).sum(dim=0)
    utilization = utilization_metrics(expert_counts)
    routed_slots = sum(item.routed_token_slots for item in metrics)
    accepted_slots = sum(item.accepted_token_slots for item in metrics)
    mean_selected = sum(
        item.mean_selected_probability * item.routed_token_slots for item in metrics
    ) / max(routed_slots, 1)
    return {
        "top_k": top_k,
        "layers": len(metrics),
        "capacity_per_expert": metrics[0].capacity,
        "tokens_per_layer": metrics[0].tokens,
        "routed_token_slots": routed_slots,
        "accepted_token_slots": accepted_slots,
        "dropped_token_slots": routed_slots - accepted_slots,
        "dropped_fraction": (routed_slots - accepted_slots) / max(routed_slots, 1),
        "expert_counts": expert_counts.tolist(),
        "raw_expert_counts": raw_expert_counts.tolist(),
        "coefficient_of_variation": utilization["coefficient_of_variation"],
        "max_to_mean_load": utilization["max_to_mean_load"],
        "zero_expert_fraction": utilization["zero_expert_fraction"],
        "expert_utilization": utilization["expert_utilization"],
        "load_balancing_loss": sum(item.load_balancing_loss for item in metrics)
        / len(metrics),
        "mean_selected_probability": mean_selected,
        "per_layer": [item.to_dict() for item in metrics],
    }


def compare_top_k_routing(
    probability_layers: Sequence[torch.Tensor],
    top_k_values: Sequence[int],
    capacity_factor: float,
) -> dict[str, Any]:
    if not probability_layers:
        raise ValueError("probability_layers must not be empty")
    results: dict[str, Any] = {}
    for top_k in top_k_values:
        layer_metrics = []
        for probabilities in probability_layers:
            capacity = expert_capacity(
                probabilities.shape[0], probabilities.shape[1], capacity_factor
            )
            layer_metrics.append(
                routing_metrics_for_top_k(probabilities, top_k, capacity)
            )
        results[str(top_k)] = aggregate_top_k_metrics(layer_metrics)
    return results


def _token_batch(
    config_path: str | Path,
    seed: int,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    validation_fraction: float,
    device: torch.device,
    corpus_path: str | Path | None,
) -> tuple[torch.Tensor, str]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if corpus_path is None:
        return (
            torch.randint(
                vocab_size,
                (batch_size, context_length),
                generator=generator,
            ).to(device),
            "seeded_synthetic_fixed_batch",
        )
    _training, validation = _load_corpus(
        corpus_path,
        vocab_size=vocab_size,
        context_length=context_length,
        validation_fraction=validation_fraction,
    )
    return (
        _corpus_batch(validation, batch_size, context_length, generator, device),
        f"held_out_uint16_fixed_batch_from_{Path(config_path).name}",
    )


def _model(
    config: ModelConfig, device: torch.device, checkpoint_path: str | Path | None
) -> GPT:
    model = GPT(config).to(device)
    if checkpoint_path is None:
        return model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    checkpoint_config = ModelConfig(**checkpoint["model_config"])
    if checkpoint_config != config:
        raise ValueError("checkpoint configuration does not match routing study")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


@torch.no_grad()
def _capture_probabilities(
    model: GPT, token_ids: torch.Tensor, device: torch.device
) -> tuple[tuple[torch.Tensor, ...], ModelOutput]:
    probability_layers: list[torch.Tensor] = []
    handles = []
    for module in model.modules():
        if isinstance(module, Top1Router):
            handles.append(
                module.register_forward_hook(
                    lambda _module, _inputs, output: probability_layers.append(
                        output.probabilities.detach().cpu()
                    )
                )
            )
    try:
        model.eval()
        output = model(token_ids)
        synchronize(device)
    finally:
        for handle in handles:
            handle.remove()
    if not probability_layers:
        raise ValueError("routing study requires a MoE model with routers")
    return tuple(probability_layers), output


def run(
    config_path: str | Path,
    output_path: str | Path,
    checkpoint_path: str | Path | None = None,
    *,
    top_k_values: Sequence[int] = (1, 2),
    seed: int | None = None,
    corpus_path: str | Path | None = None,
) -> dict[str, Any]:
    experiment = load_config(config_path)
    if experiment.model.architecture != "moe":
        raise ValueError("routing study requires model.architecture = 'moe'")
    if not top_k_values:
        raise ValueError("top_k_values must not be empty")
    if len(set(top_k_values)) != len(top_k_values):
        raise ValueError("top_k_values must be unique")

    selected_seed = experiment.train.seed if seed is None else seed
    selected_corpus = experiment.train.corpus_path if corpus_path is None else corpus_path
    seed_everything(selected_seed)
    device = select_device(experiment.train.device)
    model = _model(experiment.model, device, checkpoint_path)
    token_ids, token_source = _token_batch(
        config_path,
        selected_seed,
        experiment.train.batch_size,
        experiment.model.context_length,
        experiment.model.vocab_size,
        experiment.train.validation_fraction,
        device,
        selected_corpus,
    )
    probability_layers, model_output = _capture_probabilities(model, token_ids, device)
    top_k_measurements = compare_top_k_routing(
        probability_layers, tuple(top_k_values), experiment.model.capacity_factor
    )
    language_model_loss = float(
        F.cross_entropy(
            model_output.logits[:, :-1].reshape(-1, experiment.model.vocab_size),
            token_ids[:, 1:].reshape(-1),
        )
        .detach()
        .cpu()
    )
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    config_file = Path(config_path)
    result = {
        "schema_version": 1,
        "result_kind": "measured",
        "workload": "development_scale_fixed_batch_routing_extension",
        "extension": "top-1 versus top-2 routing at fixed capacity",
        "config": {
            "model": asdict(experiment.model),
            "train": asdict(experiment.train),
            "routing_study": {
                "top_k_values": list(top_k_values),
                "fixed_per_expert_capacity": True,
            },
        },
        "provenance": {
            "config_path": str(config_file),
            "config_sha256": sha256(config_file),
            "checkpoint_path": str(checkpoint) if checkpoint is not None else None,
            "checkpoint_sha256": sha256(checkpoint) if checkpoint is not None else None,
            "seed": selected_seed,
            "token_source": token_source,
            "batch_size": int(token_ids.shape[0]),
            "context_length": int(token_ids.shape[1]),
            "flattened_tokens_per_layer": int(token_ids.numel()),
            "captured_moe_layers": len(probability_layers),
            "capacity_factor": experiment.model.capacity_factor,
            "expert_count": experiment.model.expert_count,
        },
        "environment": environment_metadata(device),
        "measurements": {
            "top_k": top_k_measurements,
            "model_forward": {
                "language_model_loss": language_model_loss,
                "auxiliary_loss": float(model_output.auxiliary_loss.detach().cpu()),
                "existing_top1_routing": aggregate_routing_metrics(
                    model_output.routing_metrics
                ),
            },
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure top-k routing drops at fixed expert capacity"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--corpus", type=Path)
    arguments = parser.parse_args()
    result = run(
        arguments.config,
        arguments.output,
        arguments.checkpoint,
        top_k_values=tuple(arguments.top_k),
        seed=arguments.seed,
        corpus_path=arguments.corpus,
    )
    print(json.dumps(result["measurements"]["top_k"], sort_keys=True))


if __name__ == "__main__":
    main()
