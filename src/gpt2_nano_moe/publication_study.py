"""Publication routing study: matched multi-condition training comparison.

Trains identical-weight models under top1_fc (CF1.25), top1_bpr (CF1.25),
top2_fixed (CF1.25), top1_fc (CF2.0), and top2_capsel (CF1.25 × threshold grid),
each from the same initial weights and same token batches, across multiple seeds.

All results are written to a single JSON artifact with full provenance.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F

from .config import ModelConfig, RoutingMode, TrainConfig, load_config
from .metrics import aggregate_routing_metrics
from .model import GPT
from .routing_study import sha256
from .runtime import environment_metadata, seed_everything, select_device, synchronize
from .train import _batch, _corpus_batch, _load_corpus


# ──────────────────────────────────────────────────────────────────────────────
# Condition definition
# ──────────────────────────────────────────────────────────────────────────────

def _build_conditions(
    base_cf: float,
    double_cf: float,
    threshold_grid: list[float],
) -> list[dict[str, Any]]:
    """Return the ordered list of study conditions."""
    conditions: list[dict[str, Any]] = [
        {"routing_mode": "top1_fc",    "capacity_factor": base_cf,   "capsel_margin_threshold": None},
        {"routing_mode": "top1_bpr",   "capacity_factor": base_cf,   "capsel_margin_threshold": None},
        {"routing_mode": "top2_fixed", "capacity_factor": base_cf,   "capsel_margin_threshold": None},
        {"routing_mode": "top1_fc",    "capacity_factor": double_cf, "capsel_margin_threshold": None},
    ]
    for threshold in threshold_grid:
        conditions.append({
            "routing_mode": "top2_capsel",
            "capacity_factor": base_cf,
            "capsel_margin_threshold": threshold,
        })
    return conditions


def _condition_key(cond: dict[str, Any]) -> str:
    mode = cond["routing_mode"]
    cf = cond["capacity_factor"]
    threshold = cond["capsel_margin_threshold"]
    key = f"{mode}_cf{cf}"
    if threshold is not None:
        key += f"_m{threshold}"
    return key


# ──────────────────────────────────────────────────────────────────────────────
# Model construction with overridden routing config
# ──────────────────────────────────────────────────────────────────────────────

def _make_model_config(base: ModelConfig, cond: dict[str, Any]) -> ModelConfig:
    routing_mode: RoutingMode = cond["routing_mode"]
    capacity_factor: float = cond["capacity_factor"]
    threshold = cond["capsel_margin_threshold"]
    kwargs: dict[str, Any] = {
        field.name: getattr(base, field.name)
        for field in base.__dataclass_fields__.values()  # type: ignore[attr-defined]
    }
    kwargs["routing_mode"] = routing_mode
    kwargs["capacity_factor"] = capacity_factor
    if threshold is not None:
        kwargs["capsel_margin_threshold"] = threshold
    return ModelConfig(**kwargs)


def _state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def _calibrate_margin_thresholds(
    model_config: ModelConfig,
    train_cfg: TrainConfig,
    seed: int,
    device: torch.device,
    initial_state: dict[str, torch.Tensor],
    corpus: tuple | None,
    quantiles: list[float],
    batches: int,
) -> dict[str, Any]:
    if batches < 1:
        raise ValueError("threshold_calibration_batches must be at least 1")
    if not quantiles or any(not 0.0 < value < 1.0 for value in quantiles):
        raise ValueError("threshold_quantiles must contain values between 0 and 1")

    model = GPT(model_config).to(device)
    model.load_state_dict(initial_state)
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed + 3)
    margins: list[torch.Tensor] = []
    handles = []

    def capture_margin(module, args) -> None:
        inputs = args[0]
        probabilities = torch.softmax(module.projection(inputs).float(), dim=-1)
        top2 = torch.topk(probabilities, 2, dim=-1).values
        margins.append((top2[:, 0] - top2[:, 1]).detach().cpu())

    for block in model.blocks:
        router = getattr(block.mlp, "router", None)
        if router is None:
            raise RuntimeError("threshold calibration requires an MoE router")
        handles.append(router.register_forward_pre_hook(capture_margin))

    try:
        for _ in range(batches):
            if corpus is None:
                token_ids = _batch(
                    train_cfg.batch_size,
                    model_config.context_length,
                    model_config.vocab_size,
                    generator,
                    device,
                )
            else:
                token_ids = _corpus_batch(
                    corpus[0],
                    train_cfg.batch_size,
                    model_config.context_length,
                    generator,
                    device,
                )
            model(token_ids)
    finally:
        for handle in handles:
            handle.remove()

    values = torch.cat(margins)
    thresholds = [
        float(torch.quantile(values, quantile).item())
        for quantile in quantiles
    ]
    return {
        "seed": seed,
        "batches": batches,
        "samples": int(values.numel()),
        "quantiles": quantiles,
        "thresholds": thresholds,
        "margin_mean": float(values.mean().item()),
        "margin_std": float(values.std(unbiased=False).item()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Single-condition training run
# ──────────────────────────────────────────────────────────────────────────────

def _train_condition(
    model_config: ModelConfig,
    train_cfg: TrainConfig,
    seed: int,
    device: torch.device,
    initial_state: dict[str, torch.Tensor],
    corpus: tuple | None,
    validation_batches: int,
) -> dict[str, Any]:
    """Train one condition for one seed and return per-step history + final metrics."""
    model = GPT(model_config).to(device)
    # Load weights from the shared initial state for this seed.
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    # Reset generator to the same seed so batches are identical across conditions.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    validation_generator = torch.Generator(device="cpu").manual_seed(seed + 1)

    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    model.train()
    for step in range(1, train_cfg.steps + 1):
        if corpus is None:
            token_ids = _batch(
                train_cfg.batch_size,
                model_config.context_length,
                model_config.vocab_size,
                generator,
                device,
            )
        else:
            token_ids = _corpus_batch(
                corpus[0],
                train_cfg.batch_size,
                model_config.context_length,
                generator,
                device,
            )
        optimizer.zero_grad(set_to_none=True)
        loss, model_output = model.loss(token_ids)
        lm_loss = F.cross_entropy(
            model_output.logits[:, :-1].reshape(-1, model_config.vocab_size),
            token_ids[:, 1:].reshape(-1),
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % train_cfg.log_every == 0 or step == train_cfg.steps:
            val_lm_loss: float | None = None
            if corpus is not None:
                model.eval()
                with torch.no_grad():
                    val_tokens = _corpus_batch(
                        corpus[1],
                        train_cfg.batch_size,
                        model_config.context_length,
                        validation_generator,
                        device,
                    )
                    _, val_output = model.loss(val_tokens)
                    val_lm_loss = float(
                        F.cross_entropy(
                            val_output.logits[:, :-1].reshape(-1, model_config.vocab_size),
                            val_tokens[:, 1:].reshape(-1),
                        ).detach().cpu()
                    )
                model.train()
            history.append({
                "step": step,
                "optimization_loss": float(loss.detach().cpu()),
                "language_model_loss": float(lm_loss.detach().cpu()),
                "auxiliary_loss": float(model_output.auxiliary_loss.detach().cpu()),
                "gradient_norm": float(grad_norm.detach().cpu()),
                "validation_lm_loss": val_lm_loss,
                "routing": aggregate_routing_metrics(model_output.routing_metrics),
            })

    synchronize(device)
    elapsed = time.perf_counter() - started
    final_lm_loss = history[-1]["language_model_loss"] if history else None
    final_val_losses: list[float] = []
    final_val_routing = []
    final_validation_generator = torch.Generator(device="cpu").manual_seed(seed + 2)
    model.eval()
    with torch.no_grad():
        for _ in range(validation_batches):
            if corpus is None:
                val_tokens = _batch(
                    train_cfg.batch_size,
                    model_config.context_length,
                    model_config.vocab_size,
                    final_validation_generator,
                    device,
                )
            else:
                val_tokens = _corpus_batch(
                    corpus[1],
                    train_cfg.batch_size,
                    model_config.context_length,
                    final_validation_generator,
                    device,
                )
            _, val_output = model.loss(val_tokens)
            final_val_losses.append(
                float(
                    F.cross_entropy(
                        val_output.logits[:, :-1].reshape(
                            -1, model_config.vocab_size
                        ),
                        val_tokens[:, 1:].reshape(-1),
                    ).detach().cpu()
                )
            )
            final_val_routing.extend(val_output.routing_metrics)
    final_val_lm_loss = sum(final_val_losses) / len(final_val_losses)
    return {
        "training_curve": history,
        "final_lm_loss": final_lm_loss,
        "final_held_out_lm_loss": final_val_lm_loss,
        "final_validation_batches": validation_batches,
        "final_validation_routing": aggregate_routing_metrics(
            tuple(final_val_routing)
        ),
        "elapsed_seconds": elapsed,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public runner
# ──────────────────────────────────────────────────────────────────────────────

def run(
    config_path: str | Path,
    output_path: str | Path,
    *,
    seeds: list[int] | None = None,
    threshold_grid: list[float] | None = None,
    base_capacity_factor: float | None = None,
    double_capacity_factor: float | None = None,
    corpus_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the full publication routing study and write a JSON artifact.

    Parameters
    ----------
    config_path:
        TOML config with ``[model]``, ``[train]``, and optional ``[study]`` sections.
    output_path:
        Destination for the JSON artifact.
    seeds:
        Override for ``study.seeds``; falls back to TOML or ``[train.seed]``.
    threshold_grid:
        Override for ``study.threshold_grid``; falls back to TOML or ``[0.1]``.
    base_capacity_factor:
        Override for the baseline CF (default 1.25).
    double_capacity_factor:
        Override for the doubled-CF top-1 condition (default 2.0).
    corpus_path:
        Override for corpus path; falls back to ``train.corpus_path``.
    """
    config_file = Path(config_path)
    experiment = load_config(config_file)
    if experiment.model.architecture != "moe":
        raise ValueError("publication study requires model.architecture = 'moe'")

    # Read study section from raw TOML if present.
    import tomllib
    with config_file.open("rb") as fh:
        raw = tomllib.load(fh)
    study_raw: dict[str, Any] = raw.get("study", {})

    effective_seeds: list[int] = seeds or study_raw.get("seeds", [experiment.train.seed])
    if len(set(effective_seeds)) != len(effective_seeds):
        raise ValueError("study seeds must be unique")
    if threshold_grid is not None:
        explicit_thresholds = threshold_grid
        threshold_quantiles = []
    else:
        explicit_thresholds = study_raw.get("threshold_grid")
        threshold_quantiles = [
            float(value)
            for value in study_raw.get("threshold_quantiles", [])
        ]
    if explicit_thresholds is not None and threshold_quantiles:
        raise ValueError(
            "configure threshold_grid or threshold_quantiles, not both"
        )
    effective_base_cf: float = (
        base_capacity_factor
        or study_raw.get("base_capacity_factor", 1.25)
    )
    effective_double_cf: float = (
        double_capacity_factor
        or study_raw.get("double_capacity_factor", 2.0)
    )
    validation_batches = int(study_raw.get("validation_batches", 16))
    if validation_batches < 1:
        raise ValueError("study.validation_batches must be at least 1")
    effective_corpus = corpus_path or experiment.train.corpus_path

    device = select_device(experiment.train.device)
    corpus: tuple | None = None
    if effective_corpus is not None:
        corpus = _load_corpus(
            effective_corpus,
            vocab_size=experiment.model.vocab_size,
            context_length=experiment.model.context_length,
            validation_fraction=experiment.train.validation_fraction,
        )

    canonical_config = _make_model_config(
        experiment.model,
        {
            "routing_mode": "top1_fc",
            "capacity_factor": effective_base_cf,
            "capsel_margin_threshold": None,
        },
    )
    threshold_calibration: dict[str, Any] | None = None
    if threshold_quantiles:
        calibration_seed = int(
            study_raw.get("threshold_calibration_seed", effective_seeds[0])
        )
        calibration_batches = int(
            study_raw.get("threshold_calibration_batches", 16)
        )
        seed_everything(calibration_seed)
        calibration_model = GPT(canonical_config)
        calibration_state = copy.deepcopy(calibration_model.state_dict())
        del calibration_model
        threshold_calibration = _calibrate_margin_thresholds(
            canonical_config,
            experiment.train,
            calibration_seed,
            device,
            calibration_state,
            corpus,
            threshold_quantiles,
            calibration_batches,
        )
        effective_threshold_grid = threshold_calibration["thresholds"]
    else:
        effective_threshold_grid = [
            float(value) for value in (explicit_thresholds or [0.1])
        ]

    conditions = _build_conditions(
        effective_base_cf, effective_double_cf, effective_threshold_grid
    )

    # ── Run all conditions × seeds ──────────────────────────────────────────
    condition_results: dict[str, dict[str, Any]] = {}
    initialization_hashes: dict[str, set[str]] = {
        str(seed): set() for seed in effective_seeds
    }
    total_start = time.perf_counter()

    for cond in conditions:
        key = _condition_key(cond)
        cond_model_cfg = _make_model_config(experiment.model, cond)
        cond_seeds: dict[str, Any] = {}
        for seed in effective_seeds:
            seed_everything(seed)
            base_model = GPT(canonical_config)
            initial_state = copy.deepcopy(base_model.state_dict())
            del base_model
            initial_state_hash = _state_dict_sha256(initial_state)
            initialization_hashes[str(seed)].add(initial_state_hash)
            result = _train_condition(
                cond_model_cfg,
                experiment.train,
                seed,
                device,
                initial_state,
                corpus,
                validation_batches,
            )
            result["initial_state_sha256"] = initial_state_hash
            cond_seeds[str(seed)] = result

        # Aggregate across seeds for this condition.
        final_losses = [
            v["final_held_out_lm_loss"]
            for v in cond_seeds.values()
            if v["final_held_out_lm_loss"] is not None
        ]
        if not final_losses:
            final_losses = [
                v["final_lm_loss"]
                for v in cond_seeds.values()
                if v["final_lm_loss"] is not None
            ]
        mean_loss = sum(final_losses) / len(final_losses) if final_losses else None
        std_loss: float | None
        if len(final_losses) > 1:
            variance = sum((x - mean_loss) ** 2 for x in final_losses) / len(final_losses)  # type: ignore[operator]
            std_loss = variance ** 0.5
        else:
            std_loss = 0.0 if final_losses else None

        condition_results[key] = {
            "condition": cond,
            "model_config": asdict(cond_model_cfg),
            "seeds": cond_seeds,
            "aggregate": {
                "final_loss_mean": mean_loss,
                "final_loss_std": std_loss,
                "n_seeds": len(effective_seeds),
            },
        }

    total_elapsed = time.perf_counter() - total_start

    # ── Provenance ───────────────────────────────────────────────────────────
    corpus_provenance: dict[str, Any] = {}
    if effective_corpus is not None:
        corpus_path_abs = Path(effective_corpus)
        corpus_provenance = {
            "corpus_path": str(corpus_path_abs),
            "corpus_sha256": sha256(corpus_path_abs) if corpus_path_abs.is_file() else None,
        }

    artifact: dict[str, Any] = {
        "schema_version": 1,
        "result_kind": "measured",
        "workload": "publication_routing_study",
        "config": {
            "model_base": asdict(experiment.model),
            "train_base": asdict(experiment.train),
            "study": {
                "seeds": effective_seeds,
                "threshold_grid": effective_threshold_grid,
                "threshold_calibration": threshold_calibration,
                "base_capacity_factor": effective_base_cf,
                "double_capacity_factor": effective_double_cf,
                "validation_batches": validation_batches,
                "conditions": [_condition_key(c) for c in conditions],
            },
        },
        "provenance": {
            "config_path": str(config_file),
            "config_sha256": sha256(config_file),
            **corpus_provenance,
        },
        "environment": environment_metadata(device),
        "initialization_hashes": {
            seed: sorted(hashes)
            for seed, hashes in initialization_hashes.items()
        },
        "identical_initialization_across_conditions": all(
            len(hashes) == 1 for hashes in initialization_hashes.values()
        ),
        "conditions": condition_results,
        "total_elapsed_seconds": total_elapsed,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publication routing study: matched multi-condition training"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--threshold-grid", nargs="+", type=float)
    parser.add_argument("--base-cf", type=float)
    parser.add_argument("--double-cf", type=float)
    parser.add_argument("--corpus", type=Path)
    args = parser.parse_args()
    artifact = run(
        args.config,
        args.output,
        seeds=args.seeds,
        threshold_grid=args.threshold_grid,
        base_capacity_factor=args.base_cf,
        double_capacity_factor=args.double_cf,
        corpus_path=args.corpus,
    )
    summary = {
        key: val["aggregate"]
        for key, val in artifact["conditions"].items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
