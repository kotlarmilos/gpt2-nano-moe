from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from typing import Any

import torch

from .config import Architecture, ModelConfig, load_config
from .metrics import aggregate_routing_metrics
from .model import GPT
from .runtime import environment_metadata, seed_everything, select_device, synchronize
from .train import _corpus_batch, _load_corpus


@torch.no_grad()
def _measure(
    model: GPT,
    token_ids: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    for _ in range(2):
        model(token_ids)
    synchronize(device)
    baseline_memory = (
        torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    routing_metrics = []
    for _ in range(iterations):
        output = model(token_ids)
        routing_metrics.extend(output.routing_metrics)
    synchronize(device)
    elapsed = time.perf_counter() - started
    peak_memory = (
        torch.cuda.max_memory_allocated(device) - baseline_memory
        if device.type == "cuda"
        else None
    )
    tokens = token_ids.numel() * iterations
    return {
        "elapsed_seconds": elapsed,
        "iterations": iterations,
        "processed_tokens": tokens,
        "tokens_per_second": tokens / elapsed,
        "incremental_peak_memory_bytes": peak_memory,
        "parameter_counts": model.parameter_counts().to_dict(),
        "routing": aggregate_routing_metrics(tuple(routing_metrics)),
    }


def _model(
    config: ModelConfig,
    architecture: Architecture,
    device: torch.device,
    checkpoint_path: str | Path | None,
) -> GPT:
    selected = replace(config, architecture=architecture)
    model = GPT(selected).to(device)
    if checkpoint_path is None:
        return model
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    checkpoint_config = ModelConfig(**checkpoint["model_config"])
    if checkpoint_config != selected:
        raise ValueError(
            f"{architecture} checkpoint configuration does not match benchmark"
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def run(
    config_path: str | Path,
    output_path: str | Path,
    iterations: int = 10,
    *,
    dense_checkpoint: str | Path | None = None,
    moe_checkpoint: str | Path | None = None,
    corpus_path: str | Path | None = None,
) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    experiment = load_config(config_path)
    if (dense_checkpoint is None) != (moe_checkpoint is None):
        raise ValueError(
            "dense and MoE checkpoints must be supplied together"
        )
    seed_everything(experiment.train.seed)
    device = select_device(experiment.train.device)
    generator = torch.Generator(device="cpu").manual_seed(experiment.train.seed)
    if corpus_path is None:
        token_ids = torch.randint(
            experiment.model.vocab_size,
            (experiment.train.batch_size, experiment.model.context_length),
            generator=generator,
        ).to(device)
        workload = "seeded_synthetic_forward_pass"
    else:
        _training, validation = _load_corpus(
            corpus_path,
            vocab_size=experiment.model.vocab_size,
            context_length=experiment.model.context_length,
            validation_fraction=experiment.train.validation_fraction,
        )
        token_ids = _corpus_batch(
            validation,
            experiment.train.batch_size,
            experiment.model.context_length,
            generator,
            device,
        )
        workload = "held_out_uint16_corpus_forward_pass"
    measurements: dict[str, Any] = {}
    checkpoints = {
        "dense": dense_checkpoint,
        "moe": moe_checkpoint,
    }
    architectures: tuple[Architecture, ...] = ("dense", "moe")
    for architecture in architectures:
        seed_everything(experiment.train.seed)
        model = _model(
            experiment.model,
            architecture,
            device,
            checkpoints[architecture],
        )
        measurements[architecture] = _measure(
            model, token_ids, iterations, device
        )
    result = {
        "schema_version": 1,
        "result_kind": "measured",
        "workload": workload,
        "checkpoints": {
            name: str(path) if path is not None else None
            for name, path in checkpoints.items()
        },
        "config": {
            "model": asdict(experiment.model),
            "train": asdict(experiment.train),
            "benchmark_iterations": iterations,
        },
        "environment": environment_metadata(device),
        "measurements": measurements,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark dense and sparse models")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--dense-checkpoint")
    parser.add_argument("--moe-checkpoint")
    parser.add_argument("--corpus")
    arguments = parser.parse_args()
    result = run(
        arguments.config,
        arguments.output,
        arguments.iterations,
        dense_checkpoint=arguments.dense_checkpoint,
        moe_checkpoint=arguments.moe_checkpoint,
        corpus_path=arguments.corpus,
    )
    print(json.dumps(result["measurements"], sort_keys=True))


if __name__ == "__main__":
    main()
