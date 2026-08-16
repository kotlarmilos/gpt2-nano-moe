from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from typing import Any, cast

import numpy as np
import torch

from .config import Architecture, load_config
from .metrics import aggregate_routing_metrics
from .model import GPT
from .runtime import environment_metadata, seed_everything, select_device, synchronize


def _batch(
    batch_size: int,
    context_length: int,
    vocab_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return torch.randint(
        vocab_size,
        (batch_size, context_length),
        generator=generator,
        device="cpu",
    ).to(device)


def _load_corpus(
    path: str | Path,
    *,
    vocab_size: int,
    context_length: int,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    corpus_path = Path(path)
    if not corpus_path.is_file():
        raise FileNotFoundError(corpus_path)
    tokens = np.fromfile(corpus_path, dtype="<u2")
    if tokens.size < 2 * context_length:
        raise ValueError("corpus must contain at least two context windows")
    if int(tokens.max()) >= vocab_size:
        raise ValueError("corpus contains a token outside the configured vocabulary")
    split = int(tokens.size * (1.0 - validation_fraction))
    if split < context_length or tokens.size - split < context_length:
        raise ValueError("validation_fraction leaves a split shorter than one context")
    return tokens[:split], tokens[split:]


def _corpus_batch(
    tokens: np.ndarray,
    batch_size: int,
    context_length: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    starts = torch.randint(
        tokens.size - context_length + 1,
        (batch_size,),
        generator=generator,
    ).tolist()
    rows = np.stack(
        [tokens[start : start + context_length] for start in starts]
    ).astype(np.int64, copy=False)
    return torch.from_numpy(rows).to(device)


def run(
    config_path: str | Path,
    output_path: str | Path,
    checkpoint_path: str | Path | None = None,
    *,
    architecture: Architecture | None = None,
    seed: int | None = None,
    corpus_path: str | Path | None = None,
) -> dict[str, Any]:
    experiment = load_config(config_path)
    if architecture is not None:
        experiment = replace(
            experiment,
            model=replace(experiment.model, architecture=architecture),
        )
    if seed is not None or corpus_path is not None:
        experiment = replace(
            experiment,
            train=replace(
                experiment.train,
                seed=experiment.train.seed if seed is None else seed,
                corpus_path=(
                    experiment.train.corpus_path
                    if corpus_path is None
                    else str(corpus_path)
                ),
            ),
        )
    seed_everything(experiment.train.seed)
    device = select_device(experiment.train.device)
    model = GPT(experiment.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=experiment.train.learning_rate,
        weight_decay=experiment.train.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(experiment.train.seed)
    validation_generator = torch.Generator(device="cpu").manual_seed(
        experiment.train.seed + 1
    )
    corpus: tuple[np.ndarray, np.ndarray] | None = None
    if experiment.train.corpus_path is not None:
        corpus = _load_corpus(
            experiment.train.corpus_path,
            vocab_size=experiment.model.vocab_size,
            context_length=experiment.model.context_length,
            validation_fraction=experiment.train.validation_fraction,
        )
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    processed_tokens = 0
    model.train()
    for step in range(1, experiment.train.steps + 1):
        token_ids = (
            _batch(
                experiment.train.batch_size,
                experiment.model.context_length,
                experiment.model.vocab_size,
                generator,
                device,
            )
            if corpus is None
            else _corpus_batch(
                corpus[0],
                experiment.train.batch_size,
                experiment.model.context_length,
                generator,
                device,
            )
        )
        optimizer.zero_grad(set_to_none=True)
        loss, model_output = model.loss(token_ids)
        language_model_loss = torch.nn.functional.cross_entropy(
            model_output.logits[:, :-1].reshape(-1, experiment.model.vocab_size),
            token_ids[:, 1:].reshape(-1),
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        processed_tokens += token_ids.numel()
        if step % experiment.train.log_every == 0 or step == experiment.train.steps:
            validation_objective: float | None = None
            validation_language_model_loss: float | None = None
            if corpus is not None:
                model.eval()
                with torch.no_grad():
                    validation_tokens = _corpus_batch(
                        corpus[1],
                        experiment.train.batch_size,
                        experiment.model.context_length,
                        validation_generator,
                        device,
                    )
                    validation_total, validation_output = model.loss(
                        validation_tokens
                    )
                    validation_objective = float(validation_total.detach().cpu())
                    validation_language_model_loss = float(
                        torch.nn.functional.cross_entropy(
                            validation_output.logits[:, :-1].reshape(
                                -1, experiment.model.vocab_size
                            ),
                            validation_tokens[:, 1:].reshape(-1),
                        )
                        .detach()
                        .cpu()
                    )
                model.train()
            history.append(
                {
                    "step": step,
                    "optimization_loss": float(loss.detach().cpu()),
                    "language_model_loss": float(
                        language_model_loss.detach().cpu()
                    ),
                    "auxiliary_loss": float(
                        model_output.auxiliary_loss.detach().cpu()
                    ),
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                    "validation_objective": validation_objective,
                    "validation_language_model_loss": (
                        validation_language_model_loss
                    ),
                    "routing": aggregate_routing_metrics(
                        model_output.routing_metrics
                    ),
                }
            )
    synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        "schema_version": 1,
        "result_kind": "measured",
        "workload": (
            "deterministic_synthetic_next_token_training"
            if corpus is None
            else "fixed_uint16_corpus_next_token_training"
        ),
        "config": {
            "model": asdict(experiment.model),
            "train": asdict(experiment.train),
        },
        "environment": environment_metadata(device),
        "measurements": {
            "elapsed_seconds": elapsed,
            "processed_tokens": processed_tokens,
            "tokens_per_second": processed_tokens / elapsed,
            "parameter_counts": model.parameter_counts().to_dict(),
            "history": history,
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_config": asdict(experiment.model),
                "model_state_dict": model.state_dict(),
                "step": experiment.train.steps,
            },
            checkpoint,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on deterministic synthetic tokens")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--architecture", choices=("dense", "moe"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--corpus", type=Path)
    arguments = parser.parse_args()
    result = run(
        arguments.config,
        arguments.output,
        arguments.checkpoint,
        architecture=(
            cast(Architecture, arguments.architecture)
            if arguments.architecture is not None
            else None
        ),
        seed=arguments.seed,
        corpus_path=arguments.corpus,
    )
    print(json.dumps(result["measurements"], sort_keys=True))


if __name__ == "__main__":
    main()
