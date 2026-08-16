from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, TypeVar
import tomllib

Architecture = Literal["dense", "moe"]


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    context_length: int = 64
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    mlp_multiplier: int = 4
    dropout: float = 0.0
    architecture: Architecture = "moe"
    expert_count: int = 4
    capacity_factor: float = 1.25
    auxiliary_loss_weight: float = 0.01

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "mlp_multiplier": self.mlp_multiplier,
            "expert_count": self.expert_count,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.architecture not in ("dense", "moe"):
            raise ValueError("architecture must be 'dense' or 'moe'")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.capacity_factor <= 0.0:
            raise ValueError("capacity_factor must be positive")
        if self.auxiliary_loss_weight < 0.0:
            raise ValueError("auxiliary_loss_weight must be non-negative")


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 17
    steps: int = 10
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    log_every: int = 1
    corpus_path: str | None = None
    validation_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0 or self.log_every <= 0:
            raise ValueError("steps, batch_size, and log_every must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.device not in ("auto", "cpu", "mps", "cuda"):
            raise ValueError("device must be auto, cpu, mps, or cuda")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    train: TrainConfig


T = TypeVar("T")


def _construct(cls: type[T], values: dict[str, Any]) -> T:
    allowed = {field.name for field in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("rb") as handle:
        data = tomllib.load(handle)
    unknown = set(data) - {"model", "train"}
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
    return ExperimentConfig(
        model=_construct(ModelConfig, data.get("model", {})),
        train=_construct(TrainConfig, data.get("train", {})),
    )
