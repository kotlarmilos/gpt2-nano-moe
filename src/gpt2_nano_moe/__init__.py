"""Small GPT-compatible dense and sparse MoE models."""

from .config import ExperimentConfig, ModelConfig, TrainConfig, load_config
from .model import GPT, ModelOutput, ParameterCounts

__all__ = [
    "ExperimentConfig",
    "GPT",
    "ModelConfig",
    "ModelOutput",
    "ParameterCounts",
    "TrainConfig",
    "load_config",
]

__version__ = "0.1.0rc1"

