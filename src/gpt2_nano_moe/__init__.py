"""CapSel-2 routing study for a small GPT-style MoE model."""

from .config import ModelConfig, load_config
from .model import GPT, ModelOutput

__all__ = [
    "GPT",
    "ModelConfig",
    "ModelOutput",
    "load_config",
]

__version__ = "0.1.0rc1"
