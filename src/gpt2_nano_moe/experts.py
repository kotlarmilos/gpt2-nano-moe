from __future__ import annotations

import torch
from torch import nn

from .dense import DenseMLP


class Experts(nn.Module):
    def __init__(
        self, expert_count: int, d_model: int, hidden_size: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            DenseMLP(d_model, hidden_size, dropout) for _ in range(expert_count)
        )

    def forward(
        self,
        inputs: torch.Tensor,
        expert_indices: torch.Tensor,
        accepted: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(inputs)
        for expert_id, expert in enumerate(self.experts):
            selected = accepted & expert_indices.eq(expert_id)
            if selected.any():
                output[selected] = expert(inputs[selected])
        return output

