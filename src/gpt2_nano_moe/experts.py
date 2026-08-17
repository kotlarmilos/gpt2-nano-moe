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

    def forward_multi(
        self,
        inputs: torch.Tensor,
        expert_indices: torch.Tensor,
        accepted: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch tokens to multiple expert slots and combine with normalized gates.

        Args:
            inputs: [T, d_model]
            expert_indices: [T, k] integer expert assignments
            accepted: [T, k] bool mask of admitted slots
            gates: [T, k] float normalized gate weights (sum to 1 per token over accepted slots)

        Returns:
            [T, d_model] combined expert output (zero for completely unrouted tokens)
        """
        output = torch.zeros_like(inputs)
        _T, k = expert_indices.shape
        for slot in range(k):
            slot_indices = expert_indices[:, slot]
            slot_accepted = accepted[:, slot]
            slot_gates = gates[:, slot]
            for expert_id, expert in enumerate(self.experts):
                selected = slot_accepted & slot_indices.eq(expert_id)
                if selected.any():
                    output[selected] = output[selected] + (
                        expert(inputs[selected]) * slot_gates[selected].unsqueeze(-1)
                    )
        return output

