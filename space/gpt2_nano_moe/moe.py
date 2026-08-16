from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .experts import Experts
from .metrics import RoutingMetrics
from .router import Top1Router


@dataclass(frozen=True)
class MoEOutput:
    hidden_states: torch.Tensor
    auxiliary_loss: torch.Tensor
    metrics: RoutingMetrics
    expert_indices: torch.Tensor
    accepted: torch.Tensor


class SparseMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        expert_count: int,
        capacity_factor: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.router = Top1Router(d_model, expert_count, capacity_factor)
        self.experts = Experts(expert_count, d_model, hidden_size, dropout)

    def forward(self, inputs: torch.Tensor) -> MoEOutput:
        shape = inputs.shape
        flat = inputs.reshape(-1, shape[-1])
        route = self.router(flat)
        expert_output = self.experts(flat, route.expert_indices, route.accepted)
        expert_output = expert_output * route.gates.unsqueeze(-1)
        return MoEOutput(
            hidden_states=expert_output.reshape(shape),
            auxiliary_loss=route.auxiliary_loss,
            metrics=route.metrics,
            expert_indices=route.expert_indices.reshape(shape[:-1]),
            accepted=route.accepted.reshape(shape[:-1]),
        )

