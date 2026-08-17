from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .experts import Experts
from .metrics import RoutingMetrics
from .router import GeneralRouter, RoutingMode, Top1Router


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
        routing_mode: RoutingMode = "top1_fc",
        capsel_margin_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        self.routing_mode = routing_mode
        if routing_mode == "top1_fc":
            self.router: Top1Router | GeneralRouter = Top1Router(
                d_model, expert_count, capacity_factor
            )
        else:
            self.router = GeneralRouter(
                d_model,
                expert_count,
                capacity_factor,
                routing_mode,
                capsel_margin_threshold,
            )
        self.experts = Experts(expert_count, d_model, hidden_size, dropout)

    def forward(self, inputs: torch.Tensor) -> MoEOutput:
        shape = inputs.shape
        flat = inputs.reshape(-1, shape[-1])
        route = self.router(flat)

        if self.routing_mode in ("top1_fc", "top1_bpr"):
            # expert_indices and accepted are 1D [T]; gates are raw probabilities.
            expert_output = self.experts(flat, route.expert_indices, route.accepted)
            expert_output = expert_output * route.gates.unsqueeze(-1)
        else:
            # top2_fixed, top2_capsel: expert_indices/accepted/gates are 2D [T, 2].
            expert_output = self.experts.forward_multi(
                flat, route.expert_indices, route.accepted, route.gates
            )

        return MoEOutput(
            hidden_states=expert_output.reshape(shape),
            auxiliary_loss=route.auxiliary_loss,
            metrics=route.metrics,
            expert_indices=route.expert_indices.reshape(shape[:-1] + route.expert_indices.shape[1:]),
            accepted=route.accepted.reshape(shape[:-1] + route.accepted.shape[1:]),
        )

