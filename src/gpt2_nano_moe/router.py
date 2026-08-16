from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .metrics import RoutingMetrics


def expert_capacity(
    token_count: int, expert_count: int, capacity_factor: float
) -> int:
    if token_count < 0 or expert_count <= 0 or capacity_factor <= 0:
        raise ValueError("token_count must be non-negative and expert capacity positive")
    return max(1, math.ceil(capacity_factor * token_count / expert_count))


def deterministic_capacity_mask(
    expert_indices: torch.Tensor, expert_count: int, capacity: int
) -> torch.Tensor:
    if expert_count <= 0 or capacity <= 0:
        raise ValueError("expert_count and capacity must be positive")
    if expert_indices.numel() and (
        int(expert_indices.min().item()) < 0
        or int(expert_indices.max().item()) >= expert_count
    ):
        raise ValueError("expert index outside configured expert_count")
    flat_indices = expert_indices.reshape(-1)
    accepted = torch.zeros(
        flat_indices.shape[0], dtype=torch.bool, device=expert_indices.device
    )
    for expert_id in range(expert_count):
        positions = torch.nonzero(flat_indices.eq(expert_id), as_tuple=False).flatten()
        accepted[positions[:capacity]] = True
    return accepted.reshape_as(expert_indices)


@dataclass(frozen=True)
class RouterOutput:
    probabilities: torch.Tensor
    expert_indices: torch.Tensor
    gates: torch.Tensor
    accepted: torch.Tensor
    auxiliary_loss: torch.Tensor
    metrics: RoutingMetrics


class Top1Router(nn.Module):
    """Typed top-1 router with deterministic capacity admission by token order."""

    def __init__(self, d_model: int, expert_count: int, capacity_factor: float) -> None:
        super().__init__()
        if d_model <= 0 or expert_count <= 0 or capacity_factor <= 0:
            raise ValueError("router dimensions and capacity_factor must be positive")
        self.expert_count = expert_count
        self.capacity_factor = capacity_factor
        self.projection = nn.Linear(d_model, expert_count, bias=False)

    def forward(self, inputs: torch.Tensor) -> RouterOutput:
        if inputs.ndim != 2:
            raise ValueError("router inputs must have shape [tokens, d_model]")
        probabilities = torch.softmax(self.projection(inputs).float(), dim=-1)
        gates, expert_indices = probabilities.max(dim=-1)
        token_count = inputs.shape[0]
        capacity = expert_capacity(
            token_count, self.expert_count, self.capacity_factor
        )
        accepted = deterministic_capacity_mask(
            expert_indices, self.expert_count, capacity
        )

        assignment_fraction = torch.nn.functional.one_hot(
            expert_indices, self.expert_count
        ).float().mean(dim=0)
        probability_fraction = probabilities.mean(dim=0)
        auxiliary_loss = self.expert_count * torch.sum(
            assignment_fraction * probability_fraction
        )
        accepted_counts = torch.bincount(
            expert_indices[accepted], minlength=self.expert_count
        )
        entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(-1).mean()
        utilization = accepted_counts.gt(0).float().mean()
        metrics = RoutingMetrics(
            capacity=capacity,
            tokens=token_count,
            dropped_tokens=int((~accepted).sum().item()),
            expert_counts=accepted_counts,
            router_entropy=entropy,
            expert_utilization=utilization,
        )
        return RouterOutput(
            probabilities=probabilities,
            expert_indices=expert_indices,
            gates=gates.to(inputs.dtype),
            accepted=accepted,
            auxiliary_loss=auxiliary_loss,
            metrics=metrics,
        )
