from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class RoutingMetrics:
    capacity: int
    tokens: int
    dropped_tokens: int
    expert_counts: torch.Tensor
    router_entropy: torch.Tensor
    expert_utilization: torch.Tensor

    @property
    def dropped_fraction(self) -> float:
        return self.dropped_tokens / max(self.tokens, 1)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["expert_counts"] = self.expert_counts.detach().cpu().tolist()
        result["router_entropy"] = float(self.router_entropy.detach().cpu())
        result["expert_utilization"] = float(self.expert_utilization.detach().cpu())
        result["dropped_fraction"] = self.dropped_fraction
        return result


def aggregate_routing_metrics(metrics: tuple[RoutingMetrics, ...]) -> dict[str, Any]:
    if not metrics:
        return {
            "token_assignments": 0,
            "dropped_tokens": 0,
            "dropped_fraction": 0.0,
            "expert_counts": [],
            "router_entropy": None,
            "expert_utilization": None,
        }
    tokens = sum(item.tokens for item in metrics)
    dropped = sum(item.dropped_tokens for item in metrics)
    expert_counts = torch.stack(
        [item.expert_counts.detach().cpu() for item in metrics]
    ).sum(dim=0)
    return {
        "token_assignments": tokens,
        "dropped_tokens": dropped,
        "dropped_fraction": dropped / max(tokens, 1),
        "expert_counts": expert_counts.tolist(),
        "router_entropy": sum(float(item.router_entropy.detach().cpu()) for item in metrics)
        / len(metrics),
        "expert_utilization": sum(
            float(item.expert_utilization.detach().cpu()) for item in metrics
        )
        / len(metrics),
    }
