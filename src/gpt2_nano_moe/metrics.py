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
    # Extended fields; None for legacy Top1Router outputs.
    slot_drop_fraction: float | None = None
    effective_experts_per_token: float | None = None
    uncertainty_rate: float | None = None
    rescue_rate: float | None = None
    router_margin_mean: float | None = None
    router_margin_std: float | None = None
    routed_slots: int | None = None
    accepted_slots: int | None = None
    rescued_tokens: int | None = None
    rejected_primary_tokens: int | None = None

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


def _mean_of_present(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate_routing_metrics(metrics: tuple[RoutingMetrics, ...]) -> dict[str, Any]:
    if not metrics:
        return {
            "token_assignments": 0,
            "dropped_tokens": 0,
            "dropped_fraction": 0.0,
            "expert_counts": [],
            "router_entropy": None,
            "expert_utilization": None,
            "slot_drop_fraction": None,
            "effective_experts_per_token": None,
            "uncertainty_rate": None,
            "rescue_rate": None,
            "router_margin_mean": None,
            "router_margin_std": None,
            "routed_slots": None,
            "accepted_slots": None,
            "rescued_tokens": None,
            "rejected_primary_tokens": None,
        }
    tokens = sum(item.tokens for item in metrics)
    dropped = sum(item.dropped_tokens for item in metrics)
    expert_counts = torch.stack(
        [item.expert_counts.detach().cpu() for item in metrics]
    ).sum(dim=0)
    routed_slots_list = [item.routed_slots for item in metrics if item.routed_slots is not None]
    accepted_slots_list = [item.accepted_slots for item in metrics if item.accepted_slots is not None]
    total_routed = sum(routed_slots_list) if routed_slots_list else None
    total_accepted = sum(accepted_slots_list) if accepted_slots_list else None
    rescued_list = [
        item.rescued_tokens
        for item in metrics
        if item.rescued_tokens is not None
    ]
    rejected_primary_list = [
        item.rejected_primary_tokens
        for item in metrics
        if item.rejected_primary_tokens is not None
    ]
    rescued = sum(rescued_list) if rescued_list else None
    rejected_primary = (
        sum(rejected_primary_list) if rejected_primary_list else None
    )
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
        "slot_drop_fraction": (
            (total_routed - total_accepted) / max(total_routed, 1)
            if total_routed is not None and total_accepted is not None
            else None
        ),
        "effective_experts_per_token": _mean_of_present(
            [item.effective_experts_per_token for item in metrics
             if item.effective_experts_per_token is not None]
        ),
        "uncertainty_rate": _mean_of_present(
            [item.uncertainty_rate for item in metrics if item.uncertainty_rate is not None]
        ),
        "rescue_rate": (
            rescued / max(rejected_primary, 1)
            if rescued is not None and rejected_primary is not None
            else None
        ),
        "router_margin_mean": _mean_of_present(
            [item.router_margin_mean for item in metrics if item.router_margin_mean is not None]
        ),
        "router_margin_std": _mean_of_present(
            [item.router_margin_std for item in metrics if item.router_margin_std is not None]
        ),
        "routed_slots": total_routed,
        "accepted_slots": total_accepted,
        "rescued_tokens": rescued,
        "rejected_primary_tokens": rejected_primary,
    }
