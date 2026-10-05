from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .metrics import RoutingMetrics

RoutingMode = Literal["top1_fc", "top1_bpr", "top2_fixed", "top2_capsel"]


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


def score_prioritized_capacity_mask(
    expert_indices: torch.Tensor,
    scores: torch.Tensor,
    expert_count: int,
    capacity: int,
) -> torch.Tensor:
    """Admit tokens in descending score order per expert (Batch Priority Routing)."""
    if expert_count <= 0 or capacity <= 0:
        raise ValueError("expert_count and capacity must be positive")
    if expert_indices.shape != scores.shape or expert_indices.ndim != 1:
        raise ValueError("expert_indices and scores must be 1D with matching shape")
    if expert_indices.numel() and (
        int(expert_indices.min().item()) < 0
        or int(expert_indices.max().item()) >= expert_count
    ):
        raise ValueError("expert index outside configured expert_count")
    T = expert_indices.shape[0]
    if T == 0:
        return torch.zeros(0, dtype=torch.bool, device=expert_indices.device)
    sort_order = torch.argsort(scores, descending=True)
    sorted_indices = expert_indices[sort_order]
    sorted_accepted = deterministic_capacity_mask(sorted_indices, expert_count, capacity)
    accepted = torch.zeros(T, dtype=torch.bool, device=expert_indices.device)
    accepted[sort_order] = sorted_accepted
    return accepted


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
        accepted_slots = int(accepted.sum().item())
        metrics = RoutingMetrics(
            capacity=capacity,
            tokens=token_count,
            dropped_tokens=int((~accepted).sum().item()),
            expert_counts=accepted_counts,
            router_entropy=entropy,
            expert_utilization=utilization,
            routed_slots=token_count,
            accepted_slots=accepted_slots,
            slot_drop_fraction=(token_count - accepted_slots) / max(token_count, 1),
            effective_experts_per_token=accepted_slots / max(token_count, 1),
        )
        return RouterOutput(
            probabilities=probabilities,
            expert_indices=expert_indices,
            gates=gates.to(inputs.dtype),
            accepted=accepted,
            auxiliary_loss=auxiliary_loss,
            metrics=metrics,
        )


def _auxiliary_loss(
    probabilities: torch.Tensor, primary_indices: torch.Tensor, expert_count: int
) -> torch.Tensor:
    assignment_fraction = F.one_hot(primary_indices, expert_count).float().mean(dim=0)
    probability_fraction = probabilities.mean(dim=0)
    return expert_count * torch.sum(assignment_fraction * probability_fraction)


def _margin_stats(probabilities: torch.Tensor) -> tuple[float, float]:
    """Returns (mean, std) of the top-1 minus top-2 probability margin."""
    T, E = probabilities.shape
    if E < 2:
        return 0.0, 0.0
    top2, _ = torch.topk(probabilities.detach().float(), 2, dim=-1)
    margin = top2[:, 0] - top2[:, 1]
    std = float(margin.std(unbiased=False).item()) if T > 1 else 0.0
    return float(margin.mean().item()), std


def _normalize_gates_over_accepted(
    raw_probs: torch.Tensor, accepted: torch.Tensor
) -> torch.Tensor:
    """Normalize raw probabilities over accepted slots per token, zero elsewhere."""
    gate_mask = accepted.float()
    masked = raw_probs.float() * gate_mask
    gate_sum = masked.sum(dim=1, keepdim=True)
    has_route = gate_sum.gt(0)
    return torch.where(has_route, masked / gate_sum.clamp_min(1e-9), torch.zeros_like(masked))


def _build_multi_metrics(
    T: int,
    E: int,
    capacity: int,
    probabilities: torch.Tensor,
    expert_indices_2d: torch.Tensor,
    accepted_2d: torch.Tensor,
    extra: dict,
) -> RoutingMetrics:
    routed_slots = int(extra.get("routed_slots", expert_indices_2d.numel()))
    accepted_slots = int(accepted_2d.sum().item())
    dropped_tokens = int((~accepted_2d.any(dim=1)).sum().item())
    flat_accepted = accepted_2d.reshape(-1)
    flat_expert = expert_indices_2d.reshape(-1)
    accepted_expert_counts = torch.bincount(flat_expert[flat_accepted], minlength=E)
    entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(-1).mean()
    utilization = accepted_expert_counts.gt(0).float().mean()
    margin_mean, margin_std = _margin_stats(probabilities)
    effective = float(accepted_2d.float().sum(dim=1).mean().item())
    slot_drop = (routed_slots - accepted_slots) / max(routed_slots, 1)
    return RoutingMetrics(
        capacity=capacity,
        tokens=T,
        dropped_tokens=dropped_tokens,
        expert_counts=accepted_expert_counts,
        router_entropy=entropy,
        expert_utilization=utilization,
        slot_drop_fraction=slot_drop,
        effective_experts_per_token=effective,
        uncertainty_rate=extra.get("uncertainty_rate"),
        rescue_rate=extra.get("rescue_rate"),
        router_margin_mean=margin_mean,
        router_margin_std=margin_std,
        routed_slots=routed_slots,
        accepted_slots=accepted_slots,
        rescued_tokens=extra.get("rescued_tokens"),
        rejected_primary_tokens=extra.get("rejected_primary_tokens"),
    )


class GeneralRouter(nn.Module):
    """Route with score-prioritized admission and one capacity per expert.

    Fixed top-2 admits primary requests, then all second-choice requests.
    CapSel-2 admits primary requests, rescue requests for rejected primaries,
    then optional second routes for uncertain accepted primaries. Every pass
    consumes the same per-expert capacity.
    """

    def __init__(
        self,
        d_model: int,
        expert_count: int,
        capacity_factor: float,
        mode: RoutingMode,
        margin_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model <= 0 or expert_count <= 0 or capacity_factor <= 0:
            raise ValueError("d_model, expert_count, and capacity_factor must be positive")
        if mode not in ("top1_bpr", "top2_fixed", "top2_capsel"):
            raise ValueError("GeneralRouter supports top1_bpr, top2_fixed, top2_capsel")
        if expert_count < 2 and mode in ("top2_fixed", "top2_capsel"):
            raise ValueError("top2 routing modes require at least 2 experts")
        self.expert_count = expert_count
        self.capacity_factor = capacity_factor
        self.mode = mode
        self.margin_threshold = margin_threshold
        self.projection = nn.Linear(d_model, expert_count, bias=False)

    def forward(self, inputs: torch.Tensor) -> RouterOutput:
        if inputs.ndim != 2:
            raise ValueError("router inputs must have shape [tokens, d_model]")
        T = inputs.shape[0]
        E = self.expert_count
        probabilities = torch.softmax(self.projection(inputs).float(), dim=-1)
        capacity = expert_capacity(T, E, self.capacity_factor)

        if self.mode == "top1_bpr":
            return self._route_top1_bpr(inputs, probabilities, T, E, capacity)
        if self.mode == "top2_fixed":
            return self._route_top2_fixed(inputs, probabilities, T, E, capacity)
        return self._route_top2_capsel(inputs, probabilities, T, E, capacity)

    def _route_top1_bpr(
        self,
        inputs: torch.Tensor,
        probabilities: torch.Tensor,
        T: int,
        E: int,
        capacity: int,
    ) -> RouterOutput:
        gates, expert_indices = probabilities.max(dim=-1)
        accepted = score_prioritized_capacity_mask(
            expert_indices, gates.detach(), E, capacity
        )
        aux = _auxiliary_loss(probabilities, expert_indices, E)
        accepted_counts = torch.bincount(expert_indices[accepted], minlength=E)
        entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(-1).mean()
        utilization = accepted_counts.gt(0).float().mean()
        accepted_slots = int(accepted.sum().item())
        margin_mean, margin_std = _margin_stats(probabilities)
        metrics = RoutingMetrics(
            capacity=capacity,
            tokens=T,
            dropped_tokens=int((~accepted).sum().item()),
            expert_counts=accepted_counts,
            router_entropy=entropy,
            expert_utilization=utilization,
            routed_slots=T,
            accepted_slots=accepted_slots,
            slot_drop_fraction=(T - accepted_slots) / max(T, 1),
            effective_experts_per_token=accepted_slots / max(T, 1),
            router_margin_mean=margin_mean,
            router_margin_std=margin_std,
        )
        return RouterOutput(
            probabilities=probabilities,
            expert_indices=expert_indices,
            gates=gates.to(inputs.dtype),
            accepted=accepted,
            auxiliary_loss=aux,
            metrics=metrics,
        )

    def _route_top2_fixed(
        self,
        inputs: torch.Tensor,
        probabilities: torch.Tensor,
        T: int,
        E: int,
        capacity: int,
    ) -> RouterOutput:
        top2_probs, top2_idx = torch.topk(probabilities, 2, dim=-1)
        e1 = top2_idx[:, 0]
        e2 = top2_idx[:, 1]

        # Primary requests and all second-choice requests share one capacity.
        primary_accepted = score_prioritized_capacity_mask(
            e1, top2_probs[:, 0].detach(), E, capacity
        )
        expert_fill = torch.bincount(
            e1[primary_accepted], minlength=E
        )
        secondary_order = torch.argsort(top2_probs[:, 1].detach(), descending=True)
        ordered_e2 = e2[secondary_order]
        one_hot = F.one_hot(ordered_e2, E).long()
        cumulative = one_hot.cumsum(dim=0)
        position_within_expert = (one_hot * cumulative).sum(dim=-1)
        remaining = (capacity - expert_fill[ordered_e2]).clamp_min(0)
        ordered_secondary_accepted = position_within_expert.le(remaining)
        secondary_accepted = torch.zeros(T, dtype=torch.bool, device=probabilities.device)
        secondary_accepted[secondary_order] = ordered_secondary_accepted
        accepted_2d = torch.stack([primary_accepted, secondary_accepted], dim=1)
        gates_2d = _normalize_gates_over_accepted(top2_probs, accepted_2d)
        aux = _auxiliary_loss(probabilities, e1, E)
        metrics = _build_multi_metrics(
            T,
            E,
            capacity,
            probabilities,
            top2_idx,
            accepted_2d,
            {"routed_slots": 2 * T},
        )
        return RouterOutput(
            probabilities=probabilities,
            expert_indices=top2_idx,
            gates=gates_2d.to(inputs.dtype),
            accepted=accepted_2d,
            auxiliary_loss=aux,
            metrics=metrics,
        )

    def _route_top2_capsel(
        self,
        inputs: torch.Tensor,
        probabilities: torch.Tensor,
        T: int,
        E: int,
        capacity: int,
    ) -> RouterOutput:
        dev = probabilities.device
        top2_probs, top2_idx = torch.topk(probabilities, 2, dim=-1)
        p1 = top2_probs[:, 0]
        p2 = top2_probs[:, 1]
        e1 = top2_idx[:, 0]
        e2 = top2_idx[:, 1]
        margin = (p1 - p2).detach()

        # Pass 1: admit primary requests by router score.
        primary_accepted = score_prioritized_capacity_mask(e1, p1.detach(), E, capacity)

        expert_fill = torch.zeros(E, dtype=torch.long, device=dev)
        if primary_accepted.any():
            expert_fill.add_(torch.bincount(e1[primary_accepted], minlength=E))

        # Pass 2: rescue rejected primaries through their second-choice expert.
        rejected_indices = (~primary_accepted).nonzero(as_tuple=False).squeeze(1)
        R = rejected_indices.shape[0]
        if R > 0:
            rejected_p2 = p2[rejected_indices].detach()
            sort_r = torch.argsort(rejected_p2, descending=True)
            sorted_rejected = rejected_indices[sort_r]
            sorted_e2_r = e2[sorted_rejected]
            one_hot_r = F.one_hot(sorted_e2_r, E).long()
            cumcount_r = one_hot_r.cumsum(dim=0)
            token_cumcount_r = (one_hot_r * cumcount_r).sum(dim=-1)
            remaining_r = (capacity - expert_fill[sorted_e2_r]).clamp_min(0)
            sorted_rescue = token_cumcount_r.le(remaining_r)
            rescue_accepted = torch.zeros(T, dtype=torch.bool, device=dev)
            if sorted_rescue.any():
                rescue_accepted[sorted_rejected[sorted_rescue]] = True
        else:
            rescue_accepted = torch.zeros(T, dtype=torch.bool, device=dev)

        if rescue_accepted.any():
            expert_fill.add_(torch.bincount(e2[rescue_accepted], minlength=E))

        # Pass 3: use remaining capacity for optional routes, most uncertain first.
        uncertain_mask = primary_accepted & margin.lt(self.margin_threshold)
        uncertain_indices = uncertain_mask.nonzero(as_tuple=False).squeeze(1)
        U = uncertain_indices.shape[0]
        if U > 0:
            uncertain_margin = margin[uncertain_indices]
            sort_u = torch.argsort(uncertain_margin)  # ascending: most uncertain first
            sorted_uncertain = uncertain_indices[sort_u]
            sorted_e2_u = e2[sorted_uncertain]
            one_hot_u = F.one_hot(sorted_e2_u, E).long()
            cumcount_u = one_hot_u.cumsum(dim=0)
            token_cumcount_u = (one_hot_u * cumcount_u).sum(dim=-1)
            remaining_u = (capacity - expert_fill[sorted_e2_u]).clamp_min(0)
            sorted_optional = token_cumcount_u.le(remaining_u)
            optional_accepted = torch.zeros(T, dtype=torch.bool, device=dev)
            if sorted_optional.any():
                optional_accepted[sorted_uncertain[sorted_optional]] = True
        else:
            optional_accepted = torch.zeros(T, dtype=torch.bool, device=dev)

        # Slot 1 is the second-choice expert used for rescue or an optional route.
        slot1_accepted = rescue_accepted | optional_accepted
        accepted_2d = torch.stack([primary_accepted, slot1_accepted], dim=1)
        gates_2d = _normalize_gates_over_accepted(top2_probs, accepted_2d)
        aux = _auxiliary_loss(probabilities, e1, E)
        uncertainty_rate = float(margin.lt(self.margin_threshold).float().mean().item())
        rescue_rate = float(rescue_accepted.float().sum().item()) / max(R, 1)
        second_route_requested = (~primary_accepted) | uncertain_mask
        extra = {
            "uncertainty_rate": uncertainty_rate,
            "rescue_rate": rescue_rate,
            "routed_slots": T + int(second_route_requested.sum().item()),
            "rescued_tokens": int(rescue_accepted.sum().item()),
            "rejected_primary_tokens": R,
        }
        metrics = _build_multi_metrics(T, E, capacity, probabilities, top2_idx, accepted_2d, extra)
        return RouterOutput(
            probabilities=probabilities,
            expert_indices=top2_idx,
            gates=gates_2d.to(inputs.dtype),
            accepted=accepted_2d,
            auxiliary_loss=aux,
            metrics=metrics,
        )
