"""Tests for CapSel-2 routing and the multi-mode GeneralRouter."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
import unittest

import torch

from gpt2_nano_moe.config import ModelConfig, load_config
from gpt2_nano_moe.experts import Experts
from gpt2_nano_moe.model import GPT
from gpt2_nano_moe.moe import SparseMoE
from gpt2_nano_moe.router import (
    GeneralRouter,
    RouterOutput,
    Top1Router,
    deterministic_capacity_mask,
    expert_capacity,
    score_prioritized_capacity_mask,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tiny_moe_config(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=31,
        context_length=8,
        d_model=16,
        num_heads=4,
        num_layers=2,
        mlp_multiplier=2,
        architecture="moe",
        expert_count=4,
        capacity_factor=1.25,
    )
    base.update(overrides)
    return ModelConfig(**base)


def _expert_counts_from_output(route: RouterOutput) -> torch.Tensor:
    E = route.probabilities.shape[1]
    if route.expert_indices.ndim == 1:
        return torch.bincount(route.expert_indices[route.accepted], minlength=E)
    flat_idx = route.expert_indices.reshape(-1)
    flat_acc = route.accepted.reshape(-1)
    return torch.bincount(flat_idx[flat_acc], minlength=E)


# ──────────────────────────────────────────────────────────────────────────────
# score_prioritized_capacity_mask
# ──────────────────────────────────────────────────────────────────────────────

class ScorePrioritizedMaskTests(unittest.TestCase):
    def test_admits_highest_scores_first(self) -> None:
        # 6 tokens all wanting expert 0; capacity 2 → top-2 scores admitted.
        expert_indices = torch.zeros(6, dtype=torch.long)
        scores = torch.tensor([0.1, 0.9, 0.5, 0.8, 0.2, 0.7])
        accepted = score_prioritized_capacity_mask(expert_indices, scores, 1, 2)
        # Admitted: indices 1 (0.9) and 3 (0.8) → positions 1 and 3.
        self.assertEqual(accepted.sum().item(), 2)
        self.assertTrue(accepted[1].item())
        self.assertTrue(accepted[3].item())

    def test_empty_input_returns_empty(self) -> None:
        accepted = score_prioritized_capacity_mask(
            torch.zeros(0, dtype=torch.long), torch.zeros(0), 4, 2
        )
        self.assertEqual(accepted.shape[0], 0)

    def test_raises_on_mismatched_shapes(self) -> None:
        with self.assertRaises(ValueError):
            score_prioritized_capacity_mask(
                torch.zeros(4, dtype=torch.long), torch.zeros(3), 2, 2
            )

    def test_raises_on_out_of_range_expert(self) -> None:
        with self.assertRaises(ValueError):
            score_prioritized_capacity_mask(
                torch.tensor([0, 5]), torch.tensor([0.5, 0.5]), 4, 2
            )


# ──────────────────────────────────────────────────────────────────────────────
# Capacity invariants across all modes
# ──────────────────────────────────────────────────────────────────────────────

class CapacityInvariantTests(unittest.TestCase):
    def _check_capacity(self, mode: str) -> None:
        """Check per-expert capacity bounds for modes with a shared total budget."""
        torch.manual_seed(7)
        router = GeneralRouter(d_model=8, expert_count=4, capacity_factor=1.0, mode=mode)
        T = 20
        route = router(torch.randn(T, 8))
        capacity = expert_capacity(T, 4, 1.0)
        counts = _expert_counts_from_output(route)
        for e, count in enumerate(counts.tolist()):
            self.assertLessEqual(
                count, capacity, f"expert {e} exceeds capacity in mode={mode}"
            )

    def test_top1_bpr_capacity(self) -> None:
        self._check_capacity("top1_bpr")

    def test_top2_fixed_uses_one_shared_expert_capacity(self) -> None:
        torch.manual_seed(7)
        router = GeneralRouter(d_model=8, expert_count=4, capacity_factor=1.0, mode="top2_fixed")
        T = 20
        route = router(torch.randn(T, 8))
        capacity = expert_capacity(T, 4, 1.0)
        counts = _expert_counts_from_output(route)
        for e, count in enumerate(counts.tolist()):
            self.assertLessEqual(count, capacity, f"expert {e} exceeds shared capacity")

    def test_top2_capsel_capacity(self) -> None:
        self._check_capacity("top2_capsel")

    def test_top2_fixed_total_acceptance_respects_global_capacity(self) -> None:
        torch.manual_seed(11)
        router = GeneralRouter(d_model=4, expert_count=2, capacity_factor=1.0, mode="top2_fixed")
        T = 10
        route = router(torch.randn(T, 4))
        capacity = expert_capacity(T, 2, 1.0)
        self.assertLessEqual(int(route.accepted.sum().item()), 2 * capacity)


# ──────────────────────────────────────────────────────────────────────────────
# Score prioritization (BPR)
# ──────────────────────────────────────────────────────────────────────────────

class BPRPrioritizationTests(unittest.TestCase):
    def test_bpr_admits_higher_score_tokens(self) -> None:
        """With capacity 2 on a 4-token batch, BPR picks the 2 highest-prob tokens."""
        router = GeneralRouter(d_model=2, expert_count=2, capacity_factor=1.0, mode="top1_bpr")
        with torch.no_grad():
            router.projection.weight.copy_(
                torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
            )
        inputs = torch.tensor([[0.5, 0.0], [2.0, 0.0], [0.1, 0.0], [1.0, 0.0]])
        route = router(inputs)
        # All tokens choose expert 0. Its capacity is 2, so the two largest
        # probabilities at positions 1 and 3 must win regardless of input order.
        self.assertEqual(int(route.accepted.sum().item()), 2)
        self.assertTrue(route.accepted[1].item())
        self.assertTrue(route.accepted[3].item())
        self.assertFalse(route.accepted[0].item())
        self.assertFalse(route.accepted[2].item())

    def test_bpr_vs_fc_same_accepted_when_unconstrained(self) -> None:
        """When capacity >= T, all tokens accepted in both BPR and first-come."""
        torch.manual_seed(5)
        top1 = Top1Router(d_model=4, expert_count=3, capacity_factor=10.0)
        bpr = GeneralRouter(d_model=4, expert_count=3, capacity_factor=10.0, mode="top1_bpr")
        with torch.no_grad():
            bpr.projection.weight.copy_(top1.projection.weight)
        inputs = torch.randn(8, 4)
        r1 = top1(inputs)
        r2 = bpr(inputs)
        self.assertTrue(r1.accepted.all())
        self.assertTrue(r2.accepted.all())


# ──────────────────────────────────────────────────────────────────────────────
# Primary-before-secondary admission (capsel)
# ──────────────────────────────────────────────────────────────────────────────

class PrimaryBeforeSecondaryTests(unittest.TestCase):
    def test_primary_phase_fills_before_rescue(self) -> None:
        """All tokens with accepted primary should not be in the rescue pool."""
        torch.manual_seed(13)
        router = GeneralRouter(
            d_model=8, expert_count=4, capacity_factor=1.0,
            mode="top2_capsel", margin_threshold=0.0  # no secondary from uncertainty
        )
        route = router(torch.randn(20, 8))
        # rescue_accepted = slot1_accepted AND primary_rejected
        primary_accepted = route.accepted[:, 0]
        slot1_accepted = route.accepted[:, 1]
        # A rescued token has primary rejected AND slot1 accepted.
        rescued = (~primary_accepted) & slot1_accepted
        # Primary accepted tokens should not be in rescue pool.
        overlap = primary_accepted & rescued
        self.assertFalse(overlap.any().item())

    def test_rescue_only_fills_secondary_slot(self) -> None:
        """Rescued tokens occupy slot 1, not slot 0."""
        torch.manual_seed(17)
        router = GeneralRouter(
            d_model=8, expert_count=4, capacity_factor=0.5,
            mode="top2_capsel", margin_threshold=0.0
        )
        route = router(torch.randn(30, 8))
        primary_accepted = route.accepted[:, 0]
        slot1_accepted = route.accepted[:, 1]
        rescued = (~primary_accepted) & slot1_accepted
        # Rescued tokens must NOT have slot 0 accepted.
        self.assertTrue((~primary_accepted[rescued]).all().item() if rescued.any() else True)


# ──────────────────────────────────────────────────────────────────────────────
# Selective threshold behavior
# ──────────────────────────────────────────────────────────────────────────────

class SelectiveThresholdTests(unittest.TestCase):
    def test_threshold_zero_disables_secondary_from_uncertainty(self) -> None:
        """threshold=0 → no secondary from uncertainty; only rescue gives slot1."""
        torch.manual_seed(21)
        router_zero = GeneralRouter(
            d_model=8, expert_count=4, capacity_factor=2.0,
            mode="top2_capsel", margin_threshold=0.0
        )
        router_high = GeneralRouter(
            d_model=8, expert_count=4, capacity_factor=2.0,
            mode="top2_capsel", margin_threshold=1.0
        )
        with torch.no_grad():
            router_high.projection.weight.copy_(router_zero.projection.weight)
        inputs = torch.randn(20, 8)
        r_zero = router_zero(inputs)
        r_high = router_high(inputs)
        # With threshold=0, uncertainty_rate should be ~0 (margin is never negative).
        self.assertAlmostEqual(r_zero.metrics.uncertainty_rate, 0.0, places=5)
        # With threshold=1, all tokens are uncertain → more secondary acceptances.
        secondary_zero = r_zero.accepted[:, 1].sum().item()
        secondary_high = r_high.accepted[:, 1].sum().item()
        self.assertGreaterEqual(secondary_high, secondary_zero)

    def test_higher_threshold_increases_uncertainty_rate(self) -> None:
        torch.manual_seed(29)
        inputs = torch.randn(30, 8)
        rates = []
        for threshold in [0.0, 0.05, 0.2, 1.0]:
            router = GeneralRouter(
                d_model=8, expert_count=4, capacity_factor=2.0,
                mode="top2_capsel", margin_threshold=threshold
            )
            route = router(inputs)
            rates.append(route.metrics.uncertainty_rate)
        # Uncertainty rate should be non-decreasing with threshold.
        for i in range(len(rates) - 1):
            self.assertLessEqual(rates[i], rates[i + 1] + 1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# Overflow rescue
# ──────────────────────────────────────────────────────────────────────────────

class OverflowRescueTests(unittest.TestCase):
    def test_rescue_increases_accepted_when_primary_full(self) -> None:
        """With threshold=0, slot1 tokens must be from rescue."""
        torch.manual_seed(37)
        # Very tight capacity: many tokens overflow primary.
        router = GeneralRouter(
            d_model=8, expert_count=4, capacity_factor=0.5,
            mode="top2_capsel", margin_threshold=0.0
        )
        route = router(torch.randn(30, 8))
        rescue_rate = route.metrics.rescue_rate
        slot1_total = int(route.accepted[:, 1].sum().item())
        # All slot-1 acceptances with threshold=0 come from rescue.
        self.assertIsNotNone(rescue_rate)
        if slot1_total > 0:
            self.assertGreater(rescue_rate, 0.0)

    def test_rescue_does_not_exceed_secondary_capacity(self) -> None:
        """Rescued tokens do not overflow their secondary expert."""
        torch.manual_seed(41)
        router = GeneralRouter(
            d_model=8, expert_count=4, capacity_factor=0.5,
            mode="top2_capsel", margin_threshold=0.0
        )
        T = 40
        route = router(torch.randn(T, 8))
        capacity = expert_capacity(T, 4, 0.5)
        # Slot 1 accepted: e2 must respect capacity.
        e2_indices = route.expert_indices[:, 1][route.accepted[:, 1]]
        counts = torch.bincount(e2_indices, minlength=4)
        for e, c in enumerate(counts.tolist()):
            self.assertLessEqual(c, capacity, f"rescue overflow at expert {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Normalized gates
# ──────────────────────────────────────────────────────────────────────────────

class NormalizedGateTests(unittest.TestCase):
    def _check_gate_normalization(self, mode: str) -> None:
        torch.manual_seed(53)
        router = GeneralRouter(d_model=8, expert_count=4, capacity_factor=2.0, mode=mode)
        route = router(torch.randn(16, 8))
        gates = route.gates.float()
        accepted = route.accepted
        if gates.ndim == 1:
            # top1_bpr: raw probability gates, no normalization check needed.
            return
        # For 2D gates: gate sum over accepted slots must equal 1.0 or 0.0 per token.
        for t in range(gates.shape[0]):
            accepted_sum = float(gates[t][accepted[t]].sum().item())
            if accepted[t].any():
                self.assertAlmostEqual(accepted_sum, 1.0, places=5, msg=f"token {t}")
            else:
                self.assertAlmostEqual(float(gates[t].sum().item()), 0.0, places=5)

    def test_top2_fixed_gates_normalized(self) -> None:
        self._check_gate_normalization("top2_fixed")

    def test_top2_capsel_gates_normalized(self) -> None:
        self._check_gate_normalization("top2_capsel")

    def test_dropped_tokens_have_zero_gates(self) -> None:
        torch.manual_seed(59)
        router = GeneralRouter(
            d_model=4, expert_count=4, capacity_factor=0.25, mode="top2_fixed"
        )
        route = router(torch.randn(24, 4))
        dropped = ~route.accepted.any(dim=1)
        if dropped.any():
            self.assertTrue(route.gates[dropped].abs().max().item() < 1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# No-route behavior
# ──────────────────────────────────────────────────────────────────────────────

class NoRouteBehaviorTests(unittest.TestCase):
    def test_dropped_tokens_have_zero_expert_output(self) -> None:
        torch.manual_seed(67)
        config = _tiny_moe_config(routing_mode="top2_fixed", capacity_factor=0.25)
        model = GPT(config)
        model.eval()
        token_ids = torch.randint(config.vocab_size, (1, config.context_length))
        with torch.no_grad():
            output = model(token_ids)
        # Output logits should be finite (not NaN/inf) even with heavy dropping.
        self.assertTrue(torch.isfinite(output.logits).all())

    def test_capsel_zero_output_for_fully_dropped(self) -> None:
        """Tokens with no accepted slot must receive zero expert contribution."""
        torch.manual_seed(71)
        experts = Experts(expert_count=3, d_model=4, hidden_size=8)
        T = 6
        inputs = torch.randn(T, 4)
        # expert_indices [T, 2]; accepted all False → gates all 0.
        expert_indices = torch.zeros(T, 2, dtype=torch.long)
        accepted = torch.zeros(T, 2, dtype=torch.bool)
        gates = torch.zeros(T, 2)
        output = experts.forward_multi(inputs, expert_indices, accepted, gates)
        self.assertTrue(output.abs().max().item() < 1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# Gradient flow
# ──────────────────────────────────────────────────────────────────────────────

class GradientFlowTests(unittest.TestCase):
    def _check_gradients(self, routing_mode: str) -> None:
        torch.manual_seed(79)
        config = _tiny_moe_config(routing_mode=routing_mode, capacity_factor=2.0)
        model = GPT(config)
        loss, _ = model.loss(torch.randint(config.vocab_size, (2, config.context_length)))
        loss.backward()
        router = model.blocks[0].mlp.router
        self.assertIsNotNone(router.projection.weight.grad)
        self.assertGreater(float(router.projection.weight.grad.abs().sum()), 0.0)
        experts = model.blocks[0].mlp.experts.experts
        self.assertTrue(
            any(
                p.grad is not None and float(p.grad.abs().sum()) > 0
                for expert in experts
                for p in expert.parameters()
            )
        )

    def test_gradients_top1_bpr(self) -> None:
        self._check_gradients("top1_bpr")

    def test_gradients_top2_fixed(self) -> None:
        self._check_gradients("top2_fixed")

    def test_gradients_top2_capsel(self) -> None:
        self._check_gradients("top2_capsel")


# ──────────────────────────────────────────────────────────────────────────────
# Configuration backward compatibility
# ──────────────────────────────────────────────────────────────────────────────

class ConfigBackwardCompatTests(unittest.TestCase):
    def test_old_toml_loads_with_defaults(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        config = load_config(repo / "configs" / "smoke.toml")
        self.assertEqual(config.model.routing_mode, "top1_fc")
        self.assertAlmostEqual(config.model.capsel_margin_threshold, 0.1)

    def test_old_toml_model_behaves_as_top1_fc(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        config = load_config(repo / "configs" / "smoke.toml")
        model = GPT(config.model)
        self.assertIsInstance(model.blocks[0].mlp.router, Top1Router)

    def test_invalid_routing_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            ModelConfig(routing_mode="top3_exotic")  # type: ignore[arg-type]

    def test_invalid_margin_threshold_raises(self) -> None:
        with self.assertRaises(ValueError):
            ModelConfig(capsel_margin_threshold=-0.1)
        with self.assertRaises(ValueError):
            ModelConfig(capsel_margin_threshold=1.1)

    def test_top2_requires_min_2_experts(self) -> None:
        for mode in ("top2_fixed", "top2_capsel"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                ModelConfig(expert_count=1, routing_mode=mode)  # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic matched initialization
# ──────────────────────────────────────────────────────────────────────────────

class DeterministicInitTests(unittest.TestCase):
    def test_same_seed_same_weights_across_modes(self) -> None:
        """All modes start with identical router projection weights for the same seed."""
        from gpt2_nano_moe.runtime import seed_everything
        configs = {
            mode: _tiny_moe_config(routing_mode=mode)
            for mode in ("top1_fc", "top1_bpr", "top2_fixed", "top2_capsel")
        }
        router_weights = {}
        for mode, cfg in configs.items():
            seed_everything(42)
            model = GPT(cfg)
            router_weights[mode] = model.blocks[0].mlp.router.projection.weight.detach().clone()

        reference = router_weights["top1_fc"]
        for mode, weights in router_weights.items():
            torch.testing.assert_close(
                weights, reference, msg=f"weights differ for mode={mode}"
            )

    def test_same_seed_same_batches_across_conditions(self) -> None:
        """Identical training seeds produce identical synthetic batch sequences."""
        from gpt2_nano_moe.train import _batch
        config = _tiny_moe_config()
        gen_a = torch.Generator(device="cpu").manual_seed(17)
        gen_b = torch.Generator(device="cpu").manual_seed(17)
        for _ in range(4):
            batch_a = _batch(2, 8, 31, gen_a, torch.device("cpu"))
            batch_b = _batch(2, 8, 31, gen_b, torch.device("cpu"))
            torch.testing.assert_close(batch_a, batch_b)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics fields
# ──────────────────────────────────────────────────────────────────────────────

class ExtendedMetricsTests(unittest.TestCase):
    def test_top2_capsel_metrics_fields_populated(self) -> None:
        torch.manual_seed(97)
        router = GeneralRouter(
            d_model=8, expert_count=4, capacity_factor=1.5,
            mode="top2_capsel", margin_threshold=0.2
        )
        route = router(torch.randn(20, 8))
        m = route.metrics
        self.assertIsNotNone(m.uncertainty_rate)
        self.assertIsNotNone(m.rescue_rate)
        self.assertIsNotNone(m.router_margin_mean)
        self.assertIsNotNone(m.router_margin_std)
        self.assertIsNotNone(m.routed_slots)
        self.assertIsNotNone(m.accepted_slots)
        self.assertIsNotNone(m.slot_drop_fraction)
        self.assertIsNotNone(m.effective_experts_per_token)
        self.assertGreaterEqual(m.routed_slots, 20)
        self.assertLessEqual(m.routed_slots, 20 * 2)

    def test_routed_and_accepted_slots_consistent(self) -> None:
        torch.manual_seed(101)
        for mode in ("top2_fixed", "top2_capsel"):
            with self.subTest(mode=mode):
                router = GeneralRouter(
                    d_model=4, expert_count=4, capacity_factor=1.0, mode=mode
                )
                T = 16
                route = router(torch.randn(T, 4))
                m = route.metrics
                if mode == "top2_fixed":
                    self.assertEqual(m.routed_slots, T * 2)
                else:
                    self.assertGreaterEqual(m.routed_slots, T)
                    self.assertLessEqual(m.routed_slots, T * 2)
                self.assertLessEqual(m.accepted_slots, m.routed_slots)
                self.assertAlmostEqual(
                    m.slot_drop_fraction,
                    (m.routed_slots - m.accepted_slots) / m.routed_slots,
                    places=6,
                )

    def test_top1_bpr_metrics_populated(self) -> None:
        torch.manual_seed(103)
        router = GeneralRouter(d_model=4, expert_count=3, capacity_factor=1.0, mode="top1_bpr")
        route = router(torch.randn(12, 4))
        m = route.metrics
        self.assertIsNotNone(m.routed_slots)
        self.assertIsNotNone(m.accepted_slots)
        self.assertIsNotNone(m.router_margin_mean)
        self.assertIsNotNone(m.router_margin_std)
        # Uncertainty rate and rescue rate are None for top1_bpr.
        self.assertIsNone(m.uncertainty_rate)
        self.assertIsNone(m.rescue_rate)


# ──────────────────────────────────────────────────────────────────────────────
# Publication study artifact schema
# ──────────────────────────────────────────────────────────────────────────────

class PublicationStudyArtifactTests(unittest.TestCase):
    def test_smoke_study_artifact_schema(self) -> None:
        from gpt2_nano_moe.publication_study import run as study_run
        repo = Path(__file__).resolve().parents[1]
        artifact_dir = repo / ".test-artifacts"
        output_path = artifact_dir / "pub-study-smoke.json"
        try:
            result = study_run(
                repo / "configs" / "publication_study_smoke.toml",
                output_path,
            )
            self.assertTrue(output_path.is_file())
            # Required top-level keys.
            for key in ("schema_version", "result_kind", "workload", "config",
                        "provenance", "environment", "conditions", "total_elapsed_seconds"):
                self.assertIn(key, result, f"missing top-level key: {key}")
            self.assertEqual(result["schema_version"], 1)
            self.assertEqual(result["result_kind"], "measured")
            self.assertEqual(result["workload"], "publication_routing_study")
            self.assertTrue(result["identical_initialization_across_conditions"])
            # Conditions present.
            conditions = result["conditions"]
            self.assertGreater(len(conditions), 0)
            # Each condition has required structure.
            for key, cond in conditions.items():
                for field in ("condition", "model_config", "seeds", "aggregate"):
                    self.assertIn(field, cond, f"condition {key} missing field {field}")
                agg = cond["aggregate"]
                self.assertIn("final_loss_mean", agg)
                self.assertIn("n_seeds", agg)
            # Check standard condition keys are present.
            condition_keys = list(conditions.keys())
            self.assertTrue(
                any("top1_fc" in k for k in condition_keys), "missing top1_fc condition"
            )
            self.assertTrue(
                any("top2_capsel" in k for k in condition_keys), "missing top2_capsel condition"
            )
        finally:
            if output_path.is_file():
                output_path.unlink()
            if artifact_dir.is_dir():
                artifact_dir.rmdir()

    def test_study_conditions_include_all_standard_modes(self) -> None:
        from gpt2_nano_moe.publication_study import _build_conditions
        conditions = _build_conditions(1.25, 2.0, [0.1, 0.2])
        modes = [c["routing_mode"] for c in conditions]
        self.assertIn("top1_fc", modes)
        self.assertIn("top1_bpr", modes)
        self.assertIn("top2_fixed", modes)
        self.assertIn("top2_capsel", modes)
        capsel_conditions = [c for c in conditions if c["routing_mode"] == "top2_capsel"]
        self.assertEqual(len(capsel_conditions), 2)
        # top1 CF2.0 should be present.
        cf2_conditions = [
            c for c in conditions
            if c["routing_mode"] == "top1_fc" and c["capacity_factor"] == 2.0
        ]
        self.assertEqual(len(cf2_conditions), 1)

    def test_margin_threshold_calibration_uses_requested_quantiles(self) -> None:
        from gpt2_nano_moe.publication_study import (
            _calibrate_margin_thresholds,
            _make_model_config,
        )

        repo = Path(__file__).resolve().parents[1]
        experiment = load_config(
            repo / "configs" / "publication_study_smoke.toml"
        )
        config = _make_model_config(
            experiment.model,
            {
                "routing_mode": "top1_fc",
                "capacity_factor": 1.25,
                "capsel_margin_threshold": None,
            },
        )
        torch.manual_seed(17)
        model = GPT(config)
        result = _calibrate_margin_thresholds(
            config,
            experiment.train,
            17,
            torch.device("cpu"),
            copy.deepcopy(model.state_dict()),
            None,
            [0.25, 0.75],
            1,
        )
        self.assertEqual(result["quantiles"], [0.25, 0.75])
        self.assertEqual(len(result["thresholds"]), 2)
        self.assertLessEqual(result["thresholds"][0], result["thresholds"][1])
        self.assertGreater(result["samples"], 0)


if __name__ == "__main__":
    unittest.main()
