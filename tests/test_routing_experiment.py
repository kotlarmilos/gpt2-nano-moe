import hashlib
import json
from pathlib import Path
import unittest

import torch

from gpt2_nano_moe.config import ModelConfig
from gpt2_nano_moe.data import load_corpus
from gpt2_nano_moe.metrics import aggregate_routing_metrics
from gpt2_nano_moe.model import GPT
from gpt2_nano_moe.publication_study import (
    run as run_publication_study,
    sha256,
)
from gpt2_nano_moe.router import (
    GeneralRouter,
    deterministic_capacity_mask,
    score_prioritized_capacity_mask,
)
from scripts.prepare_publication_data import SOURCE, SHA256, verify


def _route(mode: str = "top2_capsel"):
    logits = torch.tensor(
        [
            [2.0, 0.0, 1.9],
            [1.8, 0.0, 1.79],
            [1.6, 1.59, 0.0],
            [0.0, 3.0, 1.0],
            [0.0, 1.0, 3.0],
        ]
    )
    router = GeneralRouter(
        d_model=5,
        expert_count=3,
        capacity_factor=1.0,
        mode=mode,
        margin_threshold=1.0,
    )
    with torch.no_grad():
        router.projection.weight.copy_(logits.T)
    return router(torch.eye(5))


class RoutingExperimentTests(unittest.TestCase):
    maxDiff = None

    def test_shared_capacity_and_deterministic_admission(self) -> None:
        token_order = deterministic_capacity_mask(
            torch.tensor([0, 0, 1, 0, 1, 1]), expert_count=2, capacity=2
        )
        self.assertEqual(
            token_order.tolist(), [True, True, True, False, True, False]
        )

        score_order = score_prioritized_capacity_mask(
            torch.tensor([0, 0, 0, 1]),
            torch.tensor([0.2, 0.9, 0.5, 0.1]),
            expert_count=2,
            capacity=2,
        )
        self.assertEqual(score_order.tolist(), [False, True, True, True])

        for mode in ("top2_fixed", "top2_capsel"):
            with self.subTest(mode=mode):
                route = _route(mode)
                self.assertEqual(route.metrics.capacity, 2)
                self.assertEqual(route.metrics.expert_counts.tolist(), [2, 2, 2])
                self.assertLessEqual(int(route.metrics.expert_counts.max()), 2)

    def test_rescue_precedes_optional_routes_and_uncertainty_orders_them(self) -> None:
        route = _route()
        self.assertEqual(
            route.accepted.tolist(),
            [
                [True, False],
                [True, True],
                [False, True],
                [True, False],
                [True, False],
            ],
        )
        self.assertEqual(route.expert_indices[2].tolist(), [0, 1])
        self.assertFalse(route.accepted[2, 0])
        self.assertTrue(route.accepted[2, 1])
        self.assertTrue(route.accepted[1, 1])
        self.assertFalse(route.accepted[0, 1])

    def test_token_slot_and_rescue_metrics_keep_distinct_denominators(self) -> None:
        metrics = _route().metrics
        self.assertEqual(metrics.dropped_tokens, 0)
        self.assertEqual(metrics.token_drop_fraction, 0.0)
        self.assertEqual(metrics.routed_slots, 10)
        self.assertEqual(metrics.accepted_slots, 6)
        self.assertAlmostEqual(metrics.slot_drop_fraction, 0.4)
        self.assertEqual(metrics.rescued_tokens, 1)
        self.assertEqual(metrics.rejected_primary_tokens, 1)
        self.assertEqual(metrics.rescue_rate, 1.0)
        self.assertEqual(metrics.to_dict()["token_drop_fraction"], 0.0)

        aggregate = aggregate_routing_metrics((metrics, metrics))
        self.assertEqual(aggregate["token_assignments"], 10)
        self.assertEqual(aggregate["dropped_tokens"], 0)
        self.assertEqual(aggregate["token_drop_fraction"], 0.0)
        self.assertEqual(aggregate["routed_slots"], 20)
        self.assertEqual(aggregate["accepted_slots"], 12)
        self.assertAlmostEqual(aggregate["slot_drop_fraction"], 0.4)
        self.assertEqual(aggregate["rescued_tokens"], 2)
        self.assertEqual(aggregate["rejected_primary_tokens"], 2)
        self.assertEqual(aggregate["rescue_rate"], 1.0)

    def test_capsel_model_backward_reaches_router_and_experts(self) -> None:
        torch.manual_seed(3)
        config = ModelConfig(
            vocab_size=31,
            context_length=6,
            d_model=12,
            num_heads=3,
            num_layers=1,
            mlp_multiplier=2,
            architecture="moe",
            expert_count=3,
            capacity_factor=1.0,
            routing_mode="top2_capsel",
            capsel_margin_threshold=1.0,
        )
        model = GPT(config)
        loss, output = model.loss(torch.randint(config.vocab_size, (2, 6)))
        loss.backward()

        self.assertEqual(output.logits.shape, (2, 6, config.vocab_size))
        self.assertEqual(len(output.routing_metrics), 1)
        router_grad = model.blocks[0].mlp.router.projection.weight.grad
        self.assertIsNotNone(router_grad)
        self.assertGreater(float(router_grad.abs().sum()), 0.0)
        self.assertTrue(
            any(
                parameter.grad is not None
                and float(parameter.grad.abs().sum()) > 0.0
                for expert in model.blocks[0].mlp.experts.experts
                for parameter in expert.parameters()
            )
        )

    def test_invalid_inputs_fail_at_release_boundaries(self) -> None:
        with self.assertRaisesRegex(ValueError, "top2 routing modes"):
            GeneralRouter(
                d_model=4,
                expert_count=1,
                capacity_factor=1.0,
                mode="top2_capsel",
            )

        repo = Path(__file__).resolve().parents[1]
        artifact_dir = repo / ".test-artifacts"
        corpus_path = artifact_dir / "truncated.bin"
        artifact_dir.mkdir(exist_ok=True)
        try:
            corpus_path.write_bytes(b"short")
            with self.assertRaisesRegex(ValueError, "truncated shard header"):
                load_corpus(
                    corpus_path,
                    vocab_size=32,
                    context_length=8,
                    validation_fraction=0.1,
                )
        finally:
            corpus_path.unlink(missing_ok=True)
            artifact_dir.rmdir()

    def test_release_hashes_and_input_record_match(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result_path = repo / "artifacts" / "publication-routing-study.json"
        config_path = repo / "configs" / "publication_study.toml"
        manifest = json.loads(
            (repo / "artifacts" / "manifest.json").read_text()
        )
        result = json.loads(result_path.read_text())

        self.assertEqual(
            sha256(result_path),
            "e697a77ef687689115667bbc9e0c6a41985b942bb8fd23efa834941ab1bb7ce0",
        )
        self.assertEqual(
            sha256(config_path),
            "24d1b2f56188340cd52eb39b214d653b490bcc704bf161fdd299e527850ba0c3",
        )
        self.assertEqual(manifest["measured"]["sha256"], sha256(result_path))
        self.assertEqual(manifest["measured"]["config_sha256"], sha256(config_path))
        self.assertEqual(result["provenance"]["config_sha256"], sha256(config_path))
        self.assertEqual(manifest["publication_input"]["source"], SOURCE)
        self.assertEqual(manifest["publication_input"]["sha256"], SHA256)
        self.assertEqual(result["provenance"]["corpus_source"], SOURCE)
        self.assertEqual(result["provenance"]["corpus_sha256"], SHA256)

        artifact_dir = repo / ".test-artifacts"
        sample_path = artifact_dir / "hash-sample.bin"
        artifact_dir.mkdir(exist_ok=True)
        try:
            sample_path.write_bytes(b"capsel-2")
            expected = hashlib.sha256(b"capsel-2").hexdigest()
            self.assertTrue(verify(sample_path, expected))
            self.assertFalse(verify(sample_path, "0" * 64))
        finally:
            sample_path.unlink(missing_ok=True)
            artifact_dir.rmdir()

    def test_cpu_path_preserves_pairing_and_provenance(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        artifact_dir = repo / ".test-artifacts"
        config_path = artifact_dir / "publication-study.toml"
        output_path = artifact_dir / "publication-study.json"
        config_text = """\
[model]
vocab_size = 32
context_length = 8
d_model = 16
num_heads = 4
num_layers = 1
mlp_multiplier = 2
dropout = 0.0
architecture = "moe"
expert_count = 4
auxiliary_loss_weight = 0.01

[train]
seed = 17
steps = 1
batch_size = 2
learning_rate = 0.0003
weight_decay = 0.01
device = "cpu"
log_every = 1

[study]
seeds = [17]
threshold_grid = [0.1]
base_capacity_factor = 1.25
double_capacity_factor = 2.0
validation_batches = 1
"""
        try:
            artifact_dir.mkdir(exist_ok=True)
            config_path.write_text(config_text)
            result = run_publication_study(config_path, output_path)
            self.assertEqual(result["schema_version"], 1)
            self.assertEqual(result["workload"], "publication_routing_study")
            expected_conditions = {
                "top1_fc_cf1.25",
                "top1_bpr_cf1.25",
                "top2_fixed_cf1.25",
                "top1_fc_cf2.0",
                "top2_capsel_cf1.25_m0.1",
            }
            self.assertEqual(set(result["conditions"]), expected_conditions)
            self.assertEqual(result["provenance"]["config_path"], str(config_path))
            self.assertEqual(result["provenance"]["config_sha256"], sha256(config_path))
            self.assertTrue(result["identical_initialization_across_conditions"])
            self.assertEqual(len(result["initialization_hashes"]["17"]), 1)

            initial_hashes = {
                condition["seeds"]["17"]["initial_state_sha256"]
                for condition in result["conditions"].values()
            }
            self.assertEqual(len(initial_hashes), 1)
            self.assertTrue(output_path.is_file())
        finally:
            output_path.unlink(missing_ok=True)
            config_path.unlink(missing_ok=True)
            artifact_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
