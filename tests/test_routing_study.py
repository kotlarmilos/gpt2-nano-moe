from pathlib import Path
import unittest

import torch

from gpt2_nano_moe.routing_study import (
    compare_top_k_routing,
    routing_metrics_for_top_k,
    run,
    utilization_metrics,
)


class RoutingStudyTests(unittest.TestCase):
    def test_top2_uses_fixed_capacity_and_drops_extra_slots(self) -> None:
        probabilities = torch.tensor(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.1, 0.9],
                [0.2, 0.8],
            ]
        )
        top1 = routing_metrics_for_top_k(probabilities, top_k=1, capacity=2)
        top2 = routing_metrics_for_top_k(probabilities, top_k=2, capacity=2)

        self.assertEqual(top1.routed_token_slots, 4)
        self.assertEqual(top1.dropped_token_slots, 0)
        self.assertEqual(top1.expert_counts.tolist(), [2, 2])
        self.assertEqual(top2.routed_token_slots, 8)
        self.assertEqual(top2.dropped_token_slots, 4)
        self.assertEqual(top2.expert_counts.tolist(), [2, 2])
        self.assertEqual(top2.raw_expert_counts.tolist(), [4, 4])
        self.assertAlmostEqual(top2.dropped_fraction, 0.5)

    def test_utilization_metrics_report_balance_shape(self) -> None:
        metrics = utilization_metrics(torch.tensor([2, 0, 4]))

        self.assertAlmostEqual(metrics["zero_expert_fraction"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["expert_utilization"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["max_to_mean_load"], 2.0)
        self.assertGreater(metrics["coefficient_of_variation"], 0.0)

    def test_compare_top_k_uses_capacity_factor_per_layer(self) -> None:
        probabilities = torch.tensor(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.1, 0.9],
                [0.2, 0.8],
            ]
        )
        comparison = compare_top_k_routing(
            [probabilities], top_k_values=(1, 2), capacity_factor=1.0
        )

        self.assertEqual(comparison["1"]["capacity_per_expert"], 2)
        self.assertEqual(comparison["2"]["capacity_per_expert"], 2)
        self.assertEqual(comparison["1"]["dropped_token_slots"], 0)
        self.assertEqual(comparison["2"]["dropped_token_slots"], 4)

    def test_runner_writes_measured_artifact(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        artifact_dir = repo / ".test-artifacts"
        output_path = artifact_dir / "routing-study.json"
        try:
            result = run(
                repo / "configs" / "smoke.toml",
                output_path,
                top_k_values=(1, 2),
                seed=19,
            )

            self.assertTrue(output_path.is_file())
            self.assertEqual(result["result_kind"], "measured")
            self.assertIn("1", result["measurements"]["top_k"])
            self.assertIn("2", result["measurements"]["top_k"])
            self.assertEqual(
                result["measurements"]["top_k"]["1"]["capacity_per_expert"],
                result["measurements"]["top_k"]["2"]["capacity_per_expert"],
            )
        finally:
            if output_path.is_file():
                output_path.unlink()
            if artifact_dir.is_dir():
                artifact_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
