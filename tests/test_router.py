import unittest

import torch

from gpt2_nano_moe.router import Top1Router


class Top1RouterTests(unittest.TestCase):
    def test_routing_shapes(self) -> None:
        router = Top1Router(d_model=3, expert_count=4, capacity_factor=2.0)
        output = router(torch.randn(7, 3))
        self.assertEqual(output.probabilities.shape, (7, 4))
        self.assertEqual(output.expert_indices.shape, (7,))
        self.assertEqual(output.gates.shape, (7,))
        self.assertEqual(output.accepted.shape, (7,))
        self.assertEqual(output.metrics.expert_counts.shape, (4,))

    def test_top1_selection_is_deterministic(self) -> None:
        router = Top1Router(d_model=2, expert_count=3, capacity_factor=3.0)
        with torch.no_grad():
            router.projection.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
            )
        inputs = torch.tensor([[3.0, 1.0], [1.0, 4.0], [-2.0, -2.0]])
        first = router(inputs)
        second = router(inputs)
        self.assertEqual(first.expert_indices.tolist(), [0, 1, 2])
        self.assertTrue(torch.equal(first.expert_indices, second.expert_indices))

    def test_capacity_drops_later_tokens(self) -> None:
        router = Top1Router(d_model=2, expert_count=2, capacity_factor=0.5)
        with torch.no_grad():
            router.projection.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
        output = router(torch.tensor([[1.0, 0.0]] * 6))
        self.assertEqual(output.metrics.capacity, 2)
        self.assertEqual(output.metrics.dropped_tokens, 4)
        self.assertEqual(output.accepted.tolist(), [True, True, False, False, False, False])
        self.assertEqual(output.metrics.expert_counts.tolist(), [2, 0])

    def test_auxiliary_loss_is_finite(self) -> None:
        output = Top1Router(4, 4, 1.0)(torch.randn(8, 4))
        self.assertTrue(torch.isfinite(output.auxiliary_loss))
        self.assertTrue(torch.isfinite(output.metrics.router_entropy))


if __name__ == "__main__":
    unittest.main()

