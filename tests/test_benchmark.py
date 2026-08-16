import unittest

import torch

from gpt2_nano_moe.benchmark import _measure
from gpt2_nano_moe.config import ModelConfig
from gpt2_nano_moe.model import GPT


class BenchmarkTests(unittest.TestCase):
    def test_routing_accumulates_all_measured_iterations(self) -> None:
        config = ModelConfig(
            vocab_size=31,
            context_length=4,
            d_model=16,
            num_heads=4,
            num_layers=2,
            mlp_multiplier=2,
            architecture="moe",
            expert_count=3,
            capacity_factor=2.0,
        )
        token_ids = torch.randint(config.vocab_size, (2, config.context_length))
        iterations = 3
        measurement = _measure(
            GPT(config), token_ids, iterations, torch.device("cpu")
        )
        self.assertEqual(
            measurement["routing"]["token_assignments"],
            token_ids.numel() * config.num_layers * iterations,
        )
        self.assertEqual(
            measurement["processed_tokens"], token_ids.numel() * iterations
        )
        self.assertEqual(
            sum(measurement["routing"]["expert_counts"]),
            measurement["routing"]["token_assignments"]
            - measurement["routing"]["dropped_tokens"],
        )


if __name__ == "__main__":
    unittest.main()
