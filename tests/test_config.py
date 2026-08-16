import unittest

from gpt2_nano_moe.config import ModelConfig, TrainConfig
from gpt2_nano_moe.router import Top1Router


class InvalidConfigTests(unittest.TestCase):
    def test_invalid_model_configs(self) -> None:
        invalid = [
            {"d_model": 15, "num_heads": 4},
            {"architecture": "wide"},
            {"expert_count": 0},
            {"capacity_factor": 0},
            {"dropout": 1.0},
            {"auxiliary_loss_weight": -1.0},
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                ModelConfig(**values)

    def test_invalid_train_config(self) -> None:
        with self.assertRaises(ValueError):
            TrainConfig(steps=0)
        with self.assertRaises(ValueError):
            TrainConfig(device="tpu")
        with self.assertRaisesRegex(ValueError, "validation_fraction"):
            TrainConfig(validation_fraction=1.0)

    def test_invalid_router_config(self) -> None:
        with self.assertRaises(ValueError):
            Top1Router(4, 0, 1.0)


if __name__ == "__main__":
    unittest.main()
