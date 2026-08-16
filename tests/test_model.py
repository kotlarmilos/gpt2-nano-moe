from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from gpt2_nano_moe.config import ModelConfig
from gpt2_nano_moe.model import GPT


def tiny_config(architecture: str = "moe") -> ModelConfig:
    return ModelConfig(
        vocab_size=31,
        context_length=8,
        d_model=16,
        num_heads=4,
        num_layers=2,
        mlp_multiplier=2,
        architecture=architecture,
        expert_count=3,
        capacity_factor=2.0,
    )


class ModelTests(unittest.TestCase):
    def test_dense_and_moe_forward(self) -> None:
        token_ids = torch.randint(31, (2, 8))
        for architecture in ("dense", "moe"):
            with self.subTest(architecture=architecture):
                model = GPT(tiny_config(architecture))
                output = model(token_ids)
                self.assertEqual(output.logits.shape, (2, 8, 31))
                self.assertTrue(torch.isfinite(output.logits).all())
                expected_routes = 0 if architecture == "dense" else 2
                self.assertEqual(len(output.routing_metrics), expected_routes)

    def test_gradient_flows_through_router_and_expert(self) -> None:
        torch.manual_seed(3)
        model = GPT(tiny_config())
        loss, _ = model.loss(torch.randint(31, (2, 8)))
        loss.backward()
        router = model.blocks[0].mlp.router
        experts = model.blocks[0].mlp.experts.experts
        self.assertIsNotNone(router.projection.weight.grad)
        self.assertGreater(float(router.projection.weight.grad.abs().sum()), 0.0)
        self.assertTrue(
            any(
                parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
                for expert in experts
                for parameter in expert.parameters()
            )
        )

    def test_parameter_accounting(self) -> None:
        dense = GPT(tiny_config("dense")).parameter_counts()
        moe_model = GPT(tiny_config("moe"))
        moe = moe_model.parameter_counts()
        self.assertEqual(dense.total, dense.active_per_token)
        self.assertGreater(moe.total, moe.active_per_token)
        self.assertEqual(moe.total, moe.trainable)
        expected_inactive = 0
        for block in moe_model.blocks:
            sizes = [
                sum(parameter.numel() for parameter in expert.parameters())
                for expert in block.mlp.experts.experts
            ]
            expected_inactive += sum(sizes[1:])
        self.assertEqual(moe.total - moe.active_per_token, expected_inactive)

    def test_generation_interface(self) -> None:
        model = GPT(replace(tiny_config("dense"), context_length=4))
        generator = torch.Generator().manual_seed(1)
        generated = model.generate(
            torch.tensor([[1, 2]]), 3, top_k=4, generator=generator
        )
        self.assertEqual(generated.shape, (1, 5))

    def test_loss_rejects_sequence_shorter_than_two(self) -> None:
        model = GPT(tiny_config("dense"))
        with self.assertRaisesRegex(ValueError, "at least 2"):
            model.loss(torch.tensor([[1]]))

    def test_generation_validates_before_eval(self) -> None:
        model = GPT(tiny_config("dense"))
        model.train()
        with self.assertRaisesRegex(ValueError, "top_k"):
            model.generate(torch.tensor([[1, 2]]), 1, top_k=0)
        self.assertTrue(model.training)

    def test_generation_restores_mode_after_forward_error(self) -> None:
        model = GPT(tiny_config("dense"))
        model.train()
        with patch.object(model, "forward", side_effect=RuntimeError("generation failed")):
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                model.generate(torch.tensor([[1, 2]]), 1)
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
