from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig
from .dense import DenseMLP
from .metrics import RoutingMetrics
from .moe import SparseMoE


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    auxiliary_loss: torch.Tensor
    routing_metrics: tuple[RoutingMetrics, ...]
    expert_indices: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class ParameterCounts:
    total: int
    trainable: int
    active_per_token: int

    def to_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "trainable": self.trainable,
            "active_per_token": self.active_per_token,
        }


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.dropout = config.dropout

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, sequence, width = inputs.shape
        qkv = self.qkv(inputs).reshape(
            batch, sequence, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query, key, value = (
            tensor.transpose(1, 2) for tensor in (query, key, value)
        )
        attention = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        return self.output(attention.transpose(1, 2).reshape(batch, sequence, width))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        hidden_size = config.d_model * config.mlp_multiplier
        if config.architecture == "moe":
            self.mlp: DenseMLP | SparseMoE = SparseMoE(
                config.d_model,
                hidden_size,
                config.expert_count,
                config.capacity_factor,
                config.dropout,
            )
        else:
            self.mlp = DenseMLP(config.d_model, hidden_size, config.dropout)

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        RoutingMetrics | None,
        torch.Tensor | None,
    ]:
        hidden = inputs + self.attention(self.attention_norm(inputs))
        normalized = self.mlp_norm(hidden)
        if isinstance(self.mlp, SparseMoE):
            result = self.mlp(normalized)
            return (
                hidden + result.hidden_states,
                result.auxiliary_loss,
                result.metrics,
                result.expert_indices,
            )
        return hidden + self.mlp(normalized), hidden.new_zeros(()), None, None


class GPT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, token_ids: torch.Tensor) -> ModelOutput:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.shape[1] > self.config.context_length:
            raise ValueError("sequence exceeds context_length")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        hidden = self.dropout(
            self.token_embedding(token_ids) + self.position_embedding(positions)
        )
        auxiliary_losses: list[torch.Tensor] = []
        routing_metrics: list[RoutingMetrics] = []
        expert_indices: list[torch.Tensor] = []
        for block in self.blocks:
            hidden, auxiliary, metrics, indices = block(hidden)
            auxiliary_losses.append(auxiliary)
            if metrics is not None and indices is not None:
                routing_metrics.append(metrics)
                expert_indices.append(indices)
        logits = self.lm_head(self.final_norm(hidden))
        auxiliary_loss = torch.stack(auxiliary_losses).mean()
        return ModelOutput(
            logits=logits,
            auxiliary_loss=auxiliary_loss,
            routing_metrics=tuple(routing_metrics),
            expert_indices=tuple(expert_indices),
        )

    def loss(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, ModelOutput]:
        if token_ids.ndim != 2 or token_ids.shape[1] < 2:
            raise ValueError(
                "token_ids must have shape [batch, sequence] with sequence length at least 2"
            )
        output = self(token_ids)
        language_model_loss = F.cross_entropy(
            output.logits[:, :-1].reshape(-1, self.config.vocab_size),
            token_ids[:, 1:].reshape(-1),
        )
        loss = (
            language_model_loss
            + self.config.auxiliary_loss_weight * output.auxiliary_loss
        )
        return loss, output

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if max_new_tokens < 0 or temperature <= 0:
            raise ValueError("max_new_tokens must be non-negative and temperature positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")
        was_training = self.training
        self.eval()
        try:
            generated = token_ids
            for _ in range(max_new_tokens):
                context = generated[:, -self.config.context_length :]
                logits = self(context).logits[:, -1] / temperature
                if top_k is not None:
                    values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                    logits = logits.masked_fill(logits < values[:, [-1]], -math.inf)
                probabilities = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(
                    probabilities, num_samples=1, generator=generator
                )
                generated = torch.cat((generated, next_token), dim=1)
            return generated
        finally:
            self.train(was_training)

    def parameter_counts(self) -> ParameterCounts:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        active = total
        for block in self.blocks:
            if isinstance(block.mlp, SparseMoE):
                expert_sizes = [
                    sum(parameter.numel() for parameter in expert.parameters())
                    for expert in block.mlp.experts.experts
                ]
                active -= sum(expert_sizes) - expert_sizes[0]
        return ParameterCounts(total=total, trainable=trainable, active_per_token=active)
