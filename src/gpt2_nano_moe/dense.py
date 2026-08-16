from __future__ import annotations

from torch import nn


class DenseMLP(nn.Module):
    def __init__(self, d_model: int, hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, inputs):
        return self.net(inputs)

