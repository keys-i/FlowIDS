"""Classification heads for M0 models."""

from __future__ import annotations

from typing import final, override

from torch import Tensor, nn


@final
class BinaryHead(nn.Module):
    """Apply the project-configured small ReLU classification MLP."""

    def __init__(self, d_model: int, dropout: float) -> None:
        """Initialize the project-configured 128-unit classifier."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    @override
    def forward(self, value: Tensor) -> Tensor:
        """Return one binary logit per representation."""
        return self.layers(value).squeeze(-1)
