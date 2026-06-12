"""Binary classification head used by M0 neural models"""

from __future__ import annotations

from typing import final, override

from torch import Tensor, nn


@final
class BinaryHead(nn.Module):
    """Apply the fixed-width ReLU classifier"""

    def __init__(self, d_model: int, dropout: float) -> None:
        """Create the 128-unit binary classifier"""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    @override
    def forward(self, value: Tensor) -> Tensor:
        """Return one binary logit per representation

        Args:
            value: Encoded flows with shape ``[..., d_model]``

        Returns:
            Logits with shape ``[...]`` where larger values favour attack
        """
        return self.layers(value).squeeze(-1)
