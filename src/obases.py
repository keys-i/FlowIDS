"""Implement classical M0 baselines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import final, override

import torch
from torch import Tensor, nn


def always_benign(size: int, device: torch.device | str) -> Tensor:
    """Return the zero attack probabilities of the always-benign control."""
    if size < 0:
        raise ValueError("size must not be negative")
    return torch.zeros(size, device=device)


@final
class Logistic(nn.Module):
    """Classify the final flow with regularized linear feature effects."""

    def __init__(self, numeric_count: int, categorical_sizes: Sequence[int]) -> None:
        """Initialize numeric, missingness, elapsed, and categorical effects."""
        super().__init__()
        if (
            numeric_count <= 0
            or not categorical_sizes
            or any(size <= 1 for size in categorical_sizes)
        ):
            raise ValueError("numeric_count and categorical sizes must be valid")
        self.linear = nn.Linear(numeric_count * 2 + 2, 1)
        self.categorical = nn.ModuleList(
            nn.Embedding(size, 1, padding_idx=0) for size in categorical_sizes
        )

    @override
    def forward(
        self,
        numeric: Tensor,
        missing: Tensor,
        categorical: Tensor,
        elapsed: Tensor,
        padding: Tensor,
        position: Tensor,
        causal: Tensor,
    ) -> Tensor:
        """Return one target-flow logit per causal batch."""
        del padding, position, causal
        target = torch.cat(
            (numeric[:, -1], missing[:, -1].to(numeric.dtype), elapsed[:, -1]), dim=-1
        )
        embedded = torch.stack(
            [
                embedding(categorical[:, -1, index]).squeeze(-1)
                for index, embedding in enumerate(self.categorical)
            ],
            dim=-1,
        ).sum(dim=-1)
        return self.linear(target).squeeze(-1) + embedded
