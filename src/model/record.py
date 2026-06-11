"""Record-level FlowTransformer input projection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import final, override

import torch
from torch import Tensor, nn


@final
class RecordEncoder(nn.Module):
    """Factorize one-hot record projection with PAD, UNK, and missingness."""

    def __init__(self, numeric_count: int, categorical_sizes: Sequence[int], d_model: int) -> None:
        """Initialize the bias-free equivalent for valid categorical values."""
        super().__init__()
        if numeric_count <= 0 or d_model <= 0 or any(size < 2 for size in categorical_sizes):
            raise ValueError("record dimensions must be positive")
        self.numeric = nn.Linear(numeric_count * 2, d_model, bias=False)
        self.categorical = nn.ModuleList(
            nn.Embedding(size, d_model, padding_idx=0) for size in categorical_sizes
        )

    @override
    def forward(
        self,
        numeric: Tensor,
        missing: Tensor,
        categorical: Tensor,
        elapsed: Tensor | None = None,
    ) -> Tensor:
        """Encode flows; elapsed is retained only for the shared batch interface."""
        del elapsed
        if numeric.shape != missing.shape or categorical.shape[:-1] != numeric.shape[:-1]:
            raise ValueError("record tensors must share batch and event dimensions")
        if numeric.shape[-1] * 2 != self.numeric.in_features:
            raise ValueError("numeric field count does not match the encoder")
        if categorical.shape[-1] != len(self.categorical):
            raise ValueError("categorical field count does not match the encoder")
        value = self.numeric(torch.cat((numeric, missing.to(numeric.dtype)), dim=-1))
        for index, embedding in enumerate(self.categorical):
            value = value + embedding(categorical[..., index])
        return value
