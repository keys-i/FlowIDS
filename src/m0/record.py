"""Record-level input encoding for M0"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final, override

import torch
from torch import Tensor, nn


@final
class RecordEncoder(nn.Module):
    """Map numeric and categorical flow fields into model-width vectors"""

    def __init__(self, numeric_count: int, categorical_sizes: Sequence[int], d_model: int) -> None:
        """Create the numeric projection and one embedding per category field

        Args:
            numeric_count: Number of numeric values supplied to forward
            categorical_sizes: Vocabulary sizes including ID zero for padding
            d_model: Shared output width for the projection and embeddings

        Raises:
            ValueError: If a numeric count, model width, or category size is invalid
        """
        super().__init__()
        if numeric_count <= 0 or d_model <= 0 or any(size < 2 for size in categorical_sizes):
            raise ValueError("record dimensions must be positive")
        self.numeric = nn.Linear(numeric_count * 2, d_model, bias=False)
        self.categorical = nn.ModuleList(
            nn.Embedding(size, d_model, padding_idx=0) for size in categorical_sizes
        )

    @override
    def forward(self, numeric: Tensor, missing: Tensor, categorical: Tensor) -> Tensor:
        """Encode a batch of flows

        Args:
            numeric: Values with shape ``[batch, event, numeric_field]``
            missing: Boolean flags with the same shape as numeric
            categorical: Category IDs with shape ``[batch, event, category_field]``

        Returns:
            Flow vectors with shape ``[batch, event, d_model]``

        Raises:
            ValueError: If input shapes or field counts do not match this encoder
        """
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
