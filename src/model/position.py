"""Position encoding blocks for the model"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LearnedPositionalEmbedding(nn.Module):
    """Add learned positional embeddings to a batch-first sequence."""

    def __init__(
        self,
        max_len: int,
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.max_len = max_len
        self.d_model = d_model

        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned position embeddings.

        Args:
            x: Tensor of shape [batch, seq_len, d_model].

        Returns:
            Tensor of shape [batch, seq_len, d_model].
        """
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, seq_len, d_model].")

        batch_size, seq_len, d_model = x.shape

        if d_model != self.d_model:
            raise ValueError(f"Expected x last dim {self.d_model}, got {d_model}.")

        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len={self.max_len}.")

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)

        x = x + self.embedding(positions)
        x = self.dropout(x)
        return x


class SinusoidalPositionalEmbedding(nn.Module):
    """Add fixed sinusoidal positional encodings to a batch-first sequence."""

    def __init__(
        self,
        max_len: int,
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add sinusoidal position encodings.

        Args:
            x: Tensor of shape [batch, seq_len, d_model].

        Returns:
            Tensor of shape [batch, seq_len, d_model].
        """
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, seq_len, d_model].")

        _, seq_len, d_model = x.shape

        if d_model != self.d_model:
            raise ValueError(f"Expected x last dim {self.d_model}, got {d_model}.")

        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len={self.max_len}.")

        x = x + self.pe[:, :seq_len]
        x = self.dropout(x)
        return x
