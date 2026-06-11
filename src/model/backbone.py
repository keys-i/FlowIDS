"""FlowTransformer self-attention blocks."""

from __future__ import annotations

from typing import final, override

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


@final
class Attention(nn.Module):
    """Apply masked multi-head self-attention without positions."""

    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        """Initialize attention projections."""
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.width = d_model // heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.output = nn.Linear(d_model, d_model)

    @override
    def forward(self, value: Tensor, padding: Tensor, causal: bool) -> Tensor:
        """Attend to non-padding keys, optionally only earlier keys."""
        batch, length, d_model = value.shape
        qkv = self.qkv(value).view(batch, length, 3, self.heads, self.width).permute(2, 0, 3, 1, 4)
        query, key, content = qkv.unbind(0)
        allowed = ~padding[:, None, None, :]
        if causal:
            allowed = (
                allowed & torch.ones(length, length, dtype=torch.bool, device=value.device).tril()
            )
        attended = functional.scaled_dot_product_attention(
            query,
            key,
            content,
            attn_mask=allowed,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.output(attended.transpose(1, 2).reshape(batch, length, d_model))


@final
class Block(nn.Module):
    """Implement a FlowTransformer-style post-layer-normalized block."""

    def __init__(self, d_model: int, heads: int, ffn: int, dropout: float) -> None:
        """Initialize attention, ReLU feed-forward, and normalizers."""
        super().__init__()
        self.attention = Attention(d_model, heads, dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, ffn), nn.ReLU(), nn.Linear(ffn, d_model)
        )
        self.feedforward_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(d_model, eps=1e-6)

    @override
    def forward(self, value: Tensor, padding: Tensor, causal: bool) -> Tensor:
        """Transform one flow sequence and retain zero padding."""
        value = self.attention_norm(
            value + self.attention_dropout(self.attention(value, padding, causal))
        )
        value = value.masked_fill(padding.unsqueeze(-1), 0)
        value = self.feedforward_norm(value + self.feedforward_dropout(self.feedforward(value)))
        return value.masked_fill(padding.unsqueeze(-1), 0)


@final
class Backbone(nn.Module):
    """Stack FlowTransformer-style post-layer-normalized blocks."""

    def __init__(self, d_model: int, layers: int, heads: int, ffn: int, dropout: float) -> None:
        """Initialize a uniform stack of blocks."""
        super().__init__()
        if layers <= 0 or ffn <= 0:
            raise ValueError("layers and ffn must be positive")
        self.blocks = nn.ModuleList(Block(d_model, heads, ffn, dropout) for _ in range(layers))

    @override
    def forward(self, value: Tensor, padding: Tensor, causal: bool) -> Tensor:
        """Encode a padded flow history."""
        if padding.shape != value.shape[:2] or padding.dtype != torch.bool:
            raise ValueError("padding must be a boolean [batch, event] tensor")
        value = value.masked_fill(padding.unsqueeze(-1), 0)
        for block in self.blocks:
            value = block(value, padding, causal)
        return value
