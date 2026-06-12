"""Sequence encoder for M0 FlowTransformer models"""

from __future__ import annotations

from typing import final, override

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


@final
class Attention(nn.Module):
    """Apply multi-head self-attention without positional embeddings"""

    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        """Create query, key, value, and output projections

        Args:
            d_model: Width of each flow vector
            heads: Number of attention heads
            dropout: Attention dropout probability

        Raises:
            ValueError: If the model width cannot divide evenly across heads
        """
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.width = d_model // heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.output = nn.Linear(d_model, d_model)

    @override
    def forward(self, value: Tensor, allowed: Tensor) -> Tensor:
        """Attend through the precomputed allowed-key mask

        Args:
            value: Flow vectors with shape ``[batch, event, d_model]``
            allowed: Boolean mask broadcastable to ``[batch, heads, event, event]``
                where True permits attention and False excludes a key

        Returns:
            Contextualized flow vectors with shape ``[batch, event, d_model]``
        """
        batch, length, d_model = value.shape
        qkv = self.qkv(value).view(batch, length, 3, self.heads, self.width).permute(2, 0, 3, 1, 4)
        query, key, content = qkv.unbind(0)
        attended = functional.scaled_dot_product_attention(
            query, key, content, attn_mask=allowed, dropout_p=self.dropout if self.training else 0.0
        )
        return self.output(attended.transpose(1, 2).reshape(batch, length, d_model))


@final
class Block(nn.Module):
    """One post-layer-normalized attention and feed-forward block"""

    def __init__(self, d_model: int, heads: int, ffn: int, dropout: float) -> None:
        """Create attention, residual normalization, and feed-forward layers"""
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
    def forward(self, value: Tensor, padding: Tensor, allowed: Tensor) -> Tensor:
        """Transform one sequence and keep padding at zero

        Args:
            value: Flow vectors with shape ``[batch, event, d_model]``
            padding: Boolean padding mask with shape ``[batch, event]``
            allowed: Boolean attention mask shared by all backbone blocks

        Returns:
            Transformed vectors with shape ``[batch, event, d_model]``
        """
        value = self.attention_norm(value + self.attention_dropout(self.attention(value, allowed)))
        value = value.masked_fill(padding.unsqueeze(-1), 0)
        value = self.feedforward_norm(value + self.feedforward_dropout(self.feedforward(value)))
        return value.masked_fill(padding.unsqueeze(-1), 0)


@final
class Backbone(nn.Module):
    """Stack attention blocks over a padded flow history"""

    def __init__(self, d_model: int, layers: int, heads: int, ffn: int, dropout: float) -> None:
        """Create a uniform stack of attention blocks

        Args:
            d_model: Width of each flow vector
            layers: Number of blocks to create
            heads: Number of attention heads in each block
            ffn: Hidden width of each feed-forward network
            dropout: Dropout probability in each block

        Raises:
            ValueError: If the layer count or feed-forward width is not positive
        """
        super().__init__()
        if layers <= 0 or ffn <= 0:
            raise ValueError("layers and ffn must be positive")
        self.blocks = nn.ModuleList(Block(d_model, heads, ffn, dropout) for _ in range(layers))

    @override
    def forward(
        self, value: Tensor, padding: Tensor, causal: bool, *, average_last: int = 0
    ) -> Tensor:
        """Encode a padded flow history

        Args:
            value: Flow vectors with shape ``[batch, event, d_model]``
            padding: Boolean mask with shape ``[batch, event]`` where True is padding
            causal: Restrict each query to its own and earlier event positions
            average_last: Return the mean normalized state from this many final
                blocks for an M1 teacher; zero returns the final block unchanged

        Returns:
            Encoded vectors with shape ``[batch, event, d_model]`` and zeros at padding

        Raises:
            ValueError: If padding is not a boolean tensor matching batch and event axes
        """
        if padding.shape != value.shape[:2] or padding.dtype != torch.bool:
            raise ValueError("padding must be a boolean [batch, event] tensor")
        if not 0 <= average_last <= len(self.blocks):
            raise ValueError("average_last must be between zero and the number of blocks")
        value = value.masked_fill(padding.unsqueeze(-1), 0)
        allowed = ~padding[:, None, None, :]
        if causal:
            allowed = (
                allowed
                & torch.ones(
                    value.shape[1], value.shape[1], dtype=torch.bool, device=value.device
                ).tril()
            )
        states: list[Tensor] = []
        for index, block in enumerate(self.blocks):
            value = block(value, padding, allowed)
            if average_last and index >= len(self.blocks) - average_last:
                states.append(functional.layer_norm(value.float(), (value.shape[-1],)))
        return torch.stack(states).mean(0) if states else value
