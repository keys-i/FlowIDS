"""Pooling blocks for the model"""

from __future__ import annotations

import torch
import torch.nn as nn


class CLSPooling(nn.Module):
    """Pool a sequence by taking the first token."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the CLS token representation.

        Args:
            x: Tensor of shape [batch, seq_len, d_model].

        Returns:
            Tensor of shape [batch, d_model].
        """
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, seq_len, d_model].")
        if x.size(1) < 1:
            raise ValueError("seq_len must be at least 1.")

        return x[:, 0]


class MeanPooling(nn.Module):
    """Pool a sequence by masked mean over valid tokens."""

    def __init__(self, has_cls_token: bool = True) -> None:
        super().__init__()
        self.has_cls_token = has_cls_token

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the masked mean representation.

        Args:
            x: Tensor of shape [batch, seq_len, d_model].
            mask: Optional boolean tensor of shape [batch, seq_len]
                where True means valid token and False means padding.

        Returns:
            Tensor of shape [batch, d_model].
        """
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, seq_len, d_model].")

        if self.has_cls_token:
            x = x[:, 1:]
            if mask is not None:
                mask = mask[:, 1:]

        if x.size(1) == 0:
            raise ValueError("No tokens available for mean pooling.")

        if mask is None:
            return x.mean(dim=1)

        if mask.ndim != 2:
            raise ValueError("mask must have shape [batch, seq_len].")
        if mask.shape[:2] != x.shape[:2]:
            raise ValueError("mask shape must match x batch and seq_len dims.")

        mask = mask.unsqueeze(-1)
        x = x.masked_fill(~mask, 0.0)

        denom = mask.sum(dim=1).clamp_min(1)
        return x.sum(dim=1) / denom


class DualPooling(nn.Module):
    """Concatenate CLS pooling and masked mean pooling."""

    def __init__(self) -> None:
        super().__init__()
        self.cls_pool = CLSPooling()
        self.mean_pool = MeanPooling(has_cls_token=True)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return concatenated CLS and mean pooled features.

        Args:
            x: Tensor of shape [batch, seq_len, d_model].
            mask: Optional boolean tensor of shape [batch, seq_len].

        Returns:
            Tensor of shape [batch, 2 * d_model].
        """
        h_cls = self.cls_pool(x)
        h_mean = self.mean_pool(x, mask)
        return torch.cat([h_cls, h_mean], dim=-1)
