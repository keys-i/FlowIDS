"""Transformer backbone for the model"""

from __future__ import annotations

import torch
import torch.nn as nn


class Backbone(nn.Module):
    """Encode a batch-first token sequence with a Transformer encoder.

    Inputs:
        x: [batch, seq_len, d_model]
        mask: [batch, seq_len] with True for valid tokens and False for padding

    Output:
        x: [batch, seq_len, d_model]
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.d_model = d_model

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode the sequence.

        Args:
            x: Tensor of shape [batch, seq_len, d_model].
            mask: Optional boolean tensor of shape [batch, seq_len]
                where True means valid token and False means padding.

        Returns:
            Tensor of shape [batch, seq_len, d_model].
        """
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, seq_len, d_model].")

        if x.size(-1) != self.d_model:
            raise ValueError(f"Expected x last dim {self.d_model}, got {x.size(-1)}.")

        src_key_padding_mask = None

        if mask is not None:
            if mask.ndim != 2:
                raise ValueError("mask must have shape [batch, seq_len].")
            if mask.shape[:2] != x.shape[:2]:
                raise ValueError("mask shape must match x batch and seq_len dims.")

            src_key_padding_mask = ~mask

        return self.encoder(
            src=x,
            src_key_padding_mask=src_key_padding_mask,
        )
