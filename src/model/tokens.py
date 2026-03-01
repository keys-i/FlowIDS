"""Global tokens for the model"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CLSToken(nn.Module):
    """Prepend a learnable CLS token to a batch-first sequence.

    Inputs:
        x: [batch, seq_len, d_model]
        mask: [batch, seq_len] with True for valid tokens and False for padding

    Outputs:
        x: [batch, seq_len + 1, d_model]
        mask: [batch, seq_len + 1] if mask is provided
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.token = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepend the learnable CLS token.

        Args:
            x: Tensor of shape [batch, seq_len, d_model].
            mask: Optional boolean tensor of shape [batch, seq_len].

        Returns:
            A tuple of:
            - x with CLS token prepended, shape [batch, seq_len + 1, d_model]
            - updated mask if provided, shape [batch, seq_len + 1]
        """
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, seq_len, d_model].")

        batch_size, _, d_model = x.shape

        if d_model != self.d_model:
            raise ValueError(f"Expected x last dim {self.d_model}, got {d_model}.")

        cls = self.token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        if mask is None:
            return x, None

        if mask.ndim != 2:
            raise ValueError("mask must have shape [batch, seq_len].")

        if mask.size(0) != batch_size:
            raise ValueError("mask batch dimension must match x batch dimension.")

        cls_mask = torch.ones(
            (batch_size, 1),
            dtype=torch.bool,
            device=mask.device,
        )
        mask = torch.cat([cls_mask, mask], dim=1)

        return x, mask
