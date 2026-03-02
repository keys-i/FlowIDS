"""Main model."""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import Backbone
from .embedding import Embeddings
from .heads import Heads
from .pooling import DualPooling
from .position import LearnedPositionalEmbedding
from .tokens import CLSToken


class FlowTransformer(nn.Module):
    """Vanilla temporal Transformer for flow windows."""

    def __init__(
        self,
        num_numeric: int,
        categorical_cardinalities: dict[str, int],
        num_families: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        max_len: int,
        dropout: float = 0.1,
        aux_class_dims: dict[str, int] | None = None,
        aux_reg_dims: dict[str, int] | None = None,
        aux_gaussian_dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__()

        self.embedding = Embeddings(
            num_numeric=num_numeric,
            categorical_cardinalities=categorical_cardinalities,
            embedding_dim=d_model,
            dropout=dropout,
        )

        self.position = LearnedPositionalEmbedding(
            max_len=max_len + 1,
            d_model=d_model,
            dropout=dropout,
        )

        self.cls_token = CLSToken(d_model=d_model)

        self.backbone = Backbone(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
        )

        self.pool = DualPooling()

        self.heads = Heads(
            in_dim=2 * d_model,
            num_families=num_families,
            dropout=dropout,
            aux_class_dims=aux_class_dims,
            aux_reg_dims=aux_reg_dims,
            aux_gaussian_dims=aux_gaussian_dims,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize model parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

        nn.init.normal_(self.cls_token.token, mean=0.0, std=0.02)

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """Run the full model forward pass.

        Args:
            x_num: [batch, seq_len, num_numeric]
            x_cat: [batch, seq_len, num_categorical]
            mask: [batch, seq_len] with True for valid tokens

        Returns:
            Dictionary of model outputs from the prediction heads.
        """
        x = self.embedding(x_num, x_cat)
        x, mask = self.cls_token(x, mask)
        x = self.position(x)
        x = self.backbone(x, mask)
        h = self.pool(x, mask)
        return self.heads(h)
