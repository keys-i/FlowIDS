"""Top-level M0 FlowTransformer"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final, override

from torch import Tensor, nn

from src.config import Config
from src.m0.backbone import Backbone
from src.m0.heads import BinaryHead
from src.m0.record import RecordEncoder


@final
class FlowTransformer(nn.Module):
    """Classify the final valid flow in a history"""

    def __init__(
        self,
        config: Config,
        numeric_count: int,
        categorical_sizes: Sequence[int],
        causal: bool = True,
    ) -> None:
        """Build the record encoder, attention backbone, and classifier"""
        super().__init__()
        model = config.model
        self.causal = causal
        self.record = RecordEncoder(numeric_count, categorical_sizes, model.d_model)
        self.backbone = Backbone(model.d_model, model.layers, model.heads, model.ffn, model.dropout)
        self.head = BinaryHead(model.d_model, model.dropout)

    @override
    def forward(
        self, numeric: Tensor, missing: Tensor, categorical: Tensor, padding: Tensor
    ) -> Tensor:
        """Return one binary logit for the final flow in every sequence

        Args:
            numeric: Numeric values with shape ``[batch, event, numeric_field]``
            missing: Boolean missing-value flags with the same shape as numeric
            categorical: Category IDs with shape ``[batch, event, category_field]``
            padding: Boolean left-padding mask with shape ``[batch, event]``

        Returns:
            Attack logits with shape ``[batch]`` where larger values favour attack

        Raises:
            ValueError: If a final target flow is marked as padding
        """
        if padding[:, -1].any():
            raise ValueError("the completed target flow cannot be padding")
        hidden = self.backbone(self.record(numeric, missing, categorical), padding, self.causal)
        return self.head(hidden[:, -1])
