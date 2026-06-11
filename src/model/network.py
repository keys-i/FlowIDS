"""FlowTransformer-style models for M0."""

from __future__ import annotations

from collections.abc import Sequence
from typing import final, override

from torch import Tensor, nn

from src.config import Config
from src.model.backbone import Backbone
from src.model.heads import BinaryHead
from src.model.record import RecordEncoder


@final
class FlowTransformer(nn.Module):
    """Classify the last flow from offline or causal chronological context."""

    def __init__(
        self,
        config: Config,
        numeric_count: int,
        categorical_sizes: Sequence[int],
        causal: bool = True,
    ) -> None:
        """Initialize the shared architecture with its sole mode difference."""
        super().__init__()
        model = config.model
        self.causal = causal
        self.record = RecordEncoder(numeric_count, categorical_sizes, model.d_model)
        self.backbone = Backbone(model.d_model, model.layers, model.heads, model.ffn, model.dropout)
        self.head = BinaryHead(model.d_model, model.dropout)

    @override
    def forward(
        self,
        numeric: Tensor,
        missing: Tensor,
        categorical: Tensor,
        elapsed: Tensor,
        padding: Tensor,
        position: Tensor,
        causal: Tensor,
    ) -> Tensor:
        """Return a logit for each final completed flow."""
        del position, causal
        if padding[:, -1].any():
            raise ValueError("the completed target flow cannot be padding")
        hidden = self.backbone(
            self.record(numeric, missing, categorical, elapsed), padding, self.causal
        )
        return self.head(hidden[:, -1])


Network = FlowTransformer
