"""Per-flow MLP baseline without sequence context."""

from __future__ import annotations

from collections.abc import Sequence
from typing import final, override

from torch import Tensor, nn

from src.config import Config
from src.model.heads import BinaryHead
from src.model.record import RecordEncoder


@final
class MLP(nn.Module):
    """Classify only the completed target flow."""

    def __init__(
        self,
        config: Config,
        numeric_count: int,
        categorical_sizes: Sequence[int],
    ) -> None:
        """Initialize the shared record encoder and binary head."""
        super().__init__()
        self.record = RecordEncoder(
            numeric_count,
            categorical_sizes,
            config.model.d_model,
        )
        self.head = BinaryHead(config.model.d_model, config.model.dropout)

    @override
    def forward(
        self,
        numeric: Tensor,
        missing: Tensor,
        categorical: Tensor,
        elapsed: Tensor,
    ) -> Tensor:
        """Classify the final flow in each batch sequence."""
        target = self.record(
            numeric[:, -1:],
            missing[:, -1:],
            categorical[:, -1:],
            elapsed[:, -1:],
        )
        return self.head(target[:, -1])
