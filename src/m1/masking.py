"""Mask related NetFlow fields together across short spans of completed flows"""

import torch
from torch import Tensor

from src.data.features import IAT_COLUMNS, NUMERIC_COLUMNS

GROUPS = (
    ("MIN_TTL", "MAX_TTL", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT"),
    (),
    (
        "IN_BYTES",
        "OUT_BYTES",
        "IN_PKTS",
        "OUT_PKTS",
        "SRC_TO_DST_SECOND_BYTES",
        "DST_TO_SRC_SECOND_BYTES",
        "SRC_TO_DST_AVG_THROUGHPUT",
        "DST_TO_SRC_AVG_THROUGHPUT",
    ),
    (
        "LONGEST_FLOW_PKT",
        "SHORTEST_FLOW_PKT",
        "MIN_IP_PKT_LEN",
        "MAX_IP_PKT_LEN",
        "RETRANSMITTED_IN_BYTES",
        "RETRANSMITTED_IN_PKTS",
        "RETRANSMITTED_OUT_BYTES",
        "RETRANSMITTED_OUT_PKTS",
        "NUM_PKTS_UP_TO_128_BYTES",
        "NUM_PKTS_128_TO_256_BYTES",
        "NUM_PKTS_256_TO_512_BYTES",
        "NUM_PKTS_512_TO_1024_BYTES",
        "NUM_PKTS_1024_TO_1514_BYTES",
    ),
    ("FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT", *IAT_COLUMNS),
)
NUMERIC_GROUPS = tuple(
    next(i for i, group in enumerate(GROUPS) if name in group) for name in NUMERIC_COLUMNS
)
CATEGORICAL_GROUPS = (0, 1, 1, 0, 0, 0, 1, 1)


def sample_mask(
    padding: Tensor, fraction: float, span: int, *, generator: torch.Generator | None = None
) -> Tensor:
    """Select feature groups in independently shifted, fixed-length spans

    Args:
        padding: Boolean ``(batch, events)`` mask with True at padding positions
        fraction: Probability of hiding each span, strictly between zero and one
        span: Positive span length in events; the default configuration uses three
        generator: Optional random stream on the same device as padding

    Returns:
        Boolean ``(batch, events, 5)`` selection excluding padding. Each example
        has at least one selected group; if random sampling selects none, one
        group on its final valid event is selected. Boundary spans are truncated

    Raises:
        ValueError: If settings or padding are invalid, or a sequence is all padding
    """
    if not 0 < fraction < 1 or span < 1:
        raise ValueError("mask fraction must be in (0, 1) and span must be positive")
    if padding.ndim != 2 or padding.dtype != torch.bool or not padding.numel():
        raise ValueError("padding must be a nonempty boolean [batch, events] tensor")
    if padding.all(1).any():
        raise ValueError("each sequence needs a valid event")
    batch, length = padding.shape
    offset = torch.randint(span, (batch, 1, 5), device=padding.device, generator=generator)
    blocks = (torch.arange(length, device=padding.device)[None, :, None] + offset) // span
    choices = torch.rand(
        batch, (length + span - 1) // span + 1, 5, device=padding.device, generator=generator
    )
    selected = (choices.gather(1, blocks) < fraction) & ~padding[..., None]
    empty = ~selected.flatten(1).any(1)
    last = ((~padding).long() * torch.arange(length, device=padding.device)).max(1).values
    group = torch.randint(5, (batch,), device=padding.device, generator=generator)
    selected[torch.arange(batch, device=padding.device), last, group] |= empty
    return selected
