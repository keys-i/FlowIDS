from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from typing import final, override

import polars as pl
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.features import (
    CATEGORICAL_COLUMNS,
    END_TIME,
    EVENT,
    NUMERIC_COLUMNS,
    PARTITION,
    ROUTING_COLUMNS,
    ROW,
    SCORE,
    SOURCE,
)

CATEGORICAL = (
    *(f"{column}_id" for column in CATEGORICAL_COLUMNS),
    "L4_SRC_PORT_range",
    "L4_DST_PORT_range",
)


@final
class FlowDataset(Dataset[dict[str, Tensor]]):
    """In-memory M0 shard; elapsed features are clipped log1p seconds."""

    def __init__(
        self,
        events: pl.DataFrame,
        contexts: pl.DataFrame,
        horizon_minutes: int,
    ) -> None:
        if horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be positive")
        required = {
            END_TIME,
            EVENT,
            "Label",
            PARTITION,
            *ROUTING_COLUMNS,
            ROW,
            SCORE,
            SOURCE,
            *NUMERIC_COLUMNS,
            *(f"{column}_missing" for column in NUMERIC_COLUMNS),
            *CATEGORICAL,
        }
        missing = required.difference(events.columns)
        if missing:
            raise ValueError(f"missing dataset columns: {', '.join(sorted(missing))}")
        if contexts.height != events.height:
            raise ValueError("contexts must contain one row per event")

        ordered = events.sort([SOURCE, PARTITION, END_TIME, ROW])
        if any(
            ordered[column].null_count()
            for column in (SOURCE, PARTITION, END_TIME, *ROUTING_COLUMNS)
        ):
            raise ValueError("routing and ordering columns cannot contain nulls")
        positions = contexts.get_column("target_position").to_list()
        starts = contexts.get_column("context_start").to_list()
        targets = contexts.get_column("target_event").to_list()
        event_ids = ordered.get_column(EVENT).to_list()
        if any(
            not isinstance(start, int)
            or not isinstance(position, int)
            or start < 0
            or start > position
            or position >= ordered.height
            or event_ids[position] != target
            for start, position, target in zip(starts, positions, targets, strict=True)
        ):
            raise ValueError("contexts do not index the supplied events")

        labels = ordered.get_column("Label")
        if labels.null_count() or not set(labels.unique()).issubset({0, 1}):
            raise ValueError("Label must contain only 0 and 1")

        keep = [index for index, position in enumerate(positions) if ordered[SCORE][position]]
        if not keep:
            raise ValueError("dataset contains no scorable targets")

        # ponytail: keep one bounded shard in memory; stream shards when profiling requires it.
        self.starts = torch.tensor([starts[index] for index in keep], dtype=torch.int64)
        self.ends = torch.tensor([positions[index] + 1 for index in keep], dtype=torch.int64)
        self.numeric = torch.tensor(ordered.select(NUMERIC_COLUMNS).to_numpy(), dtype=torch.float32)
        self.missing = torch.tensor(
            ordered.select(f"{column}_missing" for column in NUMERIC_COLUMNS).to_numpy(),
            dtype=torch.bool,
        )
        self.categorical = torch.tensor(ordered.select(CATEGORICAL).to_numpy(), dtype=torch.int64)
        self.elapsed = _elapsed(ordered, horizon_minutes * 60_000)
        self.labels = torch.tensor(labels.to_numpy(), dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.starts)

    @override
    def __getitem__(self, index: int) -> dict[str, Tensor]:
        start = int(self.starts[index])
        end = int(self.ends[index])
        return {
            "numeric": self.numeric[start:end],
            "missing": self.missing[start:end],
            "categorical": self.categorical[start:end],
            "elapsed": self.elapsed[start:end],
            "label": self.labels[end - 1],
        }


def _elapsed(events: pl.DataFrame, limit: int) -> Tensor:
    values: list[tuple[float, float]] = []
    group: tuple[object, object] | None = None
    previous: int | None = None
    endpoints: dict[object, int] = {}
    expiry: deque[tuple[int, object]] = deque()

    for source, partition, end, source_ip, destination_ip in events.select(
        SOURCE, PARTITION, END_TIME, *ROUTING_COLUMNS
    ).iter_rows():
        if not isinstance(end, int):
            raise TypeError(f"{END_TIME} must contain integer milliseconds")
        key = (source, partition)
        if key != group:
            group = key
            previous = None
            endpoints.clear()
            expiry.clear()

        while expiry and end - expiry[0][0] > limit:
            seen, endpoint = expiry.popleft()
            if endpoints.get(endpoint) == seen:
                del endpoints[endpoint]

        shared = max(
            (
                endpoints[endpoint]
                for endpoint in (source_ip, destination_ip)
                if endpoint in endpoints
            ),
            default=end,
        )
        values.append(
            (
                math.log1p(min(end - previous, limit) / 1_000) if previous is not None else 0.0,
                math.log1p(min(end - shared, limit) / 1_000),
            )
        )
        previous = end
        endpoints[source_ip] = end
        endpoints[destination_ip] = end
        expiry.extend(((end, source_ip), (end, destination_ip)))

    return torch.tensor(values, dtype=torch.float32)


def collate(samples: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
    if not samples:
        raise ValueError("cannot collate an empty batch")

    lengths = torch.tensor([len(sample["numeric"]) for sample in samples])
    width = int(lengths.max())

    def pad(name: str, value: float | int | bool = 0) -> Tensor:
        tail = samples[0][name].shape[1:]
        output = torch.full(
            (len(samples), width, *tail),
            value,
            dtype=samples[0][name].dtype,
        )
        for row, sample in enumerate(samples):
            output[row, -len(sample[name]) :] = sample[name]
        return output

    positions = torch.arange(width).expand(len(samples), -1)
    padding = positions < width - lengths[:, None]
    return {
        "numeric": pad("numeric"),
        "missing": pad("missing"),
        "categorical": pad("categorical"),
        "elapsed": pad("elapsed"),
        "padding": padding,
        "position": (positions - (width - lengths[:, None])).clamp_min(0),
        "causal": torch.triu(torch.ones(width, width, dtype=torch.bool), diagonal=1),
        "label": torch.stack([sample["label"] for sample in samples]),
    }
