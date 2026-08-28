"""Select later endpoint-related targets without adding them to student histories"""

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Sequence
from heapq import merge
from typing import cast, final, override

import polars as pl
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.dataset import FlowDataset, collate
from src.data.features import END_TIME, EVENT, PARTITION, ROUTING_COLUMNS, ROW, SOURCE

HORIZONS = (1, 4, 16)


def future_indices(events: pl.DataFrame) -> Tensor:
    """Find the 1st, 4th, and 16th later flow touching either endpoint

    Returns:
        Dataset row indices shaped (flows, 3), with -1 for missing horizons.
        Targets stay in the same source and partition. Equal completion times
        are excluded; remaining ties are ordered by event ID. Labels are unused
    """
    ordered = (
        events.select(SOURCE, PARTITION, END_TIME, ROW, EVENT, *ROUTING_COLUMNS)
        .sort([SOURCE, PARTITION, END_TIME, ROW])
        .with_row_index("position")
        .sort([SOURCE, PARTITION, END_TIME, EVENT])
        .with_row_index("rank")
    )
    required = (SOURCE, PARTITION, END_TIME, EVENT)
    if any(ordered[name].null_count() for name in required):
        raise ValueError("future ordering fields cannot contain nulls")
    if ordered[EVENT].n_unique() != len(ordered):
        raise ValueError("future event IDs must be unique")
    group = [SOURCE, PARTITION]
    later = ordered.select(
        (
            pl.col(END_TIME).search_sorted(pl.col(END_TIME), side="right").over(group)
            + pl.col("rank").first().over(group)
        ).alias("later")
    )["later"].to_list()
    rows = ordered.select(SOURCE, PARTITION, *ROUTING_COLUMNS)
    positions = ordered["position"].to_list()
    incident: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for rank, (source, partition, src, dst) in enumerate(rows.iter_rows()):
        endpoints = {str(value) for value in (src, dst) if value is not None and str(value)}
        for endpoint in endpoints:
            incident[(source, partition, endpoint)].append(rank)
    targets = [[-1, -1, -1] for _ in range(len(ordered))]
    for rank, (source, partition, src, dst) in enumerate(rows.iter_rows()):
        endpoints = {
            (source, partition, str(value))
            for value in (src, dst)
            if value is not None and str(value)
        }
        candidates: list[list[int]] = []
        for key in endpoints:
            values = incident[key]
            start = bisect_left(values, later[rank])
            candidates.append(values[start : start + HORIZONS[-1]])
        count, previous = 0, -1
        for candidate in merge(*candidates):
            if candidate == previous:
                continue
            previous = candidate
            count += 1
            if count in HORIZONS:
                targets[positions[rank]][HORIZONS.index(count)] = positions[candidate]
            if count == HORIZONS[-1]:
                break
    return torch.tensor(targets, dtype=torch.long).reshape(-1, 3)


@final
class FutureDataset(Dataset[dict[str, Tensor]]):
    """Keep eligible anchors and batch their teacher histories separately"""

    def __init__(self, dataset: FlowDataset, events: pl.DataFrame) -> None:
        """Share the original tensors and compute future target indices once"""
        super().__init__()
        ordered = events.sort([SOURCE, PARTITION, END_TIME, ROW])
        if dataset.labels is not None or not dataset.targets[EVENT].equals(ordered[EVENT]):
            raise ValueError("future selection needs every unlabelled flow in dataset order")
        self.dataset = dataset
        self.targets = future_indices(ordered)
        self.anchors = (self.targets >= 0).any(1).nonzero().flatten()
        if not len(self.anchors):
            raise ValueError("no anchors have later endpoint-related flows in this partition")

    def __len__(self) -> int:
        """Return the number of anchors with a future target"""
        return len(self.anchors)

    @override
    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return a student history and indices for its teacher targets"""
        anchor = int(self.anchors[index])
        return {**self.dataset[anchor], "future": self.targets[anchor]}

    def collate(self, samples: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
        """Encode each distinct future target history once per batch"""
        batch = collate(samples)
        targets = torch.stack([sample["future"] for sample in samples])
        valid = targets >= 0
        unique, inverse = cast(
            tuple[Tensor, Tensor], torch.unique(targets[valid], return_inverse=True)
        )
        futures = collate([self.dataset[int(index)] for index in unique])
        batch.update({f"future_{name}": value for name, value in futures.items()})
        indices = torch.full_like(targets, -1)
        indices[valid] = inverse
        batch["future_index"] = indices
        return batch


if __name__ == "__main__":
    events = pl.DataFrame(
        {
            SOURCE: ["nf3"] * 20,
            PARTITION: ["train"] * 17 + ["validation"] * 3,
            END_TIME: range(20),
            ROW: range(20),
            EVENT: [f"flow:{i}" for i in range(20)],
            ROUTING_COLUMNS[0]: ["A"] * 20,
            ROUTING_COLUMNS[1]: ["A"] + ["B"] * 19,
        }
    )
    targets = future_indices(events)
    assert targets[0].tolist() == [1, 4, 16]
    assert targets[16].tolist() == [-1, -1, -1]
    print("Future horizons and partition boundaries passed")
