"""Build bounded flow histories and expose them as PyTorch batches"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import copy
from typing import final, override

import polars as pl
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.config import Config
from src.data.features import (
    CATEGORICAL_COLUMNS,
    END_TIME,
    EVENT,
    NUMERIC_COLUMNS,
    PARTITION,
    ROW,
    SCORE,
    SOURCE,
)
from src.data.preprocess import PORT_BUCKET_START, State, transform

CATEGORICAL = (
    *(f"{column}_id" for column in CATEGORICAL_COLUMNS),
    "L4_SRC_PORT_range",
    "L4_DST_PORT_range",
)


@final
class FlowDataset(Dataset[dict[str, Tensor]]):
    """Store flow tensors once and slice histories as batches need them"""

    def __init__(
        self,
        events: pl.DataFrame,
        contexts: pl.DataFrame,
    ) -> None:
        """Validate context indices and convert feature columns to CPU tensors

        Args:
            events: Preprocessed flows with ordering fields, numeric values,
                missing flags and categorical IDs
            contexts: One row per event, produced by ``build_context`` for the
                same events and containing starts, target positions, and IDs

        Raises:
            ValueError: If fields are missing, indices are invalid,
                ordering fields contain nulls, or no target is available
        """
        required = {
            END_TIME,
            EVENT,
            PARTITION,
            ROW,
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
        if any(ordered[column].null_count() for column in (SOURCE, PARTITION, END_TIME)):
            raise ValueError("ordering columns cannot contain nulls")
        positions = contexts.get_column("target_position")
        starts = contexts.get_column("context_start")
        if (
            any(not value.dtype.is_integer() or value.null_count() for value in (starts, positions))
            or (starts < 0).any()
            or (starts > positions).any()
            or (positions >= ordered.height).any()
        ):
            raise ValueError("contexts do not index the supplied events")
        if not ordered[EVENT].gather(positions).equals(contexts["target_event"]):
            raise ValueError("contexts do not index the supplied events")

        if contexts.is_empty():
            raise ValueError("dataset contains no usable targets")
        self.labels: Tensor | None = None
        self.targets = ordered.select(pl.col(EVENT, END_TIME).gather(positions))

        # ponytail: one dataset in memory; stream shards if it exceeds the job's RAM
        self.starts = torch.tensor(contexts["context_start"].to_numpy(), dtype=torch.int64)
        self.ends = torch.tensor(contexts["target_position"].to_numpy(), dtype=torch.int64) + 1
        self.numeric = torch.tensor(ordered.select(NUMERIC_COLUMNS).to_numpy(), dtype=torch.float32)
        self.missing = torch.tensor(
            ordered.select(f"{column}_missing" for column in NUMERIC_COLUMNS).to_numpy(),
            dtype=torch.bool,
        )
        self.categorical = torch.tensor(ordered.select(CATEGORICAL).to_numpy(), dtype=torch.int64)

    def with_labels(self, events: pl.DataFrame) -> FlowDataset:
        """Select scorable targets while sharing the existing feature tensors

        Args:
            events: The same flows, with Label and score columns

        Returns:
            A labelled view; this dataset remains unchanged for pretraining
        """
        ordered = events.select(SOURCE, PARTITION, END_TIME, ROW, EVENT, "Label", SCORE).sort(
            [SOURCE, PARTITION, END_TIME, ROW]
        )
        positions = self.ends.numpy() - 1
        if len(ordered) != len(self.numeric) or not ordered[EVENT].gather(positions).equals(
            self.targets[EVENT]
        ):
            raise ValueError("labels must belong to the same ordered flows")
        labels = ordered["Label"]
        if labels.null_count() or not labels.is_in([0, 1]).all():
            raise ValueError("Label must contain only 0 and 1")
        selected = ordered[SCORE].gather(positions).fill_null(False)
        if not selected.any():
            raise ValueError("dataset contains no scorable targets")
        keep = torch.tensor(selected.to_numpy(), dtype=torch.bool)
        labelled = copy(self)
        labelled.labels = torch.tensor(labels.to_numpy(), dtype=torch.int64)
        labelled.starts, labelled.ends = self.starts[keep], self.ends[keep]
        labelled.targets = self.targets.filter(selected)
        return labelled

    def __len__(self) -> int:
        """Return the number of target flows"""
        return len(self.starts)

    @override
    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Slice a target and its earlier flows without adding padding

        Args:
            index: Position among target flows, supporting negative indices

        Returns:
            Numeric, missing, and categorical tensors shaped ``(events, fields)``
            and, in supervised mode, the final flow's scalar binary label
        """
        start = int(self.starts[index])
        end = int(self.ends[index])
        sample = {
            "numeric": self.numeric[start:end],
            "missing": self.missing[start:end],
            "categorical": self.categorical[start:end],
        }
        if self.labels is not None:
            sample["label"] = self.labels[end - 1]
        return sample


def build_context(frame: pl.DataFrame, horizon_minutes: int, max_events: int) -> pl.DataFrame:
    """Find past-only histories within each source and partition

    Args:
        frame: Flows with source, partition, event ID, end time in integer
            milliseconds, and original row as the timestamp tie-breaker
        horizon_minutes: Positive age limit, including flows exactly at the boundary
        max_events: Positive length cap including the target

    Returns:
        Sorted event IDs, global start/target indices, lengths, and elapsed times

    Raises:
        ValueError: If required values are missing, limits are invalid, or ties remain
        TypeError: If end times are not integers
    """
    if horizon_minutes <= 0 or max_events <= 0:
        raise ValueError("horizon_minutes and max_events must be positive")
    required = [SOURCE, PARTITION, END_TIME, ROW, EVENT]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"missing context columns: {', '.join(sorted(missing))}")
    if any(frame[column].null_count() for column in required):
        raise ValueError("context columns cannot contain nulls")
    if frame.height and not frame.schema[END_TIME].is_integer():
        raise TypeError(f"{END_TIME} must contain integer milliseconds")
    ordered = frame.select(required).sort(required[:-1], maintain_order=True)
    if ordered.select(required[:-1]).is_duplicated().any():
        raise ValueError("row must break equal end_time ties within a source partition")
    ordered = ordered.with_row_index("target_position")
    group = [SOURCE, PARTITION]
    time = pl.col(END_TIME).cast(pl.Int128)
    position = pl.col("target_position")
    start = time.search_sorted(time - horizon_minutes * 60_000, side="left").over(group)
    start += position.first().over(group)
    return ordered.with_columns(
        pl.max_horizontal(start.cast(pl.Int64), position.cast(pl.Int64) - max_events + 1)
        .cast(pl.UInt32)
        .alias("context_start")
    ).select(
        SOURCE,
        PARTITION,
        pl.col(EVENT).alias("target_event"),
        "context_start",
        "target_position",
        (position - pl.col("context_start") + 1).alias("context_length"),
        (time - time.gather(pl.col("context_start"))).cast(pl.Int64).alias("elapsed_ms"),
    )


def collate(samples: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Left-pad histories so each target stays last

    Args:
        samples: Nonempty labeled or unlabeled examples

    Returns:
        Feature tensors, a True-at-padding mask, and labels for supervised batches
    """
    if not samples:
        raise ValueError("cannot collate an empty batch")
    lengths = torch.tensor([len(sample["numeric"]) for sample in samples])
    batch = {
        name: pad_sequence(
            [sample[name] for sample in samples], batch_first=True, padding_side="left"
        )
        for name in ("numeric", "missing", "categorical")
    }
    batch["padding"] = torch.arange(int(lengths.max())) < lengths.max() - lengths[:, None]
    if any("label" in sample for sample in samples):
        if not all("label" in sample for sample in samples):
            raise ValueError("cannot mix labeled and unlabeled examples")
        batch["label"] = torch.stack([sample["label"] for sample in samples])
    return batch


def vocabulary_sizes(state: State) -> list[int]:
    """Return embedding sizes, including padding, unknown, and missing IDs"""
    sizes: list[int] = []
    for column in CATEGORICAL:
        if column.endswith("_range"):
            sizes.append(5)
        elif column.removesuffix("_id") in state["ports"]:
            values = state["ports"][column.removesuffix("_id")].values()
            sizes.append(max((PORT_BUCKET_START + 7, *values)) + 1)
        else:
            values = state["categorical"][column.removesuffix("_id")].values()
            sizes.append(max((2, *values)) + 1)
    return sizes


def make_datasets(
    frame: pl.DataFrame,
    state: State,
    config: Config,
    names: tuple[str, ...],
    *,
    supervised: bool = True,
) -> dict[str, FlowDataset]:
    """Build histories separately for each requested partition

    Args:
        frame: Split raw flows
        state: Training-only preprocessing values
        config: Context length and age limits
        names: Partitions to load
        supervised: Include labels and filter unscorable targets when True
    """
    events = transform(frame.filter(pl.col(PARTITION).is_in(names)).lazy(), state).collect()
    output: dict[str, FlowDataset] = {}
    for name in names:
        partition = events.filter(pl.col(PARTITION) == name)
        contexts = build_context(partition, config.data.horizon_minutes, config.data.max_events)
        dataset = FlowDataset(partition, contexts)
        output[name] = dataset.with_labels(partition) if supervised else dataset
    return output


def make_loader(
    dataset: Dataset[dict[str, Tensor]],
    config: Config,
    *,
    shuffle: bool,
    device: torch.device,
    collate_fn: Callable[[Sequence[dict[str, Tensor]]], dict[str, Tensor]] = collate,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Batch histories with left padding; shuffle targets only during training"""
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=shuffle,
        num_workers=config.data.workers,
        persistent_workers=config.data.workers > 0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
