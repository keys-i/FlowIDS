from __future__ import annotations

from collections import deque

import polars as pl

from src.data.features import END_TIME, EVENT, PARTITION, ROW, SOURCE


def build(frame: pl.DataFrame, horizon_minutes: int, max_events: int) -> pl.DataFrame:
    """Return compact causal context slices for sorted NetFlow events."""
    if horizon_minutes <= 0 or max_events <= 0:
        raise ValueError("horizon_minutes and max_events must be positive")

    required = [SOURCE, PARTITION, END_TIME, ROW, EVENT]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"missing context columns: {', '.join(sorted(missing))}")
    if any(frame[column].null_count() for column in required):
        raise ValueError("context columns cannot contain nulls")

    ordered = frame.select(required).sort(required[:-1], maintain_order=True)
    horizon = horizon_minutes * 60_000
    sources: list[object] = []
    partitions: list[object] = []
    targets: list[object] = []
    starts: list[int] = []
    positions: list[int] = []
    lengths: list[int] = []
    spans: list[int] = []
    group: tuple[object, object] | None = None
    previous: tuple[object, object, int, object] | None = None
    history: deque[tuple[int, int]] = deque()

    # ponytail: Python scan first; move this to Polars only if 67M-flow profiling requires it.
    for source, partition, end_time, row, event in ordered.iter_rows():
        if not isinstance(end_time, int):
            raise TypeError(f"{END_TIME} must contain integer milliseconds")
        key = (source, partition)
        order = (source, partition, end_time, row)
        if order == previous:
            raise ValueError("row must break equal end_time ties within a source partition")
        previous = order

        if key != group:
            group = key
            history.clear()

        target = len(targets)
        history.append((target, end_time))
        while end_time - history[0][1] > horizon:
            _ = history.popleft()
        start = max(history[0][0], target - max_events + 1)
        span = end_time - history[start - history[0][0]][1]

        sources.append(source)
        partitions.append(partition)
        targets.append(event)
        starts.append(start)
        positions.append(target)
        lengths.append(target - start + 1)
        spans.append(span)

    return pl.DataFrame(
        {
            SOURCE: sources,
            PARTITION: partitions,
            "target_event": targets,
            "context_start": starts,
            "target_position": positions,
            "context_length": lengths,
            "elapsed_ms": spans,
        },
        schema={
            SOURCE: ordered.schema[SOURCE],
            PARTITION: ordered.schema[PARTITION],
            "target_event": ordered.schema[EVENT],
            "context_start": pl.UInt32,
            "target_position": pl.UInt32,
            "context_length": pl.UInt32,
            "elapsed_ms": pl.Int64,
        },
    )
