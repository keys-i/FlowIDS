"""Create leakage-resistant chronological NetFlow splits."""

from __future__ import annotations

import polars as pl

from src.data.features import END_TIME, PARTITION, RAW_COLUMNS, ROW, SCORE, TARGETS


def _check(frame: pl.LazyFrame, *, train: float, validation: float, purge: int) -> list[str]:
    """Validate split inputs and return raw fields for duplicate checks."""
    if not 0 < train < 1 or not 0 < validation < 1 or train + validation >= 1:
        raise ValueError("train and validation fractions must be positive and total less than one")
    if purge < 0:
        raise ValueError("purge_minutes must not be negative")

    columns = frame.collect_schema().names()
    missing = {END_TIME, ROW, *RAW_COLUMNS, *TARGETS}.difference(columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    return [column for column in RAW_COLUMNS if column not in TARGETS]


def _score(frame: pl.LazyFrame, raw: list[str]) -> pl.LazyFrame:
    """Mark flows that do not have conflicting duplicate labels."""
    if not raw:
        raise ValueError("no raw non-target columns available for duplicate grouping")
    conflict = pl.any_horizontal([pl.col(target).n_unique().over(raw).gt(1) for target in TARGETS])
    return frame.with_columns((~conflict).alias(SCORE))


def chronological(
    frame: pl.LazyFrame,
    train_fraction: float,
    validation_fraction: float,
    purge_minutes: int,
) -> pl.LazyFrame:
    """Split completed flows by time, purging both sides of each boundary."""
    raw = _check(
        frame,
        train=train_fraction,
        validation=validation_fraction,
        purge=purge_minutes,
    )
    count = (
        frame.select(pl.len().alias("count"), pl.col(END_TIME).null_count().alias("nulls"))
        .collect()
        .row(0)
    )
    total, nulls = count
    if nulls:
        raise ValueError(f"{END_TIME} contains null values")

    train_row = int(total * train_fraction)
    validation_row = int(total * (train_fraction + validation_fraction))
    if not 0 < train_row < validation_row < total:
        raise ValueError("split fractions leave an empty partition")

    ordered = frame.sort([END_TIME, ROW])
    cutoffs = (
        ordered.select(pl.col(END_TIME).gather([train_row, validation_row]))
        .collect()[END_TIME]
        .to_list()
    )
    train_cutoff, validation_cutoff = cutoffs
    purge = purge_minutes * 60_000
    assigned = _score(ordered, raw).with_columns(
        pl.when(pl.col(END_TIME) < train_cutoff - purge)
        .then(pl.lit("train"))
        .when(
            (pl.col(END_TIME) > train_cutoff + purge)
            & (pl.col(END_TIME) < validation_cutoff - purge)
        )
        .then(pl.lit("validation"))
        .when(pl.col(END_TIME) > validation_cutoff + purge)
        .then(pl.lit("test"))
        .otherwise(pl.lit("purged"))
        .alias(PARTITION)
    )
    sizes = (
        assigned.select(
            [
                (pl.col(PARTITION) == name).sum().alias(name)
                for name in ("train", "validation", "test")
            ]
        )
        .collect()
        .row(0)
    )
    if not all(sizes):
        raise ValueError("time cutoffs and purge leave an empty partition")
    return assigned.filter(pl.col(PARTITION) != "purged")


def holdout(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Tag an untouched source as test data and exclude conflicting duplicates."""
    raw = _check(frame, train=0.7, validation=0.15, purge=0)
    return _score(frame.sort([END_TIME, ROW]), raw).with_columns(pl.lit("test").alias(PARTITION))
