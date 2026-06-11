"""Read one NF3 file and split it by completion time"""

from pathlib import Path

import polars as pl

from src.config import Config
from src.data.features import END_TIME, EVENT, PARTITION, RAW_COLUMNS, ROW, SCORE, SOURCE, TARGETS


def load(config: Config) -> pl.LazyFrame:
    """Read the configured Parquet file and assign stable flow IDs

    Args:
        config: Contains data.root and one data.dataset name

    Returns:
        Required NF3 columns plus dataset, row, and event IDs
    """
    source = config.data.dataset
    if not isinstance(source, str) or not source.strip():
        raise ValueError("data.dataset must name one NF3 dataset")
    path = Path(config.data.root) / f"{source}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pl.scan_parquet(path)
    missing = set(RAW_COLUMNS).difference(frame.collect_schema())
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return (
        frame.select(RAW_COLUMNS)
        .with_row_index(ROW)
        .with_columns(
            pl.lit(source).alias(SOURCE),
            pl.concat_str(pl.lit(source), pl.col(ROW).cast(pl.String), separator=":").alias(EVENT),
            pl.col("SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES").cast(pl.Float64),
        )
    )


def chronological(
    frame: pl.DataFrame,
    train_fraction: float,
    validation_fraction: float,
    purge_minutes: int,
) -> pl.DataFrame:
    """Assign train, validation, and test periods with a gap at each boundary

    Args:
        frame: Raw flows with row IDs and integer completion times in milliseconds
        train_fraction: Fraction before the first time cutoff
        validation_fraction: Fraction between cutoffs
        purge_minutes: Minutes excluded on each side of each cutoff

    Returns:
        Time-sorted flows with partition and score columns. Conflicting duplicate
        labels remain available as history but cannot be classification targets
    """
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("train and validation fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1 or purge_minutes < 0:
        raise ValueError("fractions must total less than one and purge must be nonnegative")
    missing = {ROW, *RAW_COLUMNS}.difference(frame.columns)
    if missing:
        raise ValueError(f"missing split columns: {', '.join(sorted(missing))}")
    if frame[END_TIME].null_count() or not frame.schema[END_TIME].is_integer():
        raise ValueError(f"{END_TIME} must contain integer milliseconds without nulls")
    train_row = int(len(frame) * train_fraction)
    validation_row = int(len(frame) * (train_fraction + validation_fraction))
    if not 0 < train_row < validation_row < len(frame):
        raise ValueError("split fractions leave an empty partition")

    ordered = frame.sort([END_TIME, ROW])
    train_cutoff, validation_cutoff = ordered[END_TIME].gather([train_row, validation_row])
    purge = purge_minutes * 60_000
    raw = [column for column in RAW_COLUMNS if column not in TARGETS]
    conflict = pl.any_horizontal(pl.col(target).n_unique().over(raw) > 1 for target in TARGETS)
    assigned = ordered.with_columns(
        (~conflict).alias(SCORE),
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
        .alias(PARTITION),
    ).filter(pl.col(PARTITION) != "purged")
    if set(assigned[PARTITION]) != {"train", "validation", "test"}:
        raise ValueError("time cutoffs and purge leave an empty partition")
    return assigned


def load_split(config: Config) -> pl.DataFrame:
    """Read and split once; later stages share the in-memory frame"""
    return chronological(
        load(config).collect(),
        config.split.train_fraction,
        config.split.validation_fraction,
        config.split.purge_minutes,
    )


if __name__ == "__main__":
    flows = pl.DataFrame(
        {name: range(30) if name == END_TIME else [1] * 30 for name in RAW_COLUMNS}
    )
    conflicting = flows.head(1).with_columns(pl.lit(0, dtype=pl.Int64).alias("Label"))
    flows = pl.concat([flows, conflicting]).with_row_index(ROW)
    split = chronological(flows, 0.7, 0.15, 0)
    assert set(split[PARTITION]) == {"train", "validation", "test"}
    assert (~split[SCORE]).sum() == 2
    assert split[END_TIME].is_sorted()
    print("Chronological split and conflicting-label checks passed")
