"""Load configured NF3 sources lazily."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.config import Config
from src.data.features import (
    EVENT,
    RAW_COLUMNS,
    ROW,
    SOURCE,
)


def load_one(path: str | Path, source: str) -> pl.LazyFrame:
    """Return one validated source without collecting or reordering it."""
    file = Path(path)
    if not source:
        raise ValueError("source must not be empty")
    if file.suffix != ".parquet":
        raise ValueError(f"expected a Parquet file: {file}")
    if not file.is_file():
        raise FileNotFoundError(file)

    frame = pl.scan_parquet(file)
    missing = set(RAW_COLUMNS).difference(frame.collect_schema())
    if missing:
        raise ValueError(f"{file} is missing columns: {', '.join(sorted(missing))}")

    return (
        frame.select(pl.col(RAW_COLUMNS))
        .with_row_index(ROW)
        .with_columns(
            pl.lit(source).alias(SOURCE),
            pl.concat_str([pl.lit(source), pl.col(ROW).cast(pl.String)], separator=":").alias(
                EVENT
            ),
            pl.col("SRC_TO_DST_SECOND_BYTES").cast(pl.Float64),
            pl.col("DST_TO_SRC_SECOND_BYTES").cast(pl.Float64),
        )
    )


def load(config: Config) -> dict[str, pl.LazyFrame]:
    """Load configured sources separately, preserving source-local row order."""
    root = Path(config.data.root)
    sources = (*config.data.development, config.data.holdout)
    return {source: load_one(root / f"{source}.parquet", source) for source in sources}
