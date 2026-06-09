from __future__ import annotations

from typing import TypedDict

import polars as pl

from src.data.features import (
    AUDIT_COLUMNS,
    CATEGORICAL_COLUMNS,
    CONTEXT_COLUMNS,
    HEAVY_TAIL_COLUMNS,
    IAT_COLUMNS,
    NUMERIC_COLUMNS,
    ROUTING_COLUMNS,
    TARGET_COLUMNS,
)

PAD = 0
UNK = 1
MISSING = 2
PORT_BUCKET_START = 1027


class State(TypedDict):
    numeric: dict[str, dict[str, float]]
    categorical: dict[str, dict[str, int]]
    ports: dict[str, dict[str, int]]
    keep: list[str]


def _iat_missing() -> pl.Expr:
    src_min, src_max, src_avg, src_std, dst_min, dst_max, dst_avg, dst_std = IAT_COLUMNS
    return pl.any_horizontal([pl.col(column).is_null() for column in IAT_COLUMNS]) | (
        (pl.col(src_min) > pl.col(src_avg))
        | (pl.col(src_avg) > pl.col(src_max))
        | (pl.col(dst_min) > pl.col(dst_avg))
        | (pl.col(dst_avg) > pl.col(dst_max))
        | (pl.col(src_std) < 0)
        | (pl.col(dst_std) < 0)
    )


def _numeric(column: str) -> pl.Expr:
    value = pl.col(column).cast(pl.Float64, strict=False)
    return pl.when(_iat_missing()).then(None).otherwise(value) if column in IAT_COLUMNS else value


def _value(value: object) -> float:
    if not isinstance(value, int | float):
        raise ValueError("numeric preprocessing statistics must be numbers")
    return float(value)


def _stats(frame: pl.LazyFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    expressions = []
    for column in columns:
        value = _numeric(column)
        if column in HEAVY_TAIL_COLUMNS:
            value = value.clip(lower_bound=0).log1p()
        expressions.extend(
            (
                value.median().alias(f"{column}:median"),
                value.quantile(0.01).alias(f"{column}:low"),
                value.quantile(0.99).alias(f"{column}:high"),
            )
        )
    row = frame.select(expressions).collect().row(0, named=True)
    return {
        column: {name: _value(row[f"{column}:{name}"]) for name in ("median", "low", "high")}
        for column in columns
    }


def fit(train: pl.LazyFrame) -> State:
    """Fit portable M0 preprocessing on training flows only."""
    names = set(train.collect_schema().names())
    numeric = [column for column in NUMERIC_COLUMNS if column in names]
    categorical = [column for column in CATEGORICAL_COLUMNS if column in names]
    ports = [column for column in categorical if column.startswith("L4_")]
    categoricals = [column for column in categorical if column not in ports]
    numeric_stats = _stats(train, numeric)

    expressions = []
    for column in numeric:
        value = _numeric(column)
        if column in HEAVY_TAIL_COLUMNS:
            value = value.clip(lower_bound=0).log1p()
        stats = numeric_stats[column]
        value = value.fill_null(stats["median"]).clip(stats["low"], stats["high"])
        expressions.extend(
            (value.mean().alias(f"{column}:mean"), value.std().alias(f"{column}:std"))
        )
    row = train.select(expressions).collect().row(0, named=True)
    for column in numeric:
        stats = numeric_stats[column]
        stats["mean"] = _value(row[f"{column}:mean"])
        std = _value(row[f"{column}:std"])
        stats["std"] = std if std else 1.0

    vocabularies = {
        column: {
            str(value): index
            for index, value in enumerate(
                train.select(pl.col(column).drop_nulls().unique().sort())
                .collect()
                .get_column(column)
                .to_list(),
                start=3,
            )
        }
        for column in categoricals
    }
    port_buckets: dict[str, dict[str, int]] = {}
    for column in ports:
        counts = (
            train.filter(pl.col(column).is_between(1024, 65535))
            .group_by(column)
            .len()
            .sort(["len", column], descending=[True, False])
            .collect()
        )
        values = counts.get_column(column).to_list()
        port_buckets[column] = {
            str(value): PORT_BUCKET_START + index * 8 // max(len(values), 1)
            for index, value in enumerate(values)
        }

    return {
        "numeric": numeric_stats,
        "categorical": vocabularies,
        "ports": port_buckets,
        "keep": [
            column
            for column in (
                *AUDIT_COLUMNS,
                *ROUTING_COLUMNS,
                *TARGET_COLUMNS,
                *CONTEXT_COLUMNS,
            )
            if column in names
        ],
    }


def _port(column: str, buckets: dict[str, int]) -> list[pl.Expr]:
    text = pl.col(column).cast(pl.String)
    value = pl.col(column).cast(pl.Int64, strict=False)
    ids = (
        pl.when(value.is_null())
        .then(MISSING)
        .when(value.is_between(0, 1023))
        .then(value + 3)
        .when(value.is_between(1024, 65535))
        .then(text.replace_strict(buckets, default=UNK, return_dtype=pl.Int64))
        .otherwise(UNK)
        .cast(pl.Int64)
        .alias(f"{column}_id")
    )
    ranges = (
        pl.when(value.is_null())
        .then(MISSING)
        .when(value.is_between(0, 1023))
        .then(PAD)
        .when(value.is_between(1024, 49151))
        .then(3)
        .when(value.is_between(49152, 65535))
        .then(4)
        .otherwise(UNK)
        .cast(pl.Int64)
        .alias(f"{column}_range")
    )
    return [ids, ranges]


def transform(frame: pl.LazyFrame, state: State) -> pl.LazyFrame:
    """Apply a state fitted by :func:`fit` without retaining raw model features."""
    numeric = state["numeric"]
    categorical = state["categorical"]
    ports = state["ports"]
    expressions = [pl.col(column) for column in state["keep"]]

    for column, stats in numeric.items():
        value = _numeric(column)
        missing = value.is_null()
        if column in HEAVY_TAIL_COLUMNS:
            value = value.clip(lower_bound=0).log1p()
        expressions.extend(
            (
                (
                    (
                        value.fill_null(stats["median"]).clip(stats["low"], stats["high"])
                        - stats["mean"]
                    )
                    / stats["std"]
                ).alias(column),
                missing.cast(pl.Int8).alias(f"{column}_missing"),
            )
        )
    for column, vocabulary in categorical.items():
        expressions.append(
            pl.when(pl.col(column).is_null())
            .then(MISSING)
            .otherwise(
                pl.col(column)
                .cast(pl.String)
                .replace_strict(vocabulary, default=UNK, return_dtype=pl.Int64)
            )
            .cast(pl.Int64)
            .alias(f"{column}_id")
        )
    for column, buckets in ports.items():
        expressions.extend(_port(column, buckets))
    return frame.select(expressions)
