"""Deterministic, training-partition-only preprocessing for model-view rows.

Rows have ``numeric`` and ``categorical`` mappings.  Port fields must contain
the original integer port values here; their high-port frequency buckets are
learned only by :func:`fit`.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from bisect import bisect_left
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

PAD = "PAD"
UNK = "UNK"
MISSING = "MISSING"
_SPECIAL_TOKENS = (PAD, UNK, MISSING)
_PORT_FREQUENCY_SUFFIX = "__frequency"
_PORT_FREQUENCY_TOKENS = (*_SPECIAL_TOKENS, *(f"FREQ_{bucket}" for bucket in range(8)))


@dataclass(frozen=True)
class NumericState:
    """Train-only numeric statistics after log transformation."""

    median: float
    lower: float
    upper: float
    mean: float
    std: float


@dataclass(frozen=True)
class PreprocessorState:
    """Immutable state with only deterministic JSON-compatible primitives."""

    numeric: tuple[tuple[str, NumericState], ...]
    categorical: tuple[tuple[str, tuple[str, ...]], ...]
    port_fields: tuple[str, ...]
    port_frequencies: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    port_thresholds: tuple[tuple[str, tuple[int, ...]], ...]

    def to_dict(self) -> dict[str, object]:
        """Return a canonical JSON-ready representation."""
        return {
            "version": 1,
            "numeric": {
                field: {
                    "median": values.median,
                    "lower": values.lower,
                    "upper": values.upper,
                    "mean": values.mean,
                    "std": values.std,
                }
                for field, values in self.numeric
            },
            "categorical": {field: list(vocabulary) for field, vocabulary in self.categorical},
            "port_fields": list(self.port_fields),
            "port_frequencies": {
                field: {port: count for port, count in frequencies}
                for field, frequencies in self.port_frequencies
            },
            "port_thresholds": {
                field: list(thresholds) for field, thresholds in self.port_thresholds
            },
        }


def canonical_json(state: PreprocessorState) -> str:
    """Serialize state without platform-specific ordering or whitespace."""
    return json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)


def state_hash(state: PreprocessorState) -> str:
    """Return the SHA-256 identity of the exact fitted state."""
    return hashlib.sha256(canonical_json(state).encode("utf-8")).hexdigest()


def _mapping(row: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = row.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Row must contain a {name!r} mapping.")
    raw = cast(Mapping[object, object], value)
    if not all(isinstance(field, str) for field in raw):
        raise ValueError(f"{name} field names must be strings.")
    return cast(Mapping[str, object], raw)


def _view(row: Mapping[str, object]) -> tuple[Mapping[str, object], Mapping[str, object]]:
    unexpected = sorted(set(row) - {"numeric", "categorical"})
    if unexpected:
        raise ValueError(f"Model-view rows cannot contain non-model fields: {unexpected}")
    return _mapping(row, "numeric"), _mapping(row, "categorical")


def _numeric(value: object, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Numeric field {field} must be a finite number or null.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Numeric field {field} must be finite.")
    if number < 0:
        raise ValueError(f"Numeric field {field} must be non-negative for log1p.")
    return number


def _category(value: object, field: str) -> str:
    if value is None:
        return MISSING
    if isinstance(value, str):
        return f"str:{value}"
    if isinstance(value, bool):
        return f"bool:{str(value).lower()}"
    if isinstance(value, int):
        return f"int:{value}"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Categorical field {field} must be finite.")
        return f"float:{value.hex()}"
    raise ValueError(f"Categorical field {field} must be a scalar or null.")


def _port(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Port field {field} must be an integer in 0..65535 or null.")
    if isinstance(value, int):
        port = value
    elif isinstance(value, float) and value.is_integer():
        port = int(value)
    else:
        raise ValueError(f"Port field {field} must be an integer in 0..65535 or null.")
    if not 0 <= port <= 65535:
        raise ValueError(f"Port field {field} is outside 0..65535: {port}.")
    return port


def _quantile(sorted_values: list[float], probability: float) -> float:
    """Use linear interpolation, matching a deterministic inclusive quantile."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def _numeric_state(values: list[float], field: str) -> NumericState:
    if not values:
        raise ValueError(f"Numeric training field {field} has no observed values.")
    median = float(statistics.median(values))
    transformed = sorted(math.log1p(value) for value in values)
    lower = _quantile(transformed, 0.01)
    upper = _quantile(transformed, 0.99)
    clipped = [min(max(value, lower), upper) for value in transformed]
    mean = math.fsum(clipped) / len(clipped)
    variance = math.fsum((value - mean) ** 2 for value in clipped) / len(clipped)
    return NumericState(median, lower, upper, mean, math.sqrt(variance))


def _port_vocabulary() -> tuple[str, ...]:
    return (*_SPECIAL_TOKENS, *(f"PORT_{port}" for port in range(1024)), "REGISTERED", "DYNAMIC")


def _frequency_thresholds(counts: list[int]) -> tuple[int, ...]:
    ordered = sorted(counts)
    return tuple(ordered[math.ceil(index * len(ordered) / 8) - 1] for index in range(1, 8))


def fit(
    training_rows: Iterable[Mapping[str, object]], *, port_fields: Iterable[object] = ()
) -> PreprocessorState:
    """Fit preprocessing state from training rows only.

    All rows must have identical numeric and categorical field sets.  This
    function never accepts labels, routing values, or rows outside the supplied
    training partition.
    """
    checked_ports: list[str] = []
    for field in port_fields:
        if not isinstance(field, str) or not field:
            raise ValueError("Port field names must be non-empty strings.")
        checked_ports.append(field)
    ports = tuple(sorted(set(checked_ports)))
    # ponytail: exact training quantiles retain numeric values; replace with a
    # bounded external reducer before multi-million-flow fitting.
    numeric_values: dict[str, list[float]] | None = None
    categories: dict[str, set[str]] | None = None
    high_port_counts: dict[str, dict[str, int]] = {}
    row_count = 0

    for row in training_rows:
        numeric, categorical = _view(row)
        if numeric_values is None:
            numeric_values = {field: [] for field in sorted(numeric)}
            categories = {field: set() for field in sorted(categorical)}
            overlap = sorted(set(numeric) & set(categorical))
            if overlap:
                raise ValueError(f"Fields cannot be both numeric and categorical: {overlap}")
            if not set(ports) <= set(categorical):
                missing = sorted(set(ports) - set(categorical))
                raise ValueError(f"Port fields are not categorical fields: {missing}")
            high_port_counts = {field: {} for field in ports}
        elif set(numeric) != set(numeric_values) or set(categorical) != set(categories or {}):
            raise ValueError("Every training row must have identical model-view fields.")

        for field, value in numeric.items():
            number = _numeric(value, field)
            if number is not None:
                numeric_values[field].append(number)
        for field, value in categorical.items():
            if field in high_port_counts:
                port = _port(value, field)
                if port is not None and port > 1023:
                    port_key = str(port)
                    high_port_counts[field][port_key] = high_port_counts[field].get(port_key, 0) + 1
            else:
                assert categories is not None
                categories[field].add(_category(value, field))
        row_count += 1

    if row_count == 0 or numeric_values is None or categories is None:
        raise ValueError("Training rows must not be empty.")

    numeric_state = tuple(
        (field, _numeric_state(values, field)) for field, values in sorted(numeric_values.items())
    )
    categorical_state = tuple(
        [
            (
                field,
                _port_vocabulary()
                if field in high_port_counts
                else (*_SPECIAL_TOKENS, *sorted(values - set(_SPECIAL_TOKENS))),
            )
            for field, values in sorted(categories.items())
        ]
        + [(f"{field}{_PORT_FREQUENCY_SUFFIX}", _PORT_FREQUENCY_TOKENS) for field in ports]
    )
    frequencies = tuple(
        (field, tuple(sorted(counts.items(), key=lambda item: int(item[0]))))
        for field, counts in sorted(high_port_counts.items())
    )
    thresholds = tuple(
        (field, _frequency_thresholds(list(counts.values())) if counts else ())
        for field, counts in sorted(high_port_counts.items())
    )
    return PreprocessorState(numeric_state, categorical_state, ports, frequencies, thresholds)


def fit_training_partition(
    rows_by_event_id: Mapping[str, Mapping[str, object]],
    assignments: Iterable[Mapping[str, object]],
    *,
    port_fields: Iterable[object] = (),
) -> PreprocessorState:
    """Fit only rows assigned to a training role; other partition values are never read."""
    train_ids: list[str] = []
    seen: set[str] = set()
    for assignment in assignments:
        event_id = assignment.get("event_id")
        partition = assignment.get("partition")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("Every split assignment requires a non-empty event_id.")
        if event_id in seen:
            raise ValueError(f"Split assignments repeat event_id {event_id!r}.")
        if not isinstance(partition, str) or partition not in {
            "train",
            "validation",
            "test",
            "source",
            "target",
            "sealed",
        }:
            raise ValueError(f"Unknown split partition {partition!r}.")
        seen.add(event_id)
        if partition in {"train", "source"}:
            train_ids.append(event_id)
    missing = sorted(set(train_ids) - set(rows_by_event_id))
    if missing:
        raise ValueError(f"Training rows are missing assigned event IDs: {missing}")
    return fit(
        (rows_by_event_id[event_id] for event_id in sorted(train_ids)), port_fields=port_fields
    )


def _state_maps(
    state: PreprocessorState,
) -> tuple[
    dict[str, NumericState],
    dict[str, dict[str, int]],
    dict[str, dict[str, int]],
    dict[str, tuple[int, ...]],
]:
    numeric = dict(state.numeric)
    vocabulary = {
        field: {token: index for index, token in enumerate(tokens)}
        for field, tokens in state.categorical
    }
    frequencies = {field: dict(counts) for field, counts in state.port_frequencies}
    thresholds = dict(state.port_thresholds)
    return numeric, vocabulary, frequencies, thresholds


def transform(
    state: PreprocessorState, row: Mapping[str, object]
) -> dict[str, dict[str, float | int]]:
    """Transform one row without mutating the fitted state or its input row."""
    numeric_input, categorical_input = _view(row)
    numeric_state, vocabulary, frequencies, thresholds = _state_maps(state)
    output_only = {f"{field}{_PORT_FREQUENCY_SUFFIX}" for field in state.port_fields}
    expected_categorical = set(vocabulary) - output_only
    if set(numeric_input) != set(numeric_state) or set(categorical_input) != expected_categorical:
        raise ValueError("Row fields do not match the fitted preprocessor state.")

    numeric: dict[str, float | int] = {}
    missing: dict[str, float | int] = {}
    for field, values in numeric_state.items():
        value = _numeric(numeric_input[field], field)
        missing[field] = int(value is None)
        transformed = math.log1p(values.median if value is None else value)
        clipped = min(max(transformed, values.lower), values.upper)
        numeric[field] = 0.0 if values.std == 0 else (clipped - values.mean) / values.std

    categorical: dict[str, float | int] = {}
    for field, tokens in vocabulary.items():
        if field in output_only:
            continue
        if field not in frequencies:
            categorical[field] = tokens.get(_category(categorical_input[field], field), tokens[UNK])
            continue
        port = _port(categorical_input[field], field)
        frequency_tokens = vocabulary[f"{field}{_PORT_FREQUENCY_SUFFIX}"]
        if port is None:
            categorical[field] = tokens[MISSING]
            categorical[f"{field}{_PORT_FREQUENCY_SUFFIX}"] = frequency_tokens[MISSING]
        elif port <= 1023:
            categorical[field] = tokens[f"PORT_{port}"]
            categorical[f"{field}{_PORT_FREQUENCY_SUFFIX}"] = frequency_tokens[PAD]
        else:
            categorical[field] = tokens["REGISTERED" if port <= 49151 else "DYNAMIC"]
            count = frequencies[field].get(str(port))
            categorical[f"{field}{_PORT_FREQUENCY_SUFFIX}"] = (
                frequency_tokens[UNK]
                if count is None
                else frequency_tokens[f"FREQ_{bisect_left(thresholds[field], count)}"]
            )
    return {"numeric": numeric, "missing": missing, "categorical": categorical}
