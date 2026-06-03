"""Deterministic, track-specific split manifests for flow-level evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, cast

Track = Literal["chronological", "endpoint_disjoint", "held_out_family", "cross_network"]
_TRACKS = frozenset({"chronological", "endpoint_disjoint", "held_out_family", "cross_network"})
_ANON_PRINCIPAL = re.compile(r"anon:[a-z0-9_-]+$")


@dataclass(frozen=True, slots=True)
class SplitRow:
    """Routing metadata only; canonical_family is forbidden from model and SSL inputs."""

    event_id: str
    completion_ms: int
    source_principal: str
    destination_principal: str
    source_unit_id: str
    capture_lineage_id: str
    exact_group: str
    near_group: str
    canonical_family: str | None = None
    campaign_id: str | None = None


@dataclass(frozen=True, slots=True)
class SplitSpec:
    """Frozen selectors for one evaluation track; unspecified selection is refused."""

    track: Track
    train_end_ms: int | None = None
    validation_end_ms: int | None = None
    purge_ms: int = 0
    held_principals: frozenset[str] = frozenset()
    held_families: frozenset[str] = frozenset()
    held_campaigns: frozenset[str] = frozenset()
    source_units: frozenset[str] = frozenset()
    target_units: frozenset[str] = frozenset()
    sealed_units: frozenset[str] = frozenset()


def _require_identifier(name: str, value: str | None) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _validate_row(row: SplitRow) -> None:
    _require_identifier("event_id", row.event_id)
    if _integer("completion_ms", row.completion_ms) < 0:
        raise ValueError("completion_ms must be non-negative")
    for name, value in (
        ("source_unit_id", row.source_unit_id),
        ("capture_lineage_id", row.capture_lineage_id),
        ("exact_group", row.exact_group),
        ("near_group", row.near_group),
    ):
        _require_identifier(name, value)
    for name, value in (
        ("source_principal", row.source_principal),
        ("destination_principal", row.destination_principal),
    ):
        _require_identifier(name, value)
        if not _ANON_PRINCIPAL.fullmatch(value):
            raise ValueError(
                f"{name} must be an opaque anon: identifier; raw endpoints are refused"
            )
    for name, value in (
        ("canonical_family", row.canonical_family),
        ("campaign_id", row.campaign_id),
    ):
        if value is not None:
            _require_identifier(name, value)


def _selector(name: str, values: frozenset[str]) -> None:
    if not values:
        raise ValueError(f"{name} selector is required")
    for value in values:
        _require_identifier(name, value)


def _validate_spec(spec: SplitSpec) -> None:
    if spec.track not in _TRACKS:
        raise ValueError(f"unknown split track {spec.track!r}")
    if _integer("purge_ms", spec.purge_ms) < 0:
        raise ValueError("purge_ms must be non-negative")
    selectors: tuple[tuple[str, object], ...] = (
        ("held_principals", spec.held_principals),
        ("held_families", spec.held_families),
        ("held_campaigns", spec.held_campaigns),
        ("source_units", spec.source_units),
        ("target_units", spec.target_units),
        ("sealed_units", spec.sealed_units),
    )
    if any(type(cast(object, values)) is not frozenset for _, values in selectors):
        raise ValueError("split selectors must be frozensets")
    if spec.track == "chronological":
        if spec.train_end_ms is None or spec.validation_end_ms is None:
            raise ValueError("chronological track requires both time cutoffs")
        train_end = _integer("train_end_ms", spec.train_end_ms)
        validation_end = _integer("validation_end_ms", spec.validation_end_ms)
        if train_end < 0 or validation_end < 0 or train_end >= validation_end:
            raise ValueError("chronological cutoffs must be non-negative and ordered")
        if any(values for _, values in selectors):
            raise ValueError("chronological track accepts no holdout selectors")
        return
    if spec.train_end_ms is not None or spec.validation_end_ms is not None or spec.purge_ms:
        raise ValueError(f"{spec.track} track does not accept chronological cutoffs")
    if spec.track == "endpoint_disjoint":
        _selector("held_principals", spec.held_principals)
        if any(
            (
                spec.held_families,
                spec.held_campaigns,
                spec.source_units,
                spec.target_units,
                spec.sealed_units,
            )
        ):
            raise ValueError("endpoint_disjoint track accepts only held_principals")
        return
    if spec.track == "held_out_family":
        _selector("held_families", spec.held_families)
        _selector("held_campaigns", spec.held_campaigns)
        if any((spec.held_principals, spec.source_units, spec.target_units, spec.sealed_units)):
            raise ValueError("held_out_family track accepts only family and campaign selectors")
        return
    _selector("source_units", spec.source_units)
    _selector("target_units", spec.target_units)
    _selector("sealed_units", spec.sealed_units)
    if (
        spec.source_units & spec.target_units
        or spec.source_units & spec.sealed_units
        or spec.target_units & spec.sealed_units
    ):
        raise ValueError("cross_network source, target, and sealed units must be disjoint")
    if any((spec.held_principals, spec.held_families, spec.held_campaigns)):
        raise ValueError("cross_network track accepts only source, target, and sealed unit roles")


def _partition(row: SplitRow, spec: SplitSpec) -> tuple[str | None, tuple[str, ...]]:
    if spec.track == "chronological":
        assert spec.train_end_ms is not None and spec.validation_end_ms is not None
        reasons: list[str] = []
        if abs(row.completion_ms - spec.train_end_ms) <= spec.purge_ms:
            reasons.append("purge:train_cutoff")
        if abs(row.completion_ms - spec.validation_end_ms) <= spec.purge_ms:
            reasons.append("purge:validation_cutoff")
        if reasons:
            return None, tuple(reasons)
        if row.completion_ms < spec.train_end_ms:
            return "train", ()
        if row.completion_ms < spec.validation_end_ms:
            return "validation", ()
        return "test", ()
    if spec.track == "endpoint_disjoint":
        source_held = row.source_principal in spec.held_principals
        destination_held = row.destination_principal in spec.held_principals
        if source_held and destination_held:
            return "target", ()
        if source_held or destination_held:
            return None, ("purge:endpoint_boundary",)
        return "source", ()
    if spec.track == "held_out_family":
        if row.canonical_family is None or row.campaign_id is None:
            raise ValueError("held_out_family rows require canonical_family and campaign_id")
        if row.canonical_family in spec.held_families or row.campaign_id in spec.held_campaigns:
            return "target", ()
        return "source", ()
    if row.source_unit_id in spec.source_units:
        return "source", ()
    if row.source_unit_id in spec.target_units:
        return "target", ()
    if row.source_unit_id in spec.sealed_units:
        return "sealed", ()
    return None, ("excluded:unassigned_source_unit",)


def _isolation_groups(row: SplitRow, spec: SplitSpec) -> tuple[tuple[str, str], ...]:
    groups = [("exact", row.exact_group), ("near", row.near_group)]
    if spec.track == "held_out_family" and row.campaign_id is not None:
        groups.append(("campaign", row.campaign_id))
    if spec.track == "cross_network":
        groups.append(("capture_lineage", row.capture_lineage_id))
    return tuple(groups)


def _row_json(row: SplitRow, partition: str, pretraining_visible: bool) -> dict[str, object]:
    return {
        "event_id": row.event_id,
        "completion_ms": row.completion_ms,
        "partition": partition,
        "pretraining_visible": pretraining_visible,
        "source_unit_id": row.source_unit_id,
        "capture_lineage_id": row.capture_lineage_id,
        "source_principal": row.source_principal,
        "destination_principal": row.destination_principal,
        "exact_group": row.exact_group,
        "near_group": row.near_group,
        "canonical_family": row.canonical_family,
        "campaign_id": row.campaign_id,
    }


def _canonical_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _spec_json(spec: SplitSpec) -> dict[str, object]:
    return {
        "track": spec.track,
        "train_end_ms": spec.train_end_ms,
        "validation_end_ms": spec.validation_end_ms,
        "purge_ms": spec.purge_ms,
        "held_principals": sorted(spec.held_principals),
        "held_families": sorted(spec.held_families),
        "held_campaigns": sorted(spec.held_campaigns),
        "source_units": sorted(spec.source_units),
        "target_units": sorted(spec.target_units),
        "sealed_units": sorted(spec.sealed_units),
    }


def _spec_from_json(value: Mapping[str, object]) -> SplitSpec:
    def strings(name: str) -> frozenset[str]:
        members = value.get(name)
        if not isinstance(members, list):
            raise ValueError(f"split spec {name} must be a list of strings")
        typed_members = cast(list[object], members)
        if any(not isinstance(member, str) for member in typed_members):
            raise ValueError(f"split spec {name} must be a list of strings")
        return frozenset(cast(list[str], typed_members))

    track = value.get("track")
    if not isinstance(track, str) or track not in _TRACKS:
        raise ValueError("split result has invalid track")
    for name in ("train_end_ms", "validation_end_ms"):
        if value.get(name) is not None and (
            isinstance(value[name], bool) or not isinstance(value[name], int)
        ):
            raise ValueError(f"split spec {name} must be an integer or null")
    return SplitSpec(
        track=cast(Track, track),
        train_end_ms=cast(int | None, value.get("train_end_ms")),
        validation_end_ms=cast(int | None, value.get("validation_end_ms")),
        purge_ms=_integer("split spec purge_ms", value.get("purge_ms")),
        held_principals=strings("held_principals"),
        held_families=strings("held_families"),
        held_campaigns=strings("held_campaigns"),
        source_units=strings("source_units"),
        target_units=strings("target_units"),
        sealed_units=strings("sealed_units"),
    )


def _assignment(value: object) -> tuple[SplitRow, str, bool]:
    if not isinstance(value, dict):
        raise ValueError("split assignment must be an object")
    assignment = cast(dict[str, object], value)
    required = (
        "event_id",
        "completion_ms",
        "partition",
        "pretraining_visible",
        "source_unit_id",
        "capture_lineage_id",
        "source_principal",
        "destination_principal",
        "exact_group",
        "near_group",
        "canonical_family",
        "campaign_id",
    )
    missing = [key for key in required if key not in assignment]
    if missing:
        raise ValueError(f"split assignment is missing {', '.join(missing)}")
    for key in ("canonical_family", "campaign_id"):
        if assignment[key] is not None and not isinstance(assignment[key], str):
            raise ValueError(f"split assignment {key} must be a string or null")
    if not isinstance(assignment["pretraining_visible"], bool):
        raise ValueError("split assignment pretraining_visible must be boolean")
    fields = (
        "event_id",
        "source_unit_id",
        "capture_lineage_id",
        "source_principal",
        "destination_principal",
        "exact_group",
        "near_group",
    )
    if any(not isinstance(assignment[key], str) for key in fields):
        raise ValueError("split assignment contains a non-string identifier")
    if not isinstance(assignment["partition"], str):
        raise ValueError("split assignment partition must be a string")
    return (
        SplitRow(
            event_id=cast(str, assignment["event_id"]),
            completion_ms=_integer("split assignment completion_ms", assignment["completion_ms"]),
            source_principal=cast(str, assignment["source_principal"]),
            destination_principal=cast(str, assignment["destination_principal"]),
            source_unit_id=cast(str, assignment["source_unit_id"]),
            capture_lineage_id=cast(str, assignment["capture_lineage_id"]),
            exact_group=cast(str, assignment["exact_group"]),
            near_group=cast(str, assignment["near_group"]),
            canonical_family=cast(str | None, assignment["canonical_family"]),
            campaign_id=cast(str | None, assignment["campaign_id"]),
        ),
        assignment["partition"],
        assignment["pretraining_visible"],
    )


def _allowed_partitions(track: str) -> frozenset[str]:
    return {
        "chronological": frozenset({"train", "validation", "test"}),
        "endpoint_disjoint": frozenset({"source", "target"}),
        "held_out_family": frozenset({"source", "target"}),
        "cross_network": frozenset({"source", "target", "sealed"}),
    }[track]


def validate_no_overlap(result: Mapping[str, object]) -> None:
    """Independently validate role assignment, visibility, and scored-overlap isolation."""
    spec_value = result.get("spec")
    assignments = result.get("assignments")
    if not isinstance(spec_value, Mapping) or not isinstance(assignments, list):
        raise ValueError("split result must contain spec and assignments")
    spec = _spec_from_json(cast(Mapping[str, object], spec_value))
    _validate_spec(spec)
    track = spec.track
    allowed = _allowed_partitions(track)
    ids: set[str] = set()
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    seen_partitions: set[str] = set()
    for assignment in cast(list[object], assignments):
        row, partition, visible = _assignment(assignment)
        _validate_row(row)
        if row.event_id in ids:
            raise ValueError(f"split result repeats event_id {row.event_id!r}")
        ids.add(row.event_id)
        if partition not in allowed:
            raise ValueError(f"invalid {track} partition {partition!r}")
        if visible != (partition in {"train", "source"}):
            raise ValueError("pretraining visibility does not match source partition")
        if track == "cross_network" and partition in {"target", "sealed"} and visible:
            raise ValueError("target or sealed data is pretraining-visible")
        seen_partitions.add(partition)
        for group in _isolation_groups(row, spec):
            groups[group].add(partition)
    missing = allowed - seen_partitions
    if missing:
        raise ValueError(f"required partitions are empty: {', '.join(sorted(missing))}")
    unresolved = sorted({kind for (kind, _), parts in groups.items() if len(parts) > 1})
    if unresolved:
        raise ValueError(f"unresolved retained group overlap: {', '.join(unresolved)}")


def build_split(rows: Iterable[SplitRow], spec: SplitSpec) -> dict[str, object]:
    """Build one strict track-specific manifest; never reassign a conflicting flow."""
    _validate_spec(spec)
    seen_ids: set[str] = set()
    provisional: list[tuple[SplitRow, str | None, tuple[str, ...]]] = []
    partitions_by_group: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        _validate_row(row)
        if row.event_id in seen_ids:
            raise ValueError(f"duplicate event_id {row.event_id!r}")
        seen_ids.add(row.event_id)
        partition, partition_reasons = _partition(row, spec)
        provisional.append((row, partition, partition_reasons))
        if partition is not None:
            for group in _isolation_groups(row, spec):
                partitions_by_group[group].add(partition)
    conflicts = {group: parts for group, parts in partitions_by_group.items() if len(parts) > 1}
    chronological_order = {"train": 0, "validation": 1, "test": 2}
    assignments: list[dict[str, object]] = []
    dropped: list[dict[str, object]] = []
    reasons: Counter[str] = Counter()
    for row, partition, initial_reasons in provisional:
        row_reasons = list(initial_reasons)
        if partition is not None:
            for kind, value in _isolation_groups(row, spec):
                partitions = conflicts.get((kind, value))
                if partitions is None:
                    continue
                if spec.track != "chronological" or partition != max(
                    partitions, key=chronological_order.__getitem__
                ):
                    row_reasons.append(f"overlap:{kind}")
        row_reasons = sorted(set(row_reasons))
        if row_reasons:
            dropped.append({"event_id": row.event_id, "reasons": row_reasons})
            reasons.update(row_reasons)
        elif partition is None:
            raise ValueError("non-retained row has no drop reason")
        else:
            assignments.append(_row_json(row, partition, partition in {"train", "source"}))
    assignments.sort(key=lambda value: cast(str, value["event_id"]))
    dropped.sort(key=lambda value: cast(str, value["event_id"]))
    result: dict[str, object] = {
        "spec": _spec_json(spec),
        "assignments": assignments,
        "dropped": dropped,
        "dropped_counts": dict(sorted(reasons.items())),
        "retained_counts": dict(
            sorted(Counter(cast(str, row["partition"]) for row in assignments).items())
        ),
    }
    validate_no_overlap(result)
    result["validation"] = {"zero_scored_overlap": True, "retained_rows": len(assignments)}
    result["sha256"] = _canonical_sha256(result)
    return result
