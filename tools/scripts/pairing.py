"""Fail-closed Q3 pairing manifest gate; this does not join packet data."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Literal, TypeGuard, cast

type Interval = tuple[int | float, int | float]
type FiveTuple = tuple[str, int, str, int, int]


@dataclass(frozen=True)
class PairingResolution:
    status: Literal["disabled", "accepted", "ambiguous"]
    candidates: int
    pairing: "PairingEvidence | None" = None


@dataclass(frozen=True)
class PairingEvidence:
    capture_id: str
    five_tuple: FiveTuple
    flow_interval: Interval
    packet_interval: Interval
    packet_count: int
    byte_count: int


def resolve_pairing(config: object, flow: object) -> PairingResolution:
    """Resolve one Q3 manifest candidate, accepting only an unambiguous match."""
    if not isinstance(config, Mapping):
        raise ValueError("q3_pairing must be a boolean")
    config = cast(Mapping[str, object], config)
    if not isinstance(config.get("q3_pairing"), bool):
        raise ValueError("q3_pairing must be a boolean")
    if not config["q3_pairing"]:
        return PairingResolution("disabled", 0)

    manifest = config.get("pairing_manifest")
    if not isinstance(manifest, list):
        raise ValueError("pairing_manifest must be a list when q3_pairing is enabled")
    manifest = cast(list[object], manifest)

    capture_id, five_tuple, interval, packet_count, byte_count = _flow_key(flow)
    pairs = [_validate_pairing(item, index) for index, item in enumerate(manifest)]
    matches = [
        pairing
        for pairing in pairs
        if pairing.capture_id == capture_id
        and pairing.five_tuple == five_tuple
        and pairing.flow_interval == interval
        and pairing.packet_count == packet_count
        and pairing.byte_count == byte_count
    ]
    if len(matches) != 1:
        return PairingResolution("ambiguous", len(matches))
    return PairingResolution("accepted", 1, matches[0])


def _flow_key(flow: object) -> tuple[str, FiveTuple, Interval, int, int]:
    if not isinstance(flow, Mapping):
        raise ValueError("flow must be a mapping")
    flow = cast(Mapping[str, object], flow)
    return (
        _capture_id(flow.get("capture_id"), "flow"),
        _five_tuple(flow.get("five_tuple"), "flow"),
        _interval(flow.get("interval"), "flow interval"),
        _positive_int(flow.get("packet_count"), "flow.packet_count"),
        _positive_int(flow.get("byte_count"), "flow.byte_count"),
    )


def _validate_pairing(item: object, index: int) -> PairingEvidence:
    if not isinstance(item, Mapping):
        raise ValueError(f"pairing_manifest[{index}] must be a mapping")
    item = cast(Mapping[str, object], item)
    prefix = f"pairing_manifest[{index}]"
    five_tuple = _five_tuple(item.get("five_tuple"), prefix)
    raw_five_tuple = item.get("five_tuple")
    if not isinstance(raw_five_tuple, (list, tuple)):
        raise ValueError(f"{prefix}.five_tuple must be canonical")
    raw_five_tuple = cast(list[object] | tuple[object, ...], raw_five_tuple)
    if tuple(raw_five_tuple) != five_tuple:
        raise ValueError(f"{prefix}.five_tuple must be canonical")
    flow_interval = _interval(item.get("flow_interval"), f"{prefix}.flow_interval")
    packet_interval = _interval(item.get("packet_interval"), f"{prefix}.packet_interval")
    if not _overlaps(flow_interval, packet_interval):
        raise ValueError(f"{prefix} intervals must overlap")
    return PairingEvidence(
        _capture_id(item.get("capture_id"), prefix),
        five_tuple,
        flow_interval,
        packet_interval,
        _positive_int(item.get("packet_count"), f"{prefix}.packet_count"),
        _positive_int(item.get("byte_count"), f"{prefix}.byte_count"),
    )


def _capture_id(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name}.capture_id must be a non-empty string")
    return value


def _five_tuple(value: object, name: str) -> FiveTuple:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name}.five_tuple must contain endpoint, port, endpoint, port, protocol")
    values = cast(list[object] | tuple[object, ...], value)
    if len(values) != 5:
        raise ValueError(f"{name}.five_tuple must contain endpoint, port, endpoint, port, protocol")
    source, source_port, destination, destination_port, protocol = cast(
        tuple[object, object, object, object, object], tuple(values)
    )
    if (
        not isinstance(source, str)
        or not source
        or not isinstance(destination, str)
        or not destination
    ):
        raise ValueError(f"{name}.five_tuple endpoints must be non-empty strings")
    if not _port(source_port) or not _port(destination_port):
        raise ValueError(f"{name}.five_tuple ports must be integers from 0 to 65535")
    if not isinstance(protocol, int) or isinstance(protocol, bool) or not 0 <= protocol <= 255:
        raise ValueError(f"{name}.five_tuple protocol must be an integer from 0 to 255")
    first: tuple[str, int] = (source, source_port)
    second: tuple[str, int] = (destination, destination_port)
    return (*first, *second, protocol) if first <= second else (*second, *first, protocol)


def _port(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 65535


def _interval(value: object, name: str) -> Interval:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must contain start and end")
    values = cast(list[object] | tuple[object, ...], value)
    if len(values) != 2:
        raise ValueError(f"{name} must contain start and end")
    start, end = cast(tuple[object, object], tuple(values))
    if not _finite_number(start) or not _finite_number(end):
        raise ValueError(f"{name} bounds must be finite numbers")
    if start > end:
        raise ValueError(f"{name} must not end before it starts")
    return start, end


def _finite_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _overlaps(first: Interval, second: Interval) -> bool:
    return first[0] <= second[1] and second[0] <= first[1]
