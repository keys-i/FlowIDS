"""Materialise an audited, unlabeled MAWI YAF flow view from a classic PCAP.

The input is the fixed headerless Super-Mediator CSV:
STIMEMS,ETIMEMS,SIP,DIP,SPORT,DPORT,PROTOCOL,PACKETS,RPACKETS,BYTES,RBYTES,
IFLAGS,RIFLAGS,UFLAGS,RUFLAGS,ENDREASON.  ``eof`` and ``rsrc`` rows are
excluded.  Since YAF does not export the packet time that triggered a timeout,
timeout-ended records use conservative causal availability bounds.  A receipt
stays blocked until source YAF counters reconcile exactly with the PCAP.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import ipaddress
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

import click
import polars as pl

INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "STIMEMS",
    "ETIMEMS",
    "SIP",
    "DIP",
    "SPORT",
    "DPORT",
    "PROTOCOL",
    "PACKETS",
    "RPACKETS",
    "BYTES",
    "RBYTES",
    "IFLAGS",
    "RIFLAGS",
    "UFLAGS",
    "RUFLAGS",
    "ENDREASON",
)
OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "flow_start_ms",
    "flow_end_ms",
    "flow_available_ms",
    "flow_duration_ms",
    "source_ip",
    "destination_ip",
    "source_port",
    "destination_port",
    "protocol",
    "forward_packets",
    "reverse_packets",
    "forward_bytes",
    "reverse_bytes",
    "initial_tcp_flags",
    "reverse_initial_tcp_flags",
    "union_tcp_flags",
    "reverse_union_tcp_flags",
    "end_reason",
)
PCAP_KEYS: Final[tuple[str, ...]] = (
    "records",
    "ip_packets",
    "l3_octets",
    "outer_ipv4",
    "outer_ipv6",
    "unsupported",
    "malformed",
    "header_truncated",
    "record_truncated",
    "snaplen_truncated",
    "fragment_packets",
    "fragment_octets",
    "extension_chain_unresolved",
    "extension_chain_unresolved_octets",
    "capture_start_epoch_ns",
    "capture_end_epoch_ns",
    "capture_start_ms",
    "capture_end_ms",
    "nonmonotonic_timestamps",
)
FLOW_KEYS: Final[tuple[str, ...]] = (
    "rows",
    "accepted",
    "accepted_normal",
    "accepted_idle",
    "accepted_active",
    "rejected_eof",
    "rejected_rsrc",
    "exported_packet_sum",
    "exported_l3_octet_sum",
    "accepted_packet_sum",
    "accepted_l3_octet_sum",
)
CONFIG_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "id",
        "source",
        "meter",
        "feature_contract",
        "expected_pcap",
        "expected_flows",
        "yaf_stats",
    }
)
SOURCE_CONFIG_KEYS: Final[frozenset[str]] = frozenset(
    {
        "official_page",
        "download_url",
        "terms_url",
        "terms",
        "redistribution",
        "capture_lineage",
        "observation_point",
        "collector",
        "exporter",
        "anonymization",
        "timezone",
        "label_provenance",
        "compressed_file",
        "compressed_bytes",
        "compressed_sha256",
        "pcap_file",
        "pcap_bytes",
        "pcap_sha256",
        "upstream_checksum",
    }
)
METER_CONFIG_KEYS: Final[frozenset[str]] = frozenset(
    {
        "libfixbuf_version",
        "libfixbuf_source_url",
        "libfixbuf_source_sha256",
        "yaf_version",
        "yaf_source_url",
        "yaf_source_sha256",
        "yaf_binary_sha256",
        "super_mediator_version",
        "super_mediator_source_url",
        "super_mediator_source_sha256",
        "super_mediator_binary_sha256",
        "super_mediator_command",
        "super_mediator_config",
        "build",
        "active_timeout_seconds",
        "idle_timeout_seconds",
        "fragment_policy",
        "direction_rule",
        "yaf_command",
        "super_mediator_fields",
        "yaf_ipfix_file",
        "yaf_ipfix_bytes",
        "yaf_ipfix_sha256",
        "yaf_csv_file",
        "yaf_csv_sha256",
    }
)
PCAP_MAGICS: Final[dict[bytes, tuple[str, int]]] = {
    b"\xd4\xc3\xb2\xa1": ("<", 1_000_000),
    b"\xa1\xb2\xc3\xd4": (">", 1_000_000),
    b"\x4d\x3c\xb2\xa1": ("<", 1_000_000_000),
    b"\xa1\xb2\x3c\x4d": (">", 1_000_000_000),
}
VLAN_TYPES: Final[frozenset[int]] = frozenset({0x8100, 0x88A8, 0x9100})
SCHEMA: Final = {
    "flow_start_ms": pl.Int64,
    "flow_end_ms": pl.Int64,
    "flow_available_ms": pl.Int64,
    "flow_duration_ms": pl.Int64,
    "source_ip": pl.String,
    "destination_ip": pl.String,
    "source_port": pl.Int64,
    "destination_port": pl.Int64,
    "protocol": pl.Int64,
    "forward_packets": pl.Int64,
    "reverse_packets": pl.Int64,
    "forward_bytes": pl.Int64,
    "reverse_bytes": pl.Int64,
    "initial_tcp_flags": pl.String,
    "reverse_initial_tcp_flags": pl.String,
    "union_tcp_flags": pl.String,
    "reverse_union_tcp_flags": pl.String,
    "end_reason": pl.String,
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def gzip_payload(path: Path) -> tuple[int, str]:
    """Hash the decompressed archive stream without materializing another copy."""
    hasher = hashlib.sha256()
    size = 0
    try:
        with gzip.open(path, "rb") as handle:
            while chunk := handle.read(1024 * 1024):
                size += len(chunk)
                hasher.update(chunk)
    except (gzip.BadGzipFile, EOFError, OSError) as error:
        raise ValueError("compressed MAWI archive is not a valid complete gzip stream") from error
    return size, hasher.hexdigest()


def _regular(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} must be an existing regular file: {path}")
    return path


def _counter() -> dict[str, int]:
    return {key: 0 for key in PCAP_KEYS}


def _protocol(counts: dict[str, int], protocol: int, packets: int, octets: int) -> None:
    counts[f"protocol_{protocol}_packets"] = counts.get(f"protocol_{protocol}_packets", 0) + packets
    counts[f"protocol_{protocol}_l3_octets"] = (
        counts.get(f"protocol_{protocol}_l3_octets", 0) + octets
    )


def pcap_accounting(path: Path) -> dict[str, int]:
    """Count outer Ethernet IPv4/IPv6 packets without retaining packet data."""
    path = _regular(path, "--pcap")
    counts = _counter()
    with path.open("rb") as handle:
        global_header = handle.read(24)
        if len(global_header) != 24:
            raise ValueError("PCAP global header is truncated")
        magic = global_header[:4]
        if magic not in PCAP_MAGICS:
            raise ValueError("PCAP must use classic microsecond/nanosecond format")
        order, resolution = PCAP_MAGICS[magic]
        byteorder = "little" if order == "<" else "big"
        network = int.from_bytes(global_header[20:24], byteorder)
        if network != 1:
            raise ValueError(f"PCAP link type {network} is unsupported; require Ethernet (1)")
        while True:
            record_header = handle.read(16)
            if not record_header:
                return counts
            if len(record_header) != 16:
                counts["record_truncated"] += 1
                return counts
            seconds = int.from_bytes(record_header[:4], byteorder)
            fraction = int.from_bytes(record_header[4:8], byteorder)
            if fraction >= resolution:
                counts["malformed"] += 1
                return counts
            timestamp_ns = seconds * 1_000_000_000 + fraction * (1_000_000_000 // resolution)
            timestamp_ms = timestamp_ns // 1_000_000
            if counts["records"] == 0:
                counts["capture_start_epoch_ns"] = timestamp_ns
                counts["capture_start_ms"] = timestamp_ms
            if counts["records"] and timestamp_ns < counts["capture_end_epoch_ns"]:
                counts["nonmonotonic_timestamps"] += 1
            counts["capture_end_epoch_ns"] = timestamp_ns
            counts["capture_end_ms"] = timestamp_ms
            included = int.from_bytes(record_header[8:12], byteorder)
            original = int.from_bytes(record_header[12:16], byteorder)
            counts["records"] += 1
            if included > original:
                counts["malformed"] += 1
            if included < original:
                counts["snaplen_truncated"] += 1
            frame = handle.read(included)
            if len(frame) != included:
                counts["record_truncated"] += 1
                return counts
            _account_frame(frame, original, counts)


def _account_frame(frame: bytes, original: int, counts: dict[str, int]) -> None:
    if len(frame) < 14:
        counts["header_truncated"] += 1
        return
    offset = 14
    ether_type = int.from_bytes(frame[12:14], "big")
    while ether_type in VLAN_TYPES:
        if len(frame) < offset + 4:
            counts["header_truncated"] += 1
            return
        ether_type = int.from_bytes(frame[offset + 2 : offset + 4], "big")
        offset += 4
    payload = frame[offset:]
    wire_l3 = max(original - offset, 0)
    if ether_type == 0x0800:
        _account_ipv4(payload, wire_l3, counts)
    elif ether_type == 0x86DD:
        _account_ipv6(payload, wire_l3, counts)
    else:
        counts["unsupported"] += 1


def _account_ipv4(payload: bytes, wire_l3: int, counts: dict[str, int]) -> None:
    if len(payload) < 20:
        counts["header_truncated"] += 1
        return
    if payload[0] >> 4 != 4:
        counts["malformed"] += 1
        return
    header_length = (payload[0] & 0x0F) * 4
    total_length = int.from_bytes(payload[2:4], "big")
    if header_length < 20 or total_length < header_length:
        counts["malformed"] += 1
        return
    if len(payload) < header_length:
        counts["header_truncated"] += 1
        return
    if total_length > wire_l3:
        counts["malformed"] += 1
        return
    counts["ip_packets"] += 1
    counts["outer_ipv4"] += 1
    counts["l3_octets"] += total_length
    _protocol(counts, payload[9], 1, total_length)
    if int.from_bytes(payload[6:8], "big") & 0x3FFF:
        counts["fragment_packets"] += 1
        counts["fragment_octets"] += total_length


def _account_ipv6(payload: bytes, wire_l3: int, counts: dict[str, int]) -> None:
    if len(payload) < 40:
        counts["header_truncated"] += 1
        return
    if payload[0] >> 4 != 6:
        counts["malformed"] += 1
        return
    total_length = 40 + int.from_bytes(payload[4:6], "big")
    if total_length > wire_l3:
        counts["malformed"] += 1
        return
    counts["ip_packets"] += 1
    counts["outer_ipv6"] += 1
    counts["l3_octets"] += total_length
    next_header = payload[6]
    offset = 40
    while next_header in {0, 43, 44, 51, 60}:
        if next_header == 44:
            if len(payload) < offset + 8:
                counts["extension_chain_unresolved"] += 1
                counts["extension_chain_unresolved_octets"] += total_length
                return
            _protocol(counts, payload[offset], 1, total_length)
            counts["fragment_packets"] += 1
            counts["fragment_octets"] += total_length
            return
        if len(payload) < offset + 2:
            counts["extension_chain_unresolved"] += 1
            counts["extension_chain_unresolved_octets"] += total_length
            return
        extension_type = next_header
        next_header, extension_units = payload[offset], payload[offset + 1]
        extension_length = (
            (extension_units + 2) * 4 if extension_type == 51 else (extension_units + 1) * 8
        )
        if offset + extension_length > total_length:
            counts["malformed"] += 1
            return
        if len(payload) < offset + extension_length:
            counts["extension_chain_unresolved"] += 1
            counts["extension_chain_unresolved_octets"] += total_length
            return
        offset += extension_length
    _protocol(counts, next_header, 1, total_length)


def _exact(expected: Mapping[str, object], actual: Mapping[str, int], name: str) -> None:
    if set(expected) != set(actual):
        raise ValueError(f"{name} keys must be exactly {', '.join(actual)}")
    normalized: dict[str, int] = {}
    for key, value in expected.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name}.{key} must be a non-negative integer")
        normalized[key] = value
    if normalized != actual:
        raise ValueError(f"{name} does not match the input exactly")


def _number(value: str, name: str, *, maximum: int = 2**63 - 1) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if number < 0 or number > maximum:
        raise ValueError(f"{name} must be between 0 and {maximum}")
    return number


def _ip(value: str, name: str) -> str:
    try:
        return str(ipaddress.ip_address(value))
    except ValueError as error:
        raise ValueError(f"{name} must be an IPv4 or IPv6 address") from error


def _port(value: str, name: str) -> int:
    return _number(value, name, maximum=65535)


def _flags(value: str, name: str) -> str:
    if "\x00" in value or len(value) > 64:
        raise ValueError(f"{name} must be at most 64 non-NUL characters")
    return value


def _feature_contract(value: object) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise ValueError("feature_contract does not match the MAWI schema contract")
    raw = cast(Mapping[object, object], value)
    if set(raw) != {
        "model_numeric",
        "model_categorical",
        "routing_only",
        "excluded",
        "never_model_inputs",
    }:
        raise ValueError("feature_contract does not match the MAWI schema contract")
    contract: dict[str, tuple[str, ...]] = {}
    for role, fields in raw.items():
        if (
            not isinstance(role, str)
            or not isinstance(fields, list)
            or not all(isinstance(field, str) for field in cast(list[object], fields))
        ):
            raise ValueError("feature_contract roles must contain lists of field names")
        members = tuple(cast(list[str], fields))
        if len(members) != len(set(members)):
            raise ValueError(f"feature_contract.{role} repeats a field")
        contract[role] = members
    roles = (
        contract["model_numeric"],
        contract["model_categorical"],
        contract["routing_only"],
        contract["excluded"],
    )
    flattened = tuple(field for members in roles for field in members)
    if len(flattened) != len(set(flattened)) or set(flattened) != set(OUTPUT_COLUMNS):
        raise ValueError("feature_contract must assign every output field exactly one role")
    forbidden = set(contract["routing_only"]) | set(contract["excluded"])
    if set(contract["never_model_inputs"]) != forbidden:
        raise ValueError("feature_contract never_model_inputs must equal non-model fields")
    return contract


def model_view(row: Mapping[str, object], contract_value: object) -> dict[str, dict[str, object]]:
    """Return the explicit model allow-list; routing/audit fields cannot pass."""
    contract = _feature_contract(contract_value)
    if set(row) != set(OUTPUT_COLUMNS):
        raise ValueError("MAWI row does not match the admitted output schema")
    return {
        "numeric": {field: row[field] for field in contract["model_numeric"]},
        "categorical": {field: row[field] for field in contract["model_categorical"]},
    }


def _canonical(
    row: list[str], capture_start_ms: int, capture_end_ms: int
) -> tuple[tuple[object, ...], str]:
    if len(row) != len(INPUT_COLUMNS):
        raise ValueError(f"YAF CSV row has {len(row)} fields; expected 16")
    end_reason = row[15].strip().lower()
    if end_reason not in {"", "idle", "active", "eof", "rsrc"}:
        raise ValueError(f"unsupported YAF end_reason {row[15]!r}")
    start = _number(row[0], "STIMEMS")
    end = _number(row[1], "ETIMEMS")
    if not capture_start_ms <= start <= end <= capture_end_ms:
        raise ValueError("flow interval must lie inside the PCAP capture interval")
    duration = end - start
    # YAF checks timeouts only when a later packet arrives.  The IPFIX record
    # omits that trigger time, so use a causal upper bound instead of pretending
    # the configured timeout was the export time.
    availability = {
        "": end,
        "idle": capture_end_ms,
        "active": capture_end_ms,
        "eof": end,
        "rsrc": end,
    }[end_reason]
    if availability < end or availability > 2**63 - 1 or availability > capture_end_ms:
        raise ValueError("flow availability must be between flow end and capture end")
    disposition = (
        f"rejected_{end_reason}"
        if end_reason in {"eof", "rsrc"}
        else f"accepted_{'normal' if end_reason == '' else end_reason}"
    )
    return (
        (
            start,
            end,
            availability,
            duration,
            _ip(row[2], "SIP"),
            _ip(row[3], "DIP"),
            _port(row[4], "SPORT"),
            _port(row[5], "DPORT"),
            _number(row[6], "PROTOCOL", maximum=255),
            _number(row[7], "PACKETS"),
            _number(row[8], "RPACKETS"),
            _number(row[9], "BYTES"),
            _number(row[10], "RBYTES"),
            _flags(row[11], "IFLAGS"),
            _flags(row[12], "RIFLAGS"),
            _flags(row[13], "UFLAGS"),
            _flags(row[14], "RUFLAGS"),
            end_reason,
        ),
        disposition,
    )


def temporary_file(parent: Path, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=".mawi.", suffix=suffix, dir=parent)
    os.close(descriptor)
    return Path(name)


def publish_bundle(
    temporary_parquet: Path,
    receipt_payload: bytes,
    parquet: Path,
    receipt: Path,
) -> None:
    """Publish once, recover an identical orphan, and refuse mutation."""
    output_digest = digest(temporary_parquet)
    if receipt.exists():
        if (
            receipt.is_symlink()
            or parquet.is_symlink()
            or not receipt.is_file()
            or not parquet.is_file()
        ):
            raise ValueError("existing MAWI artifact bundle is incomplete")
        if receipt.read_bytes() != receipt_payload or digest(parquet) != output_digest:
            raise ValueError("existing MAWI artifact bundle differs; refusing overwrite")
        temporary_parquet.unlink()
        return
    if parquet.exists():
        if parquet.is_symlink() or not parquet.is_file() or digest(parquet) != output_digest:
            raise ValueError("existing MAWI Parquet differs; refusing overwrite")
    else:
        try:
            os.link(temporary_parquet, parquet)
        except FileExistsError as error:
            if parquet.is_symlink() or not parquet.is_file() or digest(parquet) != output_digest:
                raise ValueError("existing MAWI Parquet differs; refusing overwrite") from error
    if temporary_parquet.exists():
        temporary_parquet.unlink()
    temporary_receipt = temporary_file(receipt.parent, ".json")
    try:
        with temporary_receipt.open("wb") as handle:
            _ = handle.write(receipt_payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_receipt, receipt)
        except FileExistsError as error:
            if (
                receipt.is_symlink()
                or not receipt.is_file()
                or receipt.read_bytes() != receipt_payload
            ):
                raise ValueError("existing MAWI receipt differs; refusing overwrite") from error
    finally:
        temporary_receipt.unlink(missing_ok=True)


def _yaf_reconciliation(
    stats: Mapping[str, object] | None,
    pcap: Mapping[str, int],
    flows: Mapping[str, int],
) -> dict[str, object]:
    """Require the declared no-frag packet equation before lifting the block."""
    if stats is None:
        return {
            "status": "blocked",
            "reason": "full YAF --no-frag counters and reconciliation equation are absent",
        }
    required = (
        "exported_flows",
        "packet_total",
        "dropped_packets",
        "ignored_packets",
        "expired_fragments",
        "assembled_fragments",
        "flow_flushes",
        "peak_flows",
        "flow_l3_octets",
        "ignored_fragments",
        "ignored_extension_headers",
        "ignored_transport_headers",
        "ignored_non_ip",
        "ignored_other",
    )
    if set(stats) != set(required):
        raise ValueError(f"yaf_stats keys must be exactly {', '.join(required)}")
    values: dict[str, int] = {}
    for key, value in stats.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"yaf_stats.{key} must be a non-negative integer")
        values[key] = value
    if values["dropped_packets"] or values["expired_fragments"] or values["assembled_fragments"]:
        raise ValueError("admitted no-frag run requires zero dropped/assembled/expired fragments")
    ignored_breakdown = (
        values["ignored_fragments"]
        + values["ignored_extension_headers"]
        + values["ignored_transport_headers"]
        + values["ignored_non_ip"]
        + values["ignored_other"]
    )
    if ignored_breakdown != values["ignored_packets"]:
        raise ValueError("YAF ignored-packet breakdown does not equal ignored_packets")
    if values["ignored_fragments"] != pcap["fragment_packets"]:
        raise ValueError("YAF ignored_fragments does not equal PCAP fragment packets")
    if values["ignored_extension_headers"] != pcap["extension_chain_unresolved"]:
        raise ValueError(
            "YAF ignored_extension_headers does not equal unresolved PCAP extension chains"
        )
    if values["ignored_non_ip"] != pcap["unsupported"]:
        raise ValueError("YAF ignored_non_ip does not equal unsupported PCAP records")
    if pcap["records"] != values["packet_total"] + values["ignored_packets"]:
        raise ValueError("YAF packet_total + ignored_packets does not equal PCAP records")
    if flows["exported_packet_sum"] != values["packet_total"]:
        raise ValueError("YAF packet_total does not equal CSV bidirectional packet sum")
    if flows["exported_l3_octet_sum"] != values["flow_l3_octets"]:
        raise ValueError("YAF flow_l3_octets does not equal CSV bidirectional byte sum")
    if flows["rows"] != values["exported_flows"]:
        raise ValueError("CSV rows do not equal YAF exported flows")
    raw_protocols = _protocol_totals(pcap)
    flow_protocols = _protocol_totals(flows)
    protocol_delta: dict[str, dict[str, int]] = {}
    for protocol in sorted(set(raw_protocols) | set(flow_protocols), key=int):
        raw = raw_protocols.get(protocol, {"packets": 0, "l3_octets": 0})
        exported = flow_protocols.get(protocol, {"packets": 0, "l3_octets": 0})
        if exported["packets"] > raw["packets"] or exported["l3_octets"] > raw["l3_octets"]:
            raise ValueError(f"CSV protocol {protocol} exceeds independent PCAP accounting")
        protocol_delta[protocol] = {
            "raw_packets": raw["packets"],
            "exported_packets": exported["packets"],
            "excluded_packets": raw["packets"] - exported["packets"],
            "raw_l3_octets": raw["l3_octets"],
            "exported_l3_octets": exported["l3_octets"],
            "excluded_l3_octets": raw["l3_octets"] - exported["l3_octets"],
        }
    excluded_l3 = pcap["l3_octets"] - values["flow_l3_octets"]
    known_l3 = pcap["fragment_octets"] + pcap["extension_chain_unresolved_octets"]
    transport_l3 = excluded_l3 - known_l3
    if transport_l3 < 0:
        raise ValueError("categorized excluded L3 octets exceed the raw-to-flow difference")
    attributed_packets = sum(item["excluded_packets"] for item in protocol_delta.values())
    attributed_octets = sum(item["excluded_l3_octets"] for item in protocol_delta.values())
    if (
        attributed_packets
        != values["ignored_packets"]
        - values["ignored_non_ip"]
        - values["ignored_extension_headers"]
    ):
        raise ValueError("per-protocol excluded packets do not match YAF ignored categories")
    if attributed_octets != excluded_l3 - pcap["extension_chain_unresolved_octets"]:
        raise ValueError("per-protocol excluded L3 octets do not match raw accounting")
    return {
        "status": "reconciled",
        "counters": values,
        "packet_equation": "PCAP records = YAF packet_total + YAF ignored_packets",
        "l3_octet_equation": "raw L3 = flow L3 + excluded L3",
        "excluded_l3_octets": {
            "fragments": pcap["fragment_octets"],
            "unresolved_extension_headers": pcap["extension_chain_unresolved_octets"],
            "incomplete_transport_headers": transport_l3,
            "total": excluded_l3,
        },
        "protocol_accounting": protocol_delta,
        "reason": None,
    }


def _protocol_totals(values: Mapping[str, int]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {}
    for key, value in values.items():
        if not key.startswith("protocol_"):
            continue
        protocol, separator, measure = key.removeprefix("protocol_").partition("_")
        if (
            not protocol.isdigit()
            or separator != "_"
            or measure
            not in {
                "packets",
                "l3_octets",
            }
        ):
            raise ValueError(f"invalid protocol accounting key {key!r}")
        totals.setdefault(protocol, {})[measure] = value
    if any(set(measures) != {"packets", "l3_octets"} for measures in totals.values()):
        raise ValueError("protocol accounting requires packet and L3-octet totals")
    return totals


def materialize(
    pcap: Path,
    yaf_csv: Path,
    parquet: Path,
    receipt: Path,
    *,
    pcap_sha256: str,
    yaf_sha256: str,
    expected_pcap: Mapping[str, object],
    expected_flows: Mapping[str, object],
    yaf_stats: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate both source artifacts, then atomically write Parquet and receipt."""
    pcap = _regular(pcap, "--pcap")
    yaf_csv = _regular(yaf_csv, "--yaf-csv")
    if len(pcap_sha256) != 64 or any(value not in "0123456789abcdef" for value in pcap_sha256):
        raise ValueError("--pcap-sha256 must be a lowercase SHA-256 digest")
    if len(yaf_sha256) != 64 or any(value not in "0123456789abcdef" for value in yaf_sha256):
        raise ValueError("--yaf-sha256 must be a lowercase SHA-256 digest")
    if digest(pcap) != pcap_sha256 or digest(yaf_csv) != yaf_sha256:
        raise ValueError("declared source SHA-256 does not match")
    pcap_counts = pcap_accounting(pcap)
    _exact(expected_pcap, pcap_counts, "expected_pcap")
    if pcap_counts["nonmonotonic_timestamps"]:
        raise ValueError("PCAP record timestamps are non-monotonic")
    capture_start_ms = pcap_counts["capture_start_ms"]
    capture_end_ms = pcap_counts["capture_end_ms"]
    if not pcap_counts["records"]:
        raise ValueError("PCAP must contain at least one timestamped record")

    parquet = Path(os.path.abspath(parquet.expanduser()))
    receipt = Path(os.path.abspath(receipt.expanduser()))
    if parquet.is_symlink() or receipt.is_symlink():
        raise ValueError("output paths must not be symbolic links")
    if parquet == receipt or parquet in {pcap, yaf_csv} or receipt in {pcap, yaf_csv}:
        raise ValueError("outputs must be distinct from source artifacts and each other")
    parquet.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    canonical = temporary_file(parquet.parent, ".csv")
    temporary_parquet: Path | None = None
    try:
        counts = {key: 0 for key in FLOW_KEYS}
        with (
            yaf_csv.open(newline="", encoding="utf-8") as source,
            canonical.open("w", newline="", encoding="utf-8") as target,
        ):
            writer = csv.writer(target, lineterminator="\n")
            writer.writerow(OUTPUT_COLUMNS)
            for row in csv.reader(source):
                counts["rows"] += 1
                output, disposition = _canonical(row, capture_start_ms, capture_end_ms)
                packets = cast(int, output[9]) + cast(int, output[10])
                octets = cast(int, output[11]) + cast(int, output[12])
                _protocol(counts, cast(int, output[8]), packets, octets)
                counts["exported_packet_sum"] += packets
                counts["exported_l3_octet_sum"] += octets
                counts[disposition] += 1
                if disposition.startswith("rejected_"):
                    continue
                counts["accepted"] += 1
                counts["accepted_packet_sum"] += packets
                counts["accepted_l3_octet_sum"] += octets
                writer.writerow(output)
        _exact(expected_flows, counts, "expected_flows")
        if counts["accepted"] == 0:
            raise ValueError("no completed YAF flows remain after censoring")
        temporary_parquet = temporary_file(parquet.parent, ".parquet")
        pl.scan_csv(canonical, schema=SCHEMA).sort(
            ["flow_available_ms", "flow_end_ms", "flow_start_ms", *OUTPUT_COLUMNS[4:]]
        ).sink_parquet(temporary_parquet, compression="zstd")
        observed = list(pl.read_parquet_schema(temporary_parquet))
        if observed != list(OUTPUT_COLUMNS) or any("label" in name.lower() for name in observed):
            raise ValueError("materialized schema is not the declared unlabeled MAWI view")
        reconciliation = _yaf_reconciliation(yaf_stats, pcap_counts, counts)
        metadata_complete = (
            metadata is not None
            and set(metadata)
            == {
                "dataset_id",
                "config_sha256",
                "source",
                "meter",
                "feature_contract",
                "archive_verified",
                "conversion_artifacts_verified",
            }
            and metadata.get("archive_verified") is True
            and metadata.get("conversion_artifacts_verified") is True
            and isinstance(metadata.get("dataset_id"), str)
            and isinstance(metadata.get("config_sha256"), str)
            and isinstance(metadata.get("source"), dict)
            and bool(metadata.get("source"))
            and isinstance(metadata.get("meter"), dict)
            and bool(metadata.get("meter"))
            and isinstance(metadata.get("feature_contract"), dict)
            and bool(metadata.get("feature_contract"))
        )
        status = reconciliation["status"] if metadata_complete else "blocked"
        payload: dict[str, object] = {
            "receipt_version": 1,
            "source": {
                "pcap": pcap.name,
                "pcap_sha256": pcap_sha256,
                "yaf_csv": yaf_csv.name,
                "yaf_sha256": yaf_sha256,
            },
            "pcap_accounting": pcap_counts,
            "flow_accounting": counts,
            "yaf_reconciliation": reconciliation,
            "admission_metadata": metadata,
            "status": status,
            "qualification": {
                "tier": "Q1" if status == "reconciled" else "Q0",
                "permitted_use": "unlabeled general SSL only",
                "prohibited_claims": [
                    "benign or clean traffic",
                    "IDS accuracy",
                    "label calibration",
                    "independent network generalization",
                ],
            },
            "output": {
                "parquet": parquet.name,
                "schema": observed,
                "parquet_sha256": digest(temporary_parquet),
                "parquet_bytes": temporary_parquet.stat().st_size,
                "schema_sha256": hashlib.sha256(
                    json.dumps(
                        [
                            (name, str(dtype))
                            for name, dtype in pl.read_parquet_schema(temporary_parquet).items()
                        ],
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                "labels_present": False,
                "addresses": "routing_only",
                "availability": ("normal=end_ms; idle=capture_end_ms; active=capture_end_ms"),
                "fragment_policy": (
                    "YAF --no-frag required; fragments counted and excluded upstream"
                ),
            },
            "tooling": {
                "mawi_sha256": digest(Path(__file__)),
                "polars_version": pl.__version__,
            },
        }
        receipt_payload = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        try:
            publish_bundle(temporary_parquet, receipt_payload, parquet, receipt)
            temporary_parquet = None
        except OSError as error:
            raise ValueError(f"cannot publish MAWI artifact bundle: {error}") from error
        return payload
    finally:
        canonical.unlink(missing_ok=True)
        if temporary_parquet is not None:
            temporary_parquet.unlink(missing_ok=True)


def _json(path: Path, name: str) -> Mapping[str, object]:
    try:
        value = cast(object, json.loads(_regular(path, name).read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} must contain JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return cast(Mapping[str, object], value)


def load_config(path: Path) -> tuple[Path, Mapping[str, object]]:
    path = _regular(path, "--config")
    config = _json(path, "--config")
    if set(config) != set(CONFIG_KEYS) or config.get("schema_version") != 1:
        raise ValueError("--config has an unsupported schema")
    if not isinstance(config.get("id"), str):
        raise ValueError("--config.id must be a string")
    for key in (
        "source",
        "meter",
        "feature_contract",
        "expected_pcap",
        "expected_flows",
        "yaf_stats",
    ):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"--config.{key} must be an object")
    source = cast(Mapping[str, object], config["source"])
    meter = cast(Mapping[str, object], config["meter"])
    if set(source) != set(SOURCE_CONFIG_KEYS) or any(value is None for value in source.values()):
        raise ValueError("--config.source does not match the source metadata contract")
    if set(meter) != set(METER_CONFIG_KEYS) or any(value is None for value in meter.values()):
        raise ValueError("--config.meter does not match the meter metadata contract")
    _ = _feature_contract(config["feature_contract"])
    if not isinstance(meter.get("super_mediator_command"), list) or not isinstance(
        meter.get("super_mediator_config"), list
    ):
        raise ValueError("--config.meter must record Super Mediator command and config lines")
    for name in ("super_mediator_command", "super_mediator_config"):
        members = cast(list[object], meter[name])
        if not members or any(not isinstance(member, str) or not member for member in members):
            raise ValueError(f"--config.meter.{name} must contain non-empty strings")
    return path, config


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config", required=True, type=click.Path(path_type=Path, dir_okay=False))
@click.option("--archive", required=True, type=click.Path(path_type=Path, dir_okay=False))
@click.option("--pcap", required=True, type=click.Path(path_type=Path, dir_okay=False))
@click.option("--yaf-bin", required=True, type=click.Path(path_type=Path, dir_okay=False))
@click.option(
    "--super-mediator-bin", required=True, type=click.Path(path_type=Path, dir_okay=False)
)
@click.option("--yaf-ipfix", required=True, type=click.Path(path_type=Path, dir_okay=False))
@click.option("--yaf-csv", required=True, type=click.Path(path_type=Path, dir_okay=False))
@click.option("--parquet", required=True, type=click.Path(path_type=Path, dir_okay=False))
@click.option("--receipt", required=True, type=click.Path(path_type=Path, dir_okay=False))
def main(
    config: Path,
    archive: Path,
    pcap: Path,
    yaf_bin: Path,
    super_mediator_bin: Path,
    yaf_ipfix: Path,
    yaf_csv: Path,
    parquet: Path,
    receipt: Path,
) -> None:
    """Materialise one verified MAWI capture; no training eligibility is granted."""
    try:
        config_path, values = load_config(config)
        source = cast(Mapping[str, object], values["source"])
        meter = cast(Mapping[str, object], values["meter"])
        archive = _regular(archive, "--archive")
        pcap = _regular(pcap, "--pcap")
        yaf_bin = _regular(yaf_bin, "--yaf-bin")
        super_mediator_bin = _regular(super_mediator_bin, "--super-mediator-bin")
        yaf_ipfix = _regular(yaf_ipfix, "--yaf-ipfix")
        yaf_csv = _regular(yaf_csv, "--yaf-csv")
        if (
            archive.name != source.get("compressed_file")
            or archive.stat().st_size != source.get("compressed_bytes")
            or digest(archive) != source.get("compressed_sha256")
        ):
            raise ValueError("compressed MAWI archive does not match --config")
        decompressed_bytes, decompressed_sha256 = gzip_payload(archive)
        if decompressed_bytes != source.get("pcap_bytes") or decompressed_sha256 != source.get(
            "pcap_sha256"
        ):
            raise ValueError("compressed MAWI archive does not produce the configured PCAP")
        if (
            pcap.name != source.get("pcap_file")
            or pcap.stat().st_size != source.get("pcap_bytes")
            or yaf_ipfix.name != meter.get("yaf_ipfix_file")
            or yaf_ipfix.stat().st_size != meter.get("yaf_ipfix_bytes")
            or digest(yaf_ipfix) != meter.get("yaf_ipfix_sha256")
            or yaf_csv.name != meter.get("yaf_csv_file")
            or digest(yaf_bin) != meter.get("yaf_binary_sha256")
            or digest(super_mediator_bin) != meter.get("super_mediator_binary_sha256")
        ):
            raise ValueError("PCAP or conversion artifact does not match --config")
        result = materialize(
            pcap,
            yaf_csv,
            parquet,
            receipt,
            pcap_sha256=cast(str, source["pcap_sha256"]),
            yaf_sha256=cast(str, meter["yaf_csv_sha256"]),
            expected_pcap=cast(Mapping[str, object], values["expected_pcap"]),
            expected_flows=cast(Mapping[str, object], values["expected_flows"]),
            yaf_stats=cast(Mapping[str, object], values["yaf_stats"]),
            metadata={
                "dataset_id": values["id"],
                "config_sha256": digest(config_path),
                "source": source,
                "meter": meter,
                "feature_contract": values["feature_contract"],
                "archive_verified": True,
                "conversion_artifacts_verified": True,
            },
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    flow_accounting = cast(Mapping[str, object], result["flow_accounting"])
    click.echo(f"status={result['status']} flows={flow_accounting['accepted']} receipt={receipt}")


if __name__ == "__main__":
    main()
