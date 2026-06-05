"""Reject data-bearing files and unsafe fields from an explicit release archive."""

from __future__ import annotations

import argparse
import json
import tarfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

BLOCKED_COMPONENTS = frozenset(
    {"data", "raw", "secret", "secrets", "evidence", "receipt", "receipts"}
)
BLOCKED_SUFFIXES = frozenset(
    {".cap", ".csv", ".key", ".parquet", ".pcap", ".pcapng", ".pem", ".raw"}
)
BLOCKED_FILENAMES = frozenset(
    {".git-credentials", ".netrc", ".npmrc", ".pypirc", "credentials.json", "id_ed25519", "id_rsa"}
)
DEPLOYABLE_SCHEMA_NAMES = frozenset(
    {"checkpoint.json", "feature-schema.json", "model-schema.json", "run-manifest.json"}
)
MAX_SCHEMA_BYTES = 1_048_576
FORBIDDEN_FIELDS = frozenset(
    {
        "attack",
        "campaign_id",
        "capture_id",
        "capture_lineage_id",
        "canonical_family",
        "collector_id",
        "completion_ms",
        "context_group",
        "destination_key",
        "destination_principal",
        "destination_ip",
        "dst_ip",
        "event_id",
        "end_reason",
        "exact_group",
        "flow_id",
        "flow_end_milliseconds",
        "flow_end_ms",
        "flow_start_milliseconds",
        "flow_start_ms",
        "flow_available_ms",
        "hostname",
        "ipv4_dst_addr",
        "ipv4_src_addr",
        "label",
        "lineage_id",
        "mac_address",
        "near_group",
        "observation_domain_id",
        "partition_time_ms",
        "pretraining_visible",
        "routing",
        "scenario_id",
        "source_key",
        "source_principal",
        "source_unit_id",
        "source_ip",
        "src_ip",
        "target",
        "template_id",
        "timestamp",
    }
)


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    reason: str


@dataclass(frozen=True)
class Result:
    findings: tuple[Finding, ...]

    @property
    def ok(self) -> bool:
        return not self.findings

    def as_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "findings": [asdict(finding) for finding in self.findings]}


def check_paths(paths: Iterable[str | Path]) -> Result:
    """Check an explicit list of release paths, reading any listed JSON file."""
    entries: list[tuple[str, bytes | None]] = []
    findings: set[Finding] = set()
    for path in paths:
        name = str(path)
        candidate = Path(path)
        payload = None
        if _is_deployable_schema(name) and candidate.is_file():
            if candidate.stat().st_size > MAX_SCHEMA_BYTES:
                findings.add(Finding(name, "deployable schema exceeds size limit"))
            else:
                payload = candidate.read_bytes()
        entries.append((name, payload))
    result = _result(entries)
    return Result(tuple(sorted(findings.union(result.findings))))


def check_archive(archive: str | Path) -> Result:
    """Check the paths and JSON schemas contained in a tar archive."""
    with tarfile.open(archive, "r:*") as tar:
        entries: list[tuple[str, bytes | None]] = []
        findings: set[Finding] = set()
        for member in tar.getmembers():
            if member.issym() or member.islnk():
                findings.add(Finding(member.name, "archive link member"))
            if _is_deployable_schema(member.name) and member.size > MAX_SCHEMA_BYTES:
                findings.add(Finding(member.name, "deployable schema exceeds size limit"))
            file = (
                tar.extractfile(member)
                if member.isfile()
                and _is_deployable_schema(member.name)
                and member.size <= MAX_SCHEMA_BYTES
                else None
            )
            payload = file.read() if file is not None else None
            entries.append((member.name, payload))
    result = _result(entries)
    return Result(tuple(sorted(findings.union(result.findings))))


def _result(entries: Iterable[tuple[str, bytes | None]]) -> Result:
    findings: set[Finding] = set()
    for path, payload in entries:
        findings.update(_path_findings(path))
        if payload is not None and _is_deployable_schema(path):
            findings.update(_schema_findings(path, payload))
    return Result(tuple(sorted(findings)))


def _path_findings(path: str) -> set[Finding]:
    normalized = path.replace("\\", "/")
    parts = tuple(part.casefold() for part in normalized.split("/"))
    name = Path(normalized).name.casefold()
    findings: set[Finding] = set()
    if normalized.startswith("/") or ".." in parts:
        findings.add(Finding(path, "unsafe archive path"))
    if BLOCKED_COMPONENTS.intersection(parts):
        findings.add(Finding(path, "data, secret, or evidence artifact"))
    if any(suffix.casefold() in BLOCKED_SUFFIXES for suffix in Path(normalized).suffixes):
        findings.add(Finding(path, "raw data or secret file type"))
    if name in BLOCKED_FILENAMES or name == ".env" or name.startswith(".env."):
        findings.add(Finding(path, "standard secret file name"))
    return findings


def _schema_findings(path: str, payload: bytes) -> set[Finding]:
    try:
        document = cast(object, json.loads(payload))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {Finding(path, "invalid JSON schema")}
    forbidden = sorted(_field_names(document).intersection(FORBIDDEN_FIELDS))
    return {Finding(path, f"forbidden schema field: {field}") for field in forbidden}


def _is_deployable_schema(path: str) -> bool:
    return Path(path.replace("\\", "/")).name.casefold() in DEPLOYABLE_SCHEMA_NAMES


def _field_names(value: object) -> set[str]:
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        keys = {_normalize_field(key) for key in mapping if isinstance(key, str)}
        return keys.union(*(_field_names(nested) for nested in mapping.values()))
    if isinstance(value, list):
        fields: set[str] = set()
        for item in cast(list[object], value):
            fields.update(_field_names(item))
        return fields
    return {_normalize_field(value)} if isinstance(value, str) else set()


def _normalize_field(value: str) -> str:
    return value.casefold().replace("-", "_").replace(" ", "_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "paths", nargs="+", type=Path, help="A tar archive or explicit release paths."
    )
    arguments = parser.parse_args()
    paths = cast(list[Path], arguments.paths)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        parser.error(f"not an existing file: {', '.join(missing)}")
    results = [
        check_archive(path) if tarfile.is_tarfile(path) else check_paths([path]) for path in paths
    ]
    result = Result(tuple(sorted({finding for item in results for finding in item.findings})))
    print(json.dumps(result.as_dict(), sort_keys=True))
    return int(not result.ok)


if __name__ == "__main__":
    raise SystemExit(main())
