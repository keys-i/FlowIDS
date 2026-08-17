"""Write a deterministic conversion-integrity receipt for one CSV/Parquet pair."""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import os
import tempfile
from dataclasses import dataclass
from ipaddress import AddressValueError, IPv4Address
from pathlib import Path
from typing import cast

import click
import polars as pl

START = "FLOW_START_MILLISECONDS"
END = "FLOW_END_MILLISECONDS"
MODEL_ROLES = {"model_numeric", "model_categorical"}
FIELD_ROLES = MODEL_ROLES | {"routing", "target", "excluded"}
type JsonObject = dict[str, object]
PARITY_BATCH_ROWS = 65_536
PROVENANCE_FIELDS = (
    "capture_lineage",
    "site",
    "collector",
    "exporter",
    "meter",
    "meter_version",
    "flow_timeout",
    "time_range",
    "timezone",
    "license",
    "label_provenance",
)
NEAR_FIELDS = (
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "PROTOCOL",
    "IN_PKTS",
    "OUT_PKTS",
    "IN_BYTES",
    "OUT_BYTES",
    "TCP_FLAGS",
    "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS",
)


@dataclass(frozen=True)
class FeatureContract:
    """Validated field roles used to construct a model-only view."""

    field_order: tuple[str, ...]
    roles: dict[str, str]
    forbidden: frozenset[str]
    port_fields: frozenset[str]


@dataclass(frozen=True, slots=True)
class RoutingView:
    """Internal label-free routing data; endpoint values are keyed per lineage."""

    event_id: str
    exact_group: str
    near_group: str
    lineage_id: str
    observation_domain_id: str
    start_ms: int
    completion_ms: int
    source_key: str
    destination_key: str


def file_digest(path: Path, algorithm: str) -> str:
    """Return a streaming digest for a regular file."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for a regular file."""
    return file_digest(path, "sha256")


def csv_header(path: Path) -> list[str]:
    """Read exactly the CSV header, preserving field order."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle), None)
    if not header or any(not field for field in header):
        raise ValueError(f"CSV has no valid header: {path}")
    if len(header) != len(set(header)):
        raise ValueError(f"CSV header has duplicate fields: {path}")
    return header


def csv_row_count(path: Path) -> int:
    """Count parsed CSV records, not physical lines."""
    return cast(int, pl.scan_csv(path).select(pl.len()).collect().item())


def parquet_description(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Return ordered Parquet columns and their Polars types."""
    schema = pl.read_parquet_schema(path)
    return list(schema), [{"name": name, "type": str(dtype)} for name, dtype in schema.items()]


def full_value_parity(csv_path: Path, parquet_path: Path) -> JsonObject:
    """Compare every ordered parsed value in bounded Polars batches."""
    schema = pl.read_parquet_schema(parquet_path)
    compared = 0
    try:
        source_batches = iter(
            pl.scan_csv(csv_path, schema_overrides=schema).collect_batches(
                chunk_size=PARITY_BATCH_ROWS,
                maintain_order=True,
                engine="streaming",
            )
        )
        converted_batches = iter(
            pl.scan_parquet(parquet_path).collect_batches(
                chunk_size=PARITY_BATCH_ROWS,
                maintain_order=True,
                engine="streaming",
            )
        )
        source = converted = None
        source_offset = converted_offset = 0
        while True:
            if source is None or source_offset == source.height:
                source = next(source_batches, None)
                source_offset = 0
            if converted is None or converted_offset == converted.height:
                converted = next(converted_batches, None)
                converted_offset = 0
            if source is None or converted is None:
                return {
                    "batch_rows": PARITY_BATCH_ROWS,
                    "rows": compared,
                    "matches": source is None and converted is None,
                    "reason": (
                        None
                        if source is None and converted is None
                        else "CSV and Parquet streams ended at different rows."
                    ),
                }
            length = min(
                source.height - source_offset,
                converted.height - converted_offset,
            )
            source_slice = source.slice(source_offset, length)
            converted_slice = converted.slice(converted_offset, length)
            if not source_slice.equals(converted_slice):
                mismatch = next(
                    index
                    for index in range(length)
                    if not source_slice.slice(index, 1).equals(converted_slice.slice(index, 1))
                )
                return {
                    "batch_rows": PARITY_BATCH_ROWS,
                    "rows": compared + mismatch,
                    "matches": False,
                    "reason": "Parsed values differ.",
                }
            compared += length
            source_offset += length
            converted_offset += length
    except Exception as error:
        return {
            "batch_rows": PARITY_BATCH_ROWS,
            "rows": compared,
            "matches": False,
            "reason": f"Streaming comparison failed: {type(error).__name__}",
        }


def _json_object(path: Path) -> dict[str, object]:
    try:
        value = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in feature config {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Feature config must contain a JSON object: {path}")
    return cast(dict[str, object], value)


def load_feature_contract(path: Path) -> FeatureContract:
    """Load and validate one explicit model-field allow-list."""
    path = _regular_file(path, "--feature-config")
    config = _json_object(path)
    schema_value = config.get("schema")
    policy_value = config.get("model_input_policy")
    if not isinstance(schema_value, dict) or not isinstance(policy_value, dict):
        raise ValueError(f"Feature config requires schema and model_input_policy objects: {path}")
    schema = cast(dict[str, object], schema_value)
    policy = cast(dict[str, object], policy_value)
    expected_value = schema.get("field_order")
    roles_value = config.get("field_roles")
    forbidden_value = policy.get("never_model_inputs")
    port_fields_value = policy.get("port_fields", [])
    if not isinstance(expected_value, list):
        raise ValueError(f"feature config field_order must be a list of strings: {path}")
    expected_objects = cast(list[object], expected_value)
    if not all(isinstance(field, str) for field in expected_objects):
        raise ValueError(f"feature config field_order must be a list of strings: {path}")
    if not isinstance(roles_value, dict):
        raise ValueError(f"feature config field_roles must map strings to strings: {path}")
    role_objects = cast(dict[object, object], roles_value)
    if not all(
        isinstance(field, str) and isinstance(role, str) for field, role in role_objects.items()
    ):
        raise ValueError(f"feature config field_roles must map strings to strings: {path}")
    if not isinstance(forbidden_value, list):
        raise ValueError(f"feature config never_model_inputs must be a list of strings: {path}")
    forbidden_objects = cast(list[object], forbidden_value)
    if not all(isinstance(field, str) for field in forbidden_objects):
        raise ValueError(f"feature config never_model_inputs must be a list of strings: {path}")
    if not isinstance(port_fields_value, list):
        raise ValueError(f"feature config port_fields must be a list of strings: {path}")
    port_objects = cast(list[object], port_fields_value)
    if not all(isinstance(field, str) for field in port_objects):
        raise ValueError(f"feature config port_fields must be a list of strings: {path}")
    expected = cast(list[str], expected_value)
    roles = cast(dict[str, str], roles_value)
    forbidden = cast(list[str], forbidden_value)
    port_fields = cast(list[str], port_fields_value)
    if set(expected) != set(roles) or len(expected) != len(roles):
        raise ValueError(f"feature config must assign exactly one role to every field: {path}")
    invalid_roles = sorted(set(roles.values()) - FIELD_ROLES)
    if invalid_roles:
        raise ValueError(f"feature config has unsupported roles {invalid_roles}: {path}")
    model_fields = [field for field in expected if roles[field] in MODEL_ROLES]
    exposed = sorted(set(forbidden) & set(model_fields))
    if exposed:
        raise ValueError(f"forbidden fields are model-visible in {path}: {exposed}")
    if not set(port_fields) <= set(model_fields):
        raise ValueError(f"feature config port_fields must be model-visible: {path}")
    return FeatureContract(
        tuple(expected), dict(roles), frozenset(forbidden), frozenset(port_fields)
    )


def feature_contract(path: Path, observed_fields: list[str]) -> JsonObject:
    """Describe a validated field contract against the observed schema."""
    contract = load_feature_contract(path)
    model_fields = [field for field in contract.field_order if contract.roles[field] in MODEL_ROLES]
    return {
        "provided": True,
        "config": path.name,
        "sha256": file_sha256(path),
        "expected_order_matches": list(contract.field_order) == observed_fields,
        "model_fields": model_fields,
        "non_model_fields": [
            field for field in contract.field_order if contract.roles[field] not in MODEL_ROLES
        ],
    }


def dataset_evidence(path: Path, parquet_path: Path, source_sha1: str) -> JsonObject:
    """Resolve one configured dataset and report exact provenance omissions."""
    config = _json_object(path)
    datasets_value = config.get("datasets")
    if not isinstance(datasets_value, list):
        raise ValueError(f"Feature config requires a datasets list: {path}")
    datasets = cast(list[object], datasets_value)
    matches: list[dict[str, object]] = []
    for value in datasets:
        if not isinstance(value, dict):
            raise ValueError(f"Every configured dataset must be an object: {path}")
        dataset = cast(dict[str, object], value)
        configured_parquet = dataset.get("parquet")
        if not isinstance(configured_parquet, str):
            raise ValueError(f"Every configured dataset requires a parquet path: {path}")
        if Path(configured_parquet).name == parquet_path.name:
            matches.append(dataset)
    if len(matches) != 1:
        raise ValueError(
            f"Feature config must identify exactly one dataset for {parquet_path.name}: {path}"
        )

    dataset = matches[0]
    dataset_id = dataset.get("id")
    official_name = dataset.get("official_name")
    expected_source_sha1 = dataset.get("bagit_csv_sha1")
    metadata_value = dataset.get("source_metadata")
    evidence_value = dataset.get("evidence")
    q3_pairing = dataset.get("q3_pairing")
    if not isinstance(dataset_id, str) or not isinstance(official_name, str):
        raise ValueError(f"Configured dataset requires string id and official_name: {path}")
    if (
        not isinstance(expected_source_sha1, str)
        or len(expected_source_sha1) != 40
        or any(character not in "0123456789abcdefABCDEF" for character in expected_source_sha1)
    ):
        raise ValueError(f"Configured dataset requires a valid bagit_csv_sha1: {path}")
    if not hmac.compare_digest(expected_source_sha1.lower(), source_sha1.lower()):
        raise ValueError("Configured dataset does not match the audited BagIt CSV payload")
    if not isinstance(metadata_value, dict):
        raise ValueError(f"Configured dataset requires source_metadata: {path}")
    metadata = cast(dict[str, object], metadata_value)
    if not isinstance(evidence_value, list) or not all(
        isinstance(value, str) for value in cast(list[object], evidence_value)
    ):
        raise ValueError(f"Configured dataset evidence must be a list of URLs: {path}")
    if not isinstance(q3_pairing, bool):
        raise ValueError(f"Configured dataset q3_pairing must be boolean: {path}")
    if q3_pairing and not isinstance(dataset.get("pairing_manifest"), str):
        raise ValueError(f"q3_pairing=true requires a pairing_manifest: {path}")

    missing = [
        field
        for field in PROVENANCE_FIELDS
        if not isinstance(metadata.get(field), str) or not cast(str, metadata[field]).strip()
    ]
    return {
        "id": dataset_id,
        "official_name": official_name,
        "bagit_csv_sha1": expected_source_sha1.lower(),
        "evidence": evidence_value,
        "missing_required_fields": missing,
        "source_tier": "Q1" if not missing else "Q0",
        "q3_pairing": q3_pairing,
    }


def _port_token(value: object) -> str:
    if value is None or value == "":
        return "MISSING"
    if isinstance(value, bool):
        return "UNK"
    if isinstance(value, int):
        port = value
    elif isinstance(value, float) and value.is_integer():
        port = int(value)
    elif isinstance(value, str) and value.isdecimal():
        port = int(value)
    else:
        return "UNK"
    if not 0 <= port <= 65535:
        return "UNK"
    if port <= 1023:
        return f"PORT_{port}"
    return "REGISTERED" if port <= 49151 else "DYNAMIC"


def _selected_view(
    raw: dict[str, object], contract: FeatureContract, *, encode_ports: bool
) -> dict[str, dict[str, object]]:
    missing = [
        field
        for field in contract.field_order
        if contract.roles[field] in MODEL_ROLES and field not in raw
    ]
    if missing:
        raise ValueError(f"Raw flow is missing model fields: {missing}")
    numeric: dict[str, object] = {}
    categorical: dict[str, object] = {}
    for field in contract.field_order:
        role = contract.roles[field]
        if role == "model_numeric":
            value = raw[field]
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int | float)
            ):
                raise ValueError(f"Numeric field {field} has non-numeric value {value!r}")
            numeric[field] = value
        elif role == "model_categorical":
            categorical[field] = (
                _port_token(raw[field])
                if encode_ports and field in contract.port_fields
                else raw[field]
            )
    return {"numeric": numeric, "categorical": categorical}


def preprocessing_view(
    raw: dict[str, object], contract: FeatureContract
) -> dict[str, dict[str, object]]:
    """Select model fields for train-only fitting; ports remain internal raw integers."""
    return _selected_view(raw, contract, encode_ports=False)


def model_view(raw: dict[str, object], contract: FeatureContract) -> dict[str, dict[str, object]]:
    """Return only explicitly model-visible fields; raw high ports become range tokens."""
    return _selected_view(raw, contract, encode_ports=True)


def _routing_integer(raw: dict[str, object], field: str) -> int:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
        raise ValueError(f"Routing field {field} must be an unsigned 63-bit integer")
    return value


def _near_integer(raw: dict[str, object], field: str) -> int | None:
    value = raw.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Near-group field {field} must be a non-negative integer or null")
    upper = (
        65_535
        if field in {"L4_SRC_PORT", "L4_DST_PORT"}
        else 255
        if field == "PROTOCOL"
        else 2**64 - 1
    )
    if value > upper:
        raise ValueError(f"Near-group field {field} exceeds {upper}")
    return value


def lineage_key(secret: object, lineage_id: object) -> tuple[bytes, str]:
    if not isinstance(secret, bytes) or len(secret) < 16:
        raise ValueError("routing secret must contain at least 16 bytes")
    if not isinstance(lineage_id, str) or not lineage_id.strip():
        raise ValueError("lineage_id must be a non-empty string")
    return hmac.digest(secret, f"flowids:{lineage_id}".encode(), "sha256"), lineage_id


def _endpoint_key(value: object, key: bytes, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Routing field {field} must be a non-empty string")
    try:
        packed = IPv4Address(value).packed
    except AddressValueError as error:
        raise ValueError(f"Routing field {field} must be a valid IPv4 address") from error
    return f"anon:{hmac.digest(key, packed, 'sha256').hex()}"


def routing_view(
    raw: dict[str, object],
    contract: FeatureContract,
    *,
    lineage_id: object,
    observation_domain_id: object,
    source_ordinal: object,
    secret: object,
) -> RoutingView:
    """Create internal routing keys and label-free exact/event identities."""
    if (
        isinstance(source_ordinal, bool)
        or not isinstance(source_ordinal, int)
        or not 0 <= source_ordinal < 2**64
    ):
        raise ValueError("source_ordinal must be an unsigned 64-bit integer")
    required = [field for field in contract.field_order if contract.roles[field] != "target"]
    missing = [field for field in required if field not in raw]
    if missing:
        raise ValueError(f"Raw flow is missing non-target fields: {missing}")
    start = _routing_integer(raw, START)
    end = _routing_integer(raw, END)
    if end < start:
        raise ValueError("FLOW_END_MILLISECONDS must not precede FLOW_START_MILLISECONDS")
    key, checked_lineage = lineage_key(secret, lineage_id)
    if not isinstance(observation_domain_id, str) or not observation_domain_id.strip():
        raise ValueError("observation_domain_id must be a non-empty string")
    source = _endpoint_key(raw.get("IPV4_SRC_ADDR"), key, "IPV4_SRC_ADDR")
    destination = _endpoint_key(raw.get("IPV4_DST_ADDR"), key, "IPV4_DST_ADDR")
    exact_values = [(field, raw[field]) for field in required]
    try:
        encoded = json.dumps(
            exact_values, ensure_ascii=True, allow_nan=False, separators=(",", ":")
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError(f"Raw flow contains a non-canonical value: {error}") from error
    exact_digest = hmac.digest(key, encoded, "sha256").hex()
    near_values = {
        "version": 1,
        "observation_domain_id": observation_domain_id,
        "source": source,
        "destination": destination,
        "start_ms": start,
        "end_ms": end,
        "fields": [(field, _near_integer(raw, field)) for field in NEAR_FIELDS],
        "presence": [raw.get(field) is not None for field in NEAR_FIELDS],
    }
    near_encoded = json.dumps(
        near_values, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()
    near_digest = hmac.digest(key, b"near-v1:" + near_encoded, "sha256").hex()
    event_identity = json.dumps(
        ["event-v1", observation_domain_id, exact_digest, source_ordinal],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    event_digest = hmac.digest(key, event_identity, "sha256").hex()
    return RoutingView(
        event_id=f"event:{event_digest}",
        exact_group=f"exact:{exact_digest}",
        near_group=f"near:{near_digest}",
        lineage_id=checked_lineage,
        observation_domain_id=observation_domain_id,
        start_ms=start,
        completion_ms=end,
        source_key=source,
        destination_key=destination,
    )


def bagit_sha1(csv_path: Path, manifest: Path) -> JsonObject:
    """Verify the manifest SHA-1 entry that resolves to ``csv_path``."""
    entries: list[tuple[str, Path]] = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if (
            len(parts) != 2
            or len(parts[0]) != 40
            or any(character not in "0123456789abcdefABCDEF" for character in parts[0])
        ):
            raise ValueError(f"Invalid BagIt SHA-1 entry at {manifest}:{line_number}")
        entries.append((parts[0].lower(), (manifest.parent / parts[1].lstrip("*")).resolve()))

    actual = file_digest(csv_path, "sha1")
    matches = [digest for digest, entry_path in entries if entry_path == csv_path.resolve()]
    if len(matches) != 1:
        return {
            "provided": True,
            "entry_found": False,
            "actual_sha1": actual,
            "sha1_matches": False,
            "reason": "Manifest must contain exactly one SHA-1 entry for --csv.",
        }

    return {
        "provided": True,
        "entry_found": True,
        "expected_sha1": matches[0],
        "actual_sha1": actual,
        "sha1_matches": actual == matches[0],
    }


def timestamp_metrics(path: Path, columns: list[str]) -> JsonObject:
    """Measure timestamp validity and source storage-order regressions in Parquet order."""
    if START not in columns or END not in columns:
        return {
            "available": False,
            "reason": f"Parquet must contain {START} and {END}.",
        }

    start = pl.col(START).cast(pl.Int64, strict=False)
    end = pl.col(END).cast(pl.Int64, strict=False)
    values = cast(
        dict[str, int | None],
        pl.scan_parquet(path)
        .select(
            start.null_count().alias("start_null_or_unparseable"),
            end.null_count().alias("end_null_or_unparseable"),
            (end < start).fill_null(False).sum().alias("end_before_start"),
            start.diff().lt(0).fill_null(False).sum().alias("start_time_backward_transitions"),
            end.diff().lt(0).fill_null(False).sum().alias("end_time_backward_transitions"),
            start.min().alias("start_min"),
            end.max().alias("end_max"),
        )
        .collect()
        .row(0, named=True),
    )
    normalized = {key: int(value) if value is not None else None for key, value in values.items()}
    return {"available": True, **normalized}


def _regular_file(path: Path, option: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{option} must name an existing regular file: {path}")
    return resolved


def audit(
    csv_path: Path,
    parquet_path: Path,
    bag_manifest: Path,
    feature_config: Path,
) -> JsonObject:
    """Build an in-memory receipt; input failures raise ``ValueError`` with an action."""
    csv_path = _regular_file(csv_path, "--csv")
    parquet_path = _regular_file(parquet_path, "--parquet")
    if csv_path == parquet_path:
        raise ValueError("--csv and --parquet must be different files.")
    bag_manifest = _regular_file(bag_manifest, "--bag-manifest")
    feature_config = _regular_file(feature_config, "--feature-config")

    header = csv_header(csv_path)
    parquet_columns, parquet_schema = parquet_description(parquet_path)
    input_rows = csv_row_count(csv_path)
    output_rows = cast(int, pl.scan_parquet(parquet_path).select(pl.len()).collect().item())
    row_difference = input_rows - output_rows
    parity = full_value_parity(csv_path, parquet_path)
    timestamps = timestamp_metrics(parquet_path, parquet_columns)
    bagit = bagit_sha1(csv_path, bag_manifest)
    contract = feature_contract(feature_config, parquet_columns)
    source = dataset_evidence(feature_config, parquet_path, cast(str, bagit["actual_sha1"]))

    timestamp_available = timestamps.get("available") is True
    checks: dict[str, bool] = {
        "csv_header_matches_parquet_order": header == parquet_columns,
        "csv_row_count_matches_parquet": input_rows == output_rows,
        "no_rows_dropped": row_difference == 0,
        "timestamp_columns_present": timestamp_available,
        "timestamp_values_valid": timestamp_available
        and timestamps.get("start_null_or_unparseable") == 0
        and timestamps.get("end_null_or_unparseable") == 0
        and timestamps.get("end_before_start") == 0,
        "bagit_sha1_valid": bagit.get("sha1_matches") is True,
        "feature_order_matches_contract": contract.get("expected_order_matches") is True,
        "full_values_match": parity.get("matches") is True,
    }
    integrity_passed = all(checks.values())
    missing_provenance = cast(list[str], source["missing_required_fields"])
    qualification_reasons = (
        ["Required source provenance is incomplete: " + ", ".join(missing_provenance)]
        if missing_provenance
        else []
    )
    if not integrity_passed:
        qualification_reasons.append("One or more conversion-integrity checks failed.")
    if timestamp_available and (
        timestamps.get("start_time_backward_transitions", 0)
        or timestamps.get("end_time_backward_transitions", 0)
    ):
        qualification_reasons.append(
            "Stored Parquet order is not chronological; causal contexts must be explicitly sorted."
        )

    return {
        "receipt_version": 4,
        "d0_status": "blocked",
        "d0_gates": {
            "conversion_integrity": "passed" if integrity_passed else "failed",
            "full_value_parity": "passed" if parity.get("matches") is True else "failed",
            "model_input_isolation": (
                "passed" if checks["feature_order_matches_contract"] else "failed"
            ),
            "source_eligibility": "blocked" if missing_provenance else "passed",
            "split_isolation": "not_run",
            "fit_scope": "not_run",
            "causal_state": "not_run",
            "q3_pairing": "disabled" if source["q3_pairing"] is False else "not_run",
            "release_isolation": "not_run",
        },
        "inputs": {
            "csv": csv_path.name,
            "parquet": parquet_path.name,
            "bag_manifest": bag_manifest.name,
        },
        "sha256": {"csv": file_sha256(csv_path), "parquet": file_sha256(parquet_path)},
        "bagit": bagit,
        "source": source,
        "feature_contract": contract,
        "full_value_parity": parity,
        "tooling": {
            "d0_sha256": file_sha256(Path(__file__)),
            "polars_version": pl.__version__,
        },
        "schema": {
            "csv_header": header,
            "parquet_columns": parquet_columns,
            "parquet_schema": parquet_schema,
        },
        "rows": {
            "csv": input_rows,
            "parquet": output_rows,
            "dropped": max(row_difference, 0),
            "added": max(-row_difference, 0),
        },
        "timestamps": timestamps,
        "checks": checks,
        "status": "integrity_passed" if integrity_passed else "integrity_failed",
        "qualification": {
            "tier": source["source_tier"],
            "q1_eligible": not missing_provenance and integrity_passed,
            "reasons": qualification_reasons,
        },
    }


def write_receipt(receipt: JsonObject, output: Path, protected_paths: tuple[Path, ...]) -> Path:
    """Atomically create an append-only canonical JSON receipt."""
    output = output.expanduser().resolve()
    protected = {path.expanduser().resolve() for path in protected_paths}
    if output in protected:
        raise ValueError("--output must not replace an audited input file")
    if output.exists() and output.is_dir():
        raise ValueError(f"--output must name a file, not a directory: {output}")
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    if output.exists():
        if output.read_bytes() == payload:
            return output
        raise ValueError(f"--output already exists with different content: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            _ = handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            if output.read_bytes() != payload:
                raise ValueError(
                    f"--output already exists with different content: {output}"
                ) from error
    finally:
        if os.path.exists(temporary):
            _ = os.unlink(temporary)
    return output


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--csv", "csv_path", required=True, type=click.Path(path_type=Path, dir_okay=False))
@click.option(
    "--parquet", "parquet_path", required=True, type=click.Path(path_type=Path, dir_okay=False)
)
@click.option(
    "--bag-manifest",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False),
    help="BagIt manifest-sha1.txt for the source CSV.",
)
@click.option(
    "--feature-config",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False),
    help="JSON field-role contract.",
)
@click.option("--output", required=True, type=click.Path(path_type=Path, dir_okay=False))
def main(
    csv_path: Path,
    parquet_path: Path,
    bag_manifest: Path,
    feature_config: Path,
    output: Path,
) -> None:
    """Write a conversion receipt; never approve model training."""
    try:
        receipt = audit(csv_path, parquet_path, bag_manifest, feature_config)
        protected = (csv_path, parquet_path, bag_manifest, feature_config)
        output = write_receipt(receipt, output, protected)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    status = cast(str, receipt["status"])
    qualification = cast(dict[str, object], receipt["qualification"])
    click.echo(
        " ".join(
            (
                f"conversion_status={status}",
                "d0_status=blocked",
                f"qualification={qualification['tier']}",
                f"receipt={output}",
            )
        )
    )
    if status != "integrity_passed":
        raise SystemExit(1)
    raise SystemExit(3)


if __name__ == "__main__":
    main()
