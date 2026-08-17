"""Build a compact, local chronological split for one admitted MAWI source."""

from __future__ import annotations

import csv
import hashlib
import hmac
import ipaddress
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

import click
import polars as pl

from tools.scripts import d0, mawi, splits

type JsonObject = dict[str, object]

TIMELINE_FIELD: Final = "flow_available_ms"
EVENT_VERSION: Final = "MAWI-YAF-event-v1"
ENDPOINT_VERSION: Final = "MAWI-YAF-endpoint-v1"
EXACT_VERSION: Final = "MAWI-YAF-exact-v1"
NEAR_VERSION: Final = "MAWI-YAF-near-v1"
BATCH_ROWS: Final = 65_536
NEAR_FIELDS: Final[tuple[str, ...]] = (
    "flow_start_ms",
    "flow_end_ms",
    "flow_available_ms",
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
)
SIDECAR_COLUMNS: Final[tuple[str, ...]] = (
    "event_id",
    "partition_time_ms",
    "partition",
    "pretraining_visible",
    "source_unit_id",
    "capture_lineage_id",
    "source_principal",
    "destination_principal",
    "exact_group",
    "near_group",
    "drop_reason",
)
SIDECAR_SCHEMA: Final = {
    "event_id": pl.String,
    "partition_time_ms": pl.Int64,
    "partition": pl.String,
    "pretraining_visible": pl.Boolean,
    "source_unit_id": pl.String,
    "capture_lineage_id": pl.String,
    "source_principal": pl.String,
    "destination_principal": pl.String,
    "exact_group": pl.String,
    "near_group": pl.String,
    "drop_reason": pl.String,
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(Mapping[str, object], value)


def _regular(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} must be an existing regular file")
    return resolved


def _json(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} must contain JSON") from error
    return _mapping(value, label)


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _schema_description(path: Path) -> list[tuple[str, str]]:
    return [(name, str(dtype)) for name, dtype in pl.read_parquet_schema(path).items()]


def _schema_sha256(description: list[tuple[str, str]]) -> str:
    return hashlib.sha256(json.dumps(description, separators=(",", ":")).encode()).hexdigest()


def _source(
    config_path: Path, source_receipt: Path, source_parquet: Path
) -> tuple[Mapping[str, object], Mapping[str, object], int, int, str, str, str]:
    config_path = _regular(config_path, "--config")
    source_receipt = _regular(source_receipt, "--source-receipt")
    source_parquet = _regular(source_parquet, "--source-parquet")
    _, config = mawi.load_config(config_path)
    receipt = _json(source_receipt, "--source-receipt")
    if receipt.get("receipt_version") != 1 or receipt.get("status") != "reconciled":
        raise ValueError("source receipt is not reconciled")
    source_evidence = _mapping(receipt.get("source"), "source receipt source")
    output = _mapping(receipt.get("output"), "source receipt output")
    qualification = _mapping(receipt.get("qualification"), "source receipt qualification")
    admission = _mapping(receipt.get("admission_metadata"), "source receipt admission_metadata")
    accounting = _mapping(receipt.get("pcap_accounting"), "source receipt pcap_accounting")
    flow_accounting = _mapping(receipt.get("flow_accounting"), "source receipt flow_accounting")
    yaf_reconciliation = _mapping(
        receipt.get("yaf_reconciliation"), "source receipt yaf_reconciliation"
    )
    tooling = _mapping(receipt.get("tooling"), "source receipt tooling")
    dataset_id = _string(config.get("id"), "config id")
    config_source = _mapping(config.get("source"), "config source")
    config_meter = _mapping(config.get("meter"), "config meter")
    if (
        qualification.get("tier") != "Q1"
        or qualification.get("permitted_use") != "unlabeled general SSL only"
    ):
        raise ValueError("source receipt is not restricted to Q1 unlabeled SSL")
    prohibited = qualification.get("prohibited_claims")
    if not isinstance(prohibited, list):
        raise ValueError("source receipt is missing required Q1 claim prohibitions")
    prohibited_values = cast(list[object], prohibited)
    if not all(isinstance(item, str) for item in prohibited_values) or not {
        "benign or clean traffic",
        "IDS accuracy",
        "label calibration",
        "independent network generalization",
    } <= set(cast(list[str], prohibited_values)):
        raise ValueError("source receipt is missing required Q1 claim prohibitions")
    if output.get("labels_present") is not False:
        raise ValueError("source receipt must prove labels are absent")
    if (
        output.get("availability") != "normal=end_ms; idle=capture_end_ms; active=capture_end_ms"
        or output.get("addresses") != "routing_only"
        or output.get("fragment_policy")
        != "YAF --no-frag required; fragments counted and excluded upstream"
    ):
        raise ValueError("source receipt has unsupported availability semantics")
    if (
        admission.get("dataset_id") != dataset_id
        or admission.get("config_sha256") != mawi.digest(config_path)
        or admission.get("archive_verified") is not True
        or admission.get("conversion_artifacts_verified") is not True
        or admission.get("source") != config_source
        or admission.get("meter") != config_meter
        or admission.get("feature_contract") != config.get("feature_contract")
        or accounting != _mapping(config.get("expected_pcap"), "config expected_pcap")
        or flow_accounting != _mapping(config.get("expected_flows"), "config expected_flows")
        or yaf_reconciliation.get("status") != "reconciled"
    ):
        raise ValueError("source receipt is not bound to the configured admitted source")
    if (
        source_evidence.get("pcap") != config_source.get("pcap_file")
        or source_evidence.get("pcap_sha256") != config_source.get("pcap_sha256")
        or source_evidence.get("yaf_csv") != config_meter.get("yaf_csv_file")
        or source_evidence.get("yaf_sha256") != config_meter.get("yaf_csv_sha256")
    ):
        raise ValueError("source receipt does not identify the configured PCAP and YAF export")
    mawi_sha256 = tooling.get("mawi_sha256")
    if (
        not isinstance(mawi_sha256, str)
        or len(mawi_sha256) != 64
        or any(character not in "0123456789abcdef" for character in mawi_sha256)
        or not isinstance(tooling.get("polars_version"), str)
    ):
        raise ValueError("source receipt lacks materializer version evidence")
    if (
        output.get("parquet") != source_parquet.name
        or output.get("parquet_bytes") != source_parquet.stat().st_size
        or output.get("parquet_sha256") != mawi.digest(source_parquet)
    ):
        raise ValueError("source receipt SHA-256 or size does not match the source Parquet")
    description = _schema_description(source_parquet)
    if [name for name, _ in description] != list(mawi.OUTPUT_COLUMNS):
        raise ValueError("source Parquet does not match the admitted MAWI schema")
    if output.get("schema") != list(mawi.OUTPUT_COLUMNS) or output.get(
        "schema_sha256"
    ) != _schema_sha256(description):
        raise ValueError("source receipt does not match the source Parquet")
    source_rows = cast(int, pl.scan_parquet(source_parquet).select(pl.len()).collect().item())
    if flow_accounting.get("accepted") != source_rows:
        raise ValueError("source receipt accepted count does not match the source Parquet")
    capture_start = _integer(accounting.get("capture_start_ms"), "capture_start_ms")
    capture_end = _integer(accounting.get("capture_end_ms"), "capture_end_ms")
    if capture_start >= capture_end:
        raise ValueError("source receipt capture range is invalid")
    lineage = _string(config_source.get("capture_lineage"), "capture lineage")
    observation = _string(config_source.get("observation_point"), "observation point")
    return config, receipt, capture_start, capture_end, dataset_id, lineage, observation


def _secret(path: Path) -> bytes:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError("--secret-file must not be a symbolic link")
    try:
        details = expanded.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise ValueError("--secret-file must be an existing regular file") from error
    if not stat.S_ISREG(details.st_mode):
        raise ValueError("--secret-file must be an existing regular file")
    if stat.S_IMODE(details.st_mode) & 0o077:
        raise ValueError("--secret-file must be readable only by its owner")
    value = expanded.read_bytes()
    if len(value) != 32:
        raise ValueError("--secret-file must contain exactly 32 bytes")
    return value


def _endpoint_key(value: object, lineage_key: bytes, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be an IPv4 or IPv6 address")
    try:
        address = ipaddress.ip_address(value)
    except ValueError as error:
        raise ValueError(f"{field} must be an IPv4 or IPv6 address") from error
    domain = b"v4:" if address.version == 4 else b"v6:"
    digest = hmac.digest(
        lineage_key, ENDPOINT_VERSION.encode() + b":" + domain + address.packed, "sha256"
    )
    return f"anon:{digest.hex()}"


def _flow_integer(row: Mapping[str, object], field: str, maximum: int = 2**63 - 1) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError(f"source field {field} must be an integer between 0 and {maximum}")
    return value


def _identities(
    row: Mapping[str, object],
    ordinal: int,
    lineage_key: bytes,
    lineage_id: str,
    observation_id: str,
    capture_end: int,
) -> tuple[str, str, str, str, str]:
    if set(row) != set(mawi.OUTPUT_COLUMNS):
        raise ValueError("source row does not match the admitted MAWI schema")
    start = _flow_integer(row, "flow_start_ms")
    end = _flow_integer(row, "flow_end_ms")
    available = _flow_integer(row, "flow_available_ms")
    duration = _flow_integer(row, "flow_duration_ms")
    if start > end or end > available or available > capture_end or duration != end - start:
        raise ValueError("source flow has inconsistent interval or availability fields")
    for field in ("source_port", "destination_port"):
        _ = _flow_integer(row, field, 65_535)
    _ = _flow_integer(row, "protocol", 255)
    for field in ("forward_packets", "reverse_packets", "forward_bytes", "reverse_bytes"):
        _ = _flow_integer(row, field)
    for field in (
        "initial_tcp_flags",
        "reverse_initial_tcp_flags",
        "union_tcp_flags",
        "reverse_union_tcp_flags",
    ):
        value = row.get(field)
        if not isinstance(value, str) or "\x00" in value or len(value) > 64:
            raise ValueError(f"source field {field} must be a bounded string")
    reason = row.get("end_reason")
    if reason not in {None, "idle", "active"}:
        raise ValueError("source flow has an unsupported retained end_reason")
    if (reason is None and available != end) or (
        reason in {"idle", "active"} and available != capture_end
    ):
        raise ValueError("source flow contradicts the admitted availability semantics")
    source = _endpoint_key(row.get("source_ip"), lineage_key, "source_ip")
    destination = _endpoint_key(row.get("destination_ip"), lineage_key, "destination_ip")
    exact = hmac.digest(
        lineage_key,
        _canonical([EXACT_VERSION, [(field, row[field]) for field in mawi.OUTPUT_COLUMNS]]),
        "sha256",
    ).hex()
    near = hmac.digest(
        lineage_key,
        _canonical(
            [
                NEAR_VERSION,
                lineage_id,
                observation_id,
                source,
                destination,
                [(field, row[field]) for field in NEAR_FIELDS],
            ]
        ),
        "sha256",
    ).hex()
    event = hmac.digest(
        lineage_key,
        _canonical([EVENT_VERSION, observation_id, ordinal]),
        "sha256",
    ).hex()
    return f"event:{event}", f"exact:{exact}", f"near:{near}", source, destination


def _validate_spec(spec: splits.SplitSpec, capture_start: int, capture_end: int) -> None:
    if spec.track != "chronological":
        raise ValueError("MAWI split requires chronological partitioning")
    if spec.train_end_ms is None or spec.validation_end_ms is None:
        raise ValueError("MAWI split requires both chronological cutoffs")
    if not (
        capture_start < spec.train_end_ms - spec.purge_ms
        and spec.train_end_ms + spec.purge_ms < spec.validation_end_ms - spec.purge_ms
        and spec.validation_end_ms + spec.purge_ms < capture_end
    ):
        raise ValueError("cutoffs and purge must leave three non-empty capture intervals")


def _output(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    resolved = Path(os.path.abspath(expanded))
    if resolved.exists() and not resolved.is_file():
        raise ValueError(f"{label} must name a file")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _partition_metrics(sidecar: Path) -> JsonObject:
    frame = pl.scan_parquet(sidecar)
    count_rows = cast(
        list[tuple[object, object]],
        frame.group_by("partition").len().collect(engine="streaming").rows(),
    )
    counts = {
        _string(partition, "sidecar partition"): _integer(count, "sidecar count")
        for partition, count in count_rows
    }
    missing = {"train", "validation", "test"} - set(counts)
    if missing:
        raise ValueError(f"required split partitions are empty: {', '.join(sorted(missing))}")
    retained = frame.filter(pl.col("partition") != "dropped")
    event_counts = frame.select(pl.len().alias("rows"), pl.col("event_id").n_unique().alias("ids"))
    rows, event_ids = cast(tuple[int, int], event_counts.collect(engine="streaming").row(0))
    if rows != event_ids:
        raise ValueError("split sidecar repeats an event_id")
    visibility_mismatch = cast(
        int,
        frame.filter(pl.col("pretraining_visible") != (pl.col("partition") == "train"))
        .select(pl.len())
        .collect(engine="streaming")
        .item(),
    )
    if visibility_mismatch:
        raise ValueError("split sidecar has invalid pretraining visibility")
    overlaps: dict[str, int] = {}
    for field, label in (("exact_group", "exact"), ("near_group", "strict_near")):
        overlaps[label] = cast(
            int,
            retained.group_by(field)
            .agg(pl.col("partition").n_unique().alias("partitions"))
            .filter(pl.col("partitions") > 1)
            .select(pl.len())
            .collect(engine="streaming")
            .item(),
        )
        if overlaps[label]:
            raise ValueError(f"split sidecar retains cross-partition {label} groups")
    ties = frame.group_by(["partition", "partition_time_ms"]).len().filter(pl.col("len") > 1)
    tie_rows = ties.group_by("partition").agg(
        pl.len().alias("timestamps"),
        pl.col("len").sum().alias("rows"),
        pl.col("len").max().alias("largest"),
    )
    tie_values = cast(
        list[tuple[object, object, object, object]],
        tie_rows.collect(engine="streaming").rows(),
    )
    tie_summary = {
        _string(partition, "tie partition"): {
            "timestamps": _integer(timestamps, "tied timestamps"),
            "rows": _integer(tied_rows, "tied rows"),
            "largest": _integer(largest, "largest tie"),
        }
        for partition, timestamps, tied_rows, largest in tie_values
    }
    dropped_values = cast(
        list[tuple[object, object]],
        frame.filter(pl.col("partition") == "dropped")
        .group_by("drop_reason")
        .len()
        .collect(engine="streaming")
        .rows(),
    )
    drop_reasons = {
        _string(reason, "drop reason"): _integer(count, "drop reason count")
        for reason, count in dropped_values
    }
    return {
        "input": rows,
        "retained": rows - counts.get("dropped", 0),
        "dropped": counts.get("dropped", 0),
        "partitions": dict(sorted(counts.items())),
        "drop_reasons": dict(sorted(drop_reasons.items())),
        "availability_ties": dict(sorted(tie_summary.items())),
        "overlaps": overlaps,
    }


def build(
    config: Path,
    source_receipt: Path,
    source_parquet: Path,
    secret_file: Path,
    spec: splits.SplitSpec,
    sidecar: Path,
    receipt: Path,
) -> JsonObject:
    """Write an immutable local assignment sidecar and small evidence receipt."""
    config = _regular(config, "--config")
    source_receipt = _regular(source_receipt, "--source-receipt")
    source_parquet = _regular(source_parquet, "--source-parquet")
    secret_path = _regular(secret_file, "--secret-file")
    secret = _secret(secret_path)
    values, admitted, capture_start, capture_end, dataset_id, lineage, observation = _source(
        config, source_receipt, source_parquet
    )
    _validate_spec(spec, capture_start, capture_end)
    sidecar = _output(sidecar, "--sidecar")
    receipt = _output(receipt, "--receipt")
    protected = {config, source_receipt, source_parquet, secret_path}
    if sidecar == receipt or sidecar in protected or receipt in protected:
        raise ValueError("outputs must be distinct from every input and each other")
    lineage_key, _ = d0.lineage_key(secret, lineage)
    key_id = hmac.digest(lineage_key, b"MAWI-YAF-key-id-v1", "sha256").hex()
    source_unit_id = f"source:{dataset_id}"
    lineage_id = f"lineage:{hashlib.sha256(lineage.encode()).hexdigest()}"
    observation_id = f"observation:{hashlib.sha256(observation.encode()).hexdigest()}"
    contract: JsonObject = {
        "event": {"version": EVENT_VERSION, "fields": ["observation_id", "row_ordinal"]},
        "endpoint": {
            "version": ENDPOINT_VERSION,
            "normalization": "ipaddress packed bytes with explicit v4/v6 domain",
        },
        "exact": {"version": EXACT_VERSION, "fields": list(mawi.OUTPUT_COLUMNS)},
        "strict_near": {
            "version": NEAR_VERSION,
            "fields": [
                "lineage_id",
                "observation_id",
                "source_principal",
                "destination_principal",
                *NEAR_FIELDS,
            ],
            "purpose": "strict duplicate protection; not semantic context similarity",
        },
        "hmac": "HMAC-SHA-256",
        "key_id": key_id,
        "key_scope": lineage_id,
        "secret_persisted": False,
    }
    contract["sha256"] = _sha256(contract)
    temporary_csv = mawi.temporary_file(sidecar.parent, ".csv")
    temporary_sidecar: Path | None = None
    timeout_rows = 0
    capture_end_timeout_rows = 0
    input_rows = 0
    try:
        with temporary_csv.open("w", newline="", encoding="utf-8") as target:
            writer = csv.writer(target, lineterminator="\n")
            writer.writerow(SIDECAR_COLUMNS)
            for batch in pl.scan_parquet(source_parquet).collect_batches(
                chunk_size=BATCH_ROWS, maintain_order=True, engine="streaming"
            ):
                for row_value in batch.iter_rows(named=True):
                    row = cast(Mapping[str, object], row_value)
                    event, exact, near, source, destination = _identities(
                        row,
                        input_rows,
                        lineage_key,
                        lineage_id,
                        observation_id,
                        capture_end,
                    )
                    available = cast(int, row[TIMELINE_FIELD])
                    split_row = splits.SplitRow(
                        event_id=event,
                        partition_time_ms=available,
                        source_principal=source,
                        destination_principal=destination,
                        source_unit_id=source_unit_id,
                        capture_lineage_id=lineage_id,
                        exact_group=exact,
                        near_group=near,
                    )
                    partition, reasons = splits.partition_row(split_row, spec)
                    reason = row["end_reason"]
                    if reason in {"idle", "active"}:
                        timeout_rows += 1
                        capture_end_timeout_rows += int(available == capture_end)
                    writer.writerow(
                        (
                            event,
                            available,
                            partition or "dropped",
                            "true" if partition == "train" else "false",
                            source_unit_id,
                            lineage_id,
                            source,
                            destination,
                            exact,
                            near,
                            ";".join(reasons),
                        )
                    )
                    input_rows += 1
        temporary_sidecar = mawi.temporary_file(sidecar.parent, ".parquet")
        pl.scan_csv(temporary_csv, schema=SIDECAR_SCHEMA, null_values="").sink_parquet(
            temporary_sidecar, compression="zstd", maintain_order=True
        )
        if list(pl.read_parquet_schema(temporary_sidecar)) != list(SIDECAR_COLUMNS):
            raise ValueError("split sidecar schema does not match its contract")
        metrics = _partition_metrics(temporary_sidecar)
        if metrics["input"] != input_rows:
            raise ValueError("split sidecar row count does not match the admitted source")
        if timeout_rows != capture_end_timeout_rows:
            raise ValueError("timeout-ended flows do not use the conservative capture-end bound")
        description = _schema_description(temporary_sidecar)
        overlaps = _mapping(metrics["overlaps"], "overlap metrics")
        output: JsonObject = {
            "sidecar": sidecar.name,
            "bytes": temporary_sidecar.stat().st_size,
            "sha256": mawi.digest(temporary_sidecar),
            "schema": list(SIDECAR_COLUMNS),
            "schema_sha256": _schema_sha256(description),
        }
        payload: JsonObject = {
            "receipt_version": 1,
            "status": "split_passed",
            "d0_status": "blocked",
            "inputs": {
                "config": {"name": config.name, "sha256": mawi.digest(config)},
                "source_receipt": {
                    "name": source_receipt.name,
                    "sha256": mawi.digest(source_receipt),
                },
                "source_parquet": {
                    "name": source_parquet.name,
                    "bytes": source_parquet.stat().st_size,
                    "sha256": mawi.digest(source_parquet),
                },
                "source_materializer_sha256": _mapping(
                    admitted.get("tooling"), "source receipt tooling"
                ).get("mawi_sha256"),
                "config_schema_version": values.get("schema_version"),
            },
            "source": {
                "dataset_id": dataset_id,
                "source_unit_id": source_unit_id,
                "capture_lineage_id": lineage_id,
                "observation_domain_id": observation_id,
                "tier": "Q1",
                "labels_present": False,
            },
            "spec": {
                "track": spec.track,
                "timeline_field": TIMELINE_FIELD,
                "capture_start_ms": capture_start,
                "capture_end_ms": capture_end,
                "train_end_ms": spec.train_end_ms,
                "validation_end_ms": spec.validation_end_ms,
                "purge_ms": spec.purge_ms,
            },
            "contracts": contract,
            "counts": {
                "input": metrics["input"],
                "retained": metrics["retained"],
                "dropped": metrics["dropped"],
                "partitions": metrics["partitions"],
                "drop_reasons": metrics["drop_reasons"],
                "timeout_rows": timeout_rows,
                "capture_end_timeout_rows": capture_end_timeout_rows,
            },
            "availability_ties": metrics["availability_ties"],
            "validation": {
                "source_binding": "passed",
                "zero_event_duplicates": True,
                "zero_exact_overlap": overlaps.get("exact") == 0,
                "zero_strict_near_overlap": overlaps.get("strict_near") == 0,
                "semantic_context_overlap": "not_run",
                "fit_scope": "not_run",
                "causal_state": "not_run",
                "context_input": "retained sidecar rows only; pre-drop contexts are forbidden",
            },
            "qualification": {
                "permitted_use": "local Q1 unlabeled SSL split plumbing only",
                "prohibited_claims": [
                    "benign or clean traffic",
                    "chronological generalization",
                    "IDS accuracy",
                    "label calibration",
                    "independent network generalization",
                ],
                "reason": (
                    "timeout-ended availability is conservatively tied at capture end; "
                    "fit and causal replay gates are not run"
                ),
            },
            "output": output,
            "tooling": {
                "d0_dependency_sha256": mawi.digest(Path(d0.__file__)),
                "mawi_dependency_sha256": mawi.digest(Path(mawi.__file__)),
                "mawi_split_sha256": mawi.digest(Path(__file__)),
                "splits_sha256": mawi.digest(Path(splits.__file__)),
                "polars_version": pl.__version__,
            },
        }
        payload["sha256"] = _sha256(payload)
        receipt_payload = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        mawi.publish_bundle(temporary_sidecar, receipt_payload, sidecar, receipt)
        temporary_sidecar = None
        return payload
    finally:
        temporary_csv.unlink(missing_ok=True)
        if temporary_sidecar is not None:
            temporary_sidecar.unlink(missing_ok=True)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config", required=True, type=click.Path(path_type=Path, dir_okay=False))
@click.option("--source-receipt", required=True, type=click.Path(path_type=Path, dir_okay=False))
@click.option("--source-parquet", required=True, type=click.Path(path_type=Path, dir_okay=False))
@click.option("--secret-file", required=True, type=click.Path(path_type=Path, dir_okay=False))
@click.option("--train-end-ms", required=True, type=click.IntRange(min=0))
@click.option("--validation-end-ms", required=True, type=click.IntRange(min=0))
@click.option("--purge-ms", required=True, type=click.IntRange(min=0))
@click.option("--sidecar", required=True, type=click.Path(path_type=Path, dir_okay=False))
@click.option("--receipt", required=True, type=click.Path(path_type=Path, dir_okay=False))
def main(
    config: Path,
    source_receipt: Path,
    source_parquet: Path,
    secret_file: Path,
    train_end_ms: int,
    validation_end_ms: int,
    purge_ms: int,
    sidecar: Path,
    receipt: Path,
) -> None:
    """Freeze one local Q1 chronological split; D0 remains blocked."""
    try:
        result = build(
            config,
            source_receipt,
            source_parquet,
            secret_file,
            splits.SplitSpec(
                "chronological",
                train_end_ms=train_end_ms,
                validation_end_ms=validation_end_ms,
                purge_ms=purge_ms,
            ),
            sidecar,
            receipt,
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    counts = _mapping(result["counts"], "counts")
    click.echo(
        " ".join(
            (
                "status=split_passed",
                "d0_status=blocked",
                f"retained={counts['retained']}",
                f"receipt={receipt}",
            )
        )
    )


if __name__ == "__main__":
    main()
