"""Acceptance tests for the sealed, chronological MAWI split materializer."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import polars as pl

from tools.scripts import mawi, mawi_split, splits

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "tools/configs/mawi.json"


class MawiSplitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.parquet"
        self.config = self.root / "mawi.json"
        self.source_receipt = self.root / "source.receipt.json"
        self.secret = self.root / "lineage.key"
        self.sidecar = self.root / "split.parquet"
        self.receipt = self.root / "split.receipt.json"
        self.secret.write_bytes(bytes(range(32)))
        self.secret.chmod(0o600)
        self._write_source()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_source(self) -> None:
        rows = [
            self._row("train", 10, 10, "192.0.2.1", "2001:db8::1", None),
            self._row("validation", 20, 20, "192.0.2.1", "2001:db8::1", None),
            self._row("test", 30, 30, "2001:db8::2", "192.0.2.2", None),
            # Availability, not end-time, owns chronological partitioning.
            self._row("timeout", 11, 100, "192.0.2.1", "2001:db8::1", "idle"),
        ]
        pl.DataFrame(rows, schema=mawi.SCHEMA).write_parquet(self.source)
        self._refresh_receipt()

    def _refresh_receipt(self) -> None:
        config = json.loads(CONFIG.read_text())
        config["expected_pcap"]["capture_start_epoch_ns"] = 0
        config["expected_pcap"]["capture_start_ms"] = 0
        config["expected_pcap"]["capture_end_epoch_ns"] = 100_000_000
        config["expected_pcap"]["capture_end_ms"] = 100
        rows = pl.read_parquet(self.source)
        for key in config["expected_flows"]:
            config["expected_flows"][key] = 0
        config["expected_flows"]["rows"] = rows.height
        config["expected_flows"]["accepted"] = rows.height
        config["expected_flows"]["accepted_normal"] = rows["end_reason"].null_count()
        config["expected_flows"]["accepted_idle"] = rows.filter(
            pl.col("end_reason") == "idle"
        ).height
        config["expected_flows"]["accepted_active"] = rows.filter(
            pl.col("end_reason") == "active"
        ).height
        self.config.write_text(json.dumps(config, sort_keys=True))
        schema = pl.read_parquet_schema(self.source)
        self.source_receipt.write_text(
            json.dumps(
                {
                    "receipt_version": 1,
                    "status": "reconciled",
                    "source": {
                        "pcap": config["source"]["pcap_file"],
                        "pcap_sha256": config["source"]["pcap_sha256"],
                        "yaf_csv": config["meter"]["yaf_csv_file"],
                        "yaf_sha256": config["meter"]["yaf_csv_sha256"],
                    },
                    "admission_metadata": {
                        "dataset_id": config["id"],
                        "config_sha256": mawi.digest(self.config),
                        "archive_verified": True,
                        "conversion_artifacts_verified": True,
                        "source": config["source"],
                        "meter": config["meter"],
                        "feature_contract": config["feature_contract"],
                    },
                    "pcap_accounting": config["expected_pcap"],
                    "output": {
                        "parquet": self.source.name,
                        "parquet_sha256": mawi.digest(self.source),
                        "parquet_bytes": self.source.stat().st_size,
                        "schema": list(mawi.OUTPUT_COLUMNS),
                        "schema_sha256": hashlib.sha256(
                            json.dumps(
                                [(name, str(dtype)) for name, dtype in schema.items()],
                                separators=(",", ":"),
                            ).encode()
                        ).hexdigest(),
                        "labels_present": False,
                        "availability": "normal=end_ms; idle=capture_end_ms; active=capture_end_ms",
                        "addresses": "routing_only",
                        "fragment_policy": (
                            "YAF --no-frag required; fragments counted and excluded upstream"
                        ),
                    },
                    "flow_accounting": config["expected_flows"],
                    "yaf_reconciliation": {"status": "reconciled"},
                    "tooling": {
                        "mawi_sha256": mawi.digest(ROOT / "tools/scripts/mawi.py"),
                        "polars_version": pl.__version__,
                    },
                    "qualification": {
                        "tier": "Q1",
                        "permitted_use": "unlabeled general SSL only",
                        "prohibited_claims": [
                            "benign or clean traffic",
                            "IDS accuracy",
                            "label calibration",
                            "independent network generalization",
                        ],
                    },
                },
                sort_keys=True,
            )
        )

    @staticmethod
    def _row(
        _: str,
        end: int,
        available: int,
        source: str,
        destination: str,
        reason: str | None,
    ) -> dict[str, object]:
        return {
            "flow_start_ms": end - 1,
            "flow_end_ms": end,
            "flow_available_ms": available,
            "flow_duration_ms": 1,
            "source_ip": source,
            "destination_ip": destination,
            "source_port": 443,
            "destination_port": 50000,
            "protocol": 6,
            "forward_packets": 1,
            "reverse_packets": 0,
            "forward_bytes": 60,
            "reverse_bytes": 0,
            "initial_tcp_flags": "S",
            "reverse_initial_tcp_flags": "",
            "union_tcp_flags": "S",
            "reverse_union_tcp_flags": "",
            "end_reason": reason,
        }

    @property
    def spec(self) -> splits.SplitSpec:
        return splits.SplitSpec(
            "chronological",
            train_end_ms=15,
            validation_end_ms=25,
        )

    def build(self) -> dict[str, object]:
        return mawi_split.build(
            self.config,
            self.source_receipt,
            self.source,
            self.secret,
            self.spec,
            self.sidecar,
            self.receipt,
        )

    def test_builds_causal_opaque_partitions_and_overlap_evidence(self) -> None:
        result = self.build()
        assignments = pl.read_parquet(self.sidecar).to_dicts()
        self.assertEqual(
            {entry["partition"] for entry in assignments},
            {"train", "validation", "test"},
        )
        self.assertEqual(
            next(entry["partition"] for entry in assignments if entry["partition_time_ms"] == 100),
            "test",
        )
        self.assertEqual(result["status"], "split_passed")
        self.assertEqual(result["d0_status"], "blocked")
        self.assertEqual(result["spec"]["timeline_field"], "flow_available_ms")
        self.assertEqual(result["counts"]["partitions"], {"test": 2, "train": 1, "validation": 1})
        self.assertEqual(result["counts"]["drop_reasons"], {})
        self.assertEqual(result["counts"]["input"], 4)
        self.assertEqual(result["counts"]["retained"] + result["counts"]["dropped"], 4)
        self.assertEqual(result["availability_ties"], {})
        self.assertEqual(result["validation"]["zero_event_duplicates"], True)
        self.assertEqual(result["validation"]["zero_exact_overlap"], True)
        self.assertEqual(result["validation"]["zero_strict_near_overlap"], True)
        self.assertEqual(result["validation"]["source_binding"], "passed")
        self.assertEqual(result["validation"]["semantic_context_overlap"], "not_run")
        self.assertEqual(result["validation"]["fit_scope"], "not_run")
        self.assertEqual(result["validation"]["causal_state"], "not_run")
        self.assertEqual(result["contracts"]["key_scope"], result["source"]["capture_lineage_id"])
        self.assertEqual(
            result["tooling"]["d0_dependency_sha256"],
            mawi.digest(ROOT / "tools/scripts/d0.py"),
        )
        self.assertEqual(
            result["tooling"]["mawi_dependency_sha256"],
            mawi.digest(ROOT / "tools/scripts/mawi.py"),
        )
        self.assertEqual(
            result["qualification"]["permitted_use"],
            "local Q1 unlabeled SSL split plumbing only",
        )
        self.assertIn("IDS accuracy", result["qualification"]["prohibited_claims"])
        self.assertTrue(self.sidecar.is_file())
        self.assertTrue(self.receipt.is_file())
        published = self.sidecar.read_bytes() + self.receipt.read_bytes()
        self.assertFalse(
            any(
                value in published
                for value in (b"192.0.2.1", b"2001:db8::1", self.secret.read_bytes())
            )
        )

    def test_endpoint_keys_are_canonical_and_refuse_invalid_addresses(self) -> None:
        key = bytes(range(32))
        self.assertEqual(
            mawi_split._endpoint_key("2001:0DB8:0:0:0:0:0:1", key, "source"),
            mawi_split._endpoint_key("2001:db8::1", key, "source"),
        )
        self.assertEqual(
            mawi_split._endpoint_key("192.0.2.1", key, "source"),
            mawi_split._endpoint_key("192.0.2.1", key, "destination"),
        )
        with self.assertRaisesRegex(ValueError, "IP"):
            mawi_split._endpoint_key("not-an-address", key, "source")

    def test_binds_config_source_receipt_and_parquet(self) -> None:
        self.source_receipt.write_text('{"status":"blocked"}')
        with self.assertRaisesRegex(ValueError, "reconciled"):
            self.build()
        self._write_source()
        pl.read_parquet(self.source).with_columns(
            pl.lit(2).alias("flow_duration_ms")
        ).write_parquet(self.source)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.build()
        self._write_source()
        receipt = json.loads(self.source_receipt.read_text())
        receipt["admission_metadata"]["config_sha256"] = "0" * 64
        self.source_receipt.write_text(json.dumps(receipt, sort_keys=True))
        with self.assertRaisesRegex(ValueError, "config"):
            self.build()
        self._write_source()
        receipt = json.loads(self.source_receipt.read_text())
        receipt["yaf_reconciliation"]["status"] = "blocked"
        self.source_receipt.write_text(json.dumps(receipt, sort_keys=True))
        with self.assertRaisesRegex(ValueError, "configured admitted source"):
            self.build()
        self._write_source()
        receipt = json.loads(self.source_receipt.read_text())
        receipt["qualification"]["prohibited_claims"] = [[]]
        self.source_receipt.write_text(json.dumps(receipt, sort_keys=True))
        with self.assertRaisesRegex(ValueError, "prohibitions"):
            self.build()

    def test_refuses_unsafe_secret_and_output_paths(self) -> None:
        self.secret.write_bytes(b"short")
        with self.assertRaisesRegex(ValueError, "32"):
            self.build()
        self.secret.write_bytes(bytes(range(32)))
        self.secret.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "owner"):
            self.build()
        self.secret.chmod(0o600)
        target = self.root / "target"
        target.write_bytes(b"keep")
        self.secret.unlink()
        self.secret.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symbolic"):
            self.build()
        self.secret.unlink()
        self.secret.write_bytes(bytes(range(32)))
        self.secret.chmod(0o600)
        self.sidecar.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symbolic"):
            self.build()
        self.sidecar.unlink()
        self.receipt.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symbolic"):
            self.build()

    def test_is_deterministic_immutable_and_rejects_bad_cutoffs(self) -> None:
        first = self.build()
        parquet = self.sidecar.read_bytes()
        receipt = self.receipt.read_bytes()
        self.assertEqual(first, self.build())
        self.assertEqual(parquet, self.sidecar.read_bytes())
        self.assertEqual(receipt, self.receipt.read_bytes())
        self.receipt.write_text("changed")
        with self.assertRaisesRegex(ValueError, "refusing overwrite"):
            self.build()
        self.receipt.unlink()
        self.sidecar.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "refusing overwrite"):
            self.build()
        for spec in (
            splits.SplitSpec(
                "chronological",
                train_end_ms=1,
                validation_end_ms=2,
            ),
            splits.SplitSpec(
                "chronological",
                train_end_ms=101,
                validation_end_ms=102,
            ),
        ):
            with self.assertRaises(ValueError):
                mawi_split.build(
                    self.config,
                    self.source_receipt,
                    self.source,
                    self.secret,
                    spec,
                    self.root / ("bad-" + hashlib.sha256(repr(spec).encode()).hexdigest()),
                    self.root / "bad.receipt",
                )

    def test_strict_and_near_groups_differ_only_by_end_reason(self) -> None:
        rows = [
            self._row("train", 10, 10, "192.0.2.1", "2001:db8::1", None),
            self._row("validation", 20, 20, "192.0.2.1", "2001:db8::1", None),
            self._row("test", 30, 30, "2001:db8::2", "192.0.2.2", None),
            self._row("normal", 100, 100, "192.0.2.1", "2001:db8::1", None),
            self._row("idle", 100, 100, "192.0.2.1", "2001:db8::1", "idle"),
        ]
        pl.DataFrame(rows, schema=mawi.SCHEMA).write_parquet(self.source)
        self._refresh_receipt()
        _ = self.build()
        entries = [
            entry
            for entry in pl.read_parquet(self.sidecar).to_dicts()
            if entry["partition_time_ms"] == 100
        ]
        self.assertEqual(len(entries), 2)
        self.assertNotEqual(entries[0]["exact_group"], entries[1]["exact_group"])
        self.assertEqual(entries[0]["near_group"], entries[1]["near_group"])


if __name__ == "__main__":
    unittest.main()
