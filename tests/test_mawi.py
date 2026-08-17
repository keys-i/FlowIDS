from __future__ import annotations

import gzip
import json
import struct
import tempfile
import unittest
from pathlib import Path

import polars as pl
from click.testing import CliRunner

from tools.scripts import mawi


def ethernet(payload: bytes, ether_type: int = 0x0800) -> bytes:
    return b"\x00" * 12 + ether_type.to_bytes(2, "big") + payload


def ipv4(total: int = 1500) -> bytes:
    return (
        b"\x45\x00" + total.to_bytes(2, "big") + b"\x00\x00\x00\x00\x40\x06\x00\x00" + b"\x00" * 8
    )


def ipv6(payload_length: int = 1460) -> bytes:
    return b"\x60\x00\x00\x00" + payload_length.to_bytes(2, "big") + b"\x3b\x40" + b"\x00" * 32


def pcap(records: list[tuple[int, bytes, int | None]], *, tail: bytes = b"") -> bytes:
    content = bytearray(b"\xd4\xc3\xb2\xa1" + struct.pack("<HHIIII", 2, 4, 0, 0, 65535, 1))
    for seconds, packet, original in records:
        original = len(packet) if original is None else original
        content.extend(struct.pack("<IIII", seconds, 0, len(packet), original))
        content.extend(packet)
    return bytes(content) + tail


def flow(start: int, end: int, reason: str = "") -> str:
    return ",".join(
        (
            str(start),
            str(end),
            "192.0.2.1",
            "2001:db8::1",
            "443",
            "51234",
            "6",
            "1",
            "1",
            "60",
            "60",
            "S",
            "A",
            "SA",
            "AP",
            reason,
        )
    )


class MawiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.capture = self.root / "capture.pcap"
        self.flows = self.root / "flows.csv"
        self.parquet = self.root / "flows.parquet"
        self.receipt = self.root / "receipt.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def materialize(self) -> dict[str, object]:
        return mawi.materialize(
            self.capture,
            self.flows,
            self.parquet,
            self.receipt,
            pcap_sha256=mawi.digest(self.capture),
            yaf_sha256=mawi.digest(self.flows),
            expected_pcap=mawi.pcap_accounting(self.capture),
            expected_flows={
                "rows": 3,
                "accepted": 2,
                "accepted_normal": 1,
                "accepted_idle": 1,
                "accepted_active": 0,
                "rejected_eof": 1,
                "rejected_rsrc": 0,
                "exported_packet_sum": 6,
                "exported_l3_octet_sum": 360,
                "accepted_packet_sum": 4,
                "accepted_l3_octet_sum": 240,
                "protocol_6_packets": 6,
                "protocol_6_l3_octets": 360,
            },
        )

    def test_actual_csv_order_and_snaplen_l3_accounting(self) -> None:
        self.capture.write_bytes(
            pcap(
                [
                    (1641013200, ethernet(ipv4()), 1514),
                    (1641013320, ethernet(ipv6(), 0x86DD), 1514),
                    (1641013380, ethernet(b"", 0x0806), None),
                ]
            )
        )
        self.flows.write_text(
            "\n".join(
                (
                    flow(1641013200094, 1641013200094),
                    flow(1641013260000, 1641013260000, "idle"),
                    flow(1641013300000, 1641013300000, "eof"),
                )
            )
            + "\n"
        )

        result = self.materialize()
        counts = mawi.pcap_accounting(self.capture)
        output = pl.read_parquet(self.parquet)

        self.assertEqual(counts["snaplen_truncated"], 2)
        self.assertEqual((counts["ip_packets"], counts["l3_octets"]), (2, 3000))
        self.assertEqual(output.columns, list(mawi.OUTPUT_COLUMNS))
        self.assertEqual(output["flow_duration_ms"].to_list(), [0, 0])
        self.assertEqual(output["flow_available_ms"].to_list(), [1641013200094, 1641013380000])
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(json.loads(self.receipt.read_text())["output"]["labels_present"])
        contract = json.loads((Path(__file__).parents[1] / "tools/configs/mawi.json").read_text())[
            "feature_contract"
        ]
        view = mawi.model_view(output.row(0, named=True), contract)
        self.assertEqual(
            set(view["numeric"]),
            {
                "flow_duration_ms",
                "forward_packets",
                "reverse_packets",
                "forward_bytes",
                "reverse_bytes",
            },
        )
        self.assertFalse(set(contract["never_model_inputs"]) & set(view["numeric"]))
        self.assertFalse(set(contract["never_model_inputs"]) & set(view["categorical"]))

    def test_timeout_availability_never_assumes_the_unexported_trigger_time(self) -> None:
        idle, _ = mawi._canonical(flow(1_000, 10_000, "idle").split(","), 0, 1_000_000)
        active, _ = mawi._canonical(flow(1_000, 301_000, "active").split(","), 0, 1_000_000)

        self.assertEqual(idle[2], 1_000_000)
        self.assertEqual(active[2], 1_000_000)
        with self.assertRaisesRegex(ValueError, "capture interval"):
            mawi._canonical(flow(999, 1_000).split(","), 1_000, 1_000_000)

    def test_accounts_malformed_unsupported_header_truncation_and_nonmonotonic_time(self) -> None:
        malformed = ethernet(b"\x55" + b"\x00" * 19)
        self.capture.write_bytes(
            pcap(
                [(2, ethernet(b"", 0x0806), None), (1, malformed, None), (3, b"\x00", None)],
                tail=b"\x00",
            )
        )
        counts = mawi.pcap_accounting(self.capture)
        self.assertEqual(counts["unsupported"], 1)
        self.assertEqual(counts["malformed"], 1)
        self.assertEqual(counts["header_truncated"], 1)
        self.assertEqual(counts["record_truncated"], 1)
        self.assertEqual(counts["nonmonotonic_timestamps"], 1)

    def test_rejects_bad_schema_end_reason_and_accounting_without_outputs(self) -> None:
        self.capture.write_bytes(pcap([(1641013400, ethernet(ipv4()), 1514)]))
        self.flows.write_text(flow(1641013200000, 1641013200000, "mystery") + "\n")
        with self.assertRaisesRegex(ValueError, "unsupported YAF end_reason"):
            self.materialize()
        self.assertFalse(self.parquet.exists())
        self.flows.write_text("1,2\n")
        with self.assertRaisesRegex(ValueError, "expected 16"):
            self.materialize()
        self.assertFalse(self.parquet.exists())

    def test_meter_stats_require_packet_and_l3_reconciliation(self) -> None:
        self.capture.write_bytes(pcap([(1, ethernet(ipv4(60)), 74), (2, ethernet(ipv4(60)), 74)]))
        rows = []
        for start in (1000, 1500):
            values = flow(start, start).split(",")
            values[7:11] = ["1", "0", "60", "0"]
            rows.append(",".join(values))
        self.flows.write_text("\n".join(rows) + "\n")
        expected = mawi.pcap_accounting(self.capture)
        flows = {
            "rows": 2,
            "accepted": 2,
            "accepted_normal": 2,
            "accepted_idle": 0,
            "accepted_active": 0,
            "rejected_eof": 0,
            "rejected_rsrc": 0,
            "exported_packet_sum": 2,
            "exported_l3_octet_sum": 120,
            "accepted_packet_sum": 2,
            "accepted_l3_octet_sum": 120,
            "protocol_6_packets": 2,
            "protocol_6_l3_octets": 120,
        }
        stats = {
            "exported_flows": 2,
            "packet_total": 2,
            "dropped_packets": 0,
            "ignored_packets": 0,
            "expired_fragments": 0,
            "assembled_fragments": 0,
            "flow_flushes": 0,
            "peak_flows": 1,
            "flow_l3_octets": 120,
            "ignored_fragments": 0,
            "ignored_extension_headers": 0,
            "ignored_transport_headers": 0,
            "ignored_non_ip": 0,
            "ignored_other": 0,
        }
        result = mawi.materialize(
            self.capture,
            self.flows,
            self.parquet,
            self.receipt,
            pcap_sha256=mawi.digest(self.capture),
            yaf_sha256=mawi.digest(self.flows),
            expected_pcap=expected,
            expected_flows=flows,
            yaf_stats=stats,
            metadata={
                "dataset_id": "fixture",
                "config_sha256": "0" * 64,
                "source": {"id": "fixture"},
                "meter": {"id": "fixture"},
                "feature_contract": json.loads(
                    (Path(__file__).parents[1] / "tools/configs/mawi.json").read_text()
                )["feature_contract"],
                "archive_verified": True,
                "conversion_artifacts_verified": True,
            },
        )
        self.assertEqual(result["status"], "reconciled")
        stats["packet_total"] = 1
        with self.assertRaisesRegex(ValueError, "packet_total"):
            mawi.materialize(
                self.capture,
                self.flows,
                self.parquet,
                self.receipt,
                pcap_sha256=mawi.digest(self.capture),
                yaf_sha256=mawi.digest(self.flows),
                expected_pcap=expected,
                expected_flows=flows,
                yaf_stats=stats,
            )
        stats["packet_total"] = 2
        self.flows.write_text("\n".join(row.replace(",6,", ",17,") for row in rows) + "\n")
        mismatched_flows = dict(flows)
        del mismatched_flows["protocol_6_packets"]
        del mismatched_flows["protocol_6_l3_octets"]
        mismatched_flows["protocol_17_packets"] = 2
        mismatched_flows["protocol_17_l3_octets"] = 120
        with self.assertRaisesRegex(ValueError, "protocol 17"):
            mawi.materialize(
                self.capture,
                self.flows,
                self.parquet,
                self.receipt,
                pcap_sha256=mawi.digest(self.capture),
                yaf_sha256=mawi.digest(self.flows),
                expected_pcap=expected,
                expected_flows=mismatched_flows,
                yaf_stats=stats,
            )

    def test_replay_is_idempotent_and_existing_artifacts_are_immutable(self) -> None:
        self.capture.write_bytes(pcap([(2, ethernet(ipv4(60)), 74)]))
        values = flow(2_000, 2_000).split(",")
        values[7:11] = ["1", "0", "60", "0"]
        self.flows.write_text(",".join(values) + "\n")
        expected_pcap = mawi.pcap_accounting(self.capture)
        expected_flows = {
            "rows": 1,
            "accepted": 1,
            "accepted_normal": 1,
            "accepted_idle": 0,
            "accepted_active": 0,
            "rejected_eof": 0,
            "rejected_rsrc": 0,
            "exported_packet_sum": 1,
            "exported_l3_octet_sum": 60,
            "accepted_packet_sum": 1,
            "accepted_l3_octet_sum": 60,
            "protocol_6_packets": 1,
            "protocol_6_l3_octets": 60,
        }

        first = mawi.materialize(
            self.capture,
            self.flows,
            self.parquet,
            self.receipt,
            pcap_sha256=mawi.digest(self.capture),
            yaf_sha256=mawi.digest(self.flows),
            expected_pcap=expected_pcap,
            expected_flows=expected_flows,
        )
        parquet_bytes = self.parquet.read_bytes()
        receipt_bytes = self.receipt.read_bytes()
        second = mawi.materialize(
            self.capture,
            self.flows,
            self.parquet,
            self.receipt,
            pcap_sha256=mawi.digest(self.capture),
            yaf_sha256=mawi.digest(self.flows),
            expected_pcap=expected_pcap,
            expected_flows=expected_flows,
        )

        self.assertEqual(first, second)
        self.assertEqual(self.parquet.read_bytes(), parquet_bytes)
        self.assertEqual(self.receipt.read_bytes(), receipt_bytes)
        self.receipt.write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "refusing overwrite"):
            mawi.materialize(
                self.capture,
                self.flows,
                self.parquet,
                self.receipt,
                pcap_sha256=mawi.digest(self.capture),
                yaf_sha256=mawi.digest(self.flows),
                expected_pcap=expected_pcap,
                expected_flows=expected_flows,
            )
        self.assertEqual(self.parquet.read_bytes(), parquet_bytes)
        self.receipt.unlink()
        target = self.root / "symlink-target"
        target.write_text("do not replace\n")
        self.receipt.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symbolic links"):
            mawi.materialize(
                self.capture,
                self.flows,
                self.parquet,
                self.receipt,
                pcap_sha256=mawi.digest(self.capture),
                yaf_sha256=mawi.digest(self.flows),
                expected_pcap=expected_pcap,
                expected_flows=expected_flows,
            )
        self.assertEqual(target.read_text(), "do not replace\n")

    def test_cli_verifies_one_source_config_and_conversion_chain(self) -> None:
        archive = self.root / "source.gz"
        binary = self.root / "meter"
        mediator = self.root / "mediator"
        ipfix = self.root / "flows.ipfix"
        config_path = self.root / "mawi.json"
        binary.write_bytes(b"yaf")
        mediator.write_bytes(b"mediator")
        ipfix.write_bytes(b"ipfix")
        self.capture.write_bytes(pcap([(2, ethernet(ipv4(60)), 74)]))
        archive.write_bytes(gzip.compress(self.capture.read_bytes(), mtime=0))
        values = flow(2000, 2000).split(",")
        values[7:11] = ["1", "0", "60", "0"]
        self.flows.write_text(",".join(values) + "\n")

        config = json.loads((Path(__file__).parents[1] / "tools/configs/mawi.json").read_text())
        source = config["source"]
        source.update(
            {
                "compressed_file": archive.name,
                "compressed_bytes": archive.stat().st_size,
                "compressed_sha256": mawi.digest(archive),
                "pcap_file": self.capture.name,
                "pcap_bytes": self.capture.stat().st_size,
                "pcap_sha256": mawi.digest(self.capture),
            }
        )
        meter = config["meter"]
        meter.update(
            {
                "yaf_binary_sha256": mawi.digest(binary),
                "super_mediator_binary_sha256": mawi.digest(mediator),
                "yaf_ipfix_file": ipfix.name,
                "yaf_ipfix_bytes": ipfix.stat().st_size,
                "yaf_ipfix_sha256": mawi.digest(ipfix),
                "yaf_csv_file": self.flows.name,
                "yaf_csv_sha256": mawi.digest(self.flows),
            }
        )
        config["expected_pcap"] = mawi.pcap_accounting(self.capture)
        config["expected_flows"] = {
            "rows": 1,
            "accepted": 1,
            "accepted_normal": 1,
            "accepted_idle": 0,
            "accepted_active": 0,
            "rejected_eof": 0,
            "rejected_rsrc": 0,
            "exported_packet_sum": 1,
            "exported_l3_octet_sum": 60,
            "accepted_packet_sum": 1,
            "accepted_l3_octet_sum": 60,
            "protocol_6_packets": 1,
            "protocol_6_l3_octets": 60,
        }
        config["yaf_stats"] = {
            "exported_flows": 1,
            "packet_total": 1,
            "dropped_packets": 0,
            "ignored_packets": 0,
            "expired_fragments": 0,
            "assembled_fragments": 0,
            "flow_flushes": 0,
            "peak_flows": 1,
            "flow_l3_octets": 60,
            "ignored_fragments": 0,
            "ignored_extension_headers": 0,
            "ignored_transport_headers": 0,
            "ignored_non_ip": 0,
            "ignored_other": 0,
        }
        config_path.write_text(json.dumps(config))

        result = CliRunner().invoke(
            mawi.main,
            [
                "--config",
                str(config_path),
                "--archive",
                str(archive),
                "--pcap",
                str(self.capture),
                "--yaf-bin",
                str(binary),
                "--super-mediator-bin",
                str(mediator),
                "--yaf-ipfix",
                str(ipfix),
                "--yaf-csv",
                str(self.flows),
                "--parquet",
                str(self.parquet),
                "--receipt",
                str(self.receipt),
            ],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("status=reconciled", result.output)
        self.assertTrue(
            json.loads(self.receipt.read_text())["admission_metadata"]["archive_verified"]
        )


if __name__ == "__main__":
    unittest.main()
