import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import polars as pl
from click.testing import CliRunner

from tools.scripts import causal, convert, d0, preprocess


def event(event_id: str, completion_ms: int, source: str, destination: str) -> causal.FlowEvent:
    return causal.FlowEvent(event_id, completion_ms, source, destination)


class CausalHistoryTest(unittest.TestCase):
    def test_equal_time_events_are_not_mutual_history(self) -> None:
        events = [
            event("a", 10, "left", "x"),
            event("b", 10, "left", "y"),
            event("c", 11, "left", "z"),
        ]

        contexts = causal.replay(events, 100, "test")

        self.assertEqual(
            [[item.event_id for item in context] for context in contexts],
            [["a"], ["b"], ["a", "b", "c"]],
        )

    def test_replay_matches_streaming_and_reset_clears_state(self) -> None:
        events = [event("a", 1, "s", "x"), event("b", 2, "s", "y"), event("c", 3, "y", "z")]
        offline = causal.replay(events, 10, "one")
        builder = causal.CausalEgoHistory(10, "one")
        streaming = tuple(builder.add(item) for item in events)
        builder.flush()

        self.assertEqual(offline, streaming)
        with self.assertRaisesRegex(ValueError, "closed"):
            builder.add(event("d", 4, "s", "z"))
        builder.reset("two")
        self.assertEqual(builder.add(event("d", 4, "s", "z")), (event("d", 4, "s", "z"),))

    def test_out_of_order_and_duplicate_ids_are_rejected(self) -> None:
        builder = causal.CausalEgoHistory(10)
        _ = builder.add(event("a", 2, "s", "d"))
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            builder.add(event("b", 1, "s", "d"))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            builder.add(event("a", 2, "s", "d"))

    def test_endpoint_limits_union_deduplication_and_total_cap(self) -> None:
        histories = [
            event(f"s-{index:03}", index, "source", f"sx-{index}") for index in range(1, 131)
        ] + [
            event(f"d-{index:03}", index + 130, f"dx-{index}", "destination")
            for index in range(1, 131)
        ]
        target = event("target", 300, "source", "destination")

        context = causal.replay(histories + [target], 1_000)[-1]

        self.assertEqual(len(context), 256)
        self.assertEqual(context[-1], target)
        self.assertNotIn("s-001", {item.event_id for item in context})
        self.assertIn("s-130", {item.event_id for item in context})

    def test_relation_encoding_uses_all_four_predicates(self) -> None:
        current = event("current", 2, "a", "b")
        same = event("same", 1, "a", "b")
        reverse = event("reverse", 1, "b", "a")

        self.assertEqual(causal.endpoint_relation(current, same), 3)
        self.assertEqual(causal.endpoint_relation(current, reverse), 12)
        self.assertEqual(causal.endpoint_relation_matrix((same, current))[1][0], 3)

    def test_equal_time_truncation_is_independent_of_input_order(self) -> None:
        simultaneous = [
            event(f"e{index:03}", 10, "shared", f"remote-{index}") for index in range(129)
        ]
        target = event("target", 11, "shared", "final")

        forward = causal.replay([*simultaneous, target], 100)
        reverse = causal.replay([*reversed(simultaneous), target], 100)

        self.assertEqual(forward, reverse)
        target_ids = {item.event_id for item in forward[-1]}
        self.assertNotIn("e000", target_ids)
        self.assertIn("e128", target_ids)


class TrainOnlyPreprocessorTest(unittest.TestCase):
    @staticmethod
    def row(value: float | None, category: object, port: object) -> dict[str, object]:
        return {
            "numeric": {"volume": value},
            "categorical": {"protocol": category, "port": port},
        }

    def test_fit_is_order_independent_and_transform_does_not_mutate_state(self) -> None:
        training = [
            self.row(0, 6, 80),
            self.row(9, 17, 50000),
            self.row(None, None, 50000),
            self.row(99, 6, 40000),
        ]
        state = preprocess.fit(training, port_fields=("port",))
        before = preprocess.state_hash(state)

        transformed = preprocess.transform(state, self.row(1e12, 99, 60000))

        self.assertEqual(before, preprocess.state_hash(state))
        self.assertEqual(state, preprocess.fit(reversed(training), port_fields=("port",)))
        self.assertEqual(transformed["categorical"]["port__frequency"], 1)
        self.assertTrue(abs(transformed["numeric"]["volume"]) < 10)

    def test_validation_and_test_values_cannot_change_training_state(self) -> None:
        assignments = [
            {"event_id": "train", "partition": "train"},
            {"event_id": "validation", "partition": "validation"},
            {"event_id": "test", "partition": "test"},
        ]
        rows = {
            "train": self.row(1, 6, 80),
            "validation": self.row(2, 17, 443),
            "test": self.row(3, 1, 53),
        }
        first = preprocess.fit_training_partition(rows, assignments, port_fields=("port",))
        rows["validation"] = self.row(1e30, "new", 65535)
        rows["test"] = self.row(None, None, None)
        second = preprocess.fit_training_partition(rows, assignments, port_fields=("port",))

        self.assertEqual(preprocess.state_hash(first), preprocess.state_hash(second))

    def test_source_role_fits_cross_track_preprocessing(self) -> None:
        source = self.row(1, 6, 80)
        assignments = [
            {"event_id": "source", "partition": "source"},
            {"event_id": "target", "partition": "target"},
            {"event_id": "sealed", "partition": "sealed"},
        ]

        fitted = preprocess.fit_training_partition(
            {"source": source}, assignments, port_fields=("port",)
        )

        self.assertEqual(fitted, preprocess.fit([source], port_fields=("port",)))

    def test_missing_values_and_invalid_training_inputs_are_explicit(self) -> None:
        state = preprocess.fit([self.row(1, 6, 80), self.row(2, 6, None)], port_fields=("port",))
        transformed = preprocess.transform(state, self.row(None, None, None))

        self.assertEqual(transformed["missing"]["volume"], 1)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            preprocess.fit([self.row(-1, 6, 80)], port_fields=("port",))
        with self.assertRaisesRegex(ValueError, "non-model"):
            preprocess.fit(
                [{**self.row(1, 6, 80), "Label": 1}],
                port_fields=("port",),
            )

    def test_nf3_preprocessing_path_never_exposes_forbidden_fields(self) -> None:
        contract = d0.load_feature_contract(Path("tools/configs/nf3.json"))
        raw = {
            field: 0
            for field, role in contract.roles.items()
            if role in {"model_numeric", "model_categorical"}
        }
        raw.update({"L4_SRC_PORT": 443, "L4_DST_PORT": 55000, "Label": 1, "Attack": "x"})
        selected = d0.preprocessing_view(raw, contract)

        state = preprocess.fit([selected], port_fields=contract.port_fields)
        transformed = preprocess.transform(state, selected)

        self.assertTrue(contract.forbidden.isdisjoint(transformed["numeric"]))
        self.assertTrue(contract.forbidden.isdisjoint(transformed["categorical"]))
        self.assertNotIn("55000", json.dumps(transformed))


class ConversionSafetyTest(unittest.TestCase):
    def test_successful_conversion_is_atomic_and_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "flows.csv"
            output = root / "flows.parquet"
            source.write_text("value\n1\n2\n", encoding="utf-8")

            status, written, _, _ = convert.convert(source, output, "zstd", None, 100, False)

            self.assertEqual((status, written), ("converted", output))
            self.assertEqual(pl.read_parquet(output)["value"].to_list(), [1, 2])
            self.assertEqual({path.name for path in root.iterdir()}, {"flows.csv", "flows.parquet"})

    def test_failed_overwrite_preserves_existing_parquet(self) -> None:
        class BrokenScan:
            def sink_parquet(self, path: Path, **_: object) -> None:
                path.write_bytes(b"partial")
                raise RuntimeError("conversion failed")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "flows.csv"
            output = root / "flows.parquet"
            source.write_text("value\n1\n", encoding="utf-8")
            output.write_bytes(b"previous")

            with patch.object(convert.pl, "scan_csv", return_value=BrokenScan()):
                with self.assertRaisesRegex(RuntimeError, "conversion failed"):
                    convert.convert(source, output, "zstd", None, 100, True)

            self.assertEqual(output.read_bytes(), b"previous")
            self.assertEqual({path.name for path in root.iterdir()}, {"flows.csv", "flows.parquet"})


class D0AuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.csv = self.root / "flows.csv"
        self.parquet = self.root / "flows.parquet"
        self.manifest = self.root / "manifest-sha1.txt"
        self.feature_config = self.root / "features.json"
        self.csv.write_text(
            "FLOW_START_MILLISECONDS,FLOW_END_MILLISECONDS,Label\n1,2,0\n3,4,1\n2,3,0\n",
            encoding="utf-8",
        )
        pl.read_csv(self.csv).write_parquet(self.parquet)
        self.manifest.write_text(
            f"{d0.file_digest(self.csv, 'sha1')}  {self.csv.name}\n",
            encoding="utf-8",
        )
        self.feature_config.write_text(
            json.dumps(
                {
                    "datasets": [
                        {
                            "id": "fixture",
                            "official_name": "Fixture",
                            "parquet": "flows.parquet",
                            "bagit_csv_sha1": d0.file_digest(self.csv, "sha1"),
                            "q3_pairing": False,
                            "source_metadata": {
                                "capture_lineage": None,
                                "site": "fixture",
                                "collector": None,
                                "exporter": None,
                                "meter": None,
                                "meter_version": None,
                                "flow_timeout": None,
                                "time_range": None,
                                "timezone": None,
                                "license": None,
                                "label_provenance": None,
                            },
                            "evidence": ["https://example.invalid/fixture"],
                        }
                    ],
                    "schema": {
                        "field_order": [
                            "FLOW_START_MILLISECONDS",
                            "FLOW_END_MILLISECONDS",
                            "Label",
                        ]
                    },
                    "field_roles": {
                        "FLOW_START_MILLISECONDS": "routing",
                        "FLOW_END_MILLISECONDS": "routing",
                        "Label": "target",
                    },
                    "model_input_policy": {"never_model_inputs": ["Label"]},
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_clean_conversion_has_full_integrity_but_remains_q0(self) -> None:
        receipt = d0.audit(self.csv, self.parquet, self.manifest, self.feature_config)

        self.assertEqual(receipt["status"], "integrity_passed")
        self.assertEqual(receipt["d0_status"], "blocked")
        self.assertEqual(receipt["d0_gates"]["conversion_integrity"], "passed")
        self.assertEqual(receipt["d0_gates"]["full_value_parity"], "passed")
        self.assertEqual(receipt["d0_gates"]["split_isolation"], "not_run")
        self.assertEqual(receipt["rows"], {"csv": 3, "parquet": 3, "dropped": 0, "added": 0})
        self.assertEqual(receipt["timestamps"]["end_time_backward_transitions"], 1)
        self.assertEqual(receipt["qualification"]["tier"], "Q0")
        self.assertFalse(receipt["qualification"]["q1_eligible"])
        self.assertIn("capture_lineage", receipt["source"]["missing_required_fields"])
        self.assertEqual(receipt["inputs"]["csv"], "flows.csv")

    def test_q3_configuration_fails_closed_without_pairing_manifest(self) -> None:
        config = json.loads(self.feature_config.read_text(encoding="utf-8"))
        config["datasets"][0]["q3_pairing"] = True
        self.feature_config.write_text(json.dumps(config), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "pairing_manifest"):
            d0.audit(self.csv, self.parquet, self.manifest, self.feature_config)

    def test_source_digest_cannot_inherit_provenance_by_filename(self) -> None:
        config = json.loads(self.feature_config.read_text(encoding="utf-8"))
        config["datasets"][0]["bagit_csv_sha1"] = "0" * 40
        self.feature_config.write_text(json.dumps(config), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "does not match"):
            d0.audit(self.csv, self.parquet, self.manifest, self.feature_config)

    def test_bad_bagit_hash_fails_integrity(self) -> None:
        self.manifest.write_text(f"{'0' * 40}  {self.csv.name}\n", encoding="utf-8")

        receipt = d0.audit(self.csv, self.parquet, self.manifest, self.feature_config)

        self.assertEqual(receipt["status"], "integrity_failed")
        self.assertFalse(receipt["checks"]["bagit_sha1_valid"])

    def test_row_or_schema_change_fails_integrity(self) -> None:
        pl.read_csv(self.csv).select("FLOW_START_MILLISECONDS", "Label").head(2).write_parquet(
            self.parquet
        )

        receipt = d0.audit(self.csv, self.parquet, self.manifest, self.feature_config)

        self.assertEqual(receipt["status"], "integrity_failed")
        self.assertEqual(receipt["rows"]["dropped"], 1)
        self.assertFalse(receipt["checks"]["csv_header_matches_parquet_order"])
        self.assertFalse(receipt["checks"]["timestamp_columns_present"])

    def test_value_change_fails_integrity(self) -> None:
        changed = pl.read_csv(self.csv)
        changed[0, "Label"] = 9
        changed.write_parquet(self.parquet)

        receipt = d0.audit(self.csv, self.parquet, self.manifest, self.feature_config)

        self.assertEqual(receipt["status"], "integrity_failed")
        self.assertFalse(receipt["checks"]["full_values_match"])

    def test_change_after_first_batch_fails_full_value_parity(self) -> None:
        rows = "".join(f"{index},{index + 1},0\n" for index in range(129))
        self.csv.write_text(
            "FLOW_START_MILLISECONDS,FLOW_END_MILLISECONDS,Label\n" + rows,
            encoding="utf-8",
        )
        converted = pl.read_csv(self.csv)
        converted[128, "Label"] = 1
        converted.write_parquet(self.parquet)
        self.manifest.write_text(
            f"{d0.file_digest(self.csv, 'sha1')}  {self.csv.name}\n",
            encoding="utf-8",
        )
        config = json.loads(self.feature_config.read_text(encoding="utf-8"))
        config["datasets"][0]["bagit_csv_sha1"] = d0.file_digest(self.csv, "sha1")
        self.feature_config.write_text(json.dumps(config), encoding="utf-8")

        with patch.object(d0, "PARITY_BATCH_ROWS", 64):
            receipt = d0.audit(self.csv, self.parquet, self.manifest, self.feature_config)

        self.assertEqual(receipt["status"], "integrity_failed")
        self.assertEqual(receipt["d0_gates"]["conversion_integrity"], "failed")
        self.assertEqual(receipt["d0_gates"]["full_value_parity"], "failed")
        self.assertEqual(receipt["d0_status"], "blocked")

    def test_receipt_serialization_is_relocatable_and_deterministic(self) -> None:
        first = self.root / "first.json"
        second = self.root / "second.json"
        receipt = d0.audit(self.csv, self.parquet, self.manifest, self.feature_config)

        protected = (self.csv, self.parquet, self.manifest, self.feature_config)
        d0.write_receipt(receipt, first, protected)
        d0.write_receipt(receipt, second, protected)

        self.assertEqual(first.read_bytes(), second.read_bytes())
        parsed = json.loads(first.read_text(encoding="utf-8"))
        self.assertNotIn(str(self.root), json.dumps(parsed))

    def test_receipt_cannot_replace_any_audited_input(self) -> None:
        receipt = d0.audit(self.csv, self.parquet, self.manifest, self.feature_config)
        original = self.csv.read_bytes()

        with self.assertRaisesRegex(ValueError, "must not replace"):
            d0.write_receipt(receipt, self.csv, (self.csv, self.parquet, self.manifest))

        self.assertEqual(self.csv.read_bytes(), original)

    def test_receipt_is_idempotent_but_refuses_different_existing_content(self) -> None:
        receipt = d0.audit(self.csv, self.parquet, self.manifest, self.feature_config)
        output = self.root / "receipt.json"
        protected = (self.csv, self.parquet, self.manifest, self.feature_config)

        d0.write_receipt(receipt, output, protected)
        original = output.read_bytes()
        d0.write_receipt(receipt, output, protected)
        output.write_text("different\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "different content"):
            d0.write_receipt(receipt, output, protected)

        self.assertEqual(output.read_text(encoding="utf-8"), "different\n")
        self.assertNotEqual(original, output.read_bytes())

    def test_cli_writes_blocked_receipt_with_distinct_exit_status(self) -> None:
        output = self.root / "receipt.json"

        result = CliRunner().invoke(
            d0.main,
            [
                "--csv",
                str(self.csv),
                "--parquet",
                str(self.parquet),
                "--bag-manifest",
                str(self.manifest),
                "--feature-config",
                str(self.feature_config),
                "--output",
                str(output),
            ],
        )

        self.assertEqual(result.exit_code, 3, result.output)
        self.assertIn("d0_status=blocked", result.output)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["d0_status"], "blocked")

    def test_cli_rejects_missing_required_evidence_as_usage_error(self) -> None:
        result = CliRunner().invoke(
            d0.main,
            [
                "--csv",
                str(self.csv),
                "--parquet",
                str(self.parquet),
                "--feature-config",
                str(self.feature_config),
                "--output",
                str(self.root / "receipt.json"),
            ],
        )

        self.assertEqual(result.exit_code, 2)
        self.assertIn("Missing option '--bag-manifest'", result.output)


class NF3SchemaTest(unittest.TestCase):
    @staticmethod
    def raw_nf3(contract: d0.FeatureContract) -> dict[str, object]:
        raw: dict[str, object] = {field: 0 for field in contract.field_order}
        raw.update(
            {
                "FLOW_START_MILLISECONDS": 100,
                "FLOW_END_MILLISECONDS": 110,
                "IPV4_SRC_ADDR": "192.0.2.1",
                "IPV4_DST_ADDR": "198.51.100.2",
                "Attack": "attack-label",
            }
        )
        return raw

    def test_every_field_has_one_role_and_forbidden_fields_are_not_model_inputs(self) -> None:
        config = json.loads(Path("tools/configs/nf3.json").read_text(encoding="utf-8"))
        fields = config["schema"]["field_order"]
        roles = config["field_roles"]
        model_fields = {
            name for name, role in roles.items() if role in {"model_numeric", "model_categorical"}
        }

        self.assertEqual(set(fields), set(roles))
        self.assertEqual(len(fields), len(roles))
        self.assertTrue(
            set(config["model_input_policy"]["never_model_inputs"]).isdisjoint(model_fields)
        )
        self.assertFalse(
            config["model_input_policy"]["port_encoding"]["raw_high_port_values_are_model_inputs"]
        )
        self.assertTrue(all(dataset["q3_pairing"] is False for dataset in config["datasets"]))
        contract = d0.feature_contract(Path("tools/configs/nf3.json"), fields)
        self.assertTrue(contract["expected_order_matches"])
        self.assertNotIn("Label", contract["model_fields"])

    def test_model_view_drops_targets_identifiers_and_raw_high_ports(self) -> None:
        contract = d0.load_feature_contract(Path("tools/configs/nf3.json"))
        raw = {
            field: 0
            for field, role in contract.roles.items()
            if role in {"model_numeric", "model_categorical"}
        }
        raw.update(
            {
                "L4_SRC_PORT": 80,
                "L4_DST_PORT": 55000,
                "IPV4_SRC_ADDR": "192.0.2.1",
                "FLOW_END_MILLISECONDS": 123,
                "Label": 1,
                "Attack": "secret",
            }
        )

        view = d0.model_view(raw, contract)
        visible = set(view["numeric"]) | set(view["categorical"])

        self.assertTrue(contract.forbidden.isdisjoint(visible))
        self.assertEqual(view["categorical"]["L4_SRC_PORT"], "PORT_80")
        self.assertEqual(view["categorical"]["L4_DST_PORT"], "DYNAMIC")
        self.assertNotIn("55000", json.dumps(view))

    def test_model_view_rejects_missing_or_mistyped_numeric_fields(self) -> None:
        contract = d0.load_feature_contract(Path("tools/configs/nf3.json"))
        raw = {
            field: 0
            for field, role in contract.roles.items()
            if role in {"model_numeric", "model_categorical"}
        }
        del raw["IN_BYTES"]
        with self.assertRaisesRegex(ValueError, "IN_BYTES"):
            d0.model_view(raw, contract)

        raw["IN_BYTES"] = "not-a-number"
        with self.assertRaisesRegex(ValueError, "non-numeric"):
            d0.model_view(raw, contract)

    def test_routing_view_is_keyed_per_lineage_and_label_free(self) -> None:
        contract = d0.load_feature_contract(Path("tools/configs/nf3.json"))
        raw = self.raw_nf3(contract)
        secret = b"0123456789abcdef"
        first = d0.routing_view(
            raw,
            contract,
            lineage_id="capture-a",
            observation_domain_id="sensor-a",
            source_ordinal=7,
            secret=secret,
        )
        raw["Label"] = 1
        raw["Attack"] = "different-label"
        relabelled = d0.routing_view(
            raw,
            contract,
            lineage_id="capture-a",
            observation_domain_id="sensor-a",
            source_ordinal=7,
            secret=secret,
        )
        next_occurrence = d0.routing_view(
            raw,
            contract,
            lineage_id="capture-a",
            observation_domain_id="sensor-a",
            source_ordinal=8,
            secret=secret,
        )
        other_lineage = d0.routing_view(
            raw,
            contract,
            lineage_id="capture-b",
            observation_domain_id="sensor-a",
            source_ordinal=7,
            secret=secret,
        )
        other_domain = d0.routing_view(
            raw,
            contract,
            lineage_id="capture-a",
            observation_domain_id="sensor-b",
            source_ordinal=7,
            secret=secret,
        )

        self.assertEqual(first, relabelled)
        self.assertEqual(first.exact_group, next_occurrence.exact_group)
        self.assertNotEqual(first.event_id, next_occurrence.event_id)
        self.assertNotEqual(first.event_id, other_domain.event_id)
        self.assertNotEqual(first.source_key, other_lineage.source_key)
        self.assertNotIn("192.0.2.1", json.dumps(asdict(first)))
        self.assertNotIn("different-label", json.dumps(asdict(first)))
        self.assertRegex(first.source_key, r"^anon:[0-9a-f]{64}$")

    def test_near_group_ignores_secondary_fields_but_not_core_evidence(self) -> None:
        contract = d0.load_feature_contract(Path("tools/configs/nf3.json"))
        raw = self.raw_nf3(contract)
        arguments = {
            "lineage_id": "capture",
            "observation_domain_id": "sensor",
            "source_ordinal": 0,
            "secret": b"0123456789abcdef",
        }
        original = d0.routing_view(raw, contract, **arguments)
        raw["L7_PROTO"] = 99
        secondary_change = d0.routing_view(raw, contract, **arguments)
        raw["IN_BYTES"] = 1
        core_change = d0.routing_view(raw, contract, **arguments)

        self.assertNotEqual(original.exact_group, secondary_change.exact_group)
        self.assertEqual(original.near_group, secondary_change.near_group)
        self.assertNotEqual(secondary_change.near_group, core_change.near_group)

    def test_routing_view_rejects_short_secrets_and_invalid_intervals(self) -> None:
        contract = d0.load_feature_contract(Path("tools/configs/nf3.json"))
        raw = self.raw_nf3(contract)
        with self.assertRaisesRegex(ValueError, "16 bytes"):
            d0.routing_view(
                raw,
                contract,
                lineage_id="capture",
                observation_domain_id="sensor",
                source_ordinal=0,
                secret=b"short",
            )
        raw["FLOW_END_MILLISECONDS"] = 99
        with self.assertRaisesRegex(ValueError, "must not precede"):
            d0.routing_view(
                raw,
                contract,
                lineage_id="capture",
                observation_domain_id="sensor",
                source_ordinal=0,
                secret=b"0123456789abcdef",
            )
        raw["FLOW_END_MILLISECONDS"] = 110
        raw["IPV4_SRC_ADDR"] = "not-an-ip"
        with self.assertRaisesRegex(ValueError, "valid IPv4"):
            d0.routing_view(
                raw,
                contract,
                lineage_id="capture",
                observation_domain_id="sensor",
                source_ordinal=0,
                secret=b"0123456789abcdef",
            )


if __name__ == "__main__":
    unittest.main()
