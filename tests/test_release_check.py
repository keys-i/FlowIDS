import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.scripts.release_check import check_archive, check_paths, main


def archive(path: Path, entries: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, content in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return path


class ReleaseCheckTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.archive_path = Path(self.temporary_directory.name) / "release.tar.gz"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_allows_safe_paths_and_schema(self) -> None:
        result = check_archive(
            archive(
                self.archive_path,
                {"flowids/model-schema.json": b'{"fields": ["duration", "packets"]}'},
            )
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.as_dict(), {"ok": True, "findings": []})

    def test_source_configuration_is_not_mistaken_for_a_deployable_schema(self) -> None:
        result = check_archive(
            archive(
                self.archive_path,
                {"flowids/tools/configs/nf3.json": b'{"field_roles": {"Label": "target"}}'},
            )
        )

        self.assertTrue(result.ok)

    def test_rejects_data_secret_and_evidence_paths(self) -> None:
        result = check_paths(
            [
                "flowids/data/flows.parquet",
                "flowids/secrets/key.pem",
                "flowids/evidence/receipt.txt",
            ]
        )
        self.assertEqual(
            {(finding.path, finding.reason) for finding in result.findings},
            {
                ("flowids/data/flows.parquet", "data, secret, or evidence artifact"),
                ("flowids/data/flows.parquet", "raw data or secret file type"),
                ("flowids/evidence/receipt.txt", "data, secret, or evidence artifact"),
                ("flowids/secrets/key.pem", "data, secret, or evidence artifact"),
                ("flowids/secrets/key.pem", "raw data or secret file type"),
            },
        )

    def test_rejects_compressed_data_and_standard_secret_names(self) -> None:
        result = check_paths(
            ["flowids/flows.csv.gz", "flowids/.env", "flowids/config/.env.production"]
        )

        self.assertEqual(
            {(finding.path, finding.reason) for finding in result.findings},
            {
                ("flowids/.env", "standard secret file name"),
                ("flowids/config/.env.production", "standard secret file name"),
                ("flowids/flows.csv.gz", "raw data or secret file type"),
            },
        )

    def test_rejects_forbidden_schema_fields(self) -> None:
        result = check_archive(
            archive(
                self.archive_path,
                {
                    "flowids/model-schema.json": (
                        b'{"fields": ["duration", "target", "src-ip", '
                        b'"FLOW_START_MILLISECONDS", "flow_available_ms", '
                        b'"end_reason", "completion_ms", "scenario_id"]}'
                    ),
                    "flowids/run-manifest.json": b'{"source_key": "opaque", "event_id": "private"}',
                },
            )
        )
        self.assertEqual(
            [(finding.path, finding.reason) for finding in result.findings],
            [
                ("flowids/model-schema.json", "forbidden schema field: completion_ms"),
                ("flowids/model-schema.json", "forbidden schema field: end_reason"),
                ("flowids/model-schema.json", "forbidden schema field: flow_available_ms"),
                ("flowids/model-schema.json", "forbidden schema field: flow_start_milliseconds"),
                ("flowids/model-schema.json", "forbidden schema field: scenario_id"),
                ("flowids/model-schema.json", "forbidden schema field: src_ip"),
                ("flowids/model-schema.json", "forbidden schema field: target"),
                ("flowids/run-manifest.json", "forbidden schema field: event_id"),
                ("flowids/run-manifest.json", "forbidden schema field: source_key"),
            ],
        )

    def test_cli_rejects_a_missing_release_file(self) -> None:
        missing = Path(self.temporary_directory.name) / "missing.tar.gz"
        with (
            patch("sys.argv", ["release-check", str(missing)]),
            self.assertRaises(SystemExit) as exit_,
        ):
            main()

        self.assertEqual(exit_.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
