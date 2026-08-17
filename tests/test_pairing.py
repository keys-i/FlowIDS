"""Tests for the manifest-only Q3 pairing gate."""

import unittest

from tools.scripts.pairing import resolve_pairing


def pairing(**overrides: object) -> dict[str, object]:
    return {
        "capture_id": "capture-1",
        "five_tuple": ["10.0.0.1", 1234, "10.0.0.2", 443, 6],
        "flow_interval": [10, 20],
        "packet_interval": [12, 18],
        "packet_count": 4,
        "byte_count": 512,
    } | overrides


def flow(**overrides: object) -> dict[str, object]:
    return {
        "capture_id": "capture-1",
        "five_tuple": ["10.0.0.2", 443, "10.0.0.1", 1234, 6],
        "interval": [10, 20],
        "packet_count": 4,
        "byte_count": 512,
    } | overrides


class PairingTests(unittest.TestCase):
    def test_disabled_does_not_require_a_manifest(self) -> None:
        result = resolve_pairing({"q3_pairing": False}, {})

        self.assertEqual((result.status, result.candidates), ("disabled", 0))

    def test_accepts_one_canonical_bidirectional_overlap(self) -> None:
        result = resolve_pairing({"q3_pairing": True, "pairing_manifest": [pairing()]}, flow())

        self.assertEqual((result.status, result.candidates), ("accepted", 1))
        self.assertIsNotNone(result.pairing)

    def test_zero_or_multiple_candidates_fail_closed(self) -> None:
        none = resolve_pairing({"q3_pairing": True, "pairing_manifest": []}, flow())
        multiple = resolve_pairing(
            {
                "q3_pairing": True,
                "pairing_manifest": [
                    pairing(),
                    pairing(packet_interval=[13, 19]),
                ],
            },
            flow(),
        )

        self.assertEqual((none.status, none.candidates), ("ambiguous", 0))
        self.assertEqual((multiple.status, multiple.candidates), ("ambiguous", 2))

    def test_rejects_noncanonical_or_unverified_evidence(self) -> None:
        cases = (
            {"q3_pairing": True, "pairing_manifest": [pairing(five_tuple=["z", 9, "a", 1, 6])]},
            {"q3_pairing": True, "pairing_manifest": [pairing(packet_interval=[21, 22])]},
        )

        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                resolve_pairing(config, flow())

    def test_packet_and_byte_evidence_must_match_the_flow(self) -> None:
        result = resolve_pairing(
            {"q3_pairing": True, "pairing_manifest": [pairing()]},
            flow(packet_count=5),
        )

        self.assertEqual((result.status, result.candidates), ("ambiguous", 0))


if __name__ == "__main__":
    unittest.main()
