import unittest
from dataclasses import replace

from tools.scripts import splits


def row(
    event_id: str,
    partition_time_ms: int,
    suffix: str,
    **changes: object,
) -> splits.SplitRow:
    value = splits.SplitRow(
        event_id=event_id,
        partition_time_ms=partition_time_ms,
        source_principal=f"anon:source-{suffix}",
        destination_principal=f"anon:destination-{suffix}",
        source_unit_id=f"unit-{suffix}",
        capture_lineage_id=f"capture-{suffix}",
        exact_group=f"exact-{suffix}",
        near_group=f"near-{suffix}",
        canonical_family="benign",
        campaign_id="campaign-benign",
    )
    return replace(value, **changes)


class TrackSpecificSplitTest(unittest.TestCase):
    def test_chronological_keeps_recurring_endpoint_and_source_unit(self) -> None:
        recurring = {"source_principal": "anon:repeat", "source_unit_id": "unit-repeat"}
        rows = [
            row("train", 5, "a", **recurring),
            row("validation", 15, "b", **recurring),
            row("test", 25, "c", **recurring),
        ]
        spec = splits.SplitSpec(
            "chronological",
            train_end_ms=10,
            validation_end_ms=20,
        )

        self.assertEqual(splits.partition_row(rows[0], spec), ("train", ()))

        forward = splits.build_split(rows, spec)
        reverse = splits.build_split(reversed(rows), spec)

        self.assertEqual(forward, reverse)
        self.assertEqual(forward["retained_counts"], {"test": 1, "train": 1, "validation": 1})
        self.assertEqual(forward["dropped"], [])
        self.assertTrue(
            all("partition_time_ms" in assignment for assignment in forward["assignments"])
        )
        splits.validate_no_overlap(forward)

    def test_endpoint_holdout_purges_boundary_and_never_leaks_to_source(self) -> None:
        held = "anon:held"
        rows = [
            row("source", 1, "source"),
            row("target", 2, "target", source_principal=held, destination_principal=held),
            row("boundary", 3, "boundary", source_principal=held),
        ]
        result = splits.build_split(
            rows, splits.SplitSpec("endpoint_disjoint", held_principals=frozenset({held}))
        )

        assignments = {entry["event_id"]: entry for entry in result["assignments"]}
        self.assertEqual(assignments["source"]["partition"], "source")
        self.assertEqual(assignments["target"]["partition"], "target")
        self.assertTrue(assignments["source"]["pretraining_visible"])
        self.assertFalse(assignments["target"]["pretraining_visible"])
        self.assertEqual(
            result["dropped"], [{"event_id": "boundary", "reasons": ["purge:endpoint_boundary"]}]
        )
        self.assertFalse(
            any(
                held in (entry["source_principal"], entry["destination_principal"])
                and entry["partition"] == "source"
                for entry in result["assignments"]
            )
        )

    def test_held_family_and_campaign_are_both_isolated(self) -> None:
        rows = [
            row("source", 1, "source"),
            row("family", 2, "family", canonical_family="ransomware", campaign_id="other"),
            row("campaign", 3, "campaign", canonical_family="other", campaign_id="campaign-held"),
            row("shared-source", 4, "shared-source", campaign_id="shared-campaign"),
            row(
                "shared-target",
                5,
                "shared-target",
                canonical_family="ransomware",
                campaign_id="shared-campaign",
            ),
        ]
        result = splits.build_split(
            rows,
            splits.SplitSpec(
                "held_out_family",
                held_families=frozenset({"ransomware"}),
                held_campaigns=frozenset({"campaign-held"}),
            ),
        )

        partitions = {entry["event_id"]: entry["partition"] for entry in result["assignments"]}
        self.assertEqual(partitions, {"campaign": "target", "family": "target", "source": "source"})
        self.assertEqual(result["dropped_counts"], {"overlap:campaign": 2})
        self.assertFalse(
            any(
                entry["canonical_family"] == "ransomware" and entry["partition"] == "source"
                for entry in result["assignments"]
            )
        )
        self.assertFalse(
            any(
                entry["campaign_id"] == "campaign-held" and entry["partition"] == "source"
                for entry in result["assignments"]
            )
        )

    def test_cross_network_target_and_sealed_are_never_pretraining_visible(self) -> None:
        rows = [
            row("source", 1, "source", source_unit_id="source-unit"),
            row("target", 2, "target", source_unit_id="target-unit"),
            row("sealed", 3, "sealed", source_unit_id="sealed-unit"),
        ]
        result = splits.build_split(
            rows,
            splits.SplitSpec(
                "cross_network",
                source_units=frozenset({"source-unit"}),
                target_units=frozenset({"target-unit"}),
                sealed_units=frozenset({"sealed-unit"}),
            ),
        )

        visible = {
            entry["partition"]: entry["pretraining_visible"] for entry in result["assignments"]
        }
        self.assertEqual(visible, {"source": True, "target": False, "sealed": False})
        splits.validate_no_overlap(result)
        with self.assertRaisesRegex(ValueError, "required partitions are empty"):
            splits.build_split(
                [
                    row(
                        "source-derivative",
                        4,
                        "source-derivative",
                        source_unit_id="source-unit",
                        capture_lineage_id="shared-capture",
                    ),
                    row(
                        "target-derivative",
                        5,
                        "target-derivative",
                        source_unit_id="target-unit",
                        capture_lineage_id="shared-capture",
                    ),
                    row("sealed-only", 6, "sealed-only", source_unit_id="sealed-unit"),
                ],
                splits.SplitSpec(
                    "cross_network",
                    source_units=frozenset({"source-unit"}),
                    target_units=frozenset({"target-unit"}),
                    sealed_units=frozenset({"sealed-unit"}),
                ),
            )

    def test_duplicate_rejected_and_exact_near_overlaps_removed(self) -> None:
        spec = splits.SplitSpec(
            "chronological",
            train_end_ms=10,
            validation_end_ms=20,
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            splits.build_split([row("duplicate", 1, "a"), row("duplicate", 2, "b")], spec)
        rows = [
            row("train-exact-conflict", 1, "a", exact_group="exact-shared"),
            row("test-exact-conflict", 25, "b", exact_group="exact-shared"),
            row("train-near-conflict", 2, "c", near_group="near-shared"),
            row("test-near-conflict", 26, "d", near_group="near-shared"),
            row("train", 3, "e"),
            row("validation", 15, "f"),
            row("test", 27, "g"),
        ]
        result = splits.build_split(rows, spec)

        self.assertEqual(result["dropped_counts"], {"overlap:exact": 1, "overlap:near": 1})
        self.assertEqual(
            {entry["event_id"] for entry in result["assignments"]},
            {"test-exact-conflict", "test-near-conflict", "train", "validation", "test"},
        )
        self.assertFalse(any("context_group" in entry for entry in result["assignments"]))
        splits.validate_no_overlap(result)

    def test_invalid_or_empty_required_partitions_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "selector is required"):
            splits.build_split([], splits.SplitSpec("endpoint_disjoint"))
        with self.assertRaisesRegex(ValueError, "selector is required"):
            splits.build_split(
                [],
                splits.SplitSpec("held_out_family", held_families=frozenset({"family"})),
            )
        with self.assertRaisesRegex(ValueError, "selector is required"):
            splits.build_split(
                [],
                splits.SplitSpec(
                    "cross_network",
                    source_units=frozenset({"source"}),
                    target_units=frozenset({"target"}),
                ),
            )
        with self.assertRaisesRegex(ValueError, "required partitions are empty"):
            splits.build_split(
                [row("train", 1, "a")],
                splits.SplitSpec(
                    "chronological",
                    train_end_ms=10,
                    validation_end_ms=20,
                ),
            )
        raw = row("raw", 1, "raw", source_principal="192.0.2.1")
        with self.assertRaisesRegex(ValueError, "opaque anon"):
            splits.build_split(
                [raw],
                splits.SplitSpec(
                    "chronological",
                    train_end_ms=10,
                    validation_end_ms=20,
                ),
            )


if __name__ == "__main__":
    unittest.main()
