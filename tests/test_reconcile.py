import unittest

from harnesscad.io.ingest.reconcile import DiscrepancyKind, Evidence, reconcile


class TestReconcile(unittest.TestCase):
    def test_matching_three_source_record(self):
        evidence = [
            Evidence("hole-1", "model", metrics={"diameter": 6.0},
                     annotations={"label": "M6"}, metadata={"material": "steel"}),
            Evidence("hole-1", "drawing", metrics={"diameter": 6.005},
                     annotations={"label": "M6"}, metadata={"material": "steel"}),
            Evidence("hole-1", "reference", metrics={"diameter": 6.0},
                     annotations={"label": "M6"}, metadata={"material": "steel"}),
        ]
        report = reconcile(evidence)
        self.assertTrue(report.ok)
        self.assertEqual(report.comparisons, 9)
        self.assertEqual(report.correspondence_ids, ("hole-1",))

    def test_numeric_discrepancy_is_typed_with_delta(self):
        report = reconcile([
            Evidence("body", "model", metrics={"volume": 110.0}),
            Evidence("body", "drawing", metrics={"volume": 100.0}),
            Evidence("body", "reference", metrics={"volume": 100.0}),
        ])
        found = report.by_kind(DiscrepancyKind.NUMERIC_MISMATCH)
        self.assertEqual(len(found), 2)
        self.assertAlmostEqual(found[0].relative_delta, 10.0 / 110.0)

    def test_annotation_and_metadata_mismatch(self):
        report = reconcile([
            Evidence("part", "model", annotations={"finish": "paint"},
                     metadata={"revision": "B"}),
            Evidence("part", "drawing", annotations={"finish": "anodize"},
                     metadata={"revision": "A"}),
            Evidence("part", "reference", annotations={"finish": "paint"},
                     metadata={"revision": "B"}),
        ])
        self.assertEqual(len(report.by_kind(DiscrepancyKind.VALUE_MISMATCH)), 4)

    def test_missing_source_correspondence_is_explicit(self):
        report = reconcile([
            Evidence("bolt-7", "model", metrics={"length": 20}),
            Evidence("bolt-7", "reference", metrics={"length": 20}),
        ])
        missing = report.by_kind(DiscrepancyKind.MISSING_CORRESPONDENCE)
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0].sources, ("drawing",))

    def test_missing_field_only_when_peer_supplies_it(self):
        report = reconcile(
            [
                Evidence("p", "model", metrics={"mass": 2.0}),
                Evidence("p", "drawing", metrics={}),
            ],
            required_sources=("model", "drawing"),
        )
        missing = report.by_kind(DiscrepancyKind.MISSING_FIELD)
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0].field, "mass")
        self.assertEqual(missing[0].sources, ("drawing",))

    def test_duplicate_source_id_is_reported_deterministically(self):
        report = reconcile(
            [Evidence("p", "model"), Evidence("p", "model")],
            required_sources=("model",),
        )
        self.assertEqual(len(report.by_kind(DiscrepancyKind.DUPLICATE_EVIDENCE)), 1)

    def test_report_order_does_not_depend_on_input_order(self):
        items = [
            Evidence("b", "model", metrics={"x": 1}),
            Evidence("a", "model", metrics={"x": 2}),
        ]
        left = reconcile(items, required_sources=("model", "drawing")).to_dict()
        right = reconcile(reversed(items), required_sources=("drawing", "model")).to_dict()
        self.assertEqual(left, right)

    def test_custom_absolute_tolerance_near_zero(self):
        report = reconcile(
            [
                Evidence("p", "a", metrics={"offset": 0.0001}),
                Evidence("p", "b", metrics={"offset": 0.0}),
            ],
            required_sources=("a", "b"),
            relative_tolerance=0.0,
            absolute_tolerance=0.001,
        )
        self.assertTrue(report.ok)

    def test_invalid_identity_and_tolerances(self):
        with self.assertRaises(ValueError):
            Evidence("", "model")
        with self.assertRaises(ValueError):
            Evidence("x", "")
        with self.assertRaises(ValueError):
            reconcile([], relative_tolerance=-1)


if __name__ == "__main__":
    unittest.main()
