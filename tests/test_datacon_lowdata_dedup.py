"""Tests for dataengine.datacon_lowdata_dedup (low-data curation protocol)."""

from __future__ import annotations

import unittest

from harnesscad.data.dataengine.datacon_lowdata_dedup import (
    canonical_signature,
    construction_report,
    dedup_by_scale,
    is_scale_variant,
    scale_normalize,
    select_training_subset,
)


class ScaleNormalizeTest(unittest.TestCase):
    def test_scale_variants_normalize_equal(self):
        a = scale_normalize([3.0, 4.0])
        b = scale_normalize([6.0, 8.0])
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y)
        # [3,4] has norm 5 -> (0.6, 0.8)
        self.assertAlmostEqual(a[0], 0.6)
        self.assertAlmostEqual(a[1], 0.8)

    def test_zero_vector_guarded(self):
        self.assertEqual(scale_normalize([0.0, 0.0, 0.0]), (0.0, 0.0, 0.0))


class CanonicalSignatureTest(unittest.TestCase):
    def test_scale_variants_collide(self):
        self.assertEqual(
            canonical_signature([3.0, 4.0]),
            canonical_signature([30.0, 40.0]),
        )

    def test_distinct_designs_differ(self):
        self.assertNotEqual(
            canonical_signature([1.0, 2.0]),
            canonical_signature([2.0, 1.0]),
        )


class IsScaleVariantTest(unittest.TestCase):
    def test_true_for_scaled(self):
        self.assertTrue(is_scale_variant([1.0, 2.0], [2.0, 4.0]))

    def test_false_for_different(self):
        self.assertFalse(is_scale_variant([1.0, 2.0], [2.0, 1.0]))


class DedupByScaleTest(unittest.TestCase):
    def test_dedup_reduces_and_preserves_order(self):
        records = [
            {"id": "a", "features": [1.0, 0.0]},
            {"id": "b", "features": [0.0, 1.0]},
            {"id": "c", "features": [2.0, 0.0]},  # scale-variant of a
            {"id": "d", "features": [1.0, 1.0]},
            {"id": "e", "features": [0.0, 5.0]},  # scale-variant of b
        ]
        result = dedup_by_scale(records)
        self.assertEqual(result["n_in"], 5)
        self.assertEqual(result["n_out"], 3)
        kept_ids = [r["id"] for r in result["kept"]]
        self.assertEqual(kept_ids, ["a", "b", "d"])  # first-seen, order preserved
        removed_ids = [r["id"] for r in result["removed"]]
        self.assertEqual(removed_ids, ["c", "e"])
        self.assertAlmostEqual(result["reduction_ratio"], 1.0 - 3.0 / 5.0)

    def test_empty_input(self):
        result = dedup_by_scale([])
        self.assertEqual(result["n_in"], 0)
        self.assertEqual(result["n_out"], 0)
        self.assertAlmostEqual(result["reduction_ratio"], 0.0)


class SelectTrainingSubsetTest(unittest.TestCase):
    def _distinct_records(self):
        return [
            {"id": "a", "features": [1.0, 0.0]},
            {"id": "b", "features": [0.0, 1.0]},
            {"id": "c", "features": [1.0, 1.0]},
            {"id": "d", "features": [-1.0, 0.0]},
            {"id": "e", "features": [-1.0, -1.0]},
        ]

    def test_returns_exact_target_size(self):
        records = self._distinct_records()
        subset = select_training_subset(records, target_size=3, seed=7)
        self.assertEqual(len(subset), 3)

    def test_deterministic_same_seed(self):
        records = self._distinct_records()
        s1 = select_training_subset(records, target_size=3, seed=42)
        s2 = select_training_subset(records, target_size=3, seed=42)
        self.assertEqual([r["id"] for r in s1], [r["id"] for r in s2])

    def test_returns_all_when_fewer_than_target(self):
        records = [
            {"id": "a", "features": [1.0, 0.0]},
            {"id": "b", "features": [2.0, 0.0]},  # scale-variant of a
        ]
        subset = select_training_subset(records, target_size=10, seed=1)
        # after dedup only 1 distinct design remains
        self.assertEqual(len(subset), 1)


class ConstructionReportTest(unittest.TestCase):
    def test_funnel_keys_and_numbers(self):
        records = [
            {"id": "a", "features": [1.0, 0.0]},
            {"id": "b", "features": [0.0, 1.0]},
            {"id": "c", "features": [2.0, 0.0]},  # scale-variant of a
            {"id": "d", "features": [1.0, 1.0]},
            {"id": "e", "features": [0.0, 5.0]},  # scale-variant of b
        ]
        report = construction_report(records, target_size=2, seed=3)
        for key in (
            "n_raw",
            "n_after_dedup",
            "n_selected",
            "reduction_ratio",
            "target_size",
            "seed",
        ):
            self.assertIn(key, report)
        self.assertEqual(report["n_raw"], 5)
        self.assertEqual(report["n_after_dedup"], 3)
        self.assertEqual(report["n_selected"], 2)
        self.assertAlmostEqual(report["reduction_ratio"], 1.0 - 3.0 / 5.0)


if __name__ == "__main__":
    unittest.main()
