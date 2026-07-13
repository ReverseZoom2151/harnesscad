"""Tests for the cadrille CAD-reconstruction evaluation protocol."""

import unittest

from harnesscad.eval.bench.cadrille_metrics import (
    normalize_to_unit_cube,
    chamfer_distance,
    median_cd,
    invalidity_ratio,
    iou_percent,
    evaluation_report,
    CD_SCALE,
)


class NormalizeTest(unittest.TestCase):
    def test_unit_cube(self):
        out = normalize_to_unit_cube([(0.0, 0.0, 0.0), (2.0, 2.0, 2.0)])
        self.assertAlmostEqual(out[0][0], -0.5)
        self.assertAlmostEqual(out[1][0], 0.5)


class ChamferTest(unittest.TestCase):
    def test_identical_zero(self):
        pts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        self.assertAlmostEqual(chamfer_distance(pts, pts), 0.0)

    def test_scaled(self):
        a = [(0.0, 0.0, 0.0)]
        b = [(1.0, 0.0, 0.0)]
        # symmetric mean distance is 1.0, scaled by CD_SCALE
        self.assertAlmostEqual(chamfer_distance(a, b), CD_SCALE)

    def test_empty(self):
        with self.assertRaises(ValueError):
            chamfer_distance([], [(0.0, 0.0, 0.0)])


class AggregateTest(unittest.TestCase):
    def test_median_odd(self):
        self.assertEqual(median_cd([3.0, 1.0, 2.0]), 2.0)

    def test_median_even(self):
        self.assertEqual(median_cd([1.0, 2.0, 3.0, 4.0]), 2.5)

    def test_invalidity_ratio(self):
        self.assertAlmostEqual(invalidity_ratio([True, True, False, True]), 25.0)

    def test_iou_percent(self):
        self.assertAlmostEqual(iou_percent(0.871), 87.1)
        with self.assertRaises(ValueError):
            iou_percent(1.2)


class ReportTest(unittest.TestCase):
    def test_report_excludes_invalid_from_cd(self):
        records = [
            {"valid": True, "cd": 0.2, "iou": 0.9},
            {"valid": True, "cd": 0.4, "iou": 0.7},
            {"valid": False},
        ]
        report = evaluation_report(records)
        self.assertEqual(report["count"], 3)
        self.assertEqual(report["valid_count"], 2)
        self.assertAlmostEqual(report["median_cd"], 0.3)  # median of 0.2, 0.4
        self.assertAlmostEqual(report["mean_iou"], 80.0)  # mean of 90, 70
        self.assertAlmostEqual(report["invalidity_ratio"], 100 / 3)

    def test_report_empty(self):
        with self.assertRaises(ValueError):
            evaluation_report([])


if __name__ == "__main__":
    unittest.main()
