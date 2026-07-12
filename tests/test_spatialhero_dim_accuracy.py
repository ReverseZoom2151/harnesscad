"""Tests for bench.spatialhero_dim_accuracy."""

import unittest

from bench.spatialhero_dim_accuracy import (
    dimension_accuracy,
    measure_and_score,
    measure_bbox_dimensions,
)


class TestMeasure(unittest.TestCase):
    def test_extents(self):
        d = measure_bbox_dimensions(0.0, 10.0, 0.0, 4.0, 0.0, 2.0)
        self.assertEqual(d["width"], 10.0)
        self.assertEqual(d["depth"], 4.0)
        self.assertEqual(d["height"], 2.0)
        self.assertEqual(d["volume"], 80.0)

    def test_negative_origin(self):
        d = measure_bbox_dimensions(-5.0, 5.0, -1.0, 1.0, 0.0, 3.0)
        self.assertEqual(d["width"], 10.0)
        self.assertEqual(d["depth"], 2.0)
        self.assertEqual(d["height"], 3.0)


class TestDimensionAccuracy(unittest.TestCase):
    def test_perfect_match(self):
        actual = {"width": 10.0, "height": 20.0}
        res = dimension_accuracy(actual, {"width": 10.0, "height": 20.0})
        self.assertAlmostEqual(res.average_accuracy, 1.0)
        self.assertTrue(res.all_within_tolerance)
        self.assertAlmostEqual(res.comparisons["width"].relative_error, 0.0)

    def test_within_tolerance(self):
        actual = {"width": 10.4}  # 4% off
        res = dimension_accuracy(actual, {"width": 10.0}, tolerance=0.05)
        self.assertTrue(res.comparisons["width"].within_tolerance)
        self.assertAlmostEqual(res.comparisons["width"].accuracy, 0.96)

    def test_outside_tolerance(self):
        actual = {"width": 12.0}  # 20% off
        res = dimension_accuracy(actual, {"width": 10.0}, tolerance=0.05)
        self.assertFalse(res.comparisons["width"].within_tolerance)
        self.assertFalse(res.all_within_tolerance)
        self.assertAlmostEqual(res.comparisons["width"].accuracy, 0.8)

    def test_accuracy_clamped_at_zero(self):
        actual = {"width": 30.0}  # 200% off -> accuracy floored to 0
        res = dimension_accuracy(actual, {"width": 10.0})
        self.assertEqual(res.comparisons["width"].accuracy, 0.0)

    def test_zero_expected_exact(self):
        actual = {"width": 0.0}
        res = dimension_accuracy(actual, {"width": 0.0})
        self.assertEqual(res.comparisons["width"].relative_error, 0.0)
        self.assertEqual(res.comparisons["width"].accuracy, 1.0)

    def test_zero_expected_nonzero_actual(self):
        actual = {"width": 5.0}
        res = dimension_accuracy(actual, {"width": 0.0})
        self.assertEqual(res.comparisons["width"].relative_error, float("inf"))
        self.assertEqual(res.comparisons["width"].accuracy, 0.0)

    def test_missing_key_ignored(self):
        actual = {"width": 10.0}
        res = dimension_accuracy(actual, {"width": 10.0, "height": 20.0})
        self.assertIn("width", res.comparisons)
        self.assertNotIn("height", res.comparisons)

    def test_no_overlap(self):
        res = dimension_accuracy({"foo": 1.0}, {"bar": 2.0})
        self.assertEqual(res.average_accuracy, 0.0)
        self.assertTrue(res.all_within_tolerance)  # vacuous
        self.assertEqual(res.comparisons, {})

    def test_average_over_multiple(self):
        actual = {"width": 10.0, "height": 22.0}  # 0% and 10% off
        res = dimension_accuracy(actual, {"width": 10.0, "height": 20.0})
        # accuracies 1.0 and 0.9 -> mean 0.95
        self.assertAlmostEqual(res.average_accuracy, 0.95)


class TestMeasureAndScore(unittest.TestCase):
    def test_end_to_end(self):
        bbox = (0.0, 10.0, 0.0, 4.0, 0.0, 2.0)
        res = measure_and_score(bbox, {"width": 10.0, "depth": 4.0, "volume": 80.0})
        self.assertAlmostEqual(res.average_accuracy, 1.0)
        self.assertTrue(res.all_within_tolerance)

    def test_bad_bbox_length(self):
        with self.assertRaises(ValueError):
            measure_and_score((0.0, 1.0, 0.0), {"width": 1.0})


if __name__ == "__main__":
    unittest.main()
