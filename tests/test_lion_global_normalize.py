"""Tests for geometry.lion_global_normalize (dataset-level normalisation)."""

import unittest

from harnesscad.domain.geometry.transforms.lion_global_normalize import (
    bounding_box,
    global_normalize,
    global_stats,
    per_shape_normalize_unit_range,
)


class GlobalStatsTest(unittest.TestCase):
    def test_mean_and_std(self):
        clouds = [[(0.0, 0.0, 0.0)], [(2.0, 2.0, 2.0)]]
        mean, std = global_stats(clouds)
        self.assertEqual(mean, [1.0, 1.0, 1.0])
        # every deviation is +/-1 => rms = 1
        self.assertAlmostEqual(std, 1.0, places=9)

    def test_zero_variance_std_one(self):
        clouds = [[(5.0, 5.0, 5.0)], [(5.0, 5.0, 5.0)]]
        mean, std = global_stats(clouds)
        self.assertEqual(mean, [5.0, 5.0, 5.0])
        self.assertEqual(std, 1.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            global_stats([])


class GlobalNormalizeTest(unittest.TestCase):
    def test_normalize_centers_dataset(self):
        clouds = [[(0.0, 0.0, 0.0)], [(2.0, 2.0, 2.0)]]
        out = global_normalize(clouds)
        self.assertAlmostEqual(out[0][0][0], -1.0, places=9)
        self.assertAlmostEqual(out[1][0][0], 1.0, places=9)

    def test_preserves_relative_scale(self):
        # A big shape and a small shape keep their size ratio under global norm.
        big = [(-4.0, 0.0, 0.0), (4.0, 0.0, 0.0)]
        small = [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        out = global_normalize([big, small])
        big_span = out[0][1][0] - out[0][0][0]
        small_span = out[1][1][0] - out[1][0][0]
        self.assertAlmostEqual(big_span / small_span, 4.0, places=9)

    def test_reuse_stats(self):
        train = [[(0.0, 0.0, 0.0)], [(2.0, 2.0, 2.0)]]
        stats = global_stats(train)
        val = [[(1.0, 1.0, 1.0)]]
        out = global_normalize(val, stats=stats)
        # (1 - 1)/1 = 0 on every axis
        self.assertEqual(out[0][0], (0.0, 0.0, 0.0))

    def test_deterministic(self):
        clouds = [[(0.3, -0.2, 0.5)], [(1.1, 0.8, -0.4)]]
        self.assertEqual(global_normalize(clouds), global_normalize(clouds))


class PerShapeUnitRangeTest(unittest.TestCase):
    def test_longest_axis_spans_unit(self):
        cloud = [(-3.0, -1.0, 0.0), (3.0, 1.0, 0.0)]
        out = per_shape_normalize_unit_range(cloud)
        xs = [p[0] for p in out]
        self.assertAlmostEqual(min(xs), -1.0, places=9)
        self.assertAlmostEqual(max(xs), 1.0, places=9)

    def test_aspect_preserved(self):
        cloud = [(-3.0, -1.0, 0.0), (3.0, 1.0, 0.0)]
        out = per_shape_normalize_unit_range(cloud)
        ys = [p[1] for p in out]
        # y extent is 1/3 of x extent => spans [-1/3, 1/3]
        self.assertAlmostEqual(max(ys), 1.0 / 3.0, places=9)

    def test_degenerate_no_scale(self):
        out = per_shape_normalize_unit_range([(2.0, 2.0, 2.0)])
        self.assertEqual(out, [(0.0, 0.0, 0.0)])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            per_shape_normalize_unit_range([])


class BoundingBoxTest(unittest.TestCase):
    def test_box(self):
        clouds = [[(-1.0, 2.0, 0.0)], [(3.0, -4.0, 5.0)]]
        lo, hi = bounding_box(clouds)
        self.assertEqual(lo, [-1.0, -4.0, 0.0])
        self.assertEqual(hi, [3.0, 2.0, 5.0])


if __name__ == "__main__":
    unittest.main()
