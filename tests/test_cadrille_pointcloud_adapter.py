"""Tests for cadrille point-cloud adapter (unit-cube normalise + FPS)."""

import unittest

from harnesscad.domain.reconstruction.fitting.pointcloud_adapter import (
    normalize_unit_cube,
    furthest_point_sampling,
    prepare_point_input,
    NUM_POINTS,
)


class NormalizeTest(unittest.TestCase):
    def test_fits_unit_cube(self):
        pts = [(0.0, 0.0, 0.0), (10.0, 4.0, 2.0), (5.0, 2.0, 1.0)]
        out = normalize_unit_cube(pts)
        for p in out:
            for c in p:
                self.assertLessEqual(abs(c), 0.5 + 1e-9)

    def test_centered(self):
        pts = [(0.0, 0.0, 0.0), (10.0, 10.0, 10.0)]
        out = normalize_unit_cube(pts)
        self.assertAlmostEqual(out[0][0], -0.5)
        self.assertAlmostEqual(out[1][0], 0.5)

    def test_degenerate(self):
        out = normalize_unit_cube([(3.0, 3.0, 3.0)])
        self.assertEqual(out, [(0.0, 0.0, 0.0)])

    def test_empty(self):
        with self.assertRaises(ValueError):
            normalize_unit_cube([])


class FpsTest(unittest.TestCase):
    def test_count(self):
        pts = [(float(i), 0.0, 0.0) for i in range(20)]
        sampled, idx = furthest_point_sampling(pts, 5, seed=0)
        self.assertEqual(len(sampled), 5)
        self.assertEqual(len(set(idx)), 5)

    def test_picks_extremes_first(self):
        # a line of points; FPS from index 0 should pick the far end next.
        pts = [(float(i), 0.0, 0.0) for i in range(10)]
        _, idx = furthest_point_sampling(pts, 2, seed=0)
        self.assertEqual(idx, [0, 9])

    def test_deterministic(self):
        pts = [(float(i), float(i % 3), 0.0) for i in range(30)]
        a = furthest_point_sampling(pts, 8, seed=3)
        b = furthest_point_sampling(pts, 8, seed=3)
        self.assertEqual(a, b)

    def test_k_exceeds_cloud(self):
        pts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        sampled, idx = furthest_point_sampling(pts, 10, seed=0)
        self.assertEqual(len(sampled), 2)

    def test_default_num_points(self):
        pts = [(float(i), float((i * 7) % 11), float((i * 3) % 5))
               for i in range(500)]
        out = prepare_point_input(pts)
        self.assertEqual(len(out), NUM_POINTS)
        for p in out:
            for c in p:
                self.assertLessEqual(abs(c), 0.5 + 1e-9)


if __name__ == "__main__":
    unittest.main()
