"""Tests for geometry.sdf.rounded_csg (ImplicitCAD rmax/rmin circular fillets).

Checks that the rounded operators (1) reduce to hard max/min when the radius is
0 or the fields are farther apart than the radius, (2) round *inward* on the
blend (a fillet only ever adds material to a union / removes it from an
intersection relative to the hard result, bounded by the radius), and (3) match
the exact ImplicitCAD closed form at the symmetric point x == y.
"""

from __future__ import annotations

import math
import unittest

from harnesscad.domain.geometry.sdf import rounded_csg as R


class TestReduceToHard(unittest.TestCase):
    def test_zero_radius_is_hard(self):
        self.assertEqual(R.rmax(0.0, 3.0, -1.0), 3.0)
        self.assertEqual(R.rmin(0.0, 3.0, -1.0), -1.0)

    def test_far_apart_is_hard(self):
        # |x - y| = 5 >= r = 1 -> plain max/min.
        self.assertEqual(R.rmax(1.0, 4.0, -1.0), 4.0)
        self.assertEqual(R.rmin(1.0, 4.0, -1.0), -1.0)

    def test_negative_radius_rejected(self):
        with self.assertRaises(ValueError):
            R.rmax(-1.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            R.rmin(-1.0, 0.0, 0.0)


class TestFilletDirection(unittest.TestCase):
    def test_rmax_at_equal_is_hard_plus_offset(self):
        # ImplicitCAD closed form at x == y: y - r*sin(pi/4) + r.
        r = 2.0
        val = R.rmax(r, 1.0, 1.0)
        expected = 1.0 - r * math.sin(math.pi / 4.0) + r
        self.assertAlmostEqual(val, expected, places=12)
        # a rounded intersection corner bulges outward: rmax >= max here.
        self.assertGreaterEqual(val, 1.0)

    def test_rmin_at_equal_is_hard_minus_offset(self):
        r = 2.0
        val = R.rmin(r, 1.0, 1.0)
        expected = 1.0 + r * math.sin(math.pi / 4.0) - r
        self.assertAlmostEqual(val, expected, places=12)
        # a rounded union corner is pulled inward: rmin <= min here.
        self.assertLessEqual(val, 1.0)

    def test_rounded_within_radius_of_hard(self):
        # the fillet never departs from the hard result by more than the radius.
        r = 1.0
        for x, y in [(0.3, -0.2), (0.0, 0.5), (-0.4, 0.4)]:
            self.assertLessEqual(abs(R.rmax(r, x, y) - max(x, y)), r)
            self.assertLessEqual(abs(R.rmin(r, x, y) - min(x, y)), r)


class TestContinuity(unittest.TestCase):
    def test_rmax_continuous_at_band_edge(self):
        # at |x - y| -> r the rounded value must meet the hard max continuously.
        r = 1.0
        eps = 1e-9
        just_in = R.rmax(r, r - eps, 0.0)
        at_edge = max(r, 0.0)
        self.assertAlmostEqual(just_in, at_edge, places=6)

    def test_rmin_continuous_at_band_edge(self):
        r = 1.0
        eps = 1e-9
        just_in = R.rmin(r, -(r - eps), 0.0)
        at_edge = min(-(r), 0.0)
        self.assertAlmostEqual(just_in, at_edge, places=6)


class TestNary(unittest.TestCase):
    def test_empty_and_single(self):
        self.assertEqual(R.rmaximum(1.0, []), 0.0)
        self.assertEqual(R.rminimum(1.0, []), 0.0)
        self.assertEqual(R.rmaximum(1.0, [3.0]), 3.0)
        self.assertEqual(R.rminimum(1.0, [3.0]), 3.0)

    def test_rounds_only_extreme_pair(self):
        # third value is far below the top two -> result depends only on top pair.
        r = 1.0
        with_extra = R.rmaximum(r, [2.0, 2.3, -10.0])
        pair_only = R.rmax(r, 2.3, 2.0)
        self.assertAlmostEqual(with_extra, pair_only, places=12)

    def test_rminimum_extreme_pair(self):
        r = 1.0
        with_extra = R.rminimum(r, [-2.0, -2.3, 10.0])
        pair_only = R.rmin(r, -2.3, -2.0)
        self.assertAlmostEqual(with_extra, pair_only, places=12)


class TestNamedOperators(unittest.TestCase):
    def test_union_intersection_difference(self):
        a, b, r = 0.4, -0.2, 1.0
        self.assertEqual(R.rounded_union(r, a, b), R.rmin(r, a, b))
        self.assertEqual(R.rounded_intersection(r, a, b), R.rmax(r, a, b))
        self.assertEqual(R.rounded_difference(r, a, b), R.rmax(r, a, -b))
        self.assertEqual(R.rounded_complement(a), -a)

    def test_difference_removes_material(self):
        # subtracting b (inside, b<0) from a (inside, a<0) turns the point outside.
        self.assertGreater(R.rounded_difference(0.0, -0.5, -0.5), 0.0)


if __name__ == "__main__":
    unittest.main()
