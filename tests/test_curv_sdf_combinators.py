"""Tests for geometry.curv_sdf_combinators.

Checks the min/max algebra, De Morgan relations, continuity of the smooth
minima (as the blend parameter -> 0 they converge to hard min), the invariant
``smooth_union <= hard union`` everywhere, and the chamfer bevel geometry.
"""

from __future__ import annotations

import math
import unittest

from harnesscad.domain.geometry.sdf import combinators as C


class TestHard(unittest.TestCase):
    def test_union_intersection(self):
        self.assertEqual(C.union(1.0, -2.0), -2.0)
        self.assertEqual(C.intersection(1.0, -2.0), 1.0)

    def test_difference(self):
        # subtract b from a: inside a AND outside b
        self.assertEqual(C.difference(-1.0, -0.5), 0.5)  # max(-1, 0.5)
        self.assertEqual(C.difference(-2.0, 3.0), -2.0)  # max(-2,-3)

    def test_demorgan(self):
        for a in (-3.0, -0.2, 0.0, 1.5):
            for b in (-2.0, 0.1, 4.0):
                # complement(union) == intersection(complement, complement)
                self.assertAlmostEqual(
                    C.complement(C.union(a, b)),
                    C.intersection(C.complement(a), C.complement(b)),
                )

    def test_nary_identity(self):
        self.assertEqual(C.union_all([]), float("inf"))
        self.assertEqual(C.intersection_all([]), float("-inf"))
        self.assertEqual(C.union_all([3.0, -1.0, 2.0]), -1.0)
        self.assertEqual(C.intersection_all([3.0, -1.0, 2.0]), 3.0)


class TestSmoothMin(unittest.TestCase):
    def test_converges_to_min(self):
        for a, b in [(1.0, 2.0), (-1.0, 0.5), (3.0, -2.0)]:
            self.assertAlmostEqual(C.smooth_min_poly(a, b, 1e-9), min(a, b), places=6)
            self.assertAlmostEqual(C.smooth_min_exp(a, b, 1e-6), min(a, b), places=4)

    def test_smooth_min_le_min(self):
        # the smooth minimum is always <= the hard minimum (it rounds downward)
        for a in (-2.0, -0.3, 0.0, 1.0, 2.5):
            for b in (-1.5, 0.2, 3.0):
                self.assertLessEqual(C.smooth_min_poly(a, b, 1.0), min(a, b) + 1e-12)
                self.assertLessEqual(C.smooth_min_exp(a, b, 0.5), min(a, b) + 1e-12)

    def test_continuity(self):
        # small change in input -> small change in output (Lipschitz-ish)
        f = lambda a: C.smooth_min_poly(a, 0.0, 1.0)
        self.assertLess(abs(f(0.5) - f(0.5001)), 1e-3)

    def test_exp_symmetry(self):
        self.assertAlmostEqual(C.smooth_min_exp(1.0, 2.0, 0.7),
                               C.smooth_min_exp(2.0, 1.0, 0.7))

    def test_power_min(self):
        # power smin requires positive args; near-min for small blend
        self.assertLessEqual(C.smooth_min_power(1.0, 2.0, 8.0), 1.0 + 1e-9)
        with self.assertRaises(ValueError):
            C.smooth_min_power(-1.0, 2.0, 4.0)

    def test_infinity_handling(self):
        # unioning with 'nothing' (+inf) leaves the field unchanged
        self.assertAlmostEqual(C.smooth_min_poly(float("inf"), 3.0, 1.0), 3.0)


class TestSmoothBooleans(unittest.TestCase):
    def test_smooth_union_le_hard_union(self):
        # sample two sphere-like fields over a grid; blend never removes material
        def f1(x):
            return abs(x) - 1.0

        def f2(x):
            return abs(x - 1.5) - 1.0

        for i in range(-30, 31):
            x = i / 10.0
            a, b = f1(x), f2(x)
            self.assertLessEqual(C.smooth_union(a, b, 0.6), C.union(a, b) + 1e-12)

    def test_smooth_self_union_is_offset(self):
        # Curv identity: smooth k .union [s, s] == offset (k/4) s == s - k/4
        for a in (-1.0, 0.0, 0.7, 2.0):
            self.assertAlmostEqual(C.smooth_union(a, a, 1.2), a - 1.2 / 4.0, places=12)

    def test_smooth_difference(self):
        # smoothly subtracting: result >= hard difference (rounds the crease)
        a, b = -0.5, -0.2
        self.assertGreaterEqual(C.smooth_difference(a, b, 0.4), C.difference(a, b) - 1e-9)


class TestChamfer(unittest.TestCase):
    def test_chamfer_min_reduces_to_min_when_far(self):
        # when |a-b| >= r there is no bevel
        self.assertEqual(C.chamfer_min(0.0, 5.0, 1.0), 0.0)

    def test_chamfer_bevel(self):
        # equal fields, bevel subtracts r/2
        self.assertAlmostEqual(C.chamfer_min(1.0, 1.0, 2.0), 1.0 - 1.0)

    def test_chamfer_le_min(self):
        for a in (-1.0, 0.0, 1.0):
            for b in (-0.5, 0.3, 2.0):
                self.assertLessEqual(C.chamfer_min(a, b, 0.8), min(a, b) + 1e-12)


if __name__ == "__main__":
    unittest.main()
