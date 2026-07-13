"""Tests for geometry.proccad_beauty_functionals."""

import unittest
from math import cos, sin, tau

from harnesscad.domain.geometry.parametric.beauty_functionals import (
    StyleSheet,
    apply_to_region,
    arc_length,
    bending_energy,
    discrete_curvature,
    minimum_variation,
)


def regular_polygon(n, radius=1.0):
    return [(radius * cos(tau * i / n), radius * sin(tau * i / n)) for i in range(n)]


class ArcLengthTest(unittest.TestCase):
    def test_open_line(self):
        self.assertAlmostEqual(arc_length([(0, 0), (3, 0), (3, 4)]), 3 + 4)

    def test_closed_square(self):
        sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
        self.assertAlmostEqual(arc_length(sq, closed=True), 4.0)


class CurvatureTest(unittest.TestCase):
    def test_straight_line_zero_curvature(self):
        k = discrete_curvature([(0, 0), (1, 0), (2, 0), (3, 0)])
        self.assertTrue(all(abs(v) < 1e-12 for v in k))

    def test_regular_polygon_constant_curvature(self):
        poly = regular_polygon(6)
        k = discrete_curvature(poly, closed=True)
        self.assertEqual(len(k), 6)
        for v in k:
            self.assertAlmostEqual(v, k[0], places=9)


class BendingEnergyTest(unittest.TestCase):
    def test_straight_line_zero(self):
        self.assertAlmostEqual(bending_energy([(0, 0), (1, 0), (2, 0)]), 0.0)

    def test_bent_curve_positive(self):
        self.assertGreater(bending_energy([(0, 0), (1, 0), (1, 1)]), 0.0)


class MinimumVariationTest(unittest.TestCase):
    def test_circle_has_zero_variation(self):
        # constant curvature -> MVS penalty ~ 0 (the paper's key premise)
        poly = regular_polygon(24)
        self.assertLess(minimum_variation(poly, closed=True), 1e-9)

    def test_straight_line_zero(self):
        self.assertAlmostEqual(minimum_variation([(0, 0), (1, 0), (2, 0), (3, 0)]), 0.0)

    def test_varying_curvature_positive(self):
        # a curve whose curvature changes sharply
        curve = [(0, 0), (1, 0), (1, 1), (3, 1), (3, 4)]
        self.assertGreater(minimum_variation(curve), 0.0)


class StyleSheetTest(unittest.TestCase):
    def test_builtin_names(self):
        ss = StyleSheet()
        self.assertIn("min_variation", ss.names())
        self.assertIn("bending", ss.names())
        self.assertIn("minimal", ss.names())

    def test_evaluate_minimal(self):
        ss = StyleSheet()
        self.assertAlmostEqual(ss.evaluate("minimal", [(0, 0), (2, 0)]), 2.0)

    def test_unknown_style(self):
        with self.assertRaises(KeyError):
            StyleSheet().evaluate("nope", [(0, 0), (1, 0)])

    def test_register_custom(self):
        ss = StyleSheet()
        ss.register("count", lambda c, cl: float(len(c)))
        self.assertEqual(ss.evaluate("count", [(0, 0), (1, 0), (2, 0)]), 3.0)

    def test_apply_to_region(self):
        ss = StyleSheet()
        curve = [(0, 0), (1, 0), (2, 0), (2, 5)]
        # region [0:3] is a straight line -> minimal (length) = 2
        self.assertAlmostEqual(apply_to_region(ss, "minimal", curve, 0, 3), 2.0)

    def test_apply_to_region_bad_range(self):
        ss = StyleSheet()
        with self.assertRaises(ValueError):
            apply_to_region(ss, "minimal", [(0, 0), (1, 0)], 1, 1)


if __name__ == "__main__":
    unittest.main()
