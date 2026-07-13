"""Tests for geometry.nurbgen_knot_insertion (Boehm knot insertion)."""

import math
import unittest

from harnesscad.domain.geometry import nurbgen_curve as nc
from harnesscad.domain.geometry import nurbgen_knot_insertion as ki
from harnesscad.domain.numeric import nurbs_basis as nb


def _sample_curve():
    cps = [(0.0, 0.0), (1.0, 3.0), (3.0, 3.0), (4.0, 0.0), (6.0, -2.0)]
    w = [1.0, 1.0, 1.0, 1.0, 1.0]
    knots = nb.uniform_clamped_knots(4, 3)
    return cps, w, 3, knots


class TestShapePreservation(unittest.TestCase):
    def test_insert_does_not_change_curve(self):
        cps, w, p, U = _sample_curve()
        u_ins = 0.5
        cps2, w2, p2, U2 = ki.insert_knot(cps, w, p, U, u_ins, 1)
        # One extra control point and one extra knot.
        self.assertEqual(len(cps2), len(cps) + 1)
        self.assertEqual(len(U2), len(U) + 1)
        self.assertEqual(p2, p)
        # Curve evaluates identically everywhere.
        for k in range(21):
            u = k / 20.0
            a = nc.curve_point(cps, w, p, U, u)
            b = nc.curve_point(cps2, w2, p2, U2, u)
            self.assertAlmostEqual(a[0], b[0], places=10)
            self.assertAlmostEqual(a[1], b[1], places=10)

    def test_insert_preserves_rational_circle(self):
        cps, w, p, U = nc.nurbs_circle_quadrant(1.0)
        cps2, w2, p2, U2 = ki.insert_knot(cps, w, p, U, 0.5, 1)
        for k in range(21):
            u = k / 20.0
            x, y = nc.curve_point(cps2, w2, p2, U2, u)
            self.assertAlmostEqual(math.hypot(x, y), 1.0, places=9)


class TestRefinement(unittest.TestCase):
    def test_refine_multiple_knots(self):
        cps, w, p, U = _sample_curve()
        new = [0.25, 0.5, 0.75]
        cps2, w2, p2, U2 = ki.refine_knots(cps, w, p, U, new)
        self.assertEqual(len(cps2), len(cps) + len(new))
        for k in range(11):
            u = k / 10.0
            a = nc.curve_point(cps, w, p, U, u)
            b = nc.curve_point(cps2, w2, p2, U2, u)
            self.assertAlmostEqual(a[0], b[0], places=10)
            self.assertAlmostEqual(a[1], b[1], places=10)


class TestMultiplicity(unittest.TestCase):
    def test_insert_twice_raises_multiplicity(self):
        cps, w, p, U = _sample_curve()
        cps2, w2, p2, U2 = ki.insert_knot(cps, w, p, U, 0.5, 2)
        self.assertEqual(ki.knot_span_multiplicity(U2, 0.5), 3)  # was 1, +2
        self.assertEqual(len(cps2), len(cps) + 2)

    def test_exceed_degree_rejected(self):
        cps, w, p, U = _sample_curve()  # degree 3, interior mult 1
        with self.assertRaises(ValueError):
            ki.insert_knot(cps, w, p, U, 0.5, 3)  # 1 + 3 > 3

    def test_decompose_to_bezier_c0(self):
        cps, w, p, U = _sample_curve()
        u = ki.distinct_interior_knots(U, p)[0]
        cps2, w2, p2, U2 = ki.decompose_span_to_bezier(cps, w, p, U, u)
        self.assertEqual(ki.knot_span_multiplicity(U2, u), p)
        for k in range(11):
            uu = k / 10.0
            a = nc.curve_point(cps, w, p, U, uu)
            b = nc.curve_point(cps2, w2, p2, U2, uu)
            self.assertAlmostEqual(a[0], b[0], places=10)


class TestInteriorKnots(unittest.TestCase):
    def test_distinct_interior(self):
        U = nb.uniform_clamped_knots(4, 3)  # one interior knot
        interior = ki.distinct_interior_knots(U, 3)
        self.assertEqual(len(interior), 1)


if __name__ == "__main__":
    unittest.main()
