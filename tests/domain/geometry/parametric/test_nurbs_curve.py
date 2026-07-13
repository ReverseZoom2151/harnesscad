"""Tests for geometry.nurbgen_curve (NURBS curve evaluation/tessellation)."""

import math
import unittest

from harnesscad.domain.geometry.parametric import nurbs_curve as nc
from harnesscad.domain.numeric import nurbs_basis as nb


class TestStraightLine(unittest.TestCase):
    def test_degree1_reproduces_line(self):
        # A degree-1 B-spline with unit weights is the control polygon itself.
        cps = [(0.0, 0.0), (1.0, 2.0), (3.0, 2.0)]
        w = [1.0, 1.0, 1.0]
        knots = nb.uniform_clamped_knots(2, 1)  # [0,0,0.5?...]; n=2,p=1
        # Midpoint of the domain lies on the polyline.
        p0 = nc.curve_point(cps, w, 1, knots, knots[1])
        self.assertAlmostEqual(p0[0], 0.0, places=12)
        self.assertAlmostEqual(p0[1], 0.0, places=12)

    def test_collinear_control_points_stay_collinear(self):
        cps = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]
        w = [1.0, 1.0, 1.0]
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        for k in range(11):
            u = k / 10.0
            x, y, z = nc.curve_point(cps, w, 2, knots, u)
            self.assertAlmostEqual(x, y, places=12)
            self.assertAlmostEqual(y, z, places=12)


class TestEndpointInterpolation(unittest.TestCase):
    def test_clamped_curve_hits_end_control_points(self):
        cps = [(0.0, 0.0), (1.0, 3.0), (4.0, 3.0), (5.0, 0.0)]
        w = [1.0, 2.0, 0.5, 1.0]
        knots = nb.uniform_clamped_knots(3, 3)  # Bezier segment
        p_start = nc.curve_point(cps, w, 3, knots, knots[3])
        p_end = nc.curve_point(cps, w, 3, knots, knots[4])
        self.assertAlmostEqual(p_start[0], 0.0, places=12)
        self.assertAlmostEqual(p_start[1], 0.0, places=12)
        self.assertAlmostEqual(p_end[0], 5.0, places=12)
        self.assertAlmostEqual(p_end[1], 0.0, places=12)


class TestNurbsCircle(unittest.TestCase):
    def test_quarter_circle_points_on_radius(self):
        cps, w, p, knots = nc.nurbs_circle_quadrant(2.0)
        for k in range(21):
            u = k / 20.0
            x, y = nc.curve_point(cps, w, p, knots, u)
            self.assertAlmostEqual(math.hypot(x, y), 2.0, places=10)

    def test_quarter_circle_midpoint(self):
        cps, w, p, knots = nc.nurbs_circle_quadrant(1.0)
        x, y = nc.curve_point(cps, w, p, knots, 0.5)
        # 45-degree point on the unit circle.
        self.assertAlmostEqual(x, math.sqrt(2) / 2, places=10)
        self.assertAlmostEqual(y, math.sqrt(2) / 2, places=10)


class TestDerivatives(unittest.TestCase):
    def test_tangent_perpendicular_to_radius_on_circle(self):
        cps, w, p, knots = nc.nurbs_circle_quadrant(1.0)
        for u in (0.2, 0.5, 0.8):
            pt = nc.curve_point(cps, w, p, knots, u)
            t = nc.curve_tangent(cps, w, p, knots, u)
            # Radius . tangent == 0 for a circle centred at the origin.
            dot = pt[0] * t[0] + pt[1] * t[1]
            self.assertAlmostEqual(dot, 0.0, places=8)

    def test_derivative_matches_finite_difference(self):
        cps = [(0.0, 0.0), (1.0, 3.0), (4.0, 3.0), (5.0, 0.0)]
        w = [1.0, 2.0, 0.5, 1.0]
        knots = nb.uniform_clamped_knots(3, 3)
        u, h = 0.4, 1e-6
        d1 = nc.curve_derivatives(cps, w, 3, knots, u, 1)[1]
        fwd = nc.curve_point(cps, w, 3, knots, u + h)
        bwd = nc.curve_point(cps, w, 3, knots, u - h)
        for c in range(2):
            fd = (fwd[c] - bwd[c]) / (2 * h)
            self.assertAlmostEqual(d1[c], fd, places=5)

    def test_zeroth_derivative_is_point(self):
        cps, w, p, knots = nc.nurbs_circle_quadrant(1.0)
        d = nc.curve_derivatives(cps, w, p, knots, 0.3, 2)
        pt = nc.curve_point(cps, w, p, knots, 0.3)
        self.assertAlmostEqual(d[0][0], pt[0], places=12)
        self.assertAlmostEqual(d[0][1], pt[1], places=12)


class TestTessellation(unittest.TestCase):
    def test_polyline_count_and_endpoints(self):
        cps, w, p, knots = nc.nurbs_circle_quadrant(1.0)
        poly = nc.tessellate_curve(cps, w, p, knots, samples=16)
        self.assertEqual(len(poly), 17)
        self.assertAlmostEqual(poly[0][0], 1.0, places=10)
        self.assertAlmostEqual(poly[-1][1], 1.0, places=10)

    def test_polyline_length_approaches_arc_length(self):
        cps, w, p, knots = nc.nurbs_circle_quadrant(1.0)
        poly = nc.tessellate_curve(cps, w, p, knots, samples=200)
        arc = nc.polyline_length(poly)
        # Quarter of unit circle circumference = pi/2.
        self.assertAlmostEqual(arc, math.pi / 2, places=4)


class TestValidation(unittest.TestCase):
    def test_negative_weight_rejected(self):
        with self.assertRaises(ValueError):
            nc.curve_point([(0.0, 0.0), (1.0, 0.0)], [1.0, -1.0], 1,
                           [0.0, 0.0, 1.0, 1.0], 0.5)

    def test_bad_knot_length_rejected(self):
        with self.assertRaises(ValueError):
            nc.curve_point([(0.0, 0.0), (1.0, 0.0)], [1.0, 1.0], 1,
                           [0.0, 1.0], 0.5)


if __name__ == "__main__":
    unittest.main()
