"""Tests for UV-Net edge U-grid sampling (validated against analytic curves)."""

import math
import unittest

from harnesscad.domain.geometry import uvnet_u_grid as ug
from harnesscad.domain.geometry import uvnet_uv_grid as uvg


class LineTest(unittest.TestCase):
    def test_points_and_tangent(self):
        line = ug.Line(origin=(1.0, 0.0, 0.0), direction=(0.0, 2.0, 0.0),
                       u_range=(0.0, 1.0))
        grid = ug.edge_feature_grid(line, 5)
        self.assertEqual(len(grid), 5)
        self.assertEqual(len(grid[0]), 6)
        self.assertAlmostEqual(grid[0][1], 0.0, places=12)
        self.assertAlmostEqual(grid[4][1], 2.0, places=12)
        for c in grid:
            self.assertAlmostEqual(c[4], 1.0, places=12)
        self.assertAlmostEqual(ug.grid_length(grid), 2.0, places=12)
        self.assertAlmostEqual(ug.tangent_turning(grid), 0.0, places=10)

    def test_determinism(self):
        line = ug.Line(origin=(0, 0, 0), direction=(1, 1, 1))
        self.assertEqual(ug.edge_feature_grid(line, 7),
                         ug.edge_feature_grid(line, 7))


class CircleTest(unittest.TestCase):
    def test_points_on_circle_and_tangent_orthogonal_to_radius(self):
        circ = ug.Circle(centre=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0),
                         radius=3.0)
        grid = ug.edge_feature_grid(circ, 16)
        for c in grid:
            p = (c[0], c[1], c[2])
            t = (c[3], c[4], c[5])
            self.assertAlmostEqual(math.hypot(p[0], p[1]), 3.0, places=10)
            self.assertAlmostEqual(p[2], 0.0, places=12)
            self.assertAlmostEqual(uvg._norm(t), 1.0, places=12)
            self.assertAlmostEqual(uvg._dot(t, p), 0.0, places=10)

    def test_grid_length_approximates_circumference(self):
        circ = ug.Circle(centre=(0, 0, 0), axis=(0, 0, 1), radius=1.0)
        length = ug.grid_length(ug.edge_feature_grid(circ, 200))
        self.assertAlmostEqual(length, 2 * math.pi, places=3)

    def test_turning_of_quarter_arc(self):
        arc = ug.Circle(centre=(0, 0, 0), axis=(0, 0, 1), radius=2.0,
                        u_range=(0.0, math.pi / 2))
        turning = ug.tangent_turning(ug.edge_feature_grid(arc, 64))
        self.assertAlmostEqual(turning, math.pi / 2, places=6)


class EllipseTest(unittest.TestCase):
    def test_points_satisfy_ellipse_equation(self):
        ell = ug.Ellipse(centre=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0),
                         major_radius=4.0, minor_radius=2.0,
                         ref_dir=(1.0, 0.0, 0.0))
        for c in ug.edge_feature_grid(ell, 12):
            x, y = c[0], c[1]
            self.assertAlmostEqual((x / 4.0) ** 2 + (y / 2.0) ** 2, 1.0,
                                   places=10)

    def test_tangent_is_unit(self):
        ell = ug.Ellipse(centre=(0, 0, 0), axis=(0, 0, 1),
                         major_radius=3.0, minor_radius=1.0)
        for c in ug.edge_feature_grid(ell, 9):
            self.assertAlmostEqual(uvg._norm((c[3], c[4], c[5])), 1.0, places=12)


class PolylineTest(unittest.TestCase):
    def test_chord_length_parameterisation(self):
        poly = ug.Polyline(points=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                   (1.0, 2.0, 0.0)])
        self.assertEqual(poly.domain(), (0.0, 3.0))
        self.assertEqual(poly.point(0.5), (0.5, 0.0, 0.0))
        self.assertEqual(poly.point(2.0), (1.0, 1.0, 0.0))
        self.assertEqual(poly.tangent(2.0), (0.0, 1.0, 0.0))
        grid = ug.edge_feature_grid(poly, 7)
        self.assertAlmostEqual(ug.grid_length(grid), 3.0, places=10)


class BSplineCurveTest(unittest.TestCase):
    def test_degree_one_bspline_is_a_line(self):
        curve = ug.BSplineCurve(control_points=[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
                                weights=[1.0, 1.0], degree=1,
                                knots=[0.0, 0.0, 1.0, 1.0])
        self.assertEqual(curve.domain(), (0.0, 1.0))
        grid = ug.edge_feature_grid(curve, 5)
        self.assertAlmostEqual(grid[-1][0], 2.0, places=10)
        for c in grid:
            self.assertAlmostEqual(c[3], 1.0, places=10)

    def test_nurbs_circle_quadrant_lies_on_unit_circle(self):
        poles, weights, degree, knots = nurbs_quadrant()
        curve = ug.BSplineCurve(poles, weights, degree, knots)
        for c in ug.edge_feature_grid(curve, 9):
            self.assertAlmostEqual(math.hypot(c[0], c[1]), 1.0, places=9)


def nurbs_quadrant():
    """The rational quadratic quarter-circle, lifted to 3D."""
    poles = [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    weights = [1.0, math.sqrt(2.0) / 2.0, 1.0]
    knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    return poles, weights, 2, knots


class DegenerateTest(unittest.TestCase):
    def test_zero_length_edge_is_degenerate(self):
        poly = ug.Polyline(points=[(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)])
        self.assertTrue(ug.is_degenerate(poly))

    def test_collapsed_parameter_range_is_degenerate(self):
        line = ug.Line(origin=(0, 0, 0), direction=(1, 0, 0), u_range=(2.0, 2.0))
        self.assertTrue(ug.is_degenerate(line))

    def test_regular_edges_are_not_degenerate(self):
        self.assertFalse(ug.is_degenerate(
            ug.Line(origin=(0, 0, 0), direction=(1, 0, 0))))
        self.assertFalse(ug.is_degenerate(
            ug.Circle(centre=(0, 0, 0), axis=(0, 0, 1), radius=1.0)))


class ReverseAndMethodTest(unittest.TestCase):
    def test_reverse_grid(self):
        line = ug.Line(origin=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
        grid = ug.edge_feature_grid(line, 3)
        rev = ug.reverse_grid(grid)
        self.assertAlmostEqual(rev[0][0], 1.0, places=12)
        self.assertAlmostEqual(rev[-1][0], 0.0, places=12)
        for c in rev:
            self.assertAlmostEqual(c[3], -1.0, places=12)
        self.assertEqual(ug.reverse_grid(rev), grid)

    def test_u_grid_channels(self):
        line = ug.Line(origin=(0, 0, 0), direction=(0, 0, 1))
        self.assertEqual(ug.u_grid(line, 3, method=ug.PARAMETER),
                         [0.0, 0.5, 1.0])
        self.assertEqual(len(ug.u_grid(line, 3, method=ug.TANGENT)), 3)
        with self.assertRaises(ValueError):
            ug.u_grid(line, 3, method="colour")


if __name__ == "__main__":
    unittest.main()
