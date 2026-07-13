"""Tests for geometry.dreamcad_rational_bezier."""

import unittest

from harnesscad.domain.geometry.dreamcad_rational_bezier import (
    bernstein,
    bernstein_basis,
    bernstein_derivative,
    bezier_surface_derivatives,
    bezier_surface_normal,
    bezier_surface_point,
    bilinear_quad_grid,
    bounded_deform,
    de_casteljau,
    softplus_weight,
    unit_weight_grid,
)


def _flat_grid(nx=3, ny=3, size=1.0, z=0.0):
    return [[(size * i / nx, size * j / ny, z) for j in range(ny + 1)]
            for i in range(nx + 1)]


class TestBernstein(unittest.TestCase):
    def test_partition_of_unity(self):
        basis = bernstein_basis(3, 0.3)
        self.assertAlmostEqual(sum(basis), 1.0)

    def test_endpoints(self):
        self.assertAlmostEqual(bernstein(3, 0, 0.0), 1.0)
        self.assertAlmostEqual(bernstein(3, 3, 1.0), 1.0)
        self.assertAlmostEqual(bernstein(3, 1, 0.0), 0.0)

    def test_derivative_sums_to_zero(self):
        derivs = [bernstein_derivative(3, i, 0.4) for i in range(4)]
        self.assertAlmostEqual(sum(derivs), 0.0)

    def test_derivative_finite_difference(self):
        h = 1e-6
        analytic = bernstein_derivative(4, 2, 0.5)
        numeric = (bernstein(4, 2, 0.5 + h) - bernstein(4, 2, 0.5 - h)) / (2 * h)
        self.assertAlmostEqual(analytic, numeric, places=5)

    def test_out_of_range(self):
        with self.assertRaises(ValueError):
            bernstein(3, 4, 0.5)
        with self.assertRaises(ValueError):
            bernstein(3, 0, 1.5)


class TestDeCasteljau(unittest.TestCase):
    def test_matches_endpoints(self):
        pts = [(0.0, 0.0), (1.0, 2.0), (2.0, 0.0)]
        self.assertEqual(de_casteljau(pts, 0.0), (0.0, 0.0))
        self.assertEqual(de_casteljau(pts, 1.0), (2.0, 0.0))

    def test_midpoint(self):
        pts = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
        mid = de_casteljau(pts, 0.5)
        self.assertAlmostEqual(mid[0], 0.5)
        self.assertAlmostEqual(mid[1], 0.75)


class TestSurfaceEval(unittest.TestCase):
    def test_corner_interpolation(self):
        grid = _flat_grid()
        w = unit_weight_grid()
        self.assertAlmostEqual(bezier_surface_point(grid, w, 0.0, 0.0)[0], 0.0)
        p = bezier_surface_point(grid, w, 1.0, 1.0)
        self.assertAlmostEqual(p[0], 1.0)
        self.assertAlmostEqual(p[1], 1.0)

    def test_flat_patch_is_planar(self):
        grid = _flat_grid(z=0.0)
        w = unit_weight_grid()
        for u, v in [(0.25, 0.75), (0.5, 0.5), (0.1, 0.9)]:
            self.assertAlmostEqual(bezier_surface_point(grid, w, u, v)[2], 0.0)

    def test_weight_pulls_surface(self):
        grid = _flat_grid(z=0.0)
        grid[1][1] = (grid[1][1][0], grid[1][1][1], 1.0)
        w_low = unit_weight_grid()
        w_high = unit_weight_grid()
        w_high[1][1] = 10.0
        z_low = bezier_surface_point(grid, w_low, 0.4, 0.4)[2]
        z_high = bezier_surface_point(grid, w_high, 0.4, 0.4)[2]
        self.assertGreater(z_high, z_low)

    def test_negative_weight_rejected(self):
        grid = _flat_grid()
        w = unit_weight_grid()
        w[0][0] = -1.0
        with self.assertRaises(ValueError):
            bezier_surface_point(grid, w, 0.5, 0.5)

    def test_normal_of_flat_patch(self):
        grid = _flat_grid(z=0.0)
        w = unit_weight_grid()
        normal = bezier_surface_normal(grid, w, 0.5, 0.5)
        self.assertAlmostEqual(abs(normal[2]), 1.0)
        self.assertAlmostEqual(normal[0], 0.0)
        self.assertAlmostEqual(normal[1], 0.0)

    def test_derivative_finite_difference(self):
        grid = _flat_grid(z=0.0)
        grid[2][1] = (grid[2][1][0], grid[2][1][1], 0.5)
        w = unit_weight_grid()
        s_u, s_v = bezier_surface_derivatives(grid, w, 0.5, 0.5)
        h = 1e-6
        pu_plus = bezier_surface_point(grid, w, 0.5 + h, 0.5)
        pu_minus = bezier_surface_point(grid, w, 0.5 - h, 0.5)
        for d in range(3):
            numeric = (pu_plus[d] - pu_minus[d]) / (2 * h)
            self.assertAlmostEqual(s_u[d], numeric, places=4)


class TestInitAndTransforms(unittest.TestCase):
    def test_bilinear_quad_grid_corners(self):
        corners = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                   (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        grid = bilinear_quad_grid(corners)
        self.assertEqual(len(grid), 4)
        self.assertEqual(len(grid[0]), 4)
        self.assertEqual(grid[0][0], (0.0, 0.0, 0.0))
        self.assertEqual(grid[3][3], (1.0, 1.0, 0.0))
        # centre control point of a unit square is at (0.?, 0.?)
        mid = grid[1][1]
        self.assertAlmostEqual(mid[0], 1 / 3)
        self.assertAlmostEqual(mid[1], 1 / 3)

    def test_softplus_positive(self):
        self.assertGreater(softplus_weight(-5.0), 0.0)
        self.assertGreater(softplus_weight(5.0), 0.0)
        self.assertGreater(softplus_weight(50.0), softplus_weight(0.0))

    def test_bounded_deform_range(self):
        moved = bounded_deform((0.0, 0.0, 0.0), (100.0, -100.0, 0.0))
        self.assertAlmostEqual(moved[0], 1.0, places=6)
        self.assertAlmostEqual(moved[1], -1.0, places=6)
        self.assertAlmostEqual(moved[2], 0.0)


if __name__ == "__main__":
    unittest.main()
