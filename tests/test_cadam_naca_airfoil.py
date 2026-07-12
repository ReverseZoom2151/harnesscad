"""Tests for geometry.cadam_naca_airfoil (NACA 4-digit airfoil geometry)."""

import math
import unittest

from geometry.cadam_naca_airfoil import (
    thickness,
    camber,
    camber_slope,
    surface_point,
    cosine_spacing,
    airfoil_polygon,
    scale_polygon,
    max_thickness_fraction,
)


class TestThickness(unittest.TestCase):
    def test_zero_at_endpoints(self):
        # Half-thickness is 0 at the leading edge; ~0 at the trailing edge.
        self.assertAlmostEqual(thickness(0.0, 0.12), 0.0)
        self.assertAlmostEqual(thickness(1.0, 0.12), 0.00126, places=4)

    def test_max_near_030(self):
        x, yt = max_thickness_fraction(0.12)
        self.assertTrue(0.25 <= x <= 0.35)
        # Full thickness ~= t; half-thickness ~= t/2.
        self.assertAlmostEqual(2 * yt, 0.12, places=2)

    def test_clamped_below_zero(self):
        # Negative x must not raise (sqrt clamp).
        self.assertEqual(thickness(-0.5, 0.12), 0.0)


class TestCamber(unittest.TestCase):
    def test_symmetric_airfoil_has_zero_camber(self):
        for x in (0.0, 0.25, 0.5, 0.75, 1.0):
            self.assertEqual(camber(x, 0.0, 0.0), 0.0)
            self.assertEqual(camber_slope(x, 0.0, 0.0), 0.0)

    def test_camber_peaks_at_p(self):
        m, p = 0.02, 0.40
        # Slope crosses zero at x = p (maximum of the camber line).
        self.assertAlmostEqual(camber_slope(p, m, p), 0.0, places=9)
        peak = camber(p, m, p)
        self.assertAlmostEqual(peak, m, places=9)  # yc(p) == m for 4-digit
        self.assertGreater(peak, camber(0.1, m, p))
        self.assertGreater(peak, camber(0.7, m, p))

    def test_continuity_at_p(self):
        m, p = 0.02, 0.40
        left = camber(p - 1e-9, m, p)
        right = camber(p + 1e-9, m, p)
        self.assertAlmostEqual(left, right, places=6)


class TestSurfacePoint(unittest.TestCase):
    def test_symmetric_upper_lower_mirror(self):
        # Symmetric airfoil: upper and lower are exact mirrors about y=0.
        up = surface_point(0.3, 0.0, 0.0, 0.12, upper=True)
        lo = surface_point(0.3, 0.0, 0.0, 0.12, upper=False)
        self.assertAlmostEqual(up[0], lo[0], places=9)
        self.assertAlmostEqual(up[1], -lo[1], places=9)
        self.assertGreater(up[1], 0.0)

    def test_cambered_upper_above_lower(self):
        up = surface_point(0.3, 0.02, 0.4, 0.12, upper=True)
        lo = surface_point(0.3, 0.02, 0.4, 0.12, upper=False)
        self.assertGreater(up[1], lo[1])


class TestSpacingAndPolygon(unittest.TestCase):
    def test_cosine_spacing_bounds(self):
        xs = cosine_spacing(10)
        self.assertEqual(len(xs), 11)
        self.assertAlmostEqual(xs[0], 0.0)
        self.assertAlmostEqual(xs[-1], 1.0)
        # Denser near the leading edge than mid-chord.
        self.assertLess(xs[1] - xs[0], xs[6] - xs[5])

    def test_spacing_monotonic(self):
        xs = cosine_spacing(20)
        for a, b in zip(xs, xs[1:]):
            self.assertLess(a, b)

    def test_polygon_point_count(self):
        # upper (n+1) + lower (n-1) = 2n points, no duplicated endpoints.
        n = 40
        poly = airfoil_polygon(0.02, 0.4, 0.12, n=n)
        self.assertEqual(len(poly), 2 * n)

    def test_polygon_no_duplicate_consecutive(self):
        poly = airfoil_polygon(0.02, 0.4, 0.12, n=30)
        for a, b in zip(poly, poly[1:]):
            self.assertNotEqual(a, b)

    def test_polygon_deterministic(self):
        a = airfoil_polygon(0.02, 0.4, 0.12, n=25)
        b = airfoil_polygon(0.02, 0.4, 0.12, n=25)
        self.assertEqual(a, b)

    def test_scale_polygon(self):
        poly = airfoil_polygon(0.0, 0.0, 0.12, n=10)
        scaled = scale_polygon(poly, 120.0)
        for (px, py), (sx, sy) in zip(poly, scaled):
            self.assertAlmostEqual(sx, px * 120.0)
            self.assertAlmostEqual(sy, py * 120.0)

    def test_invalid_n(self):
        with self.assertRaises(ValueError):
            airfoil_polygon(0.0, 0.0, 0.12, n=1)
        with self.assertRaises(ValueError):
            cosine_spacing(0)


if __name__ == "__main__":
    unittest.main()
