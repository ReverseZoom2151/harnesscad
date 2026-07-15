"""Tests for planar section properties (Open CAD Studio MASSPROP port)."""

import math
import unittest

from harnesscad.domain.geometry.sketch import section_properties as sp


RECT = [(0.0, 0.0), (60.0, 0.0), (60.0, 40.0), (0.0, 40.0)]


class BasicTest(unittest.TestCase):
    def test_area(self):
        self.assertAlmostEqual(sp.polygon_area(RECT), 2400.0)

    def test_perimeter(self):
        self.assertAlmostEqual(sp.perimeter(RECT), 200.0)

    def test_centroid(self):
        cx, cy = sp.centroid(RECT)
        self.assertAlmostEqual(cx, 30.0)
        self.assertAlmostEqual(cy, 20.0)

    def test_closed_input_equivalent(self):
        closed = RECT + [RECT[0]]
        self.assertAlmostEqual(sp.polygon_area(closed), 2400.0)

    def test_winding_independent(self):
        cw = list(reversed(RECT))
        self.assertAlmostEqual(sp.polygon_area(cw), 2400.0)

    def test_degenerate_raises(self):
        with self.assertRaises(sp.SectionError):
            sp.polygon_area([(0, 0), (1, 1)])


class MomentTest(unittest.TestCase):
    def test_rectangle_moments(self):
        ixx, iyy, ixy = sp.area_moments(RECT)
        # b*h^3/12 about centroid: width=60 (b for Iyy), height=40.
        self.assertAlmostEqual(ixx, 60.0 * 40.0 ** 3 / 12.0)
        self.assertAlmostEqual(iyy, 40.0 * 60.0 ** 3 / 12.0)
        self.assertAlmostEqual(ixy, 0.0, places=6)

    def test_moments_positive_for_cw(self):
        ixx, iyy, ixy = sp.area_moments(list(reversed(RECT)))
        self.assertGreater(ixx, 0)
        self.assertGreater(iyy, 0)

    def test_principal_of_symmetric_equals_axes(self):
        ixx, iyy, ixy = sp.area_moments(RECT)
        i1, i2, theta = sp.principal_moments(ixx, iyy, ixy)
        self.assertAlmostEqual(i1, max(ixx, iyy))
        self.assertAlmostEqual(i2, min(ixx, iyy))


class SectionTest(unittest.TestCase):
    def test_full_report(self):
        s = sp.section_properties(RECT)
        self.assertAlmostEqual(s.area, 2400.0)
        self.assertAlmostEqual(s.centroid[0], 30.0)
        self.assertAlmostEqual(s.ixx, 60.0 * 40.0 ** 3 / 12.0)
        # section modulus about x = Ixx / (h/2)
        self.assertAlmostEqual(s.section_modulus_x(), (60.0 * 40.0 ** 3 / 12.0) / 20.0)

    def test_radius_of_gyration(self):
        s = sp.section_properties(RECT)
        # rg_x = sqrt(Ixx/A) = sqrt((h^2/12)) = h/sqrt(12)
        self.assertAlmostEqual(s.rg_x, 40.0 / math.sqrt(12.0))

    def test_hole_subtracts_area(self):
        hole = [(25.0, 15.0), (35.0, 15.0), (35.0, 25.0), (25.0, 25.0)]
        s = sp.section_properties(RECT, holes=[hole])
        self.assertAlmostEqual(s.area, 2400.0 - 100.0)
        # Centroid stays at plate centre because hole is centred.
        self.assertAlmostEqual(s.centroid[0], 30.0, places=6)
        self.assertAlmostEqual(s.centroid[1], 20.0, places=6)
        # Hole reduces Ixx below the solid value.
        self.assertLess(s.ixx, 60.0 * 40.0 ** 3 / 12.0)

    def test_hole_too_big_raises(self):
        big = [(-100, -100), (100, -100), (100, 100), (-100, 100)]
        with self.assertRaises(sp.SectionError):
            sp.section_properties(RECT, holes=[big])


class OffsetTest(unittest.TestCase):
    def test_parallel_axis(self):
        # Moments about the origin corner should exceed centroidal by A*d^2.
        ixx_c, _, _ = sp.area_moments(RECT)
        ixx_o, _, _ = sp.area_moments(RECT, about=(0.0, 0.0))
        self.assertAlmostEqual(ixx_o, ixx_c + 2400.0 * 20.0 ** 2)


if __name__ == "__main__":
    unittest.main()
