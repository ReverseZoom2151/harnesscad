"""Tests for geometry.rlcad_revolve (Pappus solid-of-revolution geometry)."""

import math
import unittest

from harnesscad.domain.geometry.features import revolve as rr


class TestProfileMeasures(unittest.TestCase):
    def test_rectangle_area_and_centroid(self):
        # Rectangle r in [1,3], z in [0,4]: area 8, centroid (2, 2).
        rect = [(1.0, 0.0), (3.0, 0.0), (3.0, 4.0), (1.0, 4.0)]
        self.assertAlmostEqual(rr.profile_area(rect), 8.0)
        rc, zc = rr.area_centroid(rect)
        self.assertAlmostEqual(rc, 2.0)
        self.assertAlmostEqual(zc, 2.0)

    def test_area_sign_independent_of_orientation(self):
        rect = [(1.0, 0.0), (3.0, 0.0), (3.0, 4.0), (1.0, 4.0)]
        self.assertAlmostEqual(rr.profile_area(rect),
                               rr.profile_area(list(reversed(rect))))

    def test_closing_vertex_ignored(self):
        tri = [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (0.0, 0.0)]
        self.assertAlmostEqual(rr.profile_area(tri), 2.0)

    def test_perimeter_and_curve_centroid(self):
        rect = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
        self.assertAlmostEqual(rr.perimeter(rect), 8.0)
        rc, zc = rr.curve_centroid(rect)
        self.assertAlmostEqual(rc, 1.0)
        self.assertAlmostEqual(zc, 1.0)


class TestAxisCrossing(unittest.TestCase):
    def test_touching_axis_allowed(self):
        tri = [(0.0, 0.0), (2.0, 0.0), (0.0, 3.0)]
        self.assertFalse(rr.crosses_axis(tri))

    def test_negative_r_crosses(self):
        bad = [(-1.0, 0.0), (2.0, 0.0), (0.0, 3.0)]
        self.assertTrue(rr.crosses_axis(bad))

    def test_pappus_rejects_crossing(self):
        bad = [(-1.0, 0.0), (2.0, 0.0), (0.0, 3.0)]
        with self.assertRaises(ValueError):
            rr.pappus_volume(bad)


class TestPappusVolume(unittest.TestCase):
    def test_cylinder_exact(self):
        # Solid cylinder radius R=2, height h=5 -> rectangle [0,2]x[0,5].
        R, h = 2.0, 5.0
        rect = [(0.0, 0.0), (R, 0.0), (R, h), (0.0, h)]
        self.assertAlmostEqual(rr.pappus_volume(rect), math.pi * R * R * h)

    def test_cone_exact(self):
        # Cone radius R=3, height h=6 -> right triangle (0,0)(3,0)(0,6).
        R, h = 3.0, 6.0
        tri = [(0.0, 0.0), (R, 0.0), (0.0, h)]
        self.assertAlmostEqual(rr.pappus_volume(tri), math.pi * R * R * h / 3.0)

    def test_partial_revolution_scales(self):
        rect = [(0.0, 0.0), (2.0, 0.0), (2.0, 5.0), (0.0, 5.0)]
        full = rr.pappus_volume(rect, rr.FULL_TURN)
        half = rr.pappus_volume(rect, math.pi)
        self.assertAlmostEqual(half, full / 2.0)

    def test_annular_cylinder(self):
        # Tube inner r=1 outer r=3 height 4 -> pi*(9-1)*4.
        rect = [(1.0, 0.0), (3.0, 0.0), (3.0, 4.0), (1.0, 4.0)]
        self.assertAlmostEqual(rr.pappus_volume(rect),
                               math.pi * (9.0 - 1.0) * 4.0)


class TestPappusSurface(unittest.TestCase):
    def test_cylinder_total_surface(self):
        # Solid cylinder R=2 h=5: total = 2*pi*R*h + 2*pi*R^2.
        R, h = 2.0, 5.0
        rect = [(0.0, 0.0), (R, 0.0), (R, h), (0.0, h)]
        expected = 2 * math.pi * R * h + 2 * math.pi * R * R
        self.assertAlmostEqual(rr.pappus_surface_area(rect), expected)

    def test_cone_total_surface(self):
        # Cone R=3 h=4, slant l=5: total = pi*R*(R+l) = pi*3*8.
        R, h = 3.0, 4.0
        tri = [(0.0, 0.0), (R, 0.0), (0.0, h)]
        expected = math.pi * R * (R + 5.0)
        self.assertAlmostEqual(rr.pappus_surface_area(tri), expected)

    def test_partial_adds_caps(self):
        rect = [(1.0, 0.0), (3.0, 0.0), (3.0, 4.0), (1.0, 4.0)]
        area = rr.profile_area(rect)
        with_caps = rr.pappus_surface_area(rect, math.pi, include_caps=True)
        without = rr.pappus_surface_area(rect, math.pi, include_caps=False)
        self.assertAlmostEqual(with_caps - without, 2.0 * area)


class TestProjectToProfile(unittest.TestCase):
    def test_projection_z_axis(self):
        # Points around the z-axis; r = radial distance, z = height.
        pts = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 3.0, 5.0)]
        prof = rr.project_to_profile(pts, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        self.assertAlmostEqual(prof[0][0], 0.0)
        self.assertAlmostEqual(prof[1][0], 2.0)
        self.assertAlmostEqual(prof[1][1], 0.0)
        self.assertAlmostEqual(prof[2][0], 3.0)
        self.assertAlmostEqual(prof[2][1], 5.0)


class TestRevolveSolid(unittest.TestCase):
    def test_solid_volume_and_full(self):
        R, h = 2.0, 5.0
        rect = ((0.0, 0.0), (R, 0.0), (R, h), (0.0, h))
        solid = rr.RevolveSolid(rect)
        self.assertTrue(solid.is_full)
        self.assertAlmostEqual(solid.volume, math.pi * R * R * h)
        self.assertEqual(solid.profile_bounds(), (0.0, 2.0, 0.0, 5.0))

    def test_invalid_angle_rejected(self):
        rect = ((0.0, 0.0), (2.0, 0.0), (2.0, 5.0), (0.0, 5.0))
        with self.assertRaises(ValueError):
            rr.RevolveSolid(rect, angle=0.0)
        with self.assertRaises(ValueError):
            rr.RevolveSolid(rect, angle=7.0)

    def test_crossing_profile_rejected(self):
        bad = ((-1.0, 0.0), (2.0, 0.0), (0.0, 3.0))
        with self.assertRaises(ValueError):
            rr.RevolveSolid(bad)

    def test_bounding_cylinder(self):
        rect = ((0.0, 0.0), (2.0, 0.0), (2.0, 5.0), (0.0, 5.0))
        solid = rr.RevolveSolid(rect)
        self.assertAlmostEqual(solid.bounding_cylinder_volume(),
                               math.pi * 4.0 * 5.0)


if __name__ == "__main__":
    unittest.main()
