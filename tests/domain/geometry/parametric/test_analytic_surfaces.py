"""Tests for geometry.dreamcad_primitives."""

import math
import unittest

from harnesscad.domain.geometry.parametric.analytic_surfaces import (
    Cone,
    Cylinder,
    Plane,
    Sphere,
    Torus,
    sample_surface,
)


class TestPlane(unittest.TestCase):
    def test_point_and_normal(self):
        plane = Plane()
        self.assertEqual(plane.point(0.0, 0.0), (0.0, 0.0, 0.0))
        self.assertEqual(plane.point(1.0, 1.0), (1.0, 1.0, 0.0))
        self.assertEqual(plane.normal(0.3, 0.7), (0.0, 0.0, 1.0))


class TestCylinder(unittest.TestCase):
    def test_radius_and_normal(self):
        cyl = Cylinder(radius=2.0, height=3.0)
        p = cyl.point(0.0, 0.0)
        self.assertAlmostEqual(p[0], 2.0)
        self.assertAlmostEqual(p[1], 0.0)
        top = cyl.point(0.0, 1.0)
        self.assertAlmostEqual(top[2], 3.0)
        n = cyl.normal(0.25, 0.5)
        self.assertAlmostEqual(math.hypot(n[0], n[1]), 1.0)
        self.assertAlmostEqual(n[2], 0.0)

    def test_point_lies_on_surface(self):
        cyl = Cylinder(radius=1.5)
        for u in (0.1, 0.4, 0.9):
            p = cyl.point(u, 0.5)
            self.assertAlmostEqual(math.hypot(p[0], p[1]), 1.5)


class TestCone(unittest.TestCase):
    def test_radius_shrinks(self):
        cone = Cone(base_radius=2.0, top_radius=0.0, height=4.0)
        base = cone.point(0.0, 0.0)
        apex = cone.point(0.0, 1.0)
        self.assertAlmostEqual(base[0], 2.0)
        self.assertAlmostEqual(apex[0], 0.0)
        self.assertAlmostEqual(apex[2], 4.0)

    def test_normal_unit_length(self):
        cone = Cone(base_radius=1.0, top_radius=0.5, height=2.0)
        n = cone.normal(0.2, 0.5)
        self.assertAlmostEqual(math.sqrt(sum(c * c for c in n)), 1.0)


class TestSphere(unittest.TestCase):
    def test_on_surface_and_radial_normal(self):
        sph = Sphere(radius=3.0)
        for u, v in [(0.1, 0.2), (0.5, 0.5), (0.9, 0.8)]:
            p = sph.point(u, v)
            self.assertAlmostEqual(math.sqrt(sum(c * c for c in p)), 3.0)
            n = sph.normal(u, v)
            # normal parallel to position vector
            self.assertAlmostEqual(n[0] * 3.0, p[0], places=6)


class TestTorus(unittest.TestCase):
    def test_distance_from_axis(self):
        torus = Torus(major_radius=2.0, minor_radius=0.5)
        p = torus.point(0.0, 0.0)
        self.assertAlmostEqual(math.hypot(p[0], p[1]), 2.5)
        p2 = torus.point(0.0, 0.5)
        self.assertAlmostEqual(math.hypot(p2[0], p2[1]), 1.5)

    def test_normal_unit(self):
        torus = Torus()
        n = torus.normal(0.3, 0.6)
        self.assertAlmostEqual(math.sqrt(sum(c * c for c in n)), 1.0)


class TestSampling(unittest.TestCase):
    def test_sample_count(self):
        pts = sample_surface(Sphere(), resolution=6)
        self.assertEqual(len(pts), 36)

    def test_sample_deterministic(self):
        a = sample_surface(Cylinder(), 5)
        b = sample_surface(Cylinder(), 5)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
