"""Tests for geometry.sdfx_cam_profile."""

import math
import unittest

from harnesscad.domain.geometry.sdf.cam_profile import (
    FlatFlankCam,
    ThreeArcCam,
    make_flat_flank_cam,
    make_three_arc_cam,
)


class TestFlatFlankCam(unittest.TestCase):
    def setUp(self):
        # tangent flanks require distance >= base_radius - nose_radius.
        self.cam = FlatFlankCam(distance=6.0, base_radius=5.0, nose_radius=1.0)

    def test_base_circle_surface(self):
        # a point on the base circle along -y (away from nose) is ~on surface
        self.assertAlmostEqual(self.cam.evaluate((0.0, -5.0)), 0.0, places=6)

    def test_center_inside(self):
        self.assertLess(self.cam.evaluate((0.0, 0.0)), 0.0)

    def test_nose_tip_surface(self):
        # nose circle centered at (0, distance) radius nose_radius -> tip at
        # (0, distance + nose_radius)
        tip = (0.0, 6.0 + 1.0)
        self.assertAlmostEqual(self.cam.evaluate(tip), 0.0, places=6)

    def test_symmetry(self):
        left = self.cam.evaluate((-2.0, 1.0))
        right = self.cam.evaluate((2.0, 1.0))
        self.assertAlmostEqual(left, right)

    def test_far_point_outside(self):
        self.assertGreater(self.cam.evaluate((20.0, 20.0)), 0.0)

    def test_bad_params(self):
        with self.assertRaises(ValueError):
            FlatFlankCam(0.0, 5.0, 1.0)


class TestThreeArcCam(unittest.TestCase):
    def setUp(self):
        # a geometrically valid three-arc cam (base circle radius 8 at origin,
        # nose radius 2 centered at (0, 8), large flank arcs of radius 48).
        self.cam = ThreeArcCam(distance=8.0, base_radius=8.0,
                               nose_radius=2.0, flank_radius=48.0)

    def test_base_circle_surface(self):
        self.assertAlmostEqual(self.cam.evaluate((0.0, -8.0)), 0.0, places=6)

    def test_nose_tip_surface(self):
        tip = (0.0, 8.0 + 2.0)
        self.assertAlmostEqual(self.cam.evaluate(tip), 0.0, places=6)

    def test_center_inside(self):
        self.assertLess(self.cam.evaluate((0.0, 0.0)), 0.0)

    def test_symmetry(self):
        self.assertAlmostEqual(self.cam.evaluate((-1.5, 2.0)),
                               self.cam.evaluate((1.5, 2.0)))

    def test_flank_radius_too_small(self):
        with self.assertRaises(ValueError):
            ThreeArcCam(8.0, 8.0, 2.0, 4.0)


class TestDesignConstructors(unittest.TestCase):
    def test_make_flat_flank(self):
        cam = make_flat_flank_cam(lift=2.0, duration=math.radians(90.0),
                                  max_diameter=20.0)
        # base radius = max/2 - lift = 10 - 2 = 8
        self.assertAlmostEqual(cam.base_radius, 8.0)
        # maximum lift point: nose tip reaches base_radius + lift = 10 above
        tip = (0.0, cam.distance + cam.nose_radius)
        self.assertAlmostEqual(cam.evaluate(tip), 0.0, places=6)
        # nose tip elevation equals base_radius + lift
        self.assertAlmostEqual(cam.distance + cam.nose_radius, 10.0, places=6)

    def test_make_three_arc(self):
        cam = make_three_arc_cam(lift=2.0, duration=math.radians(90.0),
                                 max_diameter=20.0, k=1.05)
        self.assertAlmostEqual(cam.base_radius, 8.0)
        tip = (0.0, cam.distance + cam.nose_radius)
        self.assertAlmostEqual(cam.evaluate(tip), 0.0, places=6)
        self.assertLess(cam.evaluate((0.0, 0.0)), 0.0)

    def test_invalid_duration(self):
        with self.assertRaises(ValueError):
            make_flat_flank_cam(2.0, math.pi, 20.0)

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            make_three_arc_cam(2.0, math.radians(90.0), 20.0, 1.0)


if __name__ == "__main__":
    unittest.main()
