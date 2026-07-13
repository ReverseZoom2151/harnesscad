"""Tests for geometry.sdfx_spiral_sdf."""

import math
import unittest

from harnesscad.domain.geometry.sdf.spiral import ArcSpiral, polar_dist2, to_polar


class TestPolarHelpers(unittest.TestCase):
    def test_polar_dist2_same_point(self):
        self.assertAlmostEqual(polar_dist2(2.0, 0.5, 2.0, 0.5), 0.0)

    def test_polar_dist2_radial(self):
        # two points on the same ray: distance is |r0 - r1|
        d2 = polar_dist2(1.0, 0.3, 4.0, 0.3)
        self.assertAlmostEqual(math.sqrt(d2), 3.0)

    def test_polar_dist2_matches_cartesian(self):
        p0 = (1.0, 2.0)
        p1 = (-3.0, 0.5)
        r0, t0 = to_polar(p0)
        r1, t1 = to_polar(p1)
        d2 = polar_dist2(r0, t0, r1, t1)
        exp = (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2
        self.assertAlmostEqual(d2, exp, places=9)


class TestArcSpiral(unittest.TestCase):
    def setUp(self):
        # r = 1*theta + 0, over one full turn.
        self.s = ArcSpiral(a=1.0, k=0.0, start=0.0, end=2.0 * math.pi)

    def test_point_on_spiral_is_zero(self):
        # pick theta = pi/2 -> r = pi/2; the cartesian point on the curve
        theta = math.pi / 2
        r = theta
        p = (r * math.cos(theta), r * math.sin(theta))
        self.assertAlmostEqual(self.s.evaluate(p), 0.0, places=6)

    def test_point_on_spiral_various(self):
        for theta in (0.3, 1.0, 2.0, 3.0, 5.0):
            r = theta  # a=1,k=0
            p = (r * math.cos(theta), r * math.sin(theta))
            self.assertAlmostEqual(self.s.evaluate(p), 0.0, places=6)

    def test_offset_thickness(self):
        # with offset d, a point on the centerline is at distance -d (inside).
        s = ArcSpiral(a=1.0, k=0.0, start=0.0, end=2.0 * math.pi, d=0.25)
        theta = 1.0
        r = theta
        p = (r * math.cos(theta), r * math.sin(theta))
        self.assertAlmostEqual(s.evaluate(p), -0.25, places=6)

    def test_start_endpoint_candidate(self):
        # the start point of the spiral is on the curve (distance ~0)
        p = (self.s.start_r * math.cos(self.s.start),
             self.s.start_r * math.sin(self.s.start))
        self.assertAlmostEqual(self.s.evaluate(p), 0.0, places=6)

    def test_far_point_positive(self):
        self.assertGreater(self.s.evaluate((100.0, 100.0)), 0.0)

    def test_nonzero_offset_k(self):
        # spiral r = theta + 2 ; point on curve at theta=1 -> r=3
        s = ArcSpiral(a=1.0, k=2.0, start=0.0, end=2.0 * math.pi)
        theta = 1.0
        r = theta + 2.0
        p = (r * math.cos(theta), r * math.sin(theta))
        self.assertAlmostEqual(s.evaluate(p), 0.0, places=6)

    def test_bad_params(self):
        with self.assertRaises(ValueError):
            ArcSpiral(a=0.0, k=0.0, start=0.0, end=1.0)
        with self.assertRaises(ValueError):
            ArcSpiral(a=1.0, k=0.0, start=1.0, end=1.0)


if __name__ == "__main__":
    unittest.main()
