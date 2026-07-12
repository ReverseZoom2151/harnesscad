"""Tests for geometry.cadgpt_involute_gear (involute spur-gear geometry)."""

import math
import unittest

from geometry.cadgpt_involute_gear import (
    pitch_radius,
    pitch_diameter,
    base_radius,
    tip_radius,
    root_radius,
    circular_pitch,
    involute_point,
    rack_profile,
    gear_geometry,
)


class TestPitchAndRadii(unittest.TestCase):
    def test_pitch_diameter_matches_repo_example(self):
        # spur_gear example: m=2.5, z=20 -> pitch_d 50 ; z=60 -> 150
        self.assertEqual(pitch_diameter(2.5, 20), 50.0)
        self.assertEqual(pitch_diameter(2.5, 60), 150.0)

    def test_pitch_radius_with_profile_shift(self):
        self.assertEqual(pitch_radius(2.0, 20, 0.0), 20.0)
        self.assertEqual(pitch_radius(2.0, 20, 0.5), 20.5)

    def test_tip_and_root_no_clearance(self):
        # r_wk = 20; addendum/dedendum = m = 2 -> tip 22, root 18
        self.assertAlmostEqual(tip_radius(2.0, 20), 22.0)
        self.assertAlmostEqual(root_radius(2.0, 20), 18.0)

    def test_clearance_shrinks_tip_grows_root_gap(self):
        r_kk = tip_radius(2.0, 20, clearance=0.2)
        r_fk = root_radius(2.0, 20, clearance=0.2)
        # tip = 20 + 2*(1-0.1)=21.8 ; root = 20 - 2*(1+0.1)=17.8
        self.assertAlmostEqual(r_kk, 21.8)
        self.assertAlmostEqual(r_fk, 17.8)

    def test_base_radius_pressure_angle(self):
        # r_b = (m*z/2)*cos(20deg)
        self.assertAlmostEqual(
            base_radius(2.0, 20, 20.0), 20.0 * math.cos(math.radians(20.0))
        )

    def test_circular_pitch(self):
        self.assertAlmostEqual(circular_pitch(2.5), math.pi * 2.5)


class TestInvolutePoint(unittest.TestCase):
    def test_zero_roll_on_base_circle(self):
        x, y = involute_point(10.0, 0.0)
        self.assertAlmostEqual(x, 10.0)
        self.assertAlmostEqual(y, 0.0)

    def test_point_lies_outside_base_circle(self):
        rb = 10.0
        x, y = involute_point(rb, 30.0)
        self.assertGreater(math.hypot(x, y), rb)

    def test_radius_matches_closed_form(self):
        # |involute(t)| = r_b * sqrt(1 + t^2)
        rb = 7.0
        t_deg = 25.0
        x, y = involute_point(rb, t_deg)
        t = math.radians(t_deg)
        self.assertAlmostEqual(math.hypot(x, y), rb * math.sqrt(1 + t * t))


class TestRackProfile(unittest.TestCase):
    def test_vertex_count(self):
        # leading anchor + 4*(z+2) tooth points (i from -1..z) + 2 closing
        z = 10
        pts = rack_profile(2.0, z, 0.0, 20.0, 0.0)
        self.assertEqual(len(pts), 1 + 4 * (z + 2) + 2)

    def test_all_points_finite_tuples(self):
        pts = rack_profile(1.0, 8, 0.0, 20.0, 0.0)
        for p in pts:
            self.assertEqual(len(p), 2)
            self.assertTrue(math.isfinite(p[0]) and math.isfinite(p[1]))

    def test_scales_linearly_with_module(self):
        a = rack_profile(1.0, 6, 0.0, 20.0, 0.0)
        b = rack_profile(3.0, 6, 0.0, 20.0, 0.0)
        for pa, pb in zip(a, b):
            self.assertAlmostEqual(pb[0], 3.0 * pa[0])
            self.assertAlmostEqual(pb[1], 3.0 * pa[1])

    def test_deterministic(self):
        self.assertEqual(
            rack_profile(2.0, 10, 0.0, 20.0, 0.1),
            rack_profile(2.0, 10, 0.0, 20.0, 0.1),
        )


class TestGearGeometryBundle(unittest.TestCase):
    def test_bundle_fields(self):
        g = gear_geometry(2.5, 20, pressure_angle=20.0)
        self.assertEqual(g.pitch_diameter, 50.0)
        self.assertAlmostEqual(g.tip_diameter, 55.0)   # (25+2.5)*2
        self.assertAlmostEqual(g.root_diameter, 45.0)  # (25-2.5)*2
        self.assertIn("base_radius", g.to_dict())

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            gear_geometry(0.0, 20)
        with self.assertRaises(ValueError):
            gear_geometry(2.0, 0)


if __name__ == "__main__":
    unittest.main()
