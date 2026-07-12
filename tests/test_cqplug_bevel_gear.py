"""Tests for geometry.cqplug_bevel_gear."""

import math
import unittest

from geometry.cqplug_bevel_gear import (
    BevelGear,
    BevelGearError,
    BevelGearPair,
    addendum_angle,
    base_cone_angle,
    cone_distance,
    dedendum_angle,
    pitch_cone_angles,
    spherical_involute,
)


class TestConeAngles(unittest.TestCase):
    def test_base_cone_angle_formula(self):
        # asin(sin(45)*cos(20))
        expect = math.degrees(math.asin(math.sin(math.radians(45.0))
                                        * math.cos(math.radians(20.0))))
        self.assertAlmostEqual(base_cone_angle(45.0, 20.0), expect, places=9)

    def test_base_cone_below_pitch(self):
        self.assertLess(base_cone_angle(40.0, 20.0), 40.0)

    def test_addendum_and_dedendum(self):
        self.assertAlmostEqual(addendum_angle(2.0, 50.0),
                               math.degrees(math.atan2(2.0, 50.0)), places=9)
        self.assertGreater(dedendum_angle(2.0, 50.0), addendum_angle(2.0, 50.0))

    def test_invalid_inputs(self):
        with self.assertRaises(BevelGearError):
            addendum_angle(0.0, 10.0)
        with self.assertRaises(BevelGearError):
            dedendum_angle(1.0, 0.0)


class TestPitchConeAngles(unittest.TestCase):
    def test_right_angle_reduces_to_atan2(self):
        dp1, dp2 = pitch_cone_angles(15, 30)
        self.assertAlmostEqual(dp1, math.degrees(math.atan2(15, 30)), places=9)
        self.assertAlmostEqual(dp1 + dp2, 90.0, places=9)

    def test_equal_teeth_give_45(self):
        dp1, dp2 = pitch_cone_angles(20, 20)
        self.assertAlmostEqual(dp1, 45.0, places=9)
        self.assertAlmostEqual(dp2, 45.0, places=9)

    def test_swaps_so_pinion_is_smaller(self):
        self.assertEqual(pitch_cone_angles(30, 15), pitch_cone_angles(15, 30))

    def test_non_right_shaft_angle_sums_to_sigma(self):
        dp1, dp2 = pitch_cone_angles(12, 24, shaft_angle_deg=60.0)
        self.assertAlmostEqual(dp1 + dp2, 60.0, places=9)
        self.assertLess(dp1, dp2)

    def test_invalid_teeth(self):
        with self.assertRaises(BevelGearError):
            pitch_cone_angles(0, 10)


class TestConeDistance(unittest.TestCase):
    def test_formula(self):
        # R = (m*z/2)/sin(delta_p)
        r = cone_distance(2.0, 30, 60.0)
        self.assertAlmostEqual(r, 30.0 / math.sin(math.radians(60.0)), places=9)

    def test_pitch_radius_recovered(self):
        # R*sin(delta_p) == m*z/2
        m, z, dp = 3.0, 24, 55.0
        r = cone_distance(m, z, dp)
        self.assertAlmostEqual(r * math.sin(math.radians(dp)), m * z / 2.0,
                               places=9)


class TestSphericalInvolute(unittest.TestCase):
    def test_point_lies_on_sphere(self):
        R = 40.0
        p = spherical_involute(30.0, 25.0, R)
        self.assertAlmostEqual(math.sqrt(sum(c * c for c in p)), R, places=6)

    def test_at_base_angle_is_deterministic(self):
        a = spherical_involute(26.0, 25.0, 40.0)
        b = spherical_involute(26.0, 25.0, 40.0)
        self.assertEqual(a, b)

    def test_below_base_raises(self):
        with self.assertRaises(BevelGearError):
            # cos(delta)/cos(delta_b) > 1 when delta < delta_b
            spherical_involute(10.0, 25.0, 40.0)


class TestBevelGear(unittest.TestCase):
    def setUp(self):
        # delta_p from a 20/40 right-angle pair
        dp = math.degrees(math.atan2(20, 40))
        self.R = cone_distance(2.0, 20, dp)
        self.g = BevelGear(2.0, 20, self.R * 0.3, dp, self.R)

    def test_pitch_radius(self):
        self.assertAlmostEqual(self.g.pitch_radius, 2.0 * 20 / 2.0, places=9)
        self.assertAlmostEqual(
            self.R * math.sin(math.radians(self.g.pitch_cone_angle)),
            self.g.pitch_radius, places=6)

    def test_cone_angle_ordering(self):
        self.assertLess(self.g.root_cone_angle, self.g.pitch_cone_angle)
        self.assertLess(self.g.pitch_cone_angle, self.g.face_cone_angle)
        self.assertLess(self.g.base_cone_angle, self.g.pitch_cone_angle)

    def test_tip_radius_gt_root_radius(self):
        self.assertGreater(self.g.tip_radius, self.g.root_radius)

    def test_tooth_thickness_angle_positive(self):
        self.assertGreater(self.g.tooth_thickness_angle, 0.0)

    def test_inner_cone_distance(self):
        self.assertAlmostEqual(self.g.inner_cone_distance,
                               self.R - self.R * 0.3, places=9)

    def test_tooth_profile_point_on_sphere(self):
        p = self.g.tooth_profile_point(self.g.pitch_cone_angle)
        self.assertAlmostEqual(math.sqrt(sum(c * c for c in p)), self.R,
                               places=5)

    def test_bad_face_width(self):
        with self.assertRaises(BevelGearError):
            BevelGear(2.0, 20, self.R * 2.0, 26.0, self.R)


class TestBevelGearPair(unittest.TestCase):
    def setUp(self):
        self.pair = BevelGearPair(2.0, 15, 30, 8.0)

    def test_ratio(self):
        self.assertAlmostEqual(self.pair.gear_ratio, 2.0, places=9)

    def test_swap_orders_pinion_first(self):
        p = BevelGearPair(2.0, 30, 15, 8.0)
        self.assertEqual(p.pinion_teeth, 15)
        self.assertEqual(p.wheel_teeth, 30)

    def test_shared_cone_distance(self):
        pin = self.pair.pinion()
        whl = self.pair.wheel()
        self.assertAlmostEqual(pin.cone_distance, whl.cone_distance, places=9)

    def test_pitch_cones_sum_to_shaft_angle(self):
        dp1, dp2 = self.pair.pitch_cone_angles
        self.assertAlmostEqual(dp1 + dp2, 90.0, places=9)

    def test_pinion_and_wheel_pitch_radii(self):
        self.assertAlmostEqual(self.pair.pinion().pitch_radius, 15.0, places=9)
        self.assertAlmostEqual(self.pair.wheel().pitch_radius, 30.0, places=9)


if __name__ == "__main__":
    unittest.main()
