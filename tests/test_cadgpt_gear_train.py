"""Tests for geometry.cadgpt_gear_train (meshing + assembly placement)."""

import math
import unittest

from harnesscad.domain.geometry.cadgpt_gear_train import (
    gear_ratio,
    center_distance,
    meshing_phase_offset,
    inverse_helix_angle,
    helix_twist,
    bevel_scale,
    place_driving_gear,
    place_driven_gear,
)


class TestRatioAndDistance(unittest.TestCase):
    def test_gear_ratio(self):
        self.assertEqual(gear_ratio(20, 60), 3.0)

    def test_center_distance_matches_repo_example(self):
        # spur example: m=2.5, z1=20, z2=60 -> gear 2 placed at x=100
        self.assertEqual(center_distance(2.5, 20, 60), 100.0)

    def test_center_distance_helix_example(self):
        # helix params: m=2.5, z1=16, z2=24 -> a = 2.5*40/2 = 50
        self.assertEqual(center_distance(2.5, 16, 24), 50.0)

    def test_center_distance_with_shift(self):
        self.assertEqual(center_distance(2.0, 20, 40, 0.5, 0.5), 61.0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            gear_ratio(0, 10)
        with self.assertRaises(ValueError):
            center_distance(0.0, 10, 10)


class TestPhaseOffset(unittest.TestCase):
    def test_matches_repo_formula(self):
        # scripting rule: gamma = 360/(teeth*2); repo spur script used z=60
        self.assertAlmostEqual(meshing_phase_offset(60), 360.0 / (60 * 2))
        self.assertAlmostEqual(meshing_phase_offset(60), 3.0)

    def test_more_teeth_smaller_offset(self):
        self.assertGreater(meshing_phase_offset(10), meshing_phase_offset(40))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            meshing_phase_offset(0)


class TestHelixAndBevel(unittest.TestCase):
    def test_inverse_helix(self):
        self.assertEqual(inverse_helix_angle(20.0), -20.0)

    def test_helix_twist_zero_when_no_helix(self):
        # tan(90) is infinite -> twist ~0
        self.assertAlmostEqual(helix_twist(15.0, 0.0, 20.0), 0.0, places=9)

    def test_helix_twist_matches_formula(self):
        h, w, r = 15.0, 20.0, 20.0
        expected = h / math.tan(math.radians(90 - w)) / math.pi * 180.0 / r
        self.assertAlmostEqual(helix_twist(h, w, r), expected)

    def test_bevel_scale_matches_formula(self):
        r, wb, h = 20.0, 45.0, 5.0
        expected = (r - math.tan(math.radians(wb)) * h) / r
        self.assertAlmostEqual(bevel_scale(r, wb, h), expected)

    def test_bevel_scale_zero_angle_is_one(self):
        self.assertAlmostEqual(bevel_scale(20.0, 0.0, 5.0), 1.0)


class TestPlacement(unittest.TestCase):
    def test_driving_gear_no_offset(self):
        p = place_driving_gear((0.0, 0.0, 0.0), helix_angle=20.0)
        self.assertEqual(p.rotation, (0.0, 0.0, 0.0))
        self.assertEqual(p.translation, (0.0, 0.0, 0.0))

    def test_driven_spur_phase_only(self):
        p = place_driven_gear(60, (100.0, 0.0, 0.0), gear_type="spur")
        self.assertEqual(p.translation, (100.0, 0.0, 0.0))
        self.assertAlmostEqual(p.rotation[2], 3.0)   # gamma = 360/120
        self.assertEqual(p.rotation[0], 0.0)
        self.assertEqual(p.rotation[1], 0.0)
        self.assertEqual(p.helix_angle, 0.0)

    def test_driven_helix_inverts_helix(self):
        p = place_driven_gear(24, (50.0, 0.0, 0.0), gear_type="helix",
                              driving_helix_angle=20.0)
        self.assertEqual(p.helix_angle, -20.0)
        self.assertAlmostEqual(p.rotation[2], meshing_phase_offset(24))

    def test_driven_bevel_flip(self):
        p = place_driven_gear(40, (40.0, 0.0, 40.0), gear_type="bevel")
        self.assertEqual(p.rotation[1], -90.0)   # beta
        self.assertEqual(p.rotation[2], 0.0)     # gamma overridden
        self.assertEqual(p.translation, (40.0, 0.0, 40.0))

    def test_to_dict_roundtrips(self):
        p = place_driven_gear(20, (10.0, 0.0, 0.0))
        d = p.to_dict()
        self.assertEqual(d["translation"], [10.0, 0.0, 0.0])
        self.assertIn("rotation", d)


if __name__ == "__main__":
    unittest.main()
