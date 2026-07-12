"""Tests for standards.cqplug_heatsert_schedule."""

import math
import unittest

from standards.cqplug_heatsert_schedule import (
    HeatsertError,
    boss_ok,
    bore_depth,
    bore_volume,
    chamfer_angle,
    designations,
    fits_in_wall,
    heatsert_bore,
    insert_dims,
    melt_displacement,
    select_for_bolt,
    with_extra_size,
)


class TestSchedule(unittest.TestCase):
    def test_designations_ordered(self):
        self.assertEqual(designations(), ["M3", "M4", "M5", "M6"])

    def test_monotonic_bore_and_depth(self):
        prev = None
        for name in designations():
            d = insert_dims(name)
            if prev is not None:
                self.assertGreater(d.bore_diameter, prev.bore_diameter)
                self.assertGreater(d.bore_depth, prev.bore_depth)
            prev = d

    def test_bore_bigger_than_bolt(self):
        for name in designations():
            d = insert_dims(name)
            self.assertGreater(d.bore_diameter, d.bolt_diameter)

    def test_case_insensitive_lookup(self):
        self.assertEqual(insert_dims("m4").bore_diameter, 5.6)

    def test_unknown_designation(self):
        with self.assertRaises(HeatsertError):
            insert_dims("M99")

    def test_bolt_clearance_diameter(self):
        self.assertAlmostEqual(
            insert_dims("M4").bolt_clearance_diameter, 4.8, places=9)

    def test_select_for_bolt(self):
        self.assertEqual(select_for_bolt(3.5).designation, "M4")
        self.assertEqual(select_for_bolt(6.0).designation, "M6")
        with self.assertRaises(HeatsertError):
            select_for_bolt(10.0)

    def test_with_extra_size(self):
        m8 = with_extra_size("M8", 10.0, 15.0, 8.0)
        self.assertEqual(m8.designation, "M8")
        self.assertNotIn("M8", designations())  # table not mutated
        with self.assertRaises(HeatsertError):
            with_extra_size("M8", 6.0, 15.0, 8.0)


class TestBoreProfile(unittest.TestCase):
    def test_plain_bore_is_one_cylinder(self):
        s = heatsert_bore("M4")
        self.assertEqual(len(s), 1)
        self.assertAlmostEqual(s[0].d_start, 5.6, places=9)
        self.assertAlmostEqual(bore_depth(s), 8.1, places=9)
        self.assertAlmostEqual(
            bore_volume(s), math.pi * 2.8 ** 2 * 8.1, places=6)

    def test_short_bolt_clear_is_a_no_op(self):
        self.assertEqual(len(heatsert_bore("M4", bolt_clear=5.0)), 1)

    def test_long_bolt_clear_adds_a_narrow_bore(self):
        s = heatsert_bore("M4", bolt_clear=15.0)
        self.assertEqual(len(s), 2)
        self.assertAlmostEqual(bore_depth(s), 15.0, places=9)
        self.assertAlmostEqual(s[1].d_start, 4.8, places=9)
        self.assertLess(s[1].d_start, s[0].d_start)

    def test_sections_are_contiguous(self):
        s = heatsert_bore("M6", bolt_clear=20.0, chamfer=(1.0, 5.0))
        self.assertAlmostEqual(s[0].z_start, 0.0, places=9)
        for a, b in zip(s, s[1:]):
            self.assertAlmostEqual(a.z_end, b.z_start, places=9)

    def test_scalar_chamfer_is_45_degrees(self):
        s = heatsert_bore("M4", chamfer=2.0)
        self.assertEqual(len(s), 2)
        self.assertAlmostEqual(s[0].d_start, 5.6 + 4.0, places=9)
        self.assertAlmostEqual(s[0].d_end, 5.6, places=9)
        self.assertAlmostEqual(chamfer_angle(2.0), 45.0, places=9)

    def test_two_value_chamfer(self):
        s = heatsert_bore("M4", chamfer=(1.0, 5.0))
        self.assertAlmostEqual(s[0].z_end, 5.0, places=9)
        self.assertAlmostEqual(s[0].d_start, 7.6, places=9)
        self.assertLess(chamfer_angle((1.0, 5.0)), 45.0)

    def test_chamfer_deeper_than_bore_rejected(self):
        with self.assertRaises(HeatsertError):
            heatsert_bore("M3", chamfer=(1.0, 9.0))

    def test_chamfer_increases_volume(self):
        plain = bore_volume(heatsert_bore("M5"))
        chamfered = bore_volume(heatsert_bore("M5", chamfer=1.0))
        self.assertGreater(chamfered, plain)

    def test_negative_bolt_clear_rejected(self):
        with self.assertRaises(HeatsertError):
            heatsert_bore("M3", bolt_clear=-1.0)


class TestPlanningHelpers(unittest.TestCase):
    def test_fits_in_wall(self):
        self.assertTrue(fits_in_wall("M3", 8.0))
        self.assertFalse(fits_in_wall("M3", 5.0))
        self.assertFalse(fits_in_wall("M3", 8.0, bolt_clear=12.0))

    def test_boss_rule(self):
        self.assertTrue(boss_ok("M4", 12.0))
        self.assertFalse(boss_ok("M4", 8.0))
        self.assertAlmostEqual(insert_dims("M4").min_boss_diameter, 11.2,
                               places=9)

    def test_melt_displacement(self):
        v = melt_displacement("M3")
        self.assertAlmostEqual(v, math.pi * 2.0 ** 2 * 5.8, places=6)
        self.assertAlmostEqual(melt_displacement("M3", 2.9), v / 2.0, places=6)
        with self.assertRaises(HeatsertError):
            melt_displacement("M3", 0.0)


if __name__ == "__main__":
    unittest.main()
