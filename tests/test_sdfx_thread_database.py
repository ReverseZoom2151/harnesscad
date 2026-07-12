"""Tests for standards.sdfx_thread_database."""

import math
import unittest

from standards.sdfx_thread_database import (
    MM_PER_INCH,
    hex_height,
    hex_radius,
    thread_lookup,
    thread_names,
)


class TestISOEntries(unittest.TestCase):
    def test_m6_coarse(self):
        t = thread_lookup("M6x1")
        self.assertAlmostEqual(t.radius, 3.0)
        self.assertAlmostEqual(t.pitch, 1.0)
        self.assertAlmostEqual(t.taper, 0.0)
        self.assertAlmostEqual(t.hex_flat2flat, 10.0)
        self.assertEqual(t.units, "mm")

    def test_m3_coarse(self):
        t = thread_lookup("M3x0.5")
        self.assertAlmostEqual(t.radius, 1.5)
        self.assertAlmostEqual(t.pitch, 0.5)

    def test_fine_pitch_smaller(self):
        coarse = thread_lookup("M8x1.25")
        fine = thread_lookup("M8x1")
        self.assertLess(fine.pitch, coarse.pitch)
        self.assertAlmostEqual(coarse.radius, fine.radius)


class TestUTSEntries(unittest.TestCase):
    def test_quarter_unc(self):
        t = thread_lookup("unc_1/4")
        # 1/4 inch major diameter -> radius in mm
        self.assertAlmostEqual(t.radius, 0.5 * 0.25 * MM_PER_INCH)
        # 20 tpi -> pitch = 1/20 inch in mm
        self.assertAlmostEqual(t.pitch, (1.0 / 20.0) * MM_PER_INCH)
        self.assertEqual(t.units, "inch")

    def test_unf_finer_than_unc(self):
        unc = thread_lookup("unc_1/4")
        unf = thread_lookup("unf_1/4")
        self.assertLess(unf.pitch, unc.pitch)


class TestNPTEntries(unittest.TestCase):
    def test_npt_is_tapered(self):
        t = thread_lookup("npt_1/2")
        self.assertGreater(t.taper, 0.0)
        # 1/32 taper on radius
        self.assertAlmostEqual(t.taper, math.atan(1.0 / 32.0))

    def test_npt_ftof_is_mm(self):
        t = thread_lookup("npt_1/2")
        self.assertAlmostEqual(t.hex_flat2flat, 22.4)


class TestHexGeometry(unittest.TestCase):
    def test_hex_radius_gt_half_flat(self):
        t = thread_lookup("M6x1")
        r = hex_radius(t)
        # circumradius exceeds the inradius (half flat-to-flat)
        self.assertGreater(r, t.hex_flat2flat / 2.0)
        # exact: ftof / (2 cos30)
        self.assertAlmostEqual(r, 10.0 / (2.0 * math.cos(math.radians(30))))

    def test_hex_height_positive(self):
        t = thread_lookup("M6x1")
        self.assertGreater(hex_height(t), 0.0)
        self.assertAlmostEqual(hex_height(t), 2.0 * hex_radius(t) * (5.0 / 12.0))


class TestLookupAPI(unittest.TestCase):
    def test_missing_raises(self):
        with self.assertRaises(KeyError):
            thread_lookup("M999")

    def test_names_prefix_filter(self):
        iso = thread_names("M")
        self.assertIn("M6x1", iso)
        self.assertTrue(all(n.startswith("M") for n in iso))
        # both coarse and fine M6 present
        self.assertIn("M6x0.75", iso)

    def test_names_sorted(self):
        names = thread_names()
        self.assertEqual(names, sorted(names))
        self.assertGreater(len(names), 50)


if __name__ == "__main__":
    unittest.main()
