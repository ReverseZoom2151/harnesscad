"""Tests for spec.datacon_rim_spec (DATACON wheel-rim spec parser)."""

import unittest

from spec.datacon_rim_spec import (
    RimSpec,
    bolt_circle_outer_diameter,
    center_bore_inner_diameter,
    external_circle_radius,
    flange_height,
    parse_rim_spec,
    spec_summary,
    specified_diameter,
    transform_ratio,
)


class SpecifiedDiameterTests(unittest.TestCase):
    def test_known_codes(self):
        self.assertAlmostEqual(specified_diameter(17), 436.6)
        self.assertAlmostEqual(specified_diameter(15), 380.2)
        self.assertAlmostEqual(specified_diameter(10), 253.2)
        self.assertAlmostEqual(specified_diameter(30), 766.8)

    def test_unknown_code_raises(self):
        with self.assertRaises(ValueError):
            specified_diameter(11)
        with self.assertRaises(ValueError):
            specified_diameter(99)


class FlangeHeightTests(unittest.TestCase):
    def test_b_family(self):
        self.assertAlmostEqual(flange_height("B"), 14.5)

    def test_j_family(self):
        self.assertAlmostEqual(flange_height("J"), 17.5)
        self.assertAlmostEqual(flange_height("JJ"), 17.5)
        self.assertAlmostEqual(flange_height("JX"), 17.5)

    def test_case_insensitive(self):
        self.assertAlmostEqual(flange_height("j"), 17.5)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            flange_height("Z")
        with self.assertRaises(ValueError):
            flange_height(None)


class ParseRimSpecTests(unittest.TestCase):
    def test_canonical_code(self):
        spec = parse_rim_spec("17 4H PCD 114.3 7J ET34 C/B:73")
        self.assertIsInstance(spec, RimSpec)
        self.assertEqual(spec.diameter_code, 17)
        self.assertAlmostEqual(spec.specified_diameter, 436.6)
        self.assertEqual(spec.bolt_count, 4)
        self.assertAlmostEqual(spec.pcd, 114.3)
        self.assertAlmostEqual(spec.width, 7.0)
        self.assertEqual(spec.flange, "J")
        self.assertAlmostEqual(spec.et, 34.0)
        self.assertAlmostEqual(spec.center_bore, 73.0)

    def test_negative_offset(self):
        spec = parse_rim_spec("18 5H PCD 120 8.5JJ ET-12 C/B 72.6")
        self.assertEqual(spec.diameter_code, 18)
        self.assertEqual(spec.bolt_count, 5)
        self.assertAlmostEqual(spec.pcd, 120.0)
        self.assertAlmostEqual(spec.width, 8.5)
        self.assertEqual(spec.flange, "JJ")
        self.assertAlmostEqual(spec.et, -12.0)
        self.assertAlmostEqual(spec.center_bore, 72.6)

    def test_fractional_width(self):
        spec = parse_rim_spec("15 6 1/2-JJ 4H PCD 100 ET40 C/B:54")
        self.assertEqual(spec.diameter_code, 15)
        self.assertAlmostEqual(spec.width, 6.5)
        self.assertEqual(spec.flange, "JJ")
        self.assertEqual(spec.bolt_count, 4)

    def test_minimal_code(self):
        spec = parse_rim_spec("16")
        self.assertEqual(spec.diameter_code, 16)
        self.assertAlmostEqual(spec.specified_diameter, 405.6)
        self.assertIsNone(spec.bolt_count)
        self.assertIsNone(spec.pcd)
        self.assertIsNone(spec.width)
        self.assertIsNone(spec.flange)
        self.assertIsNone(spec.et)
        self.assertIsNone(spec.center_bore)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            parse_rim_spec("")
        with self.assertRaises(ValueError):
            parse_rim_spec(None)


class DerivedGeometryTests(unittest.TestCase):
    def test_external_circle_radius(self):
        self.assertAlmostEqual(external_circle_radius(436.6, 17.5), 235.8)

    def test_bolt_circle_outer_diameter(self):
        self.assertAlmostEqual(
            bolt_circle_outer_diameter(114.3, 6.0), 138.3
        )

    def test_center_bore_inner_diameter(self):
        self.assertAlmostEqual(center_bore_inner_diameter(73.0), 73.0)

    def test_transform_ratio(self):
        self.assertAlmostEqual(transform_ratio(235.8, 235.8), 1.0)
        self.assertAlmostEqual(transform_ratio(240.0, 120.0), 2.0)

    def test_transform_ratio_invalid(self):
        with self.assertRaises(ValueError):
            transform_ratio(235.8, 0.0)
        with self.assertRaises(ValueError):
            transform_ratio(235.8, -5.0)


class SpecSummaryTests(unittest.TestCase):
    def test_summary_includes_external_radius(self):
        spec = parse_rim_spec("17 4H PCD 114.3 7J ET34 C/B:73")
        summary = spec_summary(spec)
        self.assertEqual(summary["diameter_code"], 17)
        self.assertAlmostEqual(summary["pcd"], 114.3)
        self.assertIn("external_circle_radius", summary)
        self.assertAlmostEqual(summary["external_circle_radius"], 235.8)

    def test_summary_without_flange(self):
        spec = parse_rim_spec("16")
        summary = spec_summary(spec)
        self.assertNotIn("external_circle_radius", summary)


if __name__ == "__main__":
    unittest.main()
