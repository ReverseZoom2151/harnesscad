"""Tests for DBU planar layout + DRC (Vibe_Layout port)."""

import unittest

from harnesscad.domain.fabrication import planar_layout as pl


class DBUTest(unittest.TestCase):
    def test_to_dbu_rounds(self):
        self.assertEqual(pl.to_dbu(1.0, 0.001), 1000)
        self.assertEqual(pl.to_dbu(0.6, 0.001), 600)

    def test_maps_exactly(self):
        self.assertTrue(pl.maps_exactly(0.6, 0.001))
        self.assertFalse(pl.maps_exactly(0.6, 0.4))

    def test_bad_dbu_raises(self):
        with self.assertRaises(pl.LayoutError):
            pl.to_dbu(1.0, 0.0)


class RectTest(unittest.TestCase):
    def test_dimensions(self):
        r = pl.Rect("M1", 0, 0, 100, 50)
        self.assertEqual(r.width, 100)
        self.assertEqual(r.height, 50)
        self.assertEqual(r.area, 5000)

    def test_spacing_disjoint(self):
        a = pl.Rect("M1", 0, 0, 100, 100)
        b = pl.Rect("M1", 150, 0, 200, 100)
        self.assertEqual(pl.box_spacing_dbu(a, b), 50)

    def test_spacing_touching_zero(self):
        a = pl.Rect("M1", 0, 0, 100, 100)
        b = pl.Rect("M1", 100, 0, 200, 100)
        self.assertEqual(pl.box_spacing_dbu(a, b), 0)

    def test_overlap(self):
        a = pl.Rect("M1", 0, 0, 100, 100)
        b = pl.Rect("M1", 50, 50, 150, 150)
        self.assertTrue(pl.boxes_overlap(a, b))
        c = pl.Rect("M1", 200, 200, 300, 300)
        self.assertFalse(pl.boxes_overlap(a, c))


class LayoutBuildTest(unittest.TestCase):
    def setUp(self):
        self.layout = pl.PlanarLayout(dbu_um=0.001)
        self.layout.ensure_layer("MWRITER", 1, 0)

    def test_unknown_layer_raises(self):
        with self.assertRaises(pl.LayoutError):
            self.layout.add_box_um("NOPE", 0, 0, 1, 1)

    def test_centered_box(self):
        r = self.layout.add_centered_box_um("MWRITER", 10.0, 4.0)
        self.assertEqual(r.width, 10000)
        self.assertEqual(r.height, 4000)
        self.assertEqual(r.x1, -5000)

    def test_frame_has_four_strips(self):
        strips = self.layout.add_frame_um("MWRITER", 100.0, 100.0, 1.0)
        self.assertEqual(len(strips), 4)


class DRCTest(unittest.TestCase):
    def setUp(self):
        self.layout = pl.PlanarLayout(dbu_um=0.001)
        self.layout.ensure_layer("M1", 1, 0)

    def test_clean_layout_passes(self):
        self.layout.add_box_um("M1", 0, 0, 10, 10)
        self.layout.add_box_um("M1", 20, 0, 30, 10)  # 10 um gap
        report = pl.run_drc(self.layout, min_width_um=0.6, min_spacing_um=1.0)
        self.assertTrue(report.passed, report.rule_ids())

    def test_empty_layout_flagged(self):
        report = pl.run_drc(self.layout)
        self.assertFalse(report.passed)
        self.assertIn("geometry.empty", report.rule_ids())

    def test_min_width_violation(self):
        self.layout.add_box_um("M1", 0, 0, 0.3, 10)  # 0.3 um < 0.6 um
        report = pl.run_drc(self.layout, min_width_um=0.6)
        self.assertIn("drc.min_width", report.rule_ids())

    def test_min_spacing_violation(self):
        self.layout.add_box_um("M1", 0, 0, 10, 10)
        self.layout.add_box_um("M1", 10.5, 0, 20, 10)  # 0.5 um gap < 1.0
        report = pl.run_drc(self.layout, min_spacing_um=1.0, check_shorts=False)
        self.assertIn("drc.min_spacing", report.rule_ids())

    def test_short_detected(self):
        self.layout.add_box_um("M1", 0, 0, 10, 10)
        self.layout.add_box_um("M1", 5, 5, 15, 15)
        report = pl.run_drc(self.layout, check_shorts=True)
        self.assertIn("drc.short", report.rule_ids())

    def test_different_layers_no_spacing_conflict(self):
        self.layout.ensure_layer("M2", 2, 0)
        self.layout.add_box_um("M1", 0, 0, 10, 10)
        self.layout.add_box_um("M2", 10.1, 0, 20, 10)  # close but different layer
        report = pl.run_drc(self.layout, min_spacing_um=5.0)
        self.assertNotIn("drc.min_spacing", report.rule_ids())


if __name__ == "__main__":
    unittest.main()
