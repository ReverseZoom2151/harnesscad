"""Tests for skyline part nesting (Kerf nesting port)."""

import unittest

from harnesscad.domain.fabrication import nesting as nest


class PartTest(unittest.TestCase):
    def test_rejects_bad_size(self):
        with self.assertRaises(nest.NestingError):
            nest.Part("bad", 0, 10)

    def test_rejects_bad_qty(self):
        with self.assertRaises(nest.NestingError):
            nest.Part("bad", 10, 10, qty=0)


class NestBasicTest(unittest.TestCase):
    def test_single_part_one_sheet(self):
        r = nest.nest_parts([nest.Part("A", 100, 50)], 1000, 500)
        self.assertTrue(r.ok)
        self.assertEqual(r.sheets_used, 1)
        self.assertEqual(len(r.placements), 1)

    def test_quantity_expands(self):
        r = nest.nest_parts([nest.Part("A", 100, 50, qty=4)], 1000, 500)
        self.assertTrue(r.ok)
        self.assertEqual(len(r.placements), 4)

    def test_all_placements_within_sheet(self):
        r = nest.nest_parts(
            [nest.Part("A", 300, 200, qty=10), nest.Part("B", 150, 100, qty=6)],
            1000,
            500,
            kerf=3,
            margin=5,
        )
        self.assertTrue(r.ok)
        for p in r.placements:
            self.assertGreaterEqual(p.x, 5 - 1e-6)
            self.assertGreaterEqual(p.y, 5 - 1e-6)
            self.assertLessEqual(p.x + p.w, 1000 - 5 + 1e-6)
            self.assertLessEqual(p.y + p.h, 500 - 5 + 1e-6)

    def test_no_overlap_same_sheet(self):
        r = nest.nest_parts([nest.Part("A", 300, 200, qty=6)], 1000, 500, kerf=2)
        self.assertTrue(r.ok)
        by_sheet = {}
        for p in r.placements:
            by_sheet.setdefault(p.sheet, []).append(p)
        for placements in by_sheet.values():
            for i, a in enumerate(placements):
                for b in placements[i + 1 :]:
                    overlap_x = a.x < b.x + b.w - 1e-6 and b.x < a.x + a.w - 1e-6
                    overlap_y = a.y < b.y + b.h - 1e-6 and b.y < a.y + a.h - 1e-6
                    self.assertFalse(overlap_x and overlap_y, "parts overlap")


class NestPropertiesTest(unittest.TestCase):
    def test_utilization_between_zero_and_one(self):
        r = nest.nest_parts([nest.Part("A", 500, 250, qty=3)], 1000, 500)
        self.assertTrue(0.0 < r.utilization <= 1.0)

    def test_utilization_full_sheet(self):
        # Two 500x500 tiles exactly fill a 1000x500 sheet.
        r = nest.nest_parts([nest.Part("A", 500, 500, qty=2)], 1000, 500)
        self.assertTrue(r.ok)
        self.assertEqual(r.sheets_used, 1)
        self.assertAlmostEqual(r.utilization, 1.0, places=6)

    def test_cut_length_is_sum_of_perimeters(self):
        r = nest.nest_parts([nest.Part("A", 100, 50, qty=2)], 1000, 500)
        self.assertAlmostEqual(r.cut_length, 2 * 2 * (100 + 50))

    def test_multiple_sheets_when_needed(self):
        r = nest.nest_parts([nest.Part("A", 600, 400, qty=4)], 1000, 500)
        self.assertTrue(r.ok)
        self.assertGreaterEqual(r.sheets_used, 4)


class NestRotationTest(unittest.TestCase):
    def test_rotation_allows_fit(self):
        # A 400x900 part cannot fit un-rotated in 1000x500 (900 > 500), but
        # rotated to 900x400 it fits.
        r = nest.nest_parts([nest.Part("A", 400, 900)], 1000, 500, allow_rotate=True)
        self.assertTrue(r.ok)
        self.assertTrue(r.placements[0].rotated)

    def test_no_rotation_fails(self):
        r = nest.nest_parts([nest.Part("A", 900, 400)], 1000, 500, allow_rotate=False)
        # 900x400 fits un-rotated in 1000x500 so it still succeeds; use taller.
        r2 = nest.nest_parts([nest.Part("A", 400, 900)], 1000, 500, allow_rotate=False)
        self.assertFalse(r2.ok)


class NestErrorTest(unittest.TestCase):
    def test_oversized_part_reports_error(self):
        r = nest.nest_parts([nest.Part("Big", 2000, 100)], 1000, 500)
        self.assertFalse(r.ok)
        self.assertIn("exceeds", r.error)

    def test_margin_consuming_sheet_raises(self):
        with self.assertRaises(nest.NestingError):
            nest.nest_parts([nest.Part("A", 10, 10)], 100, 100, margin=60)


class NestDeterminismTest(unittest.TestCase):
    def test_same_input_same_layout(self):
        parts = [nest.Part("A", 300, 200, qty=5), nest.Part("B", 150, 100, qty=7)]
        r1 = nest.nest_parts(parts, 1000, 500, kerf=3, margin=5)
        r2 = nest.nest_parts(parts, 1000, 500, kerf=3, margin=5)
        self.assertEqual(
            [(p.name, p.x, p.y, p.rotated, p.sheet) for p in r1.placements],
            [(p.name, p.x, p.y, p.rotated, p.sheet) for p in r2.placements],
        )


class NestReportTest(unittest.TestCase):
    def test_report_contains_summary(self):
        r = nest.nest_parts([nest.Part("A", 300, 200, qty=4)], 1000, 500, kerf=3)
        text = nest.nest_report(r, material="12 mm plywood", kerf=3)
        self.assertIn("sheets used", text)
        self.assertIn("utilisation", text)
        self.assertIn("plywood", text)

    def test_report_failure(self):
        r = nest.nest_parts([nest.Part("Big", 5000, 5000)], 1000, 500)
        text = nest.nest_report(r)
        self.assertIn("FAILED", text)


if __name__ == "__main__":
    unittest.main()
