"""Tests for the flat-pack panel decomposition module.

Reproduces concrete cabinet cases from "How Can Large Language Models
Help Humans in Design and Manufacturing" (Makatura et al.):

  * the OpenJSCAD cabinet, 30in tall x 20in wide x 18in deep, 0.75in ply;
  * the SVG cabinet, 72in x 48in x 12in, 0.5in ply, 3 shelves;
  * the Figure-68 bed-fit / split case on a 12in x 24in laser bed.
"""

import unittest

from harnesscad.domain.fabrication.flatpack_panels import (
    Panel,
    decompose_cabinet,
    fits_on_bed,
    nest_report,
    split_panel_to_fit,
    total_material_area,
)


class TestFlatpackPanels(unittest.TestCase):
    def _by_name(self, panels):
        return {p.name: p for p in panels}

    def test_side_panels_full_dimensions(self):
        # OpenJSCAD cabinet: sides are full D x H, thickness t.
        panels = decompose_cabinet(30, 20, 18, 0.75)
        by_name = self._by_name(panels)
        for name in ("side_left", "side_right"):
            side = by_name[name]
            self.assertEqual(side.width, 18)   # D
            self.assertEqual(side.height, 30)  # H
            self.assertEqual(side.thickness, 0.75)

    def test_top_bottom_fit_between_sides(self):
        # top/bottom fit between sides: width = W - 2t, height = D.
        panels = decompose_cabinet(30, 20, 18, 0.75)
        by_name = self._by_name(panels)
        for name in ("top", "bottom"):
            board = by_name[name]
            self.assertEqual(board.width, 20 - 2 * 0.75)  # 18.5
            self.assertEqual(board.height, 18)            # D
            self.assertEqual(board.thickness, 0.75)

    def test_back_inner_fit_convention(self):
        # Default back fits inside: (W - 2t) x (H - 2t).
        panels = decompose_cabinet(30, 20, 18, 0.75)
        back = self._by_name(panels)["back"]
        self.assertEqual(back.width, 20 - 2 * 0.75)   # 18.5
        self.assertEqual(back.height, 30 - 2 * 0.75)  # 28.5
        self.assertEqual(back.thickness, 0.75)

    def test_back_full_cover_alternative(self):
        panels = decompose_cabinet(30, 20, 18, 0.75, back_full_cover=True)
        back = self._by_name(panels)["back"]
        self.assertEqual(back.width, 20)
        self.assertEqual(back.height, 30)

    def test_shelf_dimensions_and_count(self):
        # SVG cabinet: 72 x 48 x 12, 0.5 ply, 3 shelves.
        panels = decompose_cabinet(72, 48, 12, 0.5, num_shelves=3)
        shelves = [p for p in panels if p.name.startswith("shelf_")]
        self.assertEqual(len(shelves), 3)
        for shelf in shelves:
            self.assertEqual(shelf.width, 48 - 2 * 0.5)  # 47
            self.assertEqual(shelf.height, 12)           # D
            self.assertEqual(shelf.thickness, 0.5)
        # Total panel count: 2 sides + top + bottom + back + 3 shelves = 8.
        self.assertEqual(len(panels), 8)

    def test_positive_inner_dim_valueerror(self):
        # 2t >= W must raise (thin side boards / overlapping walls).
        with self.assertRaises(ValueError):
            decompose_cabinet(30, 1.0, 18, 0.75)  # 2*0.75 = 1.5 >= 1.0
        # 2t >= H must raise as well.
        with self.assertRaises(ValueError):
            decompose_cabinet(1.0, 20, 18, 0.75)

    def test_total_material_area(self):
        panels = decompose_cabinet(30, 20, 18, 0.75)
        expected = (
            2 * (18 * 30)          # two sides
            + 2 * (18.5 * 18)      # top + bottom
            + (18.5 * 28.5)        # back
            + (18.5 * 18)          # one shelf (default num_shelves=1)
        )
        self.assertAlmostEqual(total_material_area(panels), expected)

    def test_fits_on_bed_both_orientations(self):
        bed_w, bed_h = 12, 24
        # Fits as-is.
        self.assertTrue(fits_on_bed(Panel("a", 10, 20, 0.5), bed_w, bed_h))
        # Fits only when rotated (20 wide, 10 tall -> rotate).
        self.assertTrue(fits_on_bed(Panel("b", 20, 10, 0.5), bed_w, bed_h))
        # Does not fit in any orientation.
        self.assertFalse(fits_on_bed(Panel("c", 20, 20, 0.5), bed_w, bed_h))

    def test_split_panel_halving(self):
        # Figure-68 style: back board wider than the 12in bed is halved
        # along its longer (width) dimension.
        bed_w, bed_h = 12, 24
        board = Panel("back", 24, 20, 0.5)  # 24 wide, does not fit as-is
        self.assertFalse(fits_on_bed(board, bed_w, bed_h))
        strips = split_panel_to_fit(board, bed_w, bed_h)
        self.assertEqual(len(strips), 2)
        self.assertEqual([s.name for s in strips], ["back_1", "back_2"])
        for s in strips:
            self.assertEqual(s.width, 12)   # 24 / 2
            self.assertEqual(s.height, 20)
            self.assertTrue(fits_on_bed(s, bed_w, bed_h))

    def test_split_panel_already_fits(self):
        bed_w, bed_h = 12, 24
        board = Panel("shelf", 10, 20, 0.5)
        strips = split_panel_to_fit(board, bed_w, bed_h)
        self.assertEqual(len(strips), 1)
        self.assertIs(strips[0], board)

    def test_split_into_three(self):
        bed_w, bed_h = 12, 24
        # 30 long dimension, short dimension 24 fits bed_h -> split long
        # into ceil(30/24) = 2? 30/2 = 15 > 12 and rotate 15>24? no ->
        # rotate check: strip 15 x 24 -> as-is 15>12; rotated 15<=24 and
        # 24<=12? no. So needs 3 strips (30/3 = 10 <= 12).
        board = Panel("long", 30, 24, 0.5)
        strips = split_panel_to_fit(board, bed_w, bed_h)
        self.assertEqual(len(strips), 3)
        for s in strips:
            self.assertAlmostEqual(s.width, 10)
            self.assertEqual(s.height, 24)
            self.assertTrue(fits_on_bed(s, bed_w, bed_h))

    def test_split_valueerror_short_dim_too_big(self):
        # Short dimension 30 exceeds largest bed dimension 24: unsplittable.
        bed_w, bed_h = 12, 24
        board = Panel("huge", 40, 30, 0.5)
        with self.assertRaises(ValueError):
            split_panel_to_fit(board, bed_w, bed_h)

    def test_nest_report_end_to_end(self):
        # Figure-68 bed case: OpenJSCAD cabinet (30 x 20 x 18, 0.75 ply)
        # on a 12 x 24 laser bed.  Panels wider than the bed get split.
        bed_w, bed_h = 12, 24
        panels = decompose_cabinet(30, 20, 18, 0.75)
        report = nest_report(panels, bed_w, bed_h)
        # None of these panels fit the narrow 12in bed as-is: the sides are
        # 18 x 30 and the top/bottom/back/shelf are all 18(.5) wide > 12.
        self.assertEqual(report["fit"], [])
        self.assertIn("side_left", report["needs_split"])
        self.assertIn("side_right", report["needs_split"])
        self.assertIn("top", report["needs_split"])
        self.assertIn("back", report["needs_split"])
        # Every split panel must fit the bed.
        for p in report["split_panels"]:
            self.assertTrue(fits_on_bed(p, bed_w, bed_h))
        # Equal-strip splitting conserves total material area.
        self.assertAlmostEqual(
            report["total_area"], total_material_area(panels)
        )


if __name__ == "__main__":
    unittest.main()
