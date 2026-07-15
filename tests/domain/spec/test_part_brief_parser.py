"""Tests for the cad-agent dimensional-brief parser and SCAD emitter."""

import unittest

from harnesscad.domain.spec import part_brief_parser as pbp


class ParseTest(unittest.TestCase):
    def test_plate_dimensions(self):
        spec = pbp.parse_part_brief("mounting plate 80x40x3mm")
        self.assertEqual(spec.kind, "plate")
        self.assertEqual((spec.width, spec.depth, spec.height), (80.0, 40.0, 3.0))

    def test_box_detected(self):
        spec = pbp.parse_part_brief("electronics enclosure 100x60x40mm wall 2mm")
        self.assertEqual(spec.kind, "box")
        self.assertEqual(spec.wall, 2.0)

    def test_holes_parsed(self):
        spec = pbp.parse_part_brief("plate 80x40x3mm with 4 holes 5mm")
        self.assertEqual(spec.holes, 4)
        self.assertEqual(spec.hole_diameter, 5.0)

    def test_missing_dims_raises(self):
        with self.assertRaises(pbp.BriefParseError):
            pbp.parse_part_brief("a nice bracket please")

    def test_whitespace_and_case_tolerant(self):
        spec = pbp.parse_part_brief("PLATE 80 X 40 X 3 MM")
        self.assertEqual((spec.width, spec.depth, spec.height), (80.0, 40.0, 3.0))


class HolePositionsTest(unittest.TestCase):
    def test_no_holes(self):
        spec = pbp.PartSpec("plate", 80, 40, 3, holes=0)
        self.assertEqual(pbp.hole_positions(spec), [])

    def test_two_holes_take_two_corners(self):
        spec = pbp.PartSpec("plate", 80, 40, 3, holes=2, hole_diameter=5)
        pos = pbp.hole_positions(spec)
        self.assertEqual(len(pos), 2)

    def test_margin_uses_hole_diameter(self):
        spec = pbp.PartSpec("plate", 80, 40, 3, holes=1, hole_diameter=5)
        (x, y) = pbp.hole_positions(spec)[0]
        # margin = max(2*5, 0.15*40) = 10
        self.assertAlmostEqual(x, 10.0)
        self.assertAlmostEqual(y, 10.0)


class EmitTest(unittest.TestCase):
    def test_plate_emits_difference_and_holes(self):
        spec = pbp.PartSpec("plate", 80, 40, 3, holes=4, hole_diameter=5)
        scad = pbp.emit_openscad(spec)
        self.assertIn("difference()", scad)
        self.assertEqual(scad.count("cylinder("), 4)

    def test_box_emits_inner_cube(self):
        spec = pbp.PartSpec("box", 100, 60, 40, wall=2)
        scad = pbp.emit_openscad(spec)
        self.assertIn("difference()", scad)
        # outer + inner cube
        self.assertEqual(scad.count("cube("), 2)

    def test_brief_to_openscad_roundtrip(self):
        scad = pbp.brief_to_openscad("plate 80x40x3mm with 2 holes 5mm")
        self.assertEqual(scad.count("cylinder("), 2)

    def test_box_inner_dims_shrunk_by_wall(self):
        spec = pbp.PartSpec("box", 100, 60, 40, wall=3)
        scad = pbp.emit_openscad(spec)
        self.assertIn("94.0, 54.0", scad)  # 100-6, 60-6


if __name__ == "__main__":
    unittest.main()
