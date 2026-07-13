"""Tests for programs.cadmcp_command_parser."""

import unittest

from harnesscad.domain.programs.extract.command_parser import (
    extract_coordinates,
    extract_keyword_value,
    extract_numbers,
    identify_command,
    parse_command,
)


class ExtractionTests(unittest.TestCase):
    def test_coords_paren_and_bare(self):
        self.assertEqual(
            extract_coordinates("from (0,0) to 10,20,5"),
            [(0.0, 0.0, 0.0), (10.0, 20.0, 5.0)])

    def test_coords_negative(self):
        self.assertEqual(extract_coordinates("(-3.5, 2)"), [(-3.5, 2.0, 0.0)])

    def test_numbers(self):
        self.assertEqual(extract_numbers("r 5 and 12.5"), [5.0, 12.5])

    def test_keyword_value(self):
        self.assertEqual(extract_keyword_value("radius 7", ["radius", "r"]), 7.0)

    def test_keyword_value_missing(self):
        self.assertIsNone(extract_keyword_value("no number here", ["radius"]))

    def test_keyword_value_negative(self):
        self.assertEqual(
            extract_keyword_value("rotation -45", ["rotation"]), -45.0)


class IdentifyTests(unittest.TestCase):
    def test_draw_circle(self):
        self.assertEqual(identify_command("draw a circle"), "draw_circle")

    def test_create_rectangle(self):
        self.assertEqual(identify_command("create a rect"), "draw_rectangle")

    def test_save(self):
        self.assertEqual(identify_command("save the drawing"), "save")

    def test_unknown(self):
        self.assertEqual(identify_command("hello world"), "unknown")


class ParseTests(unittest.TestCase):
    def test_line_with_coords(self):
        r = parse_command("draw a line from (0,0) to (10,10)")
        self.assertEqual(r["type"], "draw_line")
        self.assertEqual(r["start"], (0.0, 0.0, 0.0))
        self.assertEqual(r["end"], (10.0, 10.0, 0.0))

    def test_line_defaulted(self):
        r = parse_command("draw a line")
        self.assertTrue(r.get("defaulted"))
        self.assertEqual(r["end"], (100.0, 100.0, 0.0))

    def test_circle_radius_keyword(self):
        r = parse_command("draw a circle at (5,5) radius 8")
        self.assertEqual(r["center"], (5.0, 5.0, 0.0))
        self.assertEqual(r["radius"], 8.0)

    def test_circle_default_radius(self):
        r = parse_command("draw a circle")
        self.assertEqual(r["radius"], 50.0)

    def test_arc_angles(self):
        r = parse_command("draw an arc radius 10 start angle 30 end angle 120")
        self.assertEqual(r["radius"], 10.0)
        self.assertEqual(r["start_angle"], 30.0)
        self.assertEqual(r["end_angle"], 120.0)

    def test_rectangle_two_corners(self):
        r = parse_command("draw a rectangle (0,0) (4,2)")
        self.assertEqual(r["corner1"], (0.0, 0.0, 0.0))
        self.assertEqual(r["corner2"], (4.0, 2.0, 0.0))

    def test_rectangle_width_height(self):
        r = parse_command("draw a rectangle width 30 height 20")
        self.assertEqual(r["corner1"], (0.0, 0.0, 0.0))
        self.assertEqual(r["corner2"], (30.0, 20.0, 0.0))

    def test_polyline(self):
        r = parse_command("draw a polyline (0,0) (1,0) (1,1) closed")
        self.assertEqual(r["type"], "draw_polyline")
        self.assertTrue(r["closed"])
        self.assertEqual(len(r["points"]), 3)

    def test_polyline_insufficient(self):
        r = parse_command("draw a polyline (0,0)")
        self.assertEqual(r["type"], "error")

    def test_text_quoted(self):
        r = parse_command('add text "Hello" at (2,2) height 5')
        self.assertEqual(r["text"], "Hello")
        self.assertEqual(r["height"], 5.0)

    def test_hatch(self):
        r = parse_command("fill (0,0) (1,0) (1,1) pattern ansi31 scale 2")
        self.assertEqual(r["type"], "draw_hatch")
        self.assertEqual(r["pattern_name"], "ANSI31")
        self.assertEqual(r["scale"], 2.0)

    def test_save_path(self):
        r = parse_command('save to "out.dwg"')
        self.assertEqual(r["type"], "save")
        self.assertEqual(r["file_path"], "out.dwg")

    def test_unknown(self):
        r = parse_command("do something odd")
        self.assertEqual(r["type"], "unknown")


class PipelineTests(unittest.TestCase):
    """Parsed commands feed the drawing-command builder cleanly."""

    def test_circle_pipeline(self):
        from harnesscad.domain.drawings.drawing_commands import circle
        r = parse_command("draw a circle at (0,0) radius 4")
        e = circle(r["center"], r["radius"])
        self.assertEqual(e.geometry["radius"], 4.0)

    def test_rectangle_pipeline(self):
        from harnesscad.domain.drawings.drawing_commands import rectangle
        r = parse_command("draw a rectangle (0,0) (4,2)")
        e = rectangle(r["corner1"], r["corner2"])
        self.assertEqual(len(e.geometry["points"]), 5)


if __name__ == "__main__":
    unittest.main()
