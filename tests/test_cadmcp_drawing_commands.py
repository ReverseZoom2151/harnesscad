"""Tests for drawings.cadmcp_drawing_commands."""

import math
import unittest

from harnesscad.domain.drawings.drawing_commands import (
    DrawingBuilder,
    DrawingCommandError,
    VALID_LINEWEIGHTS,
    arc,
    circle,
    ellipse,
    ensure_3d,
    extents,
    hatch,
    line,
    polyline,
    rectangle,
    text,
    validate_lineweight,
)


class Ensure3DTests(unittest.TestCase):
    def test_promote_2d(self):
        self.assertEqual(ensure_3d((1, 2)), (1.0, 2.0, 0.0))

    def test_keep_3d(self):
        self.assertEqual(ensure_3d((1, 2, 3)), (1.0, 2.0, 3.0))

    def test_bad_length(self):
        with self.assertRaises(DrawingCommandError):
            ensure_3d((1,))


class LineweightTests(unittest.TestCase):
    def test_none_passthrough(self):
        self.assertIsNone(validate_lineweight(None))

    def test_valid_kept(self):
        for lw in VALID_LINEWEIGHTS:
            self.assertEqual(validate_lineweight(lw), lw)

    def test_invalid_snaps_to_zero(self):
        self.assertEqual(validate_lineweight(7), 0)
        self.assertEqual(validate_lineweight(999), 0)


class PrimitiveTests(unittest.TestCase):
    def test_line(self):
        e = line((0, 0), (10, 5), lineweight=13)
        self.assertEqual(e.kind, "line")
        self.assertEqual(e.geometry["start"], (0.0, 0.0, 0.0))
        self.assertEqual(e.geometry["end"], (10.0, 5.0, 0.0))
        self.assertEqual(e.lineweight, 13)

    def test_circle_positive(self):
        e = circle((1, 2), 5)
        self.assertEqual(e.geometry["radius"], 5.0)
        self.assertEqual(e.geometry["center"], (1.0, 2.0, 0.0))

    def test_circle_nonpositive_raises(self):
        with self.assertRaises(DrawingCommandError):
            circle((0, 0), 0)

    def test_arc_deg_to_rad(self):
        e = arc((0, 0), 10, 0, 90)
        self.assertAlmostEqual(e.geometry["start_angle_rad"], 0.0)
        self.assertAlmostEqual(e.geometry["end_angle_rad"], math.pi / 2)
        self.assertEqual(e.geometry["end_angle_deg"], 90.0)

    def test_ellipse_vector_and_ratio(self):
        e = ellipse((0, 0), 10, 5, rotation=0)
        self.assertAlmostEqual(e.geometry["ratio"], 0.5)
        vx, vy, vz = e.geometry["major_axis_vector"]
        self.assertAlmostEqual(vx, 10.0)
        self.assertAlmostEqual(vy, 0.0)
        self.assertAlmostEqual(vz, 0.0)

    def test_ellipse_rotated(self):
        e = ellipse((0, 0), 10, 5, rotation=90)
        vx, vy, _ = e.geometry["major_axis_vector"]
        self.assertAlmostEqual(vx, 0.0, places=9)
        self.assertAlmostEqual(vy, 10.0)

    def test_ellipse_bad_axes(self):
        with self.assertRaises(DrawingCommandError):
            ellipse((0, 0), 0, 5)

    def test_text(self):
        e = text((1, 1), "hi", height=3, rotation=180)
        self.assertEqual(e.geometry["text"], "hi")
        self.assertAlmostEqual(e.geometry["rotation_rad"], math.pi)


class RectangleTests(unittest.TestCase):
    def test_closed_five_vertices(self):
        e = rectangle((0, 0), (4, 2))
        pts = e.geometry["points"]
        self.assertEqual(len(pts), 5)
        self.assertEqual(pts[0], pts[-1])
        self.assertEqual(pts, [
            (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 2.0, 0.0),
            (0.0, 2.0, 0.0), (0.0, 0.0, 0.0)])
        self.assertTrue(e.geometry["closed"])


class PolylineTests(unittest.TestCase):
    def test_needs_two_points(self):
        with self.assertRaises(DrawingCommandError):
            polyline([(0, 0)])

    def test_two_points_not_closed(self):
        e = polyline([(0, 0), (1, 1)], closed=True)
        self.assertFalse(e.geometry["closed"])  # closed only when > 2 points

    def test_three_points_closed(self):
        e = polyline([(0, 0), (1, 0), (1, 1)], closed=True)
        self.assertTrue(e.geometry["closed"])


class HatchTests(unittest.TestCase):
    def test_needs_three_points(self):
        with self.assertRaises(DrawingCommandError):
            hatch([(0, 0), (1, 1)])

    def test_pattern_upper(self):
        e = hatch([(0, 0), (1, 0), (1, 1)], pattern_name="ansi31")
        self.assertEqual(e.geometry["pattern_name"], "ANSI31")


class ExtentsTests(unittest.TestCase):
    def test_empty(self):
        self.assertIsNone(extents([]))

    def test_circle_bbox(self):
        (mn, mx) = extents([circle((0, 0), 5)])
        self.assertEqual(mn, (-5.0, -5.0, 0.0))
        self.assertEqual(mx, (5.0, 5.0, 0.0))

    def test_builder_and_extents(self):
        b = DrawingBuilder()
        b.add(line((0, 0), (10, 0)))
        b.add(circle((10, 10), 2))
        (mn, mx) = b.extents()
        self.assertEqual(mn[0], 0.0)
        self.assertEqual(mx[0], 12.0)
        self.assertEqual(mx[1], 12.0)
        self.assertEqual(len(b.to_list()), 2)


if __name__ == "__main__":
    unittest.main()
