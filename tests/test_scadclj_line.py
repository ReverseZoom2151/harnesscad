"""Tests for geometry.scadclj_line."""

import math
import unittest

from geometry.scadclj_line import (
    direction_rotation,
    line,
    lines,
    segment_length,
)
from programs.scadclj_data_ir import write_scad


class SegmentLengthTest(unittest.TestCase):
    def test_axis_aligned(self):
        self.assertAlmostEqual(segment_length([0, 0, 0], [0, 0, 10]), 10.0)

    def test_diagonal(self):
        self.assertAlmostEqual(segment_length([0, 0, 0], [3, 4, 0]), 5.0)


class DirectionRotationTest(unittest.TestCase):
    def test_along_z_is_identity(self):
        angle, axis, length = direction_rotation([0, 0, 0], [0, 0, 5])
        self.assertAlmostEqual(angle, 0.0)
        self.assertAlmostEqual(length, 5.0)

    def test_along_x_is_ninety_degrees(self):
        angle, axis, length = direction_rotation([0, 0, 0], [5, 0, 0])
        self.assertAlmostEqual(angle, math.pi / 2)
        # axis = [-dy, dx, 0] = [0, 5, 0] -> rotate +Z toward +X about +Y
        self.assertAlmostEqual(axis[0], 0.0)
        self.assertAlmostEqual(axis[1], 5.0)
        self.assertAlmostEqual(axis[2], 0.0)

    def test_negative_z_is_pi_with_fallback_axis(self):
        angle, axis, length = direction_rotation([0, 0, 0], [0, 0, -3])
        self.assertAlmostEqual(angle, math.pi)
        # degenerate perpendicular -> X-axis fallback, not [0,0,0]
        self.assertEqual(axis, [1.0, 0.0, 0.0])

    def test_coincident(self):
        angle, axis, length = direction_rotation([1, 2, 3], [1, 2, 3])
        self.assertEqual(length, 0.0)
        self.assertEqual(axis, [0.0, 0.0, 1.0])

    def test_clamp_no_domain_error(self):
        # a purely diagonal segment must not blow acos's domain
        angle, axis, length = direction_rotation([0, 0, 0], [1, 1, 1])
        self.assertTrue(0.0 <= angle <= math.pi)


class LineTest(unittest.TestCase):
    def test_line_emits_capsule(self):
        out = write_scad(line([0, 0, 0], [0, 0, 10], radius=2))
        self.assertTrue(out.startswith("union () {\n"))
        self.assertIn("sphere (r=2)", out)
        self.assertIn("cylinder (h=10, r=2", out)

    def test_line_has_two_caps(self):
        out = write_scad(line([0, 0, 0], [10, 0, 0], radius=1))
        self.assertEqual(out.count("sphere (r=1)"), 2)

    def test_zero_length_is_single_sphere(self):
        node = line([1, 1, 1], [1, 1, 1], radius=3)
        out = write_scad(node)
        self.assertEqual(out.count("sphere (r=3)"), 1)
        self.assertIn("translate ([1, 1, 1])", out)

    def test_cap_placed_at_end(self):
        out = write_scad(line([0, 0, 0], [5, 0, 0], radius=1))
        self.assertIn("translate ([5, 0, 0])", out)


class LinesTest(unittest.TestCase):
    def test_polyline_is_union_of_segments(self):
        node = lines([[0, 0, 0], [10, 0, 0], [10, 10, 0]], radius=1)
        self.assertEqual(node[0], ":union")
        # two segments
        self.assertEqual(len(node) - 1, 2)

    def test_single_point(self):
        out = write_scad(lines([[2, 2, 2]], radius=1))
        self.assertEqual(out.count("sphere (r=1)"), 1)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            lines([])

    def test_render_is_valid_text(self):
        out = write_scad(lines([[0, 0, 0], [1, 2, 3], [4, 5, 6]], radius=0.5))
        # balanced braces in the emitted source
        self.assertEqual(out.count("{"), out.count("}"))


if __name__ == "__main__":
    unittest.main()
