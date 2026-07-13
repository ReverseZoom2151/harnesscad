"""Tests for geometry.solidpy_path_offset."""

import math
import unittest

from harnesscad.domain.geometry.parametric.solidpy_path_offset import (
    LEFT_DIR,
    RIGHT_DIR,
    arc_points,
    cross_2d,
    direction_of_bend,
    fillet_corner,
    is_ccw,
    line_intersection,
    offset_point,
    offset_points,
    opposite_direction,
    path_2d,
    path_2d_paths,
    perpendicular_vector,
    round_polygon,
    signed_area,
)

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]  # counter-clockwise


class TestPrimitives(unittest.TestCase):
    def test_cross_2d(self):
        self.assertEqual(cross_2d((1, 0), (0, 1)), 1)
        self.assertEqual(cross_2d((0, 1), (1, 0)), -1)

    def test_direction_of_bend(self):
        self.assertEqual(direction_of_bend((0, 0), (1, 0), (1, 1)), LEFT_DIR)
        self.assertEqual(direction_of_bend((0, 0), (1, 0), (1, -1)), RIGHT_DIR)
        self.assertEqual(direction_of_bend((0, 0), (1, 0), (2, 0)), RIGHT_DIR)

    def test_opposite_direction(self):
        self.assertEqual(opposite_direction(LEFT_DIR), RIGHT_DIR)
        self.assertEqual(opposite_direction(RIGHT_DIR), LEFT_DIR)

    def test_perpendicular_vector(self):
        self.assertEqual(perpendicular_vector((0, 2), RIGHT_DIR), (2, 0))
        self.assertEqual(perpendicular_vector((0, 2), LEFT_DIR), (-2, 0))
        p = perpendicular_vector((0, 2), RIGHT_DIR, length=5)
        self.assertAlmostEqual(math.hypot(*p), 5.0)

    def test_perpendicular_of_zero(self):
        with self.assertRaises(ValueError):
            perpendicular_vector((0, 0), length=1)

    def test_line_intersection(self):
        self.assertEqual(line_intersection((0, 0), (1, 0), (2, -1), (0, 1)), (2, 0))
        self.assertIsNone(line_intersection((0, 0), (1, 0), (0, 1), (2, 0)))

    def test_signed_area(self):
        self.assertAlmostEqual(signed_area(SQUARE), 100.0)
        self.assertAlmostEqual(signed_area(list(reversed(SQUARE))), -100.0)
        self.assertTrue(is_ccw(SQUARE))


class TestOffset(unittest.TestCase):
    def test_offset_point(self):
        # corner of a CCW square, offset inward by 1
        p = offset_point((0, 0), (10, 0), (10, 10), offset=1, direction=LEFT_DIR)
        self.assertAlmostEqual(p[0], 9.0)
        self.assertAlmostEqual(p[1], 1.0)

    def test_offset_point_colinear(self):
        p = offset_point((0, 0), (5, 0), (10, 0), offset=1, direction=LEFT_DIR)
        self.assertAlmostEqual(p[1], 1.0)

    def test_internal_offset_shrinks_square(self):
        pts = offset_points(SQUARE, offset=1, internal=True, closed=True)
        self.assertEqual(len(pts), 4)
        self.assertAlmostEqual(signed_area(pts), 64.0)
        self.assertAlmostEqual(pts[0][0], 1.0)
        self.assertAlmostEqual(pts[0][1], 1.0)

    def test_external_offset_grows_square(self):
        pts = offset_points(SQUARE, offset=1, internal=False, closed=True)
        self.assertAlmostEqual(signed_area(pts), 144.0)

    def test_offset_is_reversible(self):
        shrunk = offset_points(SQUARE, offset=2, internal=True, closed=True)
        grown = offset_points(shrunk, offset=2, internal=False, closed=True)
        for a, b in zip(grown, SQUARE):
            self.assertAlmostEqual(a[0], b[0])
            self.assertAlmostEqual(a[1], b[1])

    def test_open_path_offset_keeps_endpoints(self):
        path = [(0, 0), (10, 0), (10, 10)]
        pts = offset_points(path, offset=1, internal=True, closed=False)
        self.assertEqual(len(pts), 3)
        # ends are the offset segment ends, not miters
        self.assertAlmostEqual(pts[0][0], 0.0)
        self.assertAlmostEqual(pts[0][1], 1.0)
        self.assertAlmostEqual(pts[-1][0], 9.0)
        self.assertAlmostEqual(pts[-1][1], 10.0)

    def test_sharp_corner_miters_far(self):
        # a spike corner miters much further out than the offset distance
        path = [(0, 0), (10, 0), (0, 1)]
        pts = offset_points(path, offset=1, internal=True, closed=False)
        miter = pts[1]
        self.assertGreater(math.hypot(miter[0] - 10, miter[1]), 10.0)

    def test_two_point_open_path(self):
        pts = offset_points([(0, 0), (10, 0)], offset=1, internal=True, closed=False)
        self.assertEqual(len(pts), 2)
        self.assertAlmostEqual(abs(pts[0][1]), 1.0)

    def test_too_few_points(self):
        with self.assertRaises(ValueError):
            offset_points([(0, 0), (1, 1)], offset=1, closed=True)
        with self.assertRaises(ValueError):
            offset_points([(0, 0)], offset=1, closed=False)

    def test_determinism(self):
        self.assertEqual(offset_points(SQUARE, 1), offset_points(SQUARE, 1))


class TestStroke(unittest.TestCase):
    def test_open_stroke_point_count_and_area(self):
        pts = path_2d([(0, 0), (10, 0)], width=2, closed=False)
        self.assertEqual(len(pts), 4)
        self.assertAlmostEqual(abs(signed_area(pts)), 20.0)

    def test_closed_stroke_makes_two_rings(self):
        pts = path_2d(SQUARE, width=2, closed=True)
        self.assertEqual(len(pts), 8)
        paths = path_2d_paths(SQUARE, closed=True)
        self.assertEqual(paths, [[0, 1, 2, 3], [4, 5, 6, 7]])
        inner = pts[:4]
        outer = pts[4:]
        self.assertLess(abs(signed_area(inner)), abs(signed_area(outer)))
        # the ring between them has the area of the wall: 4 sides * 10 * 2 + corners
        self.assertAlmostEqual(abs(signed_area(outer)) - abs(signed_area(inner)), 80.0)

    def test_open_paths_indices(self):
        self.assertEqual(path_2d_paths([(0, 0), (1, 0), (2, 0)], closed=False),
                         [list(range(8))])


class TestFillet(unittest.TestCase):
    def test_arc_points(self):
        pts = arc_points((0, 0), 1, 0, 90, segments=4)
        self.assertEqual(len(pts), 5)
        self.assertAlmostEqual(pts[0][0], 1.0)
        self.assertAlmostEqual(pts[-1][1], 1.0)
        for p in pts:
            self.assertAlmostEqual(math.hypot(*p), 1.0)

    def test_arc_takes_short_way(self):
        pts = arc_points((0, 0), 1, 350, 10, segments=2)
        self.assertAlmostEqual(pts[1][1], 0.0, places=9)  # passes through angle 0

    def test_fillet_corner_center_and_tangency(self):
        center, arc = fillet_corner((0, 0), (10, 0), (10, 10), radius=2, segments=8)
        self.assertAlmostEqual(center[0], 8.0)
        self.assertAlmostEqual(center[1], 2.0)
        self.assertEqual(len(arc), 9)
        for p in arc:
            self.assertAlmostEqual(math.hypot(p[0] - center[0], p[1] - center[1]), 2.0)
        # endpoints are the tangent points on the two segments
        self.assertAlmostEqual(arc[0][0], 8.0)
        self.assertAlmostEqual(arc[0][1], 0.0)
        self.assertAlmostEqual(arc[-1][0], 10.0)
        self.assertAlmostEqual(arc[-1][1], 2.0)

    def test_fillet_radius_too_large(self):
        with self.assertRaises(ValueError):
            fillet_corner((0, 0), (1, 0), (1, 1), radius=5)

    def test_fillet_straight_corner(self):
        with self.assertRaises(ValueError):
            fillet_corner((0, 0), (1, 0), (2, 0), radius=0.1)

    def test_fillet_bad_radius(self):
        with self.assertRaises(ValueError):
            fillet_corner((0, 0), (1, 0), (1, 1), radius=0)

    def test_round_polygon_area(self):
        # rounding the 4 corners of a 10x10 square with r=1 removes
        # (4 - pi) * r^2 of area
        r = 1.0
        pts = round_polygon(SQUARE, radius=r, segments=64)
        expected = 100.0 - (4 - math.pi) * r * r
        self.assertAlmostEqual(abs(signed_area(pts)), expected, places=3)
        self.assertEqual(len(pts), 4 * 65)

    def test_round_polygon_open(self):
        pts = round_polygon([(0, 0), (10, 0), (10, 10)], radius=1, segments=4,
                            closed=False)
        self.assertEqual(pts[0], (0.0, 0.0))
        self.assertEqual(pts[-1], (10.0, 10.0))
        self.assertEqual(len(pts), 2 + 5)

    def test_round_polygon_determinism(self):
        self.assertEqual(round_polygon(SQUARE, 1, 8), round_polygon(SQUARE, 1, 8))


if __name__ == "__main__":
    unittest.main()
