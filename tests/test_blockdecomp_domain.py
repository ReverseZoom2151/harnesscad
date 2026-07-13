"""Tests for geometry.blockdecomp_domain."""

import math
import unittest

from harnesscad.domain.geometry.blockdecomp_domain import Shape, Corner, classify_angle


class TestClassifyAngle(unittest.TestCase):
    def test_right_angle(self):
        self.assertEqual(classify_angle((1.0, 0.0), (0.0, 1.0)), "right")

    def test_acute(self):
        self.assertEqual(classify_angle((1.0, 0.0), (1.0, 0.5)), "acute")

    def test_obtuse(self):
        self.assertEqual(classify_angle((1.0, 0.0), (-1.0, 1.0)), "obtuse")

    def test_straight(self):
        self.assertEqual(classify_angle((1.0, 0.0), (-1.0, 0.0)), "straight")


class TestRectangleShape(unittest.TestCase):
    def setUp(self):
        self.sq = Shape.from_rectangles([(0.0, 0.0, 2.0, 2.0)])

    def test_area(self):
        self.assertAlmostEqual(self.sq.area(), 4.0)

    def test_bbox(self):
        self.assertEqual(self.sq.bbox(), (0.0, 0.0, 2.0, 2.0))

    def test_square_aspect_ratio_is_one(self):
        self.assertAlmostEqual(self.sq.aspect_ratio(), 1.0)

    def test_centroid(self):
        cx, cy = self.sq.centroid()
        self.assertAlmostEqual(cx, 1.0)
        self.assertAlmostEqual(cy, 1.0)

    def test_is_rectangle_and_quad(self):
        self.assertTrue(self.sq.is_rectangle())
        self.assertTrue(self.sq.is_quad())

    def test_connected(self):
        self.assertTrue(self.sq.is_connected())

    def test_four_corners_all_convex(self):
        cs = self.sq.corners()
        self.assertEqual(len(cs), 4)
        self.assertTrue(all(c.corner_type == "convex" for c in cs))
        self.assertTrue(all(abs(c.interior_angle - 90.0) < 1e-9 for c in cs))


class TestAspectRatio(unittest.TestCase):
    def test_non_square(self):
        r = Shape.from_rectangles([(0.0, 0.0, 4.0, 1.0)])
        self.assertAlmostEqual(r.aspect_ratio(), 4.0)


class TestLShape(unittest.TestCase):
    def setUp(self):
        # L-shape: 2x2 square minus its top-right 1x1 cell region.
        self.l = Shape.from_rectangles([(0.0, 0.0, 2.0, 1.0), (0.0, 0.0, 1.0, 2.0)])

    def test_area(self):
        self.assertAlmostEqual(self.l.area(), 3.0)

    def test_not_rectangle(self):
        self.assertFalse(self.l.is_rectangle())

    def test_has_one_reentrant_corner(self):
        reentrant = [c for c in self.l.corners() if c.corner_type == "reentrant"]
        self.assertEqual(len(reentrant), 1)
        self.assertEqual(reentrant[0].pos, (1.0, 1.0))

    def test_six_corners(self):
        self.assertEqual(self.l.num_corners(), 6)

    def test_connected(self):
        self.assertTrue(self.l.is_connected())


class TestFromPolygon(unittest.TestCase):
    def test_matches_rectangles(self):
        poly = [(0.0, 0.0), (3.0, 0.0), (3.0, 2.0), (0.0, 2.0)]
        s = Shape.from_polygon(poly)
        self.assertAlmostEqual(s.area(), 6.0)
        self.assertTrue(s.is_rectangle())

    def test_l_polygon(self):
        poly = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
        poly = [(float(a), float(b)) for a, b in poly]
        s = Shape.from_polygon(poly)
        self.assertAlmostEqual(s.area(), 3.0)
        self.assertFalse(s.is_rectangle())


class TestDisconnected(unittest.TestCase):
    def test_two_separate_rects(self):
        s = Shape.from_rectangles([(0.0, 0.0, 1.0, 1.0), (3.0, 0.0, 4.0, 1.0)])
        self.assertFalse(s.is_connected())
        self.assertEqual(len(s.connected_components()), 2)


if __name__ == "__main__":
    unittest.main()
