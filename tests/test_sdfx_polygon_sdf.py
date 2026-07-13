"""Tests for geometry.sdfx_polygon_sdf."""

import math
import unittest

from harnesscad.domain.geometry.sdf.polygon import (
    point_in_polygon,
    polygon_area,
    polygon_centroid,
    polygon_distance,
    polygon_sdf,
    polygon_winding,
    prepare_edges,
)

# Unit square centered at origin.
SQUARE = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]

# A concave "L" polygon.
LSHAPE = [
    (0.0, 0.0),
    (2.0, 0.0),
    (2.0, 1.0),
    (1.0, 1.0),
    (1.0, 2.0),
    (0.0, 2.0),
]


class TestSquare(unittest.TestCase):
    def test_center_is_inside(self):
        self.assertLess(polygon_sdf((0.0, 0.0), SQUARE), 0.0)
        self.assertTrue(point_in_polygon((0.0, 0.0), SQUARE))

    def test_center_distance(self):
        # center of unit square -> nearest edge at distance 1.
        self.assertAlmostEqual(polygon_sdf((0.0, 0.0), SQUARE), -1.0)

    def test_outside_point(self):
        d = polygon_sdf((3.0, 0.0), SQUARE)
        self.assertAlmostEqual(d, 2.0)
        self.assertFalse(point_in_polygon((3.0, 0.0), SQUARE))

    def test_outside_corner(self):
        # nearest feature is the (1,1) corner.
        d = polygon_sdf((2.0, 2.0), SQUARE)
        self.assertAlmostEqual(d, math.sqrt(2.0))

    def test_on_edge_is_zero(self):
        self.assertAlmostEqual(polygon_sdf((1.0, 0.0), SQUARE), 0.0)


class TestConcave(unittest.TestCase):
    def test_notch_is_outside(self):
        # The (1.5, 1.5) point sits in the removed notch of the L -> outside.
        self.assertFalse(point_in_polygon((1.5, 1.5), LSHAPE))
        self.assertGreater(polygon_sdf((1.5, 1.5), LSHAPE), 0.0)

    def test_inside_arm(self):
        self.assertTrue(point_in_polygon((0.5, 0.5), LSHAPE))
        self.assertLess(polygon_sdf((0.5, 0.5), LSHAPE), 0.0)

    def test_notch_distance(self):
        # (1.5,1.5): nearest boundary is the vertical edge x=1 (y in [1,2]).
        self.assertAlmostEqual(polygon_sdf((1.5, 1.5), LSHAPE), 0.5)


class TestWindingAndGeometry(unittest.TestCase):
    def test_winding_inside_nonzero(self):
        edges = prepare_edges(SQUARE)
        self.assertNotEqual(polygon_winding(0.0, 0.0, edges), 0)

    def test_winding_outside_zero(self):
        edges = prepare_edges(SQUARE)
        self.assertEqual(polygon_winding(5.0, 5.0, edges), 0)

    def test_unsigned_distance(self):
        edges = prepare_edges(SQUARE)
        self.assertAlmostEqual(polygon_distance(0.0, 0.0, edges), 1.0)

    def test_area(self):
        self.assertAlmostEqual(polygon_area(SQUARE), 4.0)
        self.assertAlmostEqual(polygon_area(LSHAPE), 3.0)

    def test_centroid_square(self):
        cx, cy = polygon_centroid(SQUARE)
        self.assertAlmostEqual(cx, 0.0)
        self.assertAlmostEqual(cy, 0.0)

    def test_degenerate_vertices_skipped(self):
        # repeated vertex should not break edge preparation
        verts = [(0.0, 0.0), (0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
        self.assertLess(polygon_sdf((1.0, 1.0), verts), 0.0)

    def test_too_few_vertices(self):
        with self.assertRaises(ValueError):
            polygon_sdf((0.0, 0.0), [(0.0, 0.0), (1.0, 0.0)])


if __name__ == "__main__":
    unittest.main()
