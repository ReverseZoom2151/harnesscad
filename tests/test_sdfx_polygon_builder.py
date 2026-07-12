"""Tests for geometry.sdfx_polygon_builder."""

import math
import unittest

from geometry.sdfx_polygon_builder import Polygon, nagon
from geometry.sdfx_polygon_sdf import polygon_area, polygon_sdf


class TestBasicBuilding(unittest.TestCase):
    def test_absolute_square(self):
        p = Polygon()
        p.add(0, 0)
        p.add(2, 0)
        p.add(2, 2)
        p.add(0, 2)
        verts = p.vertices()
        self.assertEqual(len(verts), 4)
        self.assertAlmostEqual(abs(polygon_area(verts)), 4.0)

    def test_relative_vertices(self):
        p = Polygon()
        p.add(0, 0)
        p.add(2, 0).rel()
        p.add(0, 2).rel()
        p.add(-2, 0).rel()
        verts = p.vertices()
        self.assertAlmostEqual(verts[1][0], 2.0)
        self.assertAlmostEqual(verts[2][0], 2.0)
        self.assertAlmostEqual(verts[2][1], 2.0)
        self.assertAlmostEqual(verts[3][0], 0.0)
        self.assertAlmostEqual(verts[3][1], 2.0)

    def test_polar_vertices(self):
        p = Polygon()
        p.add(1.0, 0.0).polar()
        p.add(1.0, math.pi / 2).polar()
        p.add(1.0, math.pi).polar()
        verts = p.vertices()
        self.assertAlmostEqual(verts[0][0], 1.0)
        self.assertAlmostEqual(verts[0][1], 0.0)
        self.assertAlmostEqual(verts[1][0], 0.0)
        self.assertAlmostEqual(verts[1][1], 1.0)
        self.assertAlmostEqual(verts[2][0], -1.0, places=6)

    def test_relative_without_reference_raises(self):
        p = Polygon()
        p.add(0, 0).rel()  # first vertex relative, prev wraps to last (also...)
        p.add(1, 0).rel()
        # first vertex's prev is the last vertex which is relative -> error
        with self.assertRaises(ValueError):
            p.vertices()

    def test_drop(self):
        p = Polygon()
        p.add(0, 0)
        p.add(1, 0)
        p.add(9, 9)
        p.drop()
        self.assertEqual(len(p.vertices()), 2 if not p.closed else 2)


class TestChamfer(unittest.TestCase):
    def test_chamfer_adds_vertices(self):
        # square with one chamfered corner: 1 facet -> corner replaced by 2 pts
        p = Polygon()
        p.add(0, 0)
        p.add(10, 0)
        p.add(10, 10).chamfer(2.0)
        p.add(0, 10)
        verts = p.vertices()
        # the chamfered corner becomes 2 vertices, so total 5
        self.assertEqual(len(verts), 5)
        # area slightly less than 100 (a corner triangle removed)
        self.assertLess(abs(polygon_area(verts)), 100.0)
        self.assertGreater(abs(polygon_area(verts)), 95.0)

    def test_chamfer_geometry_90deg(self):
        # chamfer of size s on a 90-deg corner cuts a right triangle of legs s.
        s = 2.0
        p = Polygon()
        p.add(0, 0)
        p.add(10, 0)
        p.add(10, 10).chamfer(s)
        p.add(0, 10)
        verts = p.vertices()
        area = abs(polygon_area(verts))
        # `size` is the chamfer face (hypotenuse) length on a 90-deg corner, so
        # the cut right-triangle has legs size/sqrt(2) and area size^2 / 4.
        expected = 100.0 - s * s / 4.0
        self.assertAlmostEqual(area, expected, places=5)


class TestSmooth(unittest.TestCase):
    def test_fillet_rounds_corner(self):
        # A filleted square corner. The filled region loses a corner bite.
        r = 2.0
        facets = 8
        p = Polygon()
        p.add(0, 0)
        p.add(10, 0)
        p.add(10, 10).smooth(r, facets)
        p.add(0, 10)
        verts = p.vertices()
        # corner replaced by facets+1 vertices
        self.assertEqual(len(verts), 3 + (facets + 1))
        area = abs(polygon_area(verts))
        # bite removed = square corner (r^2) minus quarter circle (pi r^2/4)
        bite = r * r - math.pi * r * r / 4.0
        self.assertAlmostEqual(area, 100.0 - bite, delta=0.1)


class TestArc(unittest.TestCase):
    def test_arc_segment_bulges(self):
        # arc replacing the segment into the 3rd vertex bulges outward,
        # increasing area vs the straight triangle.
        p = Polygon()
        p.add(0, 0)
        p.add(10, 0)
        p.add(10, 10).arc(8.0, 6)
        verts = p.vertices()
        self.assertGreater(len(verts), 3)
        # all points finite
        for x, y in verts:
            self.assertTrue(math.isfinite(x) and math.isfinite(y))


class TestNagon(unittest.TestCase):
    def test_hexagon(self):
        verts = nagon(6, 1.0)
        self.assertEqual(len(verts), 6)
        for x, y in verts:
            self.assertAlmostEqual(math.hypot(x, y), 1.0)
        # centroid inside
        self.assertLess(polygon_sdf((0.0, 0.0), verts), 0.0)

    def test_too_few_sides(self):
        with self.assertRaises(ValueError):
            nagon(2, 1.0)


if __name__ == "__main__":
    unittest.main()
