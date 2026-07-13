"""Tests for GeoCAD deterministic vertex-based captioning of simple parts."""

import math
import unittest

from harnesscad.domain.geometry import geocad_vertex_caption as vc


class TriangleTest(unittest.TestCase):
    def test_right_triangle(self):
        self.assertEqual(vc.caption_triangle([(0, 0), (3, 0), (0, 4)]),
                         "a right triangle")

    def test_isosceles_right_triangle(self):
        self.assertEqual(vc.caption_triangle([(0, 0), (4, 0), (0, 4)]),
                         "an isosceles right triangle")

    def test_equilateral(self):
        h = math.sqrt(3)
        self.assertEqual(vc.caption_triangle([(0, 0), (2, 0), (1, h)]),
                         "an equilateral triangle")

    def test_isosceles_acute(self):
        self.assertEqual(vc.caption_triangle([(0, 0), (4, 0), (2, 3)]),
                         "an isosceles triangle")

    def test_obtuse_scalene(self):
        cap = vc.caption_triangle([(0, 0), (6, 0), (5, 1)])
        self.assertEqual(cap, "an obtuse triangle")

    def test_acute_scalene(self):
        cap = vc.caption_triangle([(0, 0), (5, 0), (1, 4)])
        self.assertEqual(cap, "an acute triangle")


class QuadrilateralTest(unittest.TestCase):
    def test_square(self):
        self.assertEqual(vc.caption_quadrilateral([(0, 0), (2, 0), (2, 2), (0, 2)]),
                         "a square")

    def test_rectangle(self):
        self.assertEqual(vc.caption_quadrilateral([(0, 0), (4, 0), (4, 2), (0, 2)]),
                         "a rectangle")

    def test_rhombus(self):
        self.assertEqual(vc.caption_quadrilateral([(0, 0), (2, 1), (4, 0), (2, -1)]),
                         "a rhombus")

    def test_parallelogram(self):
        self.assertEqual(vc.caption_quadrilateral([(0, 0), (4, 0), (5, 2), (1, 2)]),
                         "a parallelogram")

    def test_isosceles_trapezoid(self):
        self.assertEqual(vc.caption_quadrilateral([(0, 0), (4, 0), (3, 2), (1, 2)]),
                         "an isosceles trapezoid")

    def test_trapezoid(self):
        self.assertEqual(vc.caption_quadrilateral([(0, 0), (5, 0), (3, 2), (1, 2)]),
                         "a trapezoid")

    def test_kite(self):
        cap = vc.caption_quadrilateral([(0, 0), (1, 1), (0, 3), (-1, 1)])
        self.assertEqual(cap, "a kite")

    def test_general_quadrilateral(self):
        cap = vc.caption_quadrilateral([(0, 0), (5, 1), (4, 4), (1, 2)])
        self.assertEqual(cap, "a quadrilateral")


class ArcLoopTest(unittest.TestCase):
    def test_circle(self):
        self.assertEqual(vc.caption_arc_loop(360), "a circle")

    def test_semicircle(self):
        self.assertEqual(vc.caption_arc_loop(180), "a semicircle")

    def test_quarter(self):
        self.assertEqual(vc.caption_arc_loop(90), "a quarter-circle")

    def test_three_quarter(self):
        self.assertEqual(vc.caption_arc_loop(270), "a three-quarter circle")

    def test_major_arc(self):
        self.assertEqual(vc.caption_arc_loop(220), "a major-arc loop")

    def test_minor_arc(self):
        self.assertEqual(vc.caption_arc_loop(60), "a minor-arc loop")


class DimensionsTest(unittest.TestCase):
    def test_circle_dims(self):
        d = vc.circle_dimensions(5)
        self.assertEqual(vc.caption_with_dimensions("a circle", d),
                         "a circle with radius 5")

    def test_square_dims(self):
        d = vc.square_dimensions([(0, 0), (3, 0), (3, 3), (0, 3)])
        self.assertEqual(vc.caption_with_dimensions("a square", d),
                         "a square with side 3")

    def test_rectangle_dims(self):
        d = vc.rectangle_dimensions([(0, 0), (4, 0), (4, 2), (0, 2)])
        cap = vc.caption_with_dimensions("a rectangle", d)
        self.assertIn("length 4", cap)
        self.assertIn("width 2", cap)


class DispatchTest(unittest.TestCase):
    def test_dispatch_triangle(self):
        self.assertIn("triangle", vc.caption_polygon([(0, 0), (3, 0), (0, 4)]))

    def test_dispatch_quad(self):
        self.assertIn("square", vc.caption_polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))

    def test_dispatch_rejects_pentagon(self):
        with self.assertRaises(ValueError):
            vc.caption_polygon([(0, 0), (1, 0), (2, 1), (1, 2), (0, 1)])

    def test_invariant_under_scaling(self):
        # A right triangle scaled 10x is still a right triangle.
        big = [(0, 0), (30, 0), (0, 40)]
        self.assertEqual(vc.caption_triangle(big), "a right triangle")


if __name__ == "__main__":
    unittest.main()
