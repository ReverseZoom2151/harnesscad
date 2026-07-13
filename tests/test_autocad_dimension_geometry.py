"""Tests for drawings.autocad_dimension_geometry."""

import math
import unittest

from harnesscad.domain.drawings.dimensions import (
    aligned_dimension,
    rotated_dimension,
    angular_dimension,
    radial_dimension,
    diametric_dimension,
    bounding_box,
    overall_dimensions,
)


class TestAligned(unittest.TestCase):
    def test_measures_true_distance(self):
        g = aligned_dimension((0.0, 0.0), (3.0, 4.0), offset=0.0)
        self.assertAlmostEqual(g.measured, 5.0)

    def test_offset_displaces_perpendicular(self):
        g = aligned_dimension((0.0, 0.0), (10.0, 0.0), offset=2.0)
        # direction +x, perpendicular ccw is +y
        self.assertAlmostEqual(g.dimension_line[1], 2.0)
        self.assertAlmostEqual(g.dimension_line[3], 2.0)
        self.assertAlmostEqual(g.text_anchor[0], 5.0)
        self.assertAlmostEqual(g.text_anchor[1], 2.0)

    def test_extension_lines_connect_points(self):
        g = aligned_dimension((0.0, 0.0), (10.0, 0.0), offset=3.0)
        self.assertEqual(g.extension_a[:2], (0.0, 0.0))
        self.assertEqual(g.extension_b[:2], (10.0, 0.0))

    def test_degenerate_raises(self):
        with self.assertRaises(ValueError):
            aligned_dimension((1.0, 1.0), (1.0, 1.0))


class TestRotated(unittest.TestCase):
    def test_horizontal_extent_at_zero(self):
        g = rotated_dimension((0.0, 0.0), (6.0, 8.0), angle=0.0)
        self.assertAlmostEqual(g.measured, 6.0)

    def test_vertical_extent_at_ninety(self):
        g = rotated_dimension((0.0, 0.0), (6.0, 8.0), angle=math.pi / 2)
        self.assertAlmostEqual(g.measured, 8.0)

    def test_diagonal_projection(self):
        # points along 45 degrees, measured along 45 degrees == full length
        g = rotated_dimension((0.0, 0.0), (5.0, 5.0), angle=math.pi / 4)
        self.assertAlmostEqual(g.measured, math.hypot(5.0, 5.0))


class TestAngular(unittest.TestCase):
    def test_right_angle(self):
        a = angular_dimension((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
        self.assertAlmostEqual(a, math.pi / 2)

    def test_straight_angle(self):
        a = angular_dimension((0.0, 0.0), (1.0, 0.0), (-1.0, 0.0))
        self.assertAlmostEqual(a, math.pi)

    def test_vertex_coincident_raises(self):
        with self.assertRaises(ValueError):
            angular_dimension((0.0, 0.0), (0.0, 0.0), (1.0, 0.0))


class TestRadialDiametric(unittest.TestCase):
    def test_radius(self):
        self.assertAlmostEqual(radial_dimension((0.0, 0.0), (0.0, 7.0)), 7.0)

    def test_diameter(self):
        self.assertAlmostEqual(diametric_dimension((-4.0, 0.0), (4.0, 0.0)), 8.0)


class TestBBoxOverall(unittest.TestCase):
    def test_bounding_box(self):
        self.assertEqual(
            bounding_box([(1.0, 2.0), (3.0, -1.0), (0.0, 5.0)]),
            (0.0, -1.0, 3.0, 5.0),
        )

    def test_bbox_empty_raises(self):
        with self.assertRaises(ValueError):
            bounding_box([])

    def test_overall_dimensions(self):
        w, h = overall_dimensions([(0.0, 0.0), (200.0, 150.0)], offset=5.0)
        self.assertAlmostEqual(w.measured, 200.0)
        self.assertAlmostEqual(h.measured, 150.0)


if __name__ == "__main__":
    unittest.main()
