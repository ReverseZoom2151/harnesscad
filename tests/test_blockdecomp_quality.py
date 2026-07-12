"""Tests for geometry.blockdecomp_quality."""

import math
import unittest

from geometry.blockdecomp_domain import Shape
from geometry.blockdecomp_cut import full_cut
from geometry.blockdecomp_quality import (
    all_quads,
    aspect_ratio,
    area_variance_ratio,
    block_scaled_jacobian,
    is_valid_decomposition,
    mean_aspect_ratio,
    orthogonality,
    quad_corners,
    quad_fraction,
    scaled_jacobian,
)


class TestAspectRatio(unittest.TestCase):
    def test_square(self):
        self.assertAlmostEqual(aspect_ratio(Shape.from_rectangles([(0, 0, 2, 2)])), 1.0)

    def test_rect(self):
        self.assertAlmostEqual(aspect_ratio(Shape.from_rectangles([(0, 0, 6, 2)])), 3.0)


class TestScaledJacobian(unittest.TestCase):
    def test_unit_square_is_one(self):
        sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
        self.assertAlmostEqual(scaled_jacobian(sq), 1.0)

    def test_rectangle_block_is_one(self):
        r = Shape.from_rectangles([(0, 0, 5, 2)])
        self.assertAlmostEqual(block_scaled_jacobian(r), 1.0)

    def test_collapsed_corner_is_zero(self):
        deg = [(0, 0), (0, 0), (1, 1), (0, 1)]
        self.assertAlmostEqual(scaled_jacobian(deg), 0.0)

    def test_skewed_less_than_one(self):
        skew = [(0, 0), (2, 0), (3, 1), (0, 1)]
        self.assertLess(scaled_jacobian(skew), 1.0)


class TestOrthogonality(unittest.TestCase):
    def test_rectangle_is_zero(self):
        self.assertAlmostEqual(orthogonality([(0, 0), (2, 0), (2, 1), (0, 1)]), 0.0)

    def test_skewed_positive(self):
        self.assertGreater(orthogonality([(0, 0), (2, 0), (3, 1), (0, 1)]), 0.0)


class TestAreaVariance(unittest.TestCase):
    def test_equal_areas_zero(self):
        r = Shape.from_rectangles([(0, 0, 4, 2)])
        parts = full_cut(r, "vertical", 2.0)
        self.assertAlmostEqual(area_variance_ratio(parts), 0.0)

    def test_unequal_positive(self):
        r = Shape.from_rectangles([(0, 0, 4, 2)])
        parts = full_cut(r, "vertical", 1.0)
        self.assertGreater(area_variance_ratio(parts), 0.0)


class TestQuadMetrics(unittest.TestCase):
    def setUp(self):
        self.l = Shape.from_rectangles([(0, 0, 2, 1), (0, 0, 1, 2)])

    def test_quad_fraction_and_all_quads(self):
        parts = full_cut(self.l, "vertical", 1.0)
        self.assertAlmostEqual(quad_fraction(parts), 1.0)
        self.assertTrue(all_quads(parts))

    def test_non_quad_present(self):
        self.assertFalse(all_quads([self.l]))
        self.assertAlmostEqual(quad_fraction([self.l]), 0.0)


class TestValidDecomposition(unittest.TestCase):
    def setUp(self):
        # Two abutting rectangles so the domain mesh already contains x = 2,
        # matching the mesh of the cut parts.
        self.dom = Shape.from_rectangles([(0, 0, 2, 2), (2, 0, 4, 2)])

    def test_valid_partition(self):
        parts = full_cut(self.dom, "vertical", 2.0)
        self.assertTrue(is_valid_decomposition(parts, self.dom))

    def test_overlap_invalid(self):
        parts = full_cut(self.dom, "vertical", 2.0)
        overlapping = list(parts) + [self.dom]
        self.assertFalse(is_valid_decomposition(overlapping, self.dom))

    def test_incomplete_cover_invalid(self):
        parts = full_cut(self.dom, "vertical", 2.0)
        self.assertFalse(is_valid_decomposition([parts[0]], self.dom))


class TestMeanAspect(unittest.TestCase):
    def test_squares_mean_one(self):
        r = Shape.from_rectangles([(0, 0, 4, 2)])
        parts = full_cut(r, "vertical", 2.0)
        self.assertAlmostEqual(mean_aspect_ratio(parts), 1.0)


if __name__ == "__main__":
    unittest.main()
