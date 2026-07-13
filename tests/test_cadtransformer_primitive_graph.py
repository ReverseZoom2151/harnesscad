"""Tests for drawings.cadtransformer_primitive_graph."""

from __future__ import annotations

import math
import unittest

from harnesscad.domain.drawings.primitive_graph import (
    Primitive,
    build_primitive_graph,
    endpoint_knn,
    node_feature,
    normalize_center,
    primitive_center,
    primitive_length,
    primitive_segment,
)


class TestSegment(unittest.TestCase):
    def test_line_segment(self):
        p = Primitive("line", (0, 0, 3, 4))
        self.assertEqual(primitive_segment(p), (0, 0, 3, 4))

    def test_circle_segment_is_diameter(self):
        p = Primitive("circle", (5, 5, 2))
        self.assertEqual(primitive_segment(p), (3, 5, 7, 5))

    def test_ellipse_segment_horizontal_diameter(self):
        p = Primitive("ellipse", (0, 0, 3, 1))
        self.assertEqual(primitive_segment(p), (-3, 0, 3, 0))

    def test_bad_kind(self):
        with self.assertRaises(ValueError):
            Primitive("spline", (0, 0))


class TestLength(unittest.TestCase):
    def test_line_length(self):
        self.assertAlmostEqual(primitive_length(Primitive("line", (0, 0, 3, 4))), 5.0)

    def test_circle_length(self):
        self.assertAlmostEqual(
            primitive_length(Primitive("circle", (0, 0, 2))), 2 * math.pi * 2)

    def test_ellipse_length_approx(self):
        # rx=3, ry=1 -> r_min=1, r_max=3
        expected = 2 * math.pi * 1 + 4 * (3 - 1)
        self.assertAlmostEqual(
            primitive_length(Primitive("ellipse", (0, 0, 3, 1))), expected)

    def test_ellipse_circle_degenerate(self):
        # rx == ry -> reduces to circle circumference
        self.assertAlmostEqual(
            primitive_length(Primitive("ellipse", (0, 0, 2, 2))), 2 * math.pi * 2)


class TestCenter(unittest.TestCase):
    def test_line_center(self):
        self.assertEqual(primitive_center(Primitive("line", (0, 0, 4, 2))), (2, 1))

    def test_circle_center(self):
        self.assertEqual(primitive_center(Primitive("circle", (5, 6, 2))), (5, 6))


class TestNodeFeature(unittest.TestCase):
    def test_feature_layout_line(self):
        p = Primitive("line", (0, 0, 10, 0))
        f = node_feature(p, 0, 0, 100, 100)
        self.assertAlmostEqual(f[0], 10 / 100)      # length / width
        self.assertAlmostEqual(f[1], 5 / 100)       # midx norm
        self.assertAlmostEqual(f[2], 0 / 100)       # midy norm
        self.assertEqual(f[3:], [1.0, 0.0, 0.0])    # one-hot line

    def test_feature_onehot_circle(self):
        f = node_feature(Primitive("circle", (0, 0, 1)), 0, 0, 10, 10)
        self.assertEqual(f[3:], [0.0, 1.0, 0.0])

    def test_feature_onehot_ellipse(self):
        f = node_feature(Primitive("ellipse", (0, 0, 2, 1)), 0, 0, 10, 10)
        self.assertEqual(f[3:], [0.0, 0.0, 1.0])

    def test_bad_dimensions(self):
        with self.assertRaises(ValueError):
            node_feature(Primitive("line", (0, 0, 1, 1)), 0, 0, 0, 10)


class TestNormalizeCenter(unittest.TestCase):
    def test_center_maps_to_origin(self):
        self.assertEqual(normalize_center((50, 50), 100, 100), (0.0, 0.0))

    def test_corner_maps_to_minus_one(self):
        self.assertEqual(normalize_center((0, 0), 100, 100), (-1.0, -1.0))

    def test_far_corner_maps_to_one(self):
        self.assertEqual(normalize_center((100, 100), 100, 100), (1.0, 1.0))


class TestEndpointKnn(unittest.TestCase):
    def test_self_is_nearest(self):
        segs = [(0, 0, 1, 0), (10, 10, 11, 10)]
        nns = endpoint_knn(segs, max_degree=1)
        self.assertEqual(nns, [[0], [1]])

    def test_endpoint_touching_neighbours(self):
        # seg1 end (1,0) touches seg2 start (1,0): they are mutual neighbours
        segs = [(0, 0, 1, 0), (1, 0, 2, 0), (50, 50, 60, 60)]
        nns = endpoint_knn(segs, max_degree=2)
        self.assertIn(1, nns[0])
        self.assertIn(0, nns[1])
        self.assertNotIn(2, nns[0])

    def test_min_over_four_pairs(self):
        # segments far by midpoint but sharing a close endpoint pair
        segs = [(0, 0, 100, 0), (0.0, 0.5, 100, 100)]
        nns = endpoint_knn(segs, max_degree=2)
        self.assertIn(1, nns[0])

    def test_deterministic_tie_break_by_index(self):
        segs = [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)]
        nns = endpoint_knn(segs, max_degree=3)
        self.assertEqual(nns[0], [0, 1, 2])

    def test_bad_degree(self):
        with self.assertRaises(ValueError):
            endpoint_knn([(0, 0, 1, 1)], max_degree=0)


class TestBuildGraph(unittest.TestCase):
    def test_full_graph_shapes(self):
        prims = [
            Primitive("line", (0, 0, 1, 0), semantic_id=33, instance_id=-1),
            Primitive("circle", (1, 0, 0.5), semantic_id=1, instance_id=7),
            Primitive("ellipse", (5, 5, 2, 1), semantic_id=7, instance_id=8),
        ]
        g = build_primitive_graph(prims, 0, 0, 10, 10, max_degree=2)
        self.assertEqual(len(g["nd_ft"]), 3)
        self.assertEqual(len(g["ct"]), 3)
        self.assertEqual(len(g["ct_norm"]), 3)
        self.assertEqual(len(g["nns"]), 3)
        self.assertEqual(g["cat"], [[33], [1], [7]])
        self.assertEqual(g["inst"], [[-1], [7], [8]])
        self.assertEqual(len(g["nns"][0]), 2)

    def test_touching_line_and_circle_are_neighbours(self):
        prims = [
            Primitive("line", (0, 0, 1, 0)),
            Primitive("circle", (1.5, 0, 0.5)),  # diameter (1,0)-(2,0) touches line end
        ]
        g = build_primitive_graph(prims, 0, 0, 10, 10, max_degree=2)
        self.assertIn(1, g["nns"][0])


if __name__ == "__main__":
    unittest.main()
