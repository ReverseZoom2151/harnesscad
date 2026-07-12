"""Tests for geometry.querycad_edge_convexity."""

import math
import unittest

from geometry.querycad_edge_convexity import (
    CONCAVE,
    CONVEX,
    SMOOTH,
    AttributedAdjacencyGraph,
    build_aag,
    classify_edge_convexity,
    dihedral_angle,
)


class ClassifyEdgeTest(unittest.TestCase):
    def test_convex_right_angle(self):
        # Two faces meeting at a convex 90-degree edge along +Z.
        # Face A normal +X, face B normal +Y, tangent +Z.
        # cross(+X,+Y) = +Z; dot(+Z,+Z) = 1 > 0 -> convex.
        self.assertEqual(
            classify_edge_convexity((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            CONVEX,
        )

    def test_concave_right_angle(self):
        # Reverse the tangent -> sign flips -> concave.
        self.assertEqual(
            classify_edge_convexity((1, 0, 0), (0, 1, 0), (0, 0, -1)),
            CONCAVE,
        )

    def test_orientation_flag_flips_label(self):
        forward = classify_edge_convexity((1, 0, 0), (0, 1, 0), (0, 0, 1), forward=True)
        reversed_ = classify_edge_convexity((1, 0, 0), (0, 1, 0), (0, 0, 1), forward=False)
        self.assertEqual(forward, CONVEX)
        self.assertEqual(reversed_, CONCAVE)

    def test_smooth_parallel_normals(self):
        # Coplanar faces: cross(n,n) = 0 -> smooth.
        self.assertEqual(
            classify_edge_convexity((0, 0, 1), (0, 0, 1), (1, 0, 0)),
            SMOOTH,
        )

    def test_non_unit_vectors_scale_out(self):
        # Scaling operands must not change the label.
        self.assertEqual(
            classify_edge_convexity((5, 0, 0), (0, 3, 0), (0, 0, 2)),
            CONVEX,
        )

    def test_degenerate_zero_vector_is_smooth(self):
        self.assertEqual(
            classify_edge_convexity((0, 0, 0), (0, 1, 0), (0, 0, 1)),
            SMOOTH,
        )

    def test_bad_dimension_raises(self):
        with self.assertRaises(ValueError):
            classify_edge_convexity((1, 0), (0, 1, 0), (0, 0, 1))


class DihedralAngleTest(unittest.TestCase):
    def test_perpendicular(self):
        self.assertAlmostEqual(dihedral_angle((1, 0, 0), (0, 1, 0)), math.pi / 2, places=9)

    def test_parallel(self):
        self.assertAlmostEqual(dihedral_angle((0, 0, 2), (0, 0, 5)), 0.0, places=9)

    def test_opposite(self):
        self.assertAlmostEqual(dihedral_angle((1, 0, 0), (-1, 0, 0)), math.pi, places=9)

    def test_zero_vector(self):
        self.assertEqual(dihedral_angle((0, 0, 0), (0, 1, 0)), 0.0)


class AAGTest(unittest.TestCase):
    def _graph(self):
        g = AttributedAdjacencyGraph()
        g.add_edge(0, 1, CONCAVE)
        g.add_edge(1, 2, CONVEX)
        g.add_edge(0, 2, SMOOTH)
        return g

    def test_nodes_and_neighbors(self):
        g = self._graph()
        self.assertEqual(sorted(g.faces), [0, 1, 2])
        self.assertEqual(sorted(g.neighbors(0)), [1, 2])

    def test_filtered_neighbors(self):
        g = self._graph()
        self.assertEqual(g.concave_neighbors(0), [1])
        self.assertEqual(g.convex_neighbors(1), [2])
        self.assertEqual(g.neighbors(0, SMOOTH), [2])

    def test_histogram(self):
        g = self._graph()
        self.assertEqual(g.convexity_histogram(), {CONVEX: 1, CONCAVE: 1, SMOOTH: 1})

    def test_neighbors_dedup_and_order(self):
        g = AttributedAdjacencyGraph()
        g.add_edge(0, 1, CONVEX)
        g.add_edge(0, 1, CONCAVE)  # second shared edge, same pair
        self.assertEqual(g.neighbors(0), [1])

    def test_bad_label_raises(self):
        g = AttributedAdjacencyGraph()
        with self.assertRaises(ValueError):
            g.add_edge(0, 1, "wobbly")


class BuildAAGTest(unittest.TestCase):
    def test_build_from_records(self):
        records = [
            {"face_a": 0, "face_b": 1, "normal_a": (1, 0, 0), "normal_b": (0, 1, 0), "tangent": (0, 0, 1)},
            {"face_a": 1, "face_b": 2, "normal_a": (1, 0, 0), "normal_b": (0, 1, 0), "tangent": (0, 0, -1)},
        ]
        g = build_aag(records)
        self.assertEqual(g.convexity_histogram(), {CONVEX: 1, CONCAVE: 1, SMOOTH: 0})
        self.assertEqual(g.convex_neighbors(0), [1])
        self.assertEqual(g.concave_neighbors(1), [2])

    def test_deterministic(self):
        records = [
            {"face_a": 0, "face_b": 1, "normal_a": (1, 0, 0), "normal_b": (0, 1, 0), "tangent": (0, 0, 1)},
        ]
        a = build_aag(records)
        b = build_aag(records)
        self.assertEqual(a.arcs[0].convexity, b.arcs[0].convexity)


if __name__ == "__main__":
    unittest.main()
