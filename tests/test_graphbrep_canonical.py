"""Tests for reconstruction.graphbrep_canonical."""

import unittest

from harnesscad.domain.reconstruction.brep import graphbrep_canonical as gc
from harnesscad.domain.reconstruction.brep import graphbrep_surface_graph as gsg


def relabel(matrix, order):
    return tuple(tuple(matrix[order[i]][order[j]] for j in range(len(order)))
                for i in range(len(order)))


class SerializeTest(unittest.TestCase):
    def test_serialize_upper_triangle(self):
        A = ((0, 2, 1), (2, 0, 3), (1, 3, 0))
        self.assertEqual(gc.serialize(A), "3|2,1,3")

    def test_serialize_empty(self):
        self.assertEqual(gc.serialize(()), "0|")


class WLTest(unittest.TestCase):
    def test_signature_permutation_invariant(self):
        A = gsg.build_from_edge_faces([(0, 1), (1, 2), (0, 2)], 3)
        B = relabel(A, (2, 0, 1))
        self.assertEqual(gc.wl_signature(A), gc.wl_signature(B))

    def test_signature_distinguishes_shapes(self):
        triangle = gsg.build_from_edge_faces([(0, 1), (1, 2), (0, 2)], 3)
        path = gsg.build_from_edge_faces([(0, 1), (1, 2)], 3)
        self.assertNotEqual(gc.wl_signature(triangle), gc.wl_signature(path))

    def test_wl_colors_empty(self):
        self.assertEqual(gc.wl_colors(()), ())

    def test_wl_colors_regular_uniform(self):
        # a triangle is regular: all nodes same colour
        A = gsg.build_from_edge_faces([(0, 1), (1, 2), (0, 2)], 3)
        colors = gc.wl_colors(A)
        self.assertEqual(len(set(colors)), 1)


class CanonicalTest(unittest.TestCase):
    def test_canonical_key_invariant_under_relabelling(self):
        A = gsg.build_from_edge_faces([(0, 1), (0, 1), (1, 2), (0, 3)], 4)
        for order in [(0, 1, 2, 3), (3, 2, 1, 0), (1, 0, 3, 2), (2, 3, 0, 1)]:
            B = relabel(A, order)
            self.assertEqual(gc.canonical_key(A), gc.canonical_key(B))

    def test_canonical_matrix_is_permutation(self):
        A = gsg.build_from_edge_faces([(0, 1), (1, 2), (0, 2)], 3)
        C = gc.canonical_matrix(A)
        # same multiset of edge weights preserved
        self.assertEqual(gsg.total_edges(C), gsg.total_edges(A))
        self.assertEqual(len(C), len(A))

    def test_canonical_empty(self):
        self.assertEqual(gc.canonical_labelling(()), ())
        self.assertEqual(gc.canonical_key(()), "0|")

    def test_guard_raises_on_large_symmetric(self):
        # An empty (edgeless) graph on 9 nodes: all one colour, 9! orderings.
        big = tuple(tuple(0 for _ in range(9)) for _ in range(9))
        with self.assertRaises(ValueError):
            gc.canonical_key(big)


class IsomorphismTest(unittest.TestCase):
    def test_isomorphic_relabelled(self):
        A = gsg.build_from_edge_faces([(0, 1), (0, 1), (1, 2), (0, 3), (2, 3)], 4)
        B = relabel(A, (2, 3, 0, 1))
        self.assertTrue(gc.are_isomorphic(A, B))

    def test_not_isomorphic_different_weights(self):
        A = gsg.build_from_edge_faces([(0, 1), (1, 2), (0, 2)], 3)
        B = gsg.build_from_edge_faces([(0, 1), (0, 1), (1, 2)], 3)
        self.assertFalse(gc.are_isomorphic(A, B))

    def test_not_isomorphic_different_size(self):
        A = gsg.build_from_edge_faces([(0, 1)], 2)
        B = gsg.build_from_edge_faces([(0, 1), (1, 2)], 3)
        self.assertFalse(gc.are_isomorphic(A, B))

    def test_not_isomorphic_same_size_different_topology(self):
        triangle = gsg.build_from_edge_faces([(0, 1), (1, 2), (0, 2)], 3)
        path = gsg.build_from_edge_faces([(0, 1), (1, 2)], 3)
        self.assertFalse(gc.are_isomorphic(triangle, path))


if __name__ == "__main__":
    unittest.main()
