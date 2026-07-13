"""Tests for reconstruction.graphbrep_surface_graph."""

import unittest

from harnesscad.domain.reconstruction.brep import graphbrep_surface_graph as gsg


class BuildAdjacencyTest(unittest.TestCase):
    def test_shared_edge_counts(self):
        # Two faces sharing edges e1, e2; a third sharing only e2 with face 1.
        face_edges = [
            ["e1", "e2", "e3"],
            ["e1", "e2", "e4"],
            ["e2", "e5"],
        ]
        A = gsg.build_surface_adjacency(face_edges)
        self.assertEqual(A[0][1], 2)  # e1, e2
        self.assertEqual(A[1][2], 1)  # e2
        self.assertEqual(A[0][2], 1)  # e2
        # symmetric and zero diagonal
        self.assertEqual(A[1][0], 2)
        self.assertEqual(A[0][0], 0)

    def test_empty(self):
        self.assertEqual(gsg.build_surface_adjacency([]), ())

    def test_from_edge_faces(self):
        edges = [(0, 1), (0, 1), (1, 2)]
        A = gsg.build_from_edge_faces(edges, 3)
        self.assertEqual(A[0][1], 2)
        self.assertEqual(A[1][2], 1)
        self.assertEqual(A[0][2], 0)

    def test_from_edge_faces_rejects_self_edge(self):
        with self.assertRaises(ValueError):
            gsg.build_from_edge_faces([(0, 0)], 2)

    def test_from_edge_faces_range(self):
        with self.assertRaises(ValueError):
            gsg.build_from_edge_faces([(0, 5)], 3)


class PostProcessTest(unittest.TestCase):
    def test_symmetrise(self):
        raw = [[0.0, 1.0], [3.0, 0.0]]
        sym = gsg.symmetrise(raw)
        self.assertEqual(sym[0][1], 2.0)
        self.assertEqual(sym[1][0], 2.0)

    def test_symmetrise_non_square(self):
        with self.assertRaises(ValueError):
            gsg.symmetrise([[1.0, 2.0]])

    def test_finalise_clips_rounds_masks(self):
        raw = [
            [9.0, 0.4, 2.6],
            [0.4, 9.0, -1.0],
            [2.6, -1.0, 9.0],
        ]
        A = gsg.finalise_predicted(raw, e_max=2)
        self.assertEqual(A[0][0], 0)          # diagonal masked
        self.assertEqual(A[0][1], 0)          # 0.4 rounds to 0
        self.assertEqual(A[0][2], 2)          # 2.6 rounds to 3, clipped to e_max=2
        self.assertEqual(A[1][2], 0)          # negative clipped to 0
        # symmetric
        for i in range(3):
            for j in range(3):
                self.assertEqual(A[i][j], A[j][i])

    def test_finalise_negative_emax(self):
        with self.assertRaises(ValueError):
            gsg.finalise_predicted([[0.0]], e_max=-1)


class SparseTest(unittest.TestCase):
    def test_round_trip(self):
        A = gsg.build_from_edge_faces([(0, 1), (0, 1), (1, 2)], 3)
        sparse = gsg.dense_to_sparse(A)
        self.assertIn((0, 1, 2), sparse)
        self.assertIn((1, 2, 1), sparse)
        dense = gsg.sparse_to_dense(sparse, 3)
        self.assertEqual(dense, A)

    def test_sparse_to_dense_diagonal(self):
        with self.assertRaises(ValueError):
            gsg.sparse_to_dense([(0, 0, 1)], 2)


class EdgeRecoveryTest(unittest.TestCase):
    def test_recover_edges_expands_weights(self):
        A = gsg.build_from_edge_faces([(0, 1), (0, 1), (1, 2)], 3)
        edges = gsg.recover_edges(A)
        self.assertEqual(edges.count((0, 1)), 2)
        self.assertEqual(edges.count((1, 2)), 1)
        self.assertEqual(len(edges), 3)

    def test_total_edges_matches_recovered(self):
        A = gsg.build_from_edge_faces([(0, 1), (0, 1), (1, 2), (0, 2)], 3)
        self.assertEqual(gsg.total_edges(A), len(gsg.recover_edges(A)))
        self.assertEqual(gsg.total_edges(A), 4)

    def test_surface_degrees(self):
        A = gsg.build_from_edge_faces([(0, 1), (0, 1), (1, 2)], 3)
        self.assertEqual(gsg.surface_degrees(A), (2, 3, 1))


class ValidityTest(unittest.TestCase):
    def test_valid_cube_like(self):
        # 3 mutually adjacent faces, each sharing one edge -> a triangle graph.
        A = gsg.build_from_edge_faces([(0, 1), (1, 2), (0, 2)], 3)
        ok, diags = gsg.check_graph(A, e_max=1)
        self.assertTrue(ok)
        self.assertEqual(diags, ())

    def test_isolated_surface_flagged(self):
        A = gsg.build_from_edge_faces([(0, 1)], 3)  # surface 2 isolated
        ok, diags = gsg.check_graph(A)
        self.assertFalse(ok)
        self.assertTrue(any(d.code == "isolated-surface" for d in diags))

    def test_over_e_max_flagged(self):
        A = gsg.build_from_edge_faces([(0, 1), (0, 1), (0, 1)], 2)
        ok, diags = gsg.check_graph(A, e_max=2)
        self.assertFalse(ok)
        self.assertTrue(any(d.code == "over-e-max" for d in diags))

    def test_asymmetry_and_diagonal(self):
        bad = ((5, 1), (2, 0))  # non-zero diagonal + asymmetric
        ok, diags = gsg.check_graph(bad)
        self.assertFalse(ok)
        codes = {d.code for d in diags}
        self.assertIn("non-zero-diagonal", codes)
        self.assertIn("asymmetric", codes)

    def test_connectivity(self):
        # two disjoint edges among 4 surfaces -> disconnected
        A = gsg.build_from_edge_faces([(0, 1), (2, 3)], 4)
        self.assertFalse(gsg.is_connected(A))
        ok, diags = gsg.check_graph(A, require_connected=True)
        self.assertTrue(any(d.code == "disconnected" for d in diags))
        # connect them
        B = gsg.build_from_edge_faces([(0, 1), (2, 3), (1, 2)], 4)
        self.assertTrue(gsg.is_connected(B))

    def test_is_connected_small(self):
        self.assertTrue(gsg.is_connected(()))
        self.assertTrue(gsg.is_connected(((0,),)))

    def test_check_graph_non_square(self):
        with self.assertRaises(ValueError):
            gsg.check_graph(((0, 1), (1,)))

    def test_is_valid_helper(self):
        A = gsg.build_from_edge_faces([(0, 1), (1, 2), (0, 2)], 3)
        self.assertTrue(gsg.is_valid(A, e_max=1))


if __name__ == "__main__":
    unittest.main()
