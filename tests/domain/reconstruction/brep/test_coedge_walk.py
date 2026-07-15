"""Tests for the A2Z half-edge (co-edge) topological walk."""

import unittest

from harnesscad.domain.reconstruction.brep import coedge_walk as cw


# Two triangles sharing edge (1,2): face 0 = 0->1->2->0, face 1 = 2->1->3->2.
TWO_TRIANGLES = {
    0: [(0, 1), (1, 2), (2, 0)],
    1: [(2, 1), (1, 3), (3, 2)],
}


class BuildTest(unittest.TestCase):
    def setUp(self):
        self.s = cw.build_half_edges(TWO_TRIANGLES)

    def test_coedge_count(self):
        self.assertEqual(len(self.s.co_edges), 6)

    def test_loop_walk_cycles(self):
        # first co-edge of face 0
        loop = cw.walk_loop(self.s, 0)
        self.assertEqual(len(loop), 3)
        self.assertEqual(loop[0], 0)

    def test_next_prev_consistent(self):
        for ce in self.s.co_edges:
            self.assertEqual(self.s.co_edge(ce.next).prev, ce.id)

    def test_mate_of_shared_edge(self):
        # co-edge 1 is face 0 edge 1->2 ; its mate is face 1 edge 2->1.
        ce = self.s.co_edge(1)
        self.assertEqual((ce.v_start, ce.v_end), (1, 2))
        self.assertIsNotNone(ce.mate)
        mate = self.s.co_edge(ce.mate)
        self.assertEqual((mate.v_start, mate.v_end), (2, 1))

    def test_mate_face_query(self):
        # A2Z F[M[e]] for the shared co-edge is the other face.
        self.assertEqual(cw.mate_face(self.s, 1), 1)

    def test_boundary_edges_present(self):
        # unshared edges have no mate.
        b = cw.boundary_co_edges(self.s)
        self.assertTrue(len(b) > 0)
        self.assertFalse(cw.is_manifold(self.s))


class SingleLoopAsSequenceTest(unittest.TestCase):
    def test_single_loop_input_normalized(self):
        s = cw.build_half_edges({0: [(0, 1), (1, 2), (2, 0)]})
        self.assertEqual(len(s.co_edges), 3)
        self.assertEqual(cw.walk_loop(s, 0), (0, 1, 2))


class ManifoldTest(unittest.TestCase):
    def test_closed_pair_is_manifold(self):
        # Two faces both directions of every edge -> fully mated.
        faces = {0: [(0, 1), (1, 0)]}
        s = cw.build_half_edges(faces)
        self.assertTrue(cw.is_manifold(s))


if __name__ == "__main__":
    unittest.main()
