"""Tests for geometry.topology.coedge_walks."""

import unittest

from harnesscad.domain.geometry.topology.coedge_walks import (
    CoedgeTopology,
    WINGED_EDGE_KERNEL,
    SIMPLE_EDGE_KERNEL,
)


def cube_faces():
    # outward CCW quad loops of a unit cube (vertex ids 0..7)
    return [
        (0, 3, 2, 1),  # bottom z=0
        (4, 5, 6, 7),  # top z=1
        (0, 1, 5, 4),  # front y=0
        (1, 2, 6, 5),  # right x=1
        (2, 3, 7, 6),  # back y=1
        (3, 0, 4, 7),  # left x=0
    ]


class TestConstruction(unittest.TestCase):
    def setUp(self):
        self.t = CoedgeTopology.from_faces(cube_faces())

    def test_counts(self):
        self.assertEqual(self.t.num_faces, 6)
        self.assertEqual(self.t.num_coedges, 24)   # 6 quads * 4
        self.assertEqual(self.t.num_edges, 12)     # cube edges
        self.assertEqual(self.t.boundary_coedges, ())  # closed solid

    def test_next_prev_are_inverse(self):
        for c in range(self.t.num_coedges):
            self.assertEqual(self.t.p[self.t.n[c]], c)
            self.assertEqual(self.t.n[self.t.p[c]], c)

    def test_mate_is_involution_and_opposite(self):
        for c in range(self.t.num_coedges):
            m = self.t.m[c]
            self.assertNotEqual(m, c)                 # closed: every coedge mated
            self.assertEqual(self.t.m[m], c)          # involution
            a, b = self.t.coedge_verts[c]
            self.assertEqual(self.t.coedge_verts[m], (b, a))  # opposite direction

    def test_mate_crosses_faces(self):
        for c in range(self.t.num_coedges):
            self.assertNotEqual(self.t.f[c], self.t.f[self.t.m[c]])


class TestWalks(unittest.TestCase):
    def setUp(self):
        self.t = CoedgeTopology.from_faces(cube_faces())

    def test_empty_walk_identity(self):
        self.assertEqual(self.t.walk(5, ""), 5)

    def test_face_and_edge_terminal(self):
        self.assertEqual(self.t.walk(0, "f"), self.t.f[0])
        self.assertEqual(self.t.walk(0, "e"), self.t.e[0])

    def test_face_instruction_must_be_last(self):
        with self.assertRaises(ValueError):
            self.t.walk(0, "fn")

    def test_mate_face_is_neighbour_face(self):
        # walk "mf" = mate's parent face; must differ from own face
        for c in range(self.t.num_coedges):
            self.assertNotEqual(self.t.walk(c, "mf"), self.t.walk(c, "f"))

    def test_walk_composition(self):
        # "n" then "n" three more times returns to start on a quad loop
        c = 0
        cc = c
        for _ in range(4):
            cc = self.t.walk(cc, "n")
        self.assertEqual(cc, c)

    def test_kernel_neighbourhood_shapes(self):
        nb = self.t.kernel_neighbourhood(0, WINGED_EDGE_KERNEL)
        self.assertEqual(len(nb["faces"]), 2)
        self.assertEqual(len(nb["edges"]), 5)
        self.assertEqual(len(nb["coedges"]), 6)
        # coedge walk "" is identity
        self.assertEqual(nb["coedges"][0], 0)

    def test_simple_edge_kernel(self):
        nb = self.t.kernel_neighbourhood(3, SIMPLE_EDGE_KERNEL)
        self.assertEqual(nb["coedges"], [3, self.t.m[3]])


class TestSignatures(unittest.TestCase):
    def test_cube_faces_all_equivalent(self):
        t = CoedgeTopology.from_faces(cube_faces())
        sigs = t.face_signature()
        self.assertEqual(len(set(sigs)), 1)  # cube: all 6 faces symmetric

    def test_signature_relabelling_invariant(self):
        faces = cube_faces()
        # permute face order and rotate each loop; signatures (as a multiset)
        # must be unchanged
        perm = [faces[i] for i in (3, 0, 5, 2, 4, 1)]
        rotated = [tuple(f[1:] + f[:1]) for f in perm]
        s1 = sorted(CoedgeTopology.from_faces(faces).face_signature())
        s2 = sorted(CoedgeTopology.from_faces(rotated).face_signature())
        self.assertEqual(s1, s2)

    def test_prism_distinguishes_faces(self):
        # triangular prism: 2 triangles + 3 quads -> two signature classes
        faces = [
            (0, 1, 2),        # bottom triangle
            (3, 5, 4),        # top triangle
            (0, 3, 4, 1),
            (1, 4, 5, 2),
            (2, 5, 3, 0),
        ]
        sigs = CoedgeTopology.from_faces(faces).face_signature()
        self.assertEqual(len(set(sigs)), 2)


class TestCanonicalOrder(unittest.TestCase):
    def test_is_permutation(self):
        t = CoedgeTopology.from_faces(cube_faces())
        order = t.canonical_bfs_face_order()
        self.assertEqual(sorted(order), list(range(6)))

    def test_deterministic(self):
        t = CoedgeTopology.from_faces(cube_faces())
        self.assertEqual(t.canonical_bfs_face_order(),
                         t.canonical_bfs_face_order())

    def test_open_surface_has_boundary(self):
        # single quad: 4 boundary coedges, each its own mate
        t = CoedgeTopology.from_faces([(0, 1, 2, 3)])
        self.assertEqual(len(t.boundary_coedges), 4)
        for c in t.boundary_coedges:
            self.assertEqual(t.m[c], c)


if __name__ == "__main__":
    unittest.main()
