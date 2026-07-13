"""Tests for geometry.manifold_halfedge."""

import unittest

from harnesscad.domain.geometry.mesh.manifold_halfedge import (
    HalfedgeMesh,
    next_halfedge,
    prev_halfedge,
    tetrahedron_mesh,
    cube_mesh,
)


class TestHalfedgeIndexing(unittest.TestCase):
    def test_next_prev_wrap(self):
        self.assertEqual(next_halfedge(0), 1)
        self.assertEqual(next_halfedge(1), 2)
        self.assertEqual(next_halfedge(2), 0)
        self.assertEqual(prev_halfedge(0), 2)
        self.assertEqual(prev_halfedge(1), 0)
        self.assertEqual(prev_halfedge(2), 1)

    def test_next_prev_inverse(self):
        for h in range(12):
            self.assertEqual(prev_halfedge(next_halfedge(h)), h)


class TestTetrahedron(unittest.TestCase):
    def setUp(self):
        self.m = tetrahedron_mesh()

    def test_counts(self):
        self.assertEqual(self.m.num_vert(), 4)
        self.assertEqual(self.m.num_tri(), 4)
        self.assertEqual(self.m.num_halfedge(), 12)
        self.assertEqual(self.m.num_edge(), 6)

    def test_is_manifold(self):
        self.assertTrue(self.m.is_manifold())

    def test_is_2manifold(self):
        ok, issues = self.m.is_2manifold()
        self.assertTrue(ok, issues)
        self.assertEqual(issues, [])

    def test_pair_involution(self):
        for h in range(self.m.num_halfedge()):
            p = self.m.pair(h)
            self.assertGreaterEqual(p, 0)
            self.assertEqual(self.m.pair(p), h)
            self.assertEqual(self.m.start(h), self.m.end(p))
            self.assertEqual(self.m.end(h), self.m.start(p))

    def test_closed_no_boundary(self):
        self.assertTrue(self.m.is_closed())
        self.assertEqual(self.m.boundary_halfedges(), [])
        self.assertEqual(self.m.boundary_loops(), [])

    def test_euler_and_genus(self):
        # Closed genus-0 surface: chi = 2, genus 0.
        self.assertEqual(self.m.euler_characteristic(), 2)
        self.assertEqual(self.m.genus(), 0)

    def test_vertex_ring_closes(self):
        ring = self.m.vertex_ring(0)
        v = self.m.start(0)
        self.assertTrue(all(self.m.start(h) == v for h in ring))
        # Each interior vertex of a tetra has degree 3.
        self.assertEqual(len(ring), 3)


class TestCube(unittest.TestCase):
    def setUp(self):
        self.m = cube_mesh()

    def test_counts(self):
        self.assertEqual(self.m.num_vert(), 8)
        self.assertEqual(self.m.num_tri(), 12)
        self.assertEqual(self.m.num_edge(), 18)

    def test_2manifold(self):
        ok, issues = self.m.is_2manifold()
        self.assertTrue(ok, issues)

    def test_euler_genus(self):
        # V - E + F = 8 - 18 + 12 = 2.
        self.assertEqual(self.m.euler_characteristic(), 2)
        self.assertEqual(self.m.genus(), 0)

    def test_all_paired(self):
        self.assertTrue(self.m.is_closed())


class TestOpenMesh(unittest.TestCase):
    def setUp(self):
        # Two triangles sharing edge (0,2) forming an open square.
        v = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        self.m = HalfedgeMesh(v, [(0, 1, 2), (0, 2, 3)])

    def test_has_boundary(self):
        self.assertFalse(self.m.is_closed())
        bnd = self.m.boundary_halfedges()
        self.assertEqual(len(bnd), 4)

    def test_manifold_but_open(self):
        # Manifold-consistent internally, but open -> IsManifold flags boundary.
        self.assertFalse(self.m.is_manifold())

    def test_boundary_loop(self):
        loops = self.m.boundary_loops()
        self.assertEqual(len(loops), 1)
        self.assertEqual(set(loops[0]), {0, 1, 2, 3})


class TestNonManifold(unittest.TestCase):
    def test_edge_shared_by_three_faces(self):
        # Edge (0,1) shared by three triangles -> non-manifold.
        v = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, -1, 0)]
        m = HalfedgeMesh(v, [(0, 1, 2), (0, 1, 3), (0, 1, 4)])
        ok, issues = m.is_2manifold()
        self.assertFalse(ok)
        self.assertTrue(any(i.code == "nonmanifold-edge" for i in issues))


if __name__ == "__main__":
    unittest.main()
