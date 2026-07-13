"""Tests for the uniform tetrahedral grid."""

import unittest

from harnesscad.domain.geometry.volumes.meshdiff_tet_grid import TetGrid


class TetGridTest(unittest.TestCase):
    def test_vertex_count(self):
        g = TetGrid(2)
        self.assertEqual(g.num_vertices, 3 * 3 * 3)

    def test_tet_count_six_per_cube(self):
        for n in (1, 2, 3):
            g = TetGrid(n)
            self.assertEqual(g.num_tets, 6 * n * n * n)

    def test_vertices_span_cube(self):
        g = TetGrid(2, lo=-1.0, hi=1.0)
        xs = [v[0] for v in g.vertices]
        self.assertAlmostEqual(min(xs), -1.0)
        self.assertAlmostEqual(max(xs), 1.0)
        # midpoint plane exists at 0.0
        self.assertTrue(any(abs(v[0]) < 1e-12 for v in g.vertices))

    def test_all_indices_in_range(self):
        g = TetGrid(2)
        for tet in g.tets:
            self.assertEqual(len(tet), 4)
            self.assertEqual(len(set(tet)), 4)  # non-degenerate
            for v in tet:
                self.assertTrue(0 <= v < g.num_vertices)

    def test_tets_have_positive_volume(self):
        g = TetGrid(2)
        for tet in g.tets:
            a, b, c, d = (g.vertices[i] for i in tet)
            self.assertGreater(abs(_signed_volume(a, b, c, d)), 1e-9)

    def test_tets_partition_cube_volume(self):
        # Sum of |tet volume| must equal the cube volume exactly.
        g = TetGrid(3, lo=-1.0, hi=1.0)
        total = sum(
            abs(_signed_volume(*(g.vertices[i] for i in tet))) for tet in g.tets
        )
        self.assertAlmostEqual(total, 2.0 ** 3, places=9)

    def test_edges_unique_and_sorted(self):
        g = TetGrid(2)
        edges = g.edges()
        self.assertEqual(edges, sorted(set(edges)))
        for a, b in edges:
            self.assertLess(a, b)

    def test_adjacency_symmetric(self):
        g = TetGrid(2)
        adj = g.tet_adjacency()
        self.assertEqual(len(adj), g.num_tets)
        for t, nbrs in enumerate(adj):
            for u in nbrs:
                self.assertIn(t, adj[u])

    def test_bad_resolution(self):
        with self.assertRaises(ValueError):
            TetGrid(0)
        with self.assertRaises(ValueError):
            TetGrid(2, lo=1.0, hi=1.0)


def _signed_volume(a, b, c, d):
    ab = [b[i] - a[i] for i in range(3)]
    ac = [c[i] - a[i] for i in range(3)]
    ad = [d[i] - a[i] for i in range(3)]
    cross = (
        ac[1] * ad[2] - ac[2] * ad[1],
        ac[2] * ad[0] - ac[0] * ad[2],
        ac[0] * ad[1] - ac[1] * ad[0],
    )
    return (ab[0] * cross[0] + ab[1] * cross[1] + ab[2] * cross[2]) / 6.0


if __name__ == "__main__":
    unittest.main()
