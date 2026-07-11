"""Tests for marching-tets edge-crossing sensitivity metric."""

import math
import unittest

from geometry.meshdiff_tet_grid import TetGrid
from geometry.meshdiff_dmtet import DMTet
from geometry.meshdiff_edge_sensitivity import (
    mesh_generating_edges,
    edge_crossing_sensitivity,
    max_crossing_sensitivity,
)


def _sphere_sdf(grid, radius):
    return [
        math.sqrt(x * x + y * y + z * z) - radius for (x, y, z) in grid.vertices
    ]


class SingleTetTest(unittest.TestCase):
    def setUp(self):
        # One tetrahedron, unit-ish edges.
        self.verts = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        self.tets = [(0, 1, 2, 3)]

    def test_no_crossing_when_all_same_sign(self):
        sdf = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(mesh_generating_edges(self.tets, sdf), [])
        res = edge_crossing_sensitivity(self.verts, self.tets, sdf)
        self.assertEqual(res["count"], 0)
        self.assertEqual(res["max"], 0.0)
        self.assertEqual(res["mean"], 0.0)

    def test_factor_is_length_over_gap(self):
        # v0 inside (-1), others outside (+1): crossing edges (0,1),(0,2),(0,3).
        sdf = [-1.0, 1.0, 1.0, 1.0]
        edges = mesh_generating_edges(self.tets, sdf)
        self.assertEqual(edges, [(0, 1), (0, 2), (0, 3)])
        res = edge_crossing_sensitivity(self.verts, self.tets, sdf)
        # each edge length 1.0, gap |1 - (-1)| = 2 -> factor 0.5
        for f in res["per_edge"].values():
            self.assertAlmostEqual(f, 0.5)
        self.assertAlmostEqual(res["max"], 0.5)

    def test_small_gap_amplifies(self):
        # A tiny SDF gap across the same geometric edge blows up the factor.
        sdf = [-0.01, 0.01, 1.0, 1.0]
        res = edge_crossing_sensitivity(self.verts, self.tets, sdf)
        # edge (0,1): length 1, gap 0.02 -> factor 50
        self.assertAlmostEqual(res["per_edge"][(0, 1)], 50.0)
        self.assertGreater(res["max"], 10.0)


class NormalizationReducesSensitivityTest(unittest.TestCase):
    def test_normalization_bounds_worst_case(self):
        g = TetGrid(6, lo=-1.5, hi=1.5)
        sdf = _sphere_sdf(g, 1.0)
        raw = max_crossing_sensitivity(g.vertices, g.tets, sdf)
        norm = DMTet(g, sdf).normalized()
        normed = max_crossing_sensitivity(g.vertices, g.tets, norm.sdf)
        # After normalization gap is exactly 2, so factor = length/2, bounded by
        # half the longest crossing edge -- and no worse than the raw worst case.
        self.assertLessEqual(normed, raw + 1e-9)
        cell_diag = math.sqrt(3) * (3.0 / 6)
        self.assertLessEqual(normed, cell_diag / 2 + 1e-9)

    def test_same_crossing_edges_after_normalization(self):
        g = TetGrid(6, lo=-1.5, hi=1.5)
        sdf = _sphere_sdf(g, 1.0)
        norm = DMTet(g, sdf).normalized()
        self.assertEqual(
            mesh_generating_edges(g.tets, sdf),
            mesh_generating_edges(g.tets, norm.sdf),
        )


if __name__ == "__main__":
    unittest.main()
