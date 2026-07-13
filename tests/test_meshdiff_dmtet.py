"""Tests for the DMTet deformable-tet encoding."""

import math
import random
import unittest

from harnesscad.domain.geometry.volumes.meshdiff_tet_grid import TetGrid
from harnesscad.domain.geometry.volumes.meshdiff_marching_tets import is_watertight, signed_volume
from harnesscad.domain.geometry.volumes.meshdiff_dmtet import (
    DMTet,
    barycentric_coords,
    interpolate_sdf_in_tet,
)


def _sphere_sdf(grid, radius):
    out = []
    for (x, y, z) in grid.vertices:
        out.append(math.sqrt(x * x + y * y + z * z) - radius)
    return out


class DeformationTest(unittest.TestCase):
    def test_zero_deformation_matches_grid(self):
        g = TetGrid(3)
        d = DMTet(g, [1.0] * g.num_vertices)
        self.assertEqual(d.deformed_vertices(), g.vertices)

    def test_deformation_is_clipped(self):
        g = TetGrid(2, lo=-1.0, hi=1.0)  # cell size 1.0, clip 0.5
        huge = [(10.0, -10.0, 10.0)] * g.num_vertices
        d = DMTet(g, [1.0] * g.num_vertices, deformation=huge)
        for dv in d.deformation:
            self.assertLessEqual(abs(dv[0]), 0.5 + 1e-12)
            self.assertLessEqual(abs(dv[1]), 0.5 + 1e-12)

    def test_small_deformation_no_inversion(self):
        g = TetGrid(4, lo=-1.0, hi=1.0)
        rng = random.Random(7)
        cell = 2.0 / 4
        defo = [
            tuple(rng.uniform(-0.1 * cell, 0.1 * cell) for _ in range(3))
            for _ in range(g.num_vertices)
        ]
        d = DMTet(g, [1.0] * g.num_vertices, deformation=defo)
        self.assertEqual(d.inverted_tet_count(), 0)
        self.assertTrue(d.is_valid())


class ExtractTest(unittest.TestCase):
    def test_sphere_roundtrip_watertight(self):
        g = TetGrid(10, lo=-1.5, hi=1.5)
        d = DMTet(g, _sphere_sdf(g, 1.0))
        verts, tris = d.extract_surface()
        self.assertTrue(len(tris) > 0)
        self.assertTrue(is_watertight(tris))
        self.assertGreater(signed_volume(verts, tris), 0.0)

    def test_normalization_preserves_topology(self):
        g = TetGrid(8, lo=-1.5, hi=1.5)
        d = DMTet(g, _sphere_sdf(g, 1.0))
        _, tris = d.extract_surface()
        _, tris_norm = d.normalized().extract_surface()
        # Same connectivity: identical number of triangles and welded vertices.
        self.assertEqual(len(tris), len(tris_norm))
        self.assertTrue(is_watertight(tris_norm))

    def test_normalization_values_are_pm1(self):
        g = TetGrid(4)
        d = DMTet(g, [(-3.0 if i % 2 else 2.5) for i in range(g.num_vertices)])
        nd = d.normalized()
        for s in nd.sdf:
            self.assertIn(s, (1.0, -1.0))
        # occupancy (s>0) is preserved for every vertex
        for s, ns in zip(d.sdf, nd.sdf):
            self.assertEqual(s > 0, ns > 0)


class BarycentricTest(unittest.TestCase):
    def setUp(self):
        self.tet = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    def test_corner_coords(self):
        b = barycentric_coords(self.tet, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(b[0], 1.0)
        for i in (1, 2, 3):
            self.assertAlmostEqual(b[i], 0.0)

    def test_centroid_coords(self):
        centroid = tuple(sum(p[r] for p in self.tet) / 4 for r in range(3))
        b = barycentric_coords(self.tet, centroid)
        for bi in b:
            self.assertAlmostEqual(bi, 0.25)
        self.assertAlmostEqual(sum(b), 1.0)

    def test_sdf_interpolation_linear(self):
        sdf = (-1.0, 1.0, 1.0, 1.0)
        # midpoint of edge (v0, v1): sdf should be 0 (the zero-crossing)
        mid = (0.5, 0.0, 0.0)
        self.assertAlmostEqual(interpolate_sdf_in_tet(self.tet, sdf, mid), 0.0)
        # at v0 it is -1
        self.assertAlmostEqual(
            interpolate_sdf_in_tet(self.tet, sdf, (0.0, 0.0, 0.0)), -1.0
        )

    def test_degenerate_raises(self):
        flat = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            barycentric_coords(flat, (0.5, 0.0, 0.0))


class ValidationTest(unittest.TestCase):
    def test_bad_sdf_length(self):
        g = TetGrid(2)
        with self.assertRaises(ValueError):
            DMTet(g, [0.0])

    def test_bad_grid_type(self):
        with self.assertRaises(TypeError):
            DMTet(object(), [0.0])


if __name__ == "__main__":
    unittest.main()
