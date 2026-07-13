"""Tests for marching tetrahedra, validated against analytic SDFs."""

import math
import unittest

from harnesscad.domain.geometry.volumes.meshdiff_tet_grid import TetGrid
from harnesscad.domain.geometry.volumes.meshdiff_marching_tets import (
    marching_tets,
    edge_manifold_stats,
    is_watertight,
    signed_volume,
)


def _plane_sdf(grid, normal, offset):
    # Signed distance to plane n.x = offset (n unit); positive on +n side.
    nx, ny, nz = normal
    return [nx * x + ny * y + nz * z - offset for (x, y, z) in grid.vertices]


def _sphere_sdf(grid, radius, center=(0.0, 0.0, 0.0)):
    cx, cy, cz = center
    out = []
    for (x, y, z) in grid.vertices:
        d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
        out.append(d - radius)  # >0 outside the sphere
    return out


class PlaneTest(unittest.TestCase):
    def test_axis_plane_is_planar_cut(self):
        # Plane z = 0.13 through the grid -> all surface points have z == 0.13.
        g = TetGrid(6, lo=-1.0, hi=1.0)
        sdf = _plane_sdf(g, (0.0, 0.0, 1.0), 0.13)
        verts, tris = marching_tets(g.vertices, g.tets, sdf)
        self.assertTrue(len(verts) > 0)
        self.assertTrue(len(tris) > 0)
        for (x, y, z) in verts:
            self.assertAlmostEqual(z, 0.13, places=9)

    def test_diagonal_plane_is_planar(self):
        g = TetGrid(5, lo=-1.0, hi=1.0)
        n = (1.0 / math.sqrt(3),) * 3
        offset = 0.2
        sdf = _plane_sdf(g, n, offset)
        verts, tris = marching_tets(g.vertices, g.tets, sdf)
        self.assertTrue(len(tris) > 0)
        for p in verts:
            val = sum(n[i] * p[i] for i in range(3))
            self.assertAlmostEqual(val, offset, places=9)

    def test_plane_outside_grid_empty(self):
        g = TetGrid(3, lo=-1.0, hi=1.0)
        sdf = _plane_sdf(g, (0.0, 0.0, 1.0), 5.0)  # never crosses
        verts, tris = marching_tets(g.vertices, g.tets, sdf)
        self.assertEqual(verts, [])
        self.assertEqual(tris, [])


class SphereTest(unittest.TestCase):
    def setUp(self):
        self.g = TetGrid(10, lo=-1.5, hi=1.5)
        self.radius = 1.0
        self.sdf = _sphere_sdf(self.g, self.radius)
        self.verts, self.tris = marching_tets(self.g.vertices, self.g.tets, self.sdf)

    def test_nonempty(self):
        self.assertTrue(len(self.verts) > 0)
        self.assertTrue(len(self.tris) > 0)

    def test_surface_points_near_radius(self):
        # Every extracted vertex lies (approximately) on the sphere.
        for (x, y, z) in self.verts:
            r = math.sqrt(x * x + y * y + z * z)
            # linear interpolation slightly under-estimates a convex surface
            self.assertLess(abs(r - self.radius), 0.06)

    def test_closed_watertight_surface(self):
        boundary, nonmanifold = edge_manifold_stats(self.tris)
        self.assertEqual(boundary, 0)
        self.assertEqual(nonmanifold, 0)
        self.assertTrue(is_watertight(self.tris))

    def test_outward_orientation_positive_volume(self):
        vol = signed_volume(self.verts, self.tris)
        self.assertGreater(vol, 0.0)
        # roughly the volume of a unit sphere (4/3 pi ~ 4.19), under-estimated
        self.assertLess(vol, 4.19)
        self.assertGreater(vol, 3.0)

    def test_inside_outside_consistency(self):
        # The grid origin is inside (sdf<0); a far corner is outside (sdf>0).
        origin_idx = self.g.vertices.index((0.0, 0.0, 0.0))
        self.assertLess(self.sdf[origin_idx], 0.0)
        self.assertGreater(max(self.sdf), 0.0)


class ValidationTest(unittest.TestCase):
    def test_sdf_length_mismatch(self):
        g = TetGrid(2)
        with self.assertRaises(ValueError):
            marching_tets(g.vertices, g.tets, [0.0])

    def test_empty_not_watertight(self):
        self.assertFalse(is_watertight([]))


if __name__ == "__main__":
    unittest.main()
