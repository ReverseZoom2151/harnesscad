"""Tests for geometry.sdfcsg_surface_nets.

Reconstructs a sphere with Naive Surface Nets and checks that every extracted
vertex lies close to the true surface, that the mesh is closed (each edge shared
by exactly two triangles), and that the attribute blend / STL writer behave.
"""

from __future__ import annotations

import math
import unittest

from harnesscad.domain.geometry import sdfcsg_surface_nets as SN


def sphere_field(r):
    return lambda p: math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) - r


class TestSampleGrid(unittest.TestCase):
    def test_shape_and_values(self):
        g = SN.sample_sdf_grid(sphere_field(1.0), (-2, -2, -2), (2, 2, 2), (8, 8, 8))
        self.assertEqual(g.shape, (9, 9, 9))
        # Centre sample is the field at (or near) the origin -> about -1.
        c = g.get(4, 4, 4)
        self.assertAlmostEqual(c, -1.0, places=6)

    def test_world_mapping(self):
        g = SN.sample_sdf_grid(sphere_field(1.0), (-2, -2, -2), (2, 2, 2), (8, 8, 8))
        self.assertAlmostEqual(g.world(0, 0, 0)[0], -2.0, places=6)
        self.assertAlmostEqual(g.world(8, 8, 8)[2], 2.0, places=6)


class TestSurfaceNetsSphere(unittest.TestCase):
    def setUp(self):
        self.r = 1.0
        res = 24
        g = SN.sample_sdf_grid(
            sphere_field(self.r), (-1.5, -1.5, -1.5), (1.5, 1.5, 1.5), (res, res, res)
        )
        self.cell = 3.0 / res
        self.verts, self.faces = SN.surface_nets(g, 0.0)

    def test_nonempty(self):
        self.assertGreater(len(self.verts), 200)
        self.assertGreater(len(self.faces), 400)

    def test_vertices_near_surface(self):
        # Every surface-nets vertex must lie within ~one cell of the sphere.
        tol = 1.2 * self.cell
        worst = 0.0
        for v in self.verts:
            d = abs(math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) - self.r)
            worst = max(worst, d)
        self.assertLess(worst, tol)

    def test_closed_manifold(self):
        # Every undirected edge is shared by exactly two triangles.
        edge_count = {}
        for f in self.faces:
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                key = (a, b) if a < b else (b, a)
                edge_count[key] = edge_count.get(key, 0) + 1
        bad = [k for k, c in edge_count.items() if c != 2]
        self.assertEqual(bad, [])

    def test_indices_in_range(self):
        n = len(self.verts)
        for f in self.faces:
            for idx in f:
                self.assertTrue(0 <= idx < n)


class TestAttributeBlend(unittest.TestCase):
    def test_inverse_distance_weight(self):
        fa = sphere_field(1.0)  # centred at origin
        fb = lambda p: sphere_field(1.0)((p[0] - 4.0, p[1], p[2]))  # centred at x=4
        # A point on primitive A's surface should read almost pure attr_a.
        out = SN.interpolate_attribute(fa, [1.0, 0.0, 0.0], fb, [0.0, 0.0, 1.0], (1.0, 0.0, 0.0))
        self.assertAlmostEqual(out[0], 1.0, places=6)
        self.assertAlmostEqual(out[2], 0.0, places=6)

    def test_midpoint(self):
        fa = sphere_field(1.0)
        fb = lambda p: sphere_field(1.0)((p[0] - 2.0, p[1], p[2]))
        out = SN.interpolate_attribute(fa, [0.0], fb, [1.0], (1.0, 0.0, 0.0))
        # Equidistant point -> equal blend.
        self.assertAlmostEqual(out[0], 0.5, places=6)


class TestStl(unittest.TestCase):
    def test_roundtrip_counts(self):
        verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        faces = [(0, 1, 2)]
        text = SN.mesh_to_stl(verts, faces, "t")
        self.assertTrue(text.startswith("solid t"))
        self.assertEqual(text.count("facet normal"), 1)
        self.assertEqual(text.count("vertex"), 3)
        self.assertIn("endsolid t", text)


if __name__ == "__main__":
    unittest.main()
