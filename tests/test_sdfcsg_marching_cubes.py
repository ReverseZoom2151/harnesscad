"""Tests for geometry.sdfcsg_marching_cubes.

Checks the lookup tables are self-consistent (Bourke invariants) and that
Marching Cubes reconstructs a sphere: every vertex lies on the surface, the mesh
is watertight (each edge shared by exactly two triangles), and the triangle
normals point outward.
"""

from __future__ import annotations

import math
import unittest

from harnesscad.domain.geometry.volumes import sdfcsg_marching_cubes as MC
from harnesscad.domain.geometry.volumes import sdfcsg_surface_nets as SN


def sphere_field(r):
    return lambda p: math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) - r


class TestTables(unittest.TestCase):
    def test_lengths(self):
        self.assertEqual(len(MC.EDGE_TABLE), 256)
        self.assertEqual(len(MC.TRI_TABLE), 256)

    def test_edge_tri_consistency(self):
        # The set of edges referenced by TRI_TABLE[i] must equal the bits set in
        # EDGE_TABLE[i], for every one of the 256 configurations.
        for i in range(256):
            tri_edges = set(MC.TRI_TABLE[i])
            mask = MC.EDGE_TABLE[i]
            mask_edges = {e for e in range(12) if mask & (1 << e)}
            self.assertEqual(tri_edges, mask_edges, "config %d mismatch" % i)

    def test_triangles_are_triples(self):
        for i in range(256):
            self.assertEqual(len(MC.TRI_TABLE[i]) % 3, 0, "config %d not triples" % i)

    def test_complement_symmetry(self):
        # Inverting inside/outside must not change which edges are cut.
        for i in range(256):
            self.assertEqual(MC.EDGE_TABLE[i], MC.EDGE_TABLE[255 - i])


class TestSphere(unittest.TestCase):
    def setUp(self):
        self.r = 1.0
        res = 20
        g = SN.sample_sdf_grid(
            sphere_field(self.r), (-1.5, -1.5, -1.5), (1.5, 1.5, 1.5), (res, res, res)
        )
        self.cell = 3.0 / res
        self.verts, self.faces = MC.marching_cubes(g, 0.0)

    def test_nonempty(self):
        self.assertGreater(len(self.verts), 200)
        self.assertGreater(len(self.faces), 400)

    def test_vertices_on_surface(self):
        # MC vertices are exact edge crossings -> should sit very close to r.
        worst = 0.0
        for v in self.verts:
            d = abs(math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) - self.r)
            worst = max(worst, d)
        # Linear interpolation of a smooth field: error is well under a cell.
        self.assertLess(worst, 0.05 * self.cell + 1e-6)

    def test_watertight(self):
        edge_count = {}
        for f in self.faces:
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                key = (a, b) if a < b else (b, a)
                edge_count[key] = edge_count.get(key, 0) + 1
        counts = set(edge_count.values())
        self.assertEqual(counts, {2}, "non-manifold edges present: %s" % (counts,))

    def test_normals_outward(self):
        # For a sphere centred at origin, each triangle's geometric normal must
        # have positive dot with its centroid (points away from centre).
        bad = 0
        for f in self.faces:
            a, b, c = self.verts[f[0]], self.verts[f[1]], self.verts[f[2]]
            ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
            vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
            nx = uy * vz - uz * vy
            ny = uz * vx - ux * vz
            nz = ux * vy - uy * vx
            cx = (a[0] + b[0] + c[0]) / 3.0
            cy = (a[1] + b[1] + c[1]) / 3.0
            cz = (a[2] + b[2] + c[2]) / 3.0
            if nx * cx + ny * cy + nz * cz < 0.0:
                bad += 1
        self.assertEqual(bad, 0)


class TestBox(unittest.TestCase):
    def test_box_watertight(self):
        # An axis-aligned box exercises flat faces and edges.
        def box(p):
            dx = abs(p[0]) - 0.6
            dy = abs(p[1]) - 0.6
            dz = abs(p[2]) - 0.6
            ox, oy, oz = max(dx, 0.0), max(dy, 0.0), max(dz, 0.0)
            return math.sqrt(ox * ox + oy * oy + oz * oz) + min(max(dx, max(dy, dz)), 0.0)

        g = SN.sample_sdf_grid(box, (-1, -1, -1), (1, 1, 1), (16, 16, 16))
        verts, faces = MC.marching_cubes(g, 0.0)
        self.assertGreater(len(faces), 0)
        edge_count = {}
        for f in faces:
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                key = (a, b) if a < b else (b, a)
                edge_count[key] = edge_count.get(key, 0) + 1
        self.assertEqual(set(edge_count.values()), {2})


if __name__ == "__main__":
    unittest.main()
