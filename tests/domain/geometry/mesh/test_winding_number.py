"""Tests for geometry.manifold_winding."""

import math
import unittest

from harnesscad.domain.geometry.mesh.winding_number import (
    solid_angle,
    winding_number,
    is_inside,
    signed_volume,
    surface_area,
)
from harnesscad.domain.geometry.mesh.halfedge import tetrahedron_mesh, cube_mesh


def _mesh_arrays(m):
    return m.vertices, m.tris


class TestSolidAngle(unittest.TestCase):
    def test_degenerate_at_vertex(self):
        self.assertEqual(solid_angle((0, 0, 0), (0, 0, 0), (1, 0, 0), (0, 1, 0)), 0.0)

    def test_full_sphere_from_closed_mesh(self):
        # Sum of solid angles over a closed mesh, seen from inside, is +-4*pi.
        m = tetrahedron_mesh()
        # centroid of the tetra vertices
        cx = sum(v[0] for v in m.vertices) / 4
        cy = sum(v[1] for v in m.vertices) / 4
        cz = sum(v[2] for v in m.vertices) / 4
        total = sum(solid_angle((cx, cy, cz), m.vertices[i], m.vertices[j], m.vertices[k])
                    for (i, j, k) in m.tris)
        self.assertAlmostEqual(abs(total), 4.0 * math.pi, places=6)


class TestWindingCube(unittest.TestCase):
    def setUp(self):
        self.m = cube_mesh(2.0)  # cube [0,2]^3
        self.v, self.t = _mesh_arrays(self.m)

    def test_inside_is_one(self):
        w = winding_number((1.0, 1.0, 1.0), self.v, self.t)
        self.assertAlmostEqual(abs(w), 1.0, places=6)

    def test_outside_is_zero(self):
        w = winding_number((5.0, 5.0, 5.0), self.v, self.t)
        self.assertAlmostEqual(w, 0.0, places=6)

    def test_is_inside_true(self):
        self.assertTrue(is_inside((1.0, 1.0, 1.0), self.v, self.t))
        self.assertTrue(is_inside((0.1, 0.1, 0.1), self.v, self.t))

    def test_is_inside_false(self):
        self.assertFalse(is_inside((-1.0, 1.0, 1.0), self.v, self.t))
        self.assertFalse(is_inside((3.0, 3.0, 3.0), self.v, self.t))
        self.assertFalse(is_inside((1.0, 1.0, 5.0), self.v, self.t))


class TestWindingTetra(unittest.TestCase):
    def test_inside_outside(self):
        m = tetrahedron_mesh(3.0)
        v, t = _mesh_arrays(m)
        # near vertex 0 origin, interior sample
        self.assertTrue(is_inside((0.3, 0.3, 0.3), v, t))
        self.assertFalse(is_inside((2.0, 2.0, 2.0), v, t))


class TestMassProperties(unittest.TestCase):
    def test_cube_volume(self):
        m = cube_mesh(2.0)
        v, t = _mesh_arrays(m)
        self.assertAlmostEqual(abs(signed_volume(v, t)), 8.0, places=9)

    def test_cube_area(self):
        m = cube_mesh(2.0)
        v, t = _mesh_arrays(m)
        self.assertAlmostEqual(surface_area(v, t), 6 * 4.0, places=9)

    def test_tetra_volume(self):
        # tetra with legs of length s has volume s^3 / 6
        m = tetrahedron_mesh(2.0)
        v, t = _mesh_arrays(m)
        self.assertAlmostEqual(abs(signed_volume(v, t)), 8.0 / 6.0, places=9)

    def test_volume_sign_flips_with_winding(self):
        m = cube_mesh(1.0)
        v = m.vertices
        t = m.tris
        flipped = [(a, c, b) for (a, b, c) in t]
        self.assertAlmostEqual(signed_volume(v, t), -signed_volume(v, flipped), places=12)


if __name__ == "__main__":
    unittest.main()
