"""Tests for the DeepCAD sketch-plane orientation + extrusion decode."""

import math
import unittest

from harnesscad.domain.reconstruction import deepcad_sketch_plane as sp


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def _vclose(a, b, tol=1e-9):
    return all(_close(x, y, tol) for x, y in zip(a, b))


class TestRotation(unittest.TestCase):
    def test_identity_at_zero(self):
        x, y, n = sp.plane_axes(0.0, 0.0, 0.0)
        self.assertTrue(_vclose(x, (1, 0, 0)))
        self.assertTrue(_vclose(y, (0, 1, 0)))
        self.assertTrue(_vclose(n, (0, 0, 1)))

    def test_orthonormal_basis(self):
        x, y, n = sp.plane_axes(0.7, 1.1, -0.4)
        for v in (x, y, n):
            self.assertAlmostEqual(math.sqrt(sum(c * c for c in v)), 1.0)
        self.assertAlmostEqual(sum(a * b for a, b in zip(x, y)), 0.0)
        self.assertAlmostEqual(sum(a * b for a, b in zip(x, n)), 0.0)
        # Right-handed: x cross y == n.
        cross = (x[1] * y[2] - x[2] * y[1],
                 x[2] * y[0] - x[0] * y[2],
                 x[0] * y[1] - x[1] * y[0])
        self.assertTrue(_vclose(cross, n, 1e-9))

    def test_normal_matches_spherical(self):
        _, _, n = sp.plane_axes(0.9, 0.5, 1.3)
        self.assertTrue(_vclose(n, sp.plane_normal(0.9, 0.5)))

    def test_euler_roundtrip(self):
        for theta, phi, gamma in [(0.3, 0.4, 0.5), (1.2, -0.7, 2.0), (2.5, 3.0, -1.0)]:
            m = sp.rotation_matrix(theta, phi, gamma)
            t2, p2, g2 = sp.euler_from_matrix(m)
            m2 = sp.rotation_matrix(t2, p2, g2)
            for i in range(3):
                self.assertTrue(_vclose(m[i], m2[i], 1e-7))

    def test_euler_gimbal(self):
        # Normal along +z: theta ~ 0, gamma pinned to 0.
        m = sp.rotation_matrix(0.0, 0.0, 0.9)
        t, p, g = sp.euler_from_matrix(m)
        self.assertAlmostEqual(t, 0.0)
        self.assertEqual(g, 0.0)


class TestTransforms(unittest.TestCase):
    def test_local_to_world_identity_plane(self):
        w = sp.local_to_world((2.0, 3.0), 0.0, 0.0, 0.0)
        self.assertTrue(_vclose(w, (2.0, 3.0, 0.0)))

    def test_origin_and_scale(self):
        w = sp.local_to_world((1.0, 0.0), 0.0, 0.0, 0.0, origin=(5, 6, 7), scale=2.0)
        self.assertTrue(_vclose(w, (7.0, 6.0, 7.0)))

    def test_world_local_roundtrip(self):
        pt = (1.5, -2.5)
        w = sp.local_to_world(pt, 0.6, 0.4, -0.2, origin=(1, 2, 3), scale=1.5)
        u, v, wn = sp.world_to_local(w, 0.6, 0.4, -0.2, origin=(1, 2, 3), scale=1.5)
        self.assertAlmostEqual(u, pt[0])
        self.assertAlmostEqual(v, pt[1])
        self.assertAlmostEqual(wn, 0.0)

    def test_world_local_zero_scale(self):
        with self.assertRaises(ValueError):
            sp.world_to_local((0, 0, 0), 0.1, 0.1, 0.1, scale=0.0)


class TestExtrusion(unittest.TestCase):
    def test_one_sided(self):
        self.assertEqual(sp.extrusion_extents(0.7, 0.3, sp.ONE_SIDED), (0.0, 0.7))

    def test_symmetric(self):
        self.assertEqual(sp.extrusion_extents(0.5, 0.9, sp.SYMMETRIC), (-0.5, 0.5))

    def test_two_sided(self):
        self.assertEqual(sp.extrusion_extents(0.4, 0.6, sp.TWO_SIDED), (-0.6, 0.4))

    def test_bad_type(self):
        with self.assertRaises(ValueError):
            sp.extrusion_extents(0.1, 0.1, 9)

    def test_extrude_point_moves_along_normal(self):
        # On identity plane the normal is +z, so offset moves z.
        p = sp.extrude_point((1.0, 2.0), 0.5, 0.0, 0.0, 0.0)
        self.assertTrue(_vclose(p, (1.0, 2.0, 0.5)))


if __name__ == "__main__":
    unittest.main()
