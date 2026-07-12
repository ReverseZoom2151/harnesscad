"""Tests for geometry.nurbgen_surface (tensor-product NURBS surface)."""

import math
import unittest

from geometry import nurbgen_surface as ns


def _flat_plane():
    # Bilinear (degree 1x1) unit square in the z=0 plane, unit weights.
    poles = [[(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
             [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
    weights = [[1.0, 1.0], [1.0, 1.0]]
    U = [0.0, 0.0, 1.0, 1.0]
    V = [0.0, 0.0, 1.0, 1.0]
    return poles, weights, 1, 1, U, V


class TestFlatPlane(unittest.TestCase):
    def test_bilinear_interpolation(self):
        poles, w, pu, pv, U, V = _flat_plane()
        # Centre of the plane is (0.5, 0.5, 0).
        x, y, z = ns.surface_point(poles, w, pu, pv, U, V, 0.5, 0.5)
        self.assertAlmostEqual(x, 0.5, places=12)
        self.assertAlmostEqual(y, 0.5, places=12)
        self.assertAlmostEqual(z, 0.0, places=12)

    def test_corners_interpolated(self):
        poles, w, pu, pv, U, V = _flat_plane()
        self.assertAlmostEqual(
            ns.surface_point(poles, w, pu, pv, U, V, 0.0, 0.0)[0], 0.0)
        self.assertAlmostEqual(
            ns.surface_point(poles, w, pu, pv, U, V, 1.0, 1.0)[0], 1.0)

    def test_plane_normal_is_z(self):
        poles, w, pu, pv, U, V = _flat_plane()
        nrm = ns.surface_normal(poles, w, pu, pv, U, V, 0.3, 0.7)
        self.assertAlmostEqual(abs(nrm[2]), 1.0, places=12)
        self.assertAlmostEqual(nrm[0], 0.0, places=12)
        self.assertAlmostEqual(nrm[1], 0.0, places=12)


class TestCylinderPatch(unittest.TestCase):
    def test_all_points_on_cylinder(self):
        poles, w, pu, pv, U, V = ns.nurbs_cylinder_quadrant(2.0, 3.0)
        for a in range(6):
            for b in range(6):
                u, v = a / 5.0, b / 5.0
                x, y, z = ns.surface_point(poles, w, pu, pv, U, V, u, v)
                self.assertAlmostEqual(math.hypot(x, y), 2.0, places=9)
                self.assertTrue(-1e-9 <= z <= 3.0 + 1e-9)

    def test_cylinder_normal_is_radial(self):
        poles, w, pu, pv, U, V = ns.nurbs_cylinder_quadrant(1.0, 1.0)
        x, y, z = ns.surface_point(poles, w, pu, pv, U, V, 0.5, 0.5)
        nrm = ns.surface_normal(poles, w, pu, pv, U, V, 0.5, 0.5)
        # Outward normal is parallel to the radial direction (x, y, 0).
        rad = (x, y, 0.0)
        rlen = math.hypot(x, y)
        cosang = (nrm[0] * rad[0] + nrm[1] * rad[1]) / rlen
        self.assertAlmostEqual(abs(cosang), 1.0, places=8)
        self.assertAlmostEqual(nrm[2], 0.0, places=8)


class TestDerivatives(unittest.TestCase):
    def test_partials_match_finite_difference(self):
        poles, w, pu, pv, U, V = ns.nurbs_cylinder_quadrant(1.0, 2.0)
        u0, v0, h = 0.4, 0.6, 1e-6
        S, S_u, S_v = ns.surface_derivatives(poles, w, pu, pv, U, V, u0, v0)
        su_fd = tuple(
            (ns.surface_point(poles, w, pu, pv, U, V, u0 + h, v0)[c]
             - ns.surface_point(poles, w, pu, pv, U, V, u0 - h, v0)[c]) / (2 * h)
            for c in range(3))
        sv_fd = tuple(
            (ns.surface_point(poles, w, pu, pv, U, V, u0, v0 + h)[c]
             - ns.surface_point(poles, w, pu, pv, U, V, u0, v0 - h)[c]) / (2 * h)
            for c in range(3))
        for c in range(3):
            self.assertAlmostEqual(S_u[c], su_fd[c], places=5)
            self.assertAlmostEqual(S_v[c], sv_fd[c], places=5)


class TestTessellation(unittest.TestCase):
    def test_vertex_and_triangle_counts(self):
        poles, w, pu, pv, U, V = _flat_plane()
        verts, tris = ns.tessellate_surface(poles, w, pu, pv, U, V, 4, 3)
        self.assertEqual(len(verts), 5 * 4)
        self.assertEqual(len(tris), 4 * 3 * 2)
        # Indices in range.
        for t in tris:
            for idx in t:
                self.assertTrue(0 <= idx < len(verts))

    def test_flat_plane_area(self):
        poles, w, pu, pv, U, V = _flat_plane()
        verts, tris = ns.tessellate_surface(poles, w, pu, pv, U, V, 4, 4)
        self.assertAlmostEqual(ns.mesh_area(verts, tris), 1.0, places=10)

    def test_cylinder_quadrant_area(self):
        # Quarter cylinder radius 1, height 1: area = (pi/2) * 1.
        poles, w, pu, pv, U, V = ns.nurbs_cylinder_quadrant(1.0, 1.0)
        verts, tris = ns.tessellate_surface(poles, w, pu, pv, U, V, 2, 64)
        self.assertAlmostEqual(ns.mesh_area(verts, tris), math.pi / 2, places=2)


class TestValidation(unittest.TestCase):
    def test_ragged_grid_rejected(self):
        poles = [[(0.0, 0.0, 0.0)], [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
        with self.assertRaises(ValueError):
            ns.surface_point(poles, [[1.0], [1.0, 1.0]], 1, 1,
                             [0, 0, 1, 1], [0, 0, 1, 1], 0.5, 0.5)


if __name__ == "__main__":
    unittest.main()
