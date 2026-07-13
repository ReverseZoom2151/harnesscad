"""Tests for ComplexGen-style 3D surface primitive fitting (analytic shapes)."""

import math
import unittest

from harnesscad.domain.geometry import complexgen_surface_fit as sf


def _plane_points(normal, offset, n=6):
    normal = sf._normalize(normal)
    u, v = sf._basis_for(normal)
    base = sf._scale(normal, offset)
    pts = []
    for i in range(n):
        for j in range(n):
            a = -1.0 + 2.0 * i / (n - 1)
            b = -1.0 + 2.0 * j / (n - 1)
            pts.append(sf._add(base, sf._add(sf._scale(u, a), sf._scale(v, b))))
    return pts


def _sphere_points(centre, radius, n=8):
    pts = []
    for i in range(1, n):
        theta = math.pi * i / n
        for j in range(2 * n):
            phi = 2.0 * math.pi * j / (2 * n)
            pts.append((centre[0] + radius * math.sin(theta) * math.cos(phi),
                        centre[1] + radius * math.sin(theta) * math.sin(phi),
                        centre[2] + radius * math.cos(theta)))
    return pts


def _cylinder_points(axis_point, axis, radius, n=16, m=5, half_length=1.5):
    axis = sf._normalize(axis)
    u, v = sf._basis_for(axis)
    pts, normals = [], []
    for i in range(n):
        phi = 2.0 * math.pi * i / n
        radial = sf._add(sf._scale(u, radius * math.cos(phi)),
                         sf._scale(v, radius * math.sin(phi)))
        for k in range(m):
            t = -half_length + 2.0 * half_length * k / (m - 1)
            pts.append(sf._add(sf._add(axis_point, radial), sf._scale(axis, t)))
            normals.append(sf._normalize(radial))
    return pts, normals


def _cone_points(apex, axis, half_angle, n=16, m=5):
    axis = sf._normalize(axis)
    u, v = sf._basis_for(axis)
    pts, normals = [], []
    for i in range(n):
        phi = 2.0 * math.pi * i / n
        radial = sf._add(sf._scale(u, math.cos(phi)), sf._scale(v, math.sin(phi)))
        for k in range(m):
            t = 0.5 + k * 0.4
            p = sf._add(apex, sf._add(sf._scale(axis, t * math.cos(half_angle)),
                                      sf._scale(radial, t * math.sin(half_angle))))
            pts.append(p)
            n_vec = sf._sub(sf._scale(radial, math.cos(half_angle)),
                            sf._scale(axis, math.sin(half_angle)))
            normals.append(sf._normalize(n_vec))
    return pts, normals


class TestLinearAlgebra(unittest.TestCase):
    def test_jacobi_eigen_diagonal(self):
        vals, vecs = sf.jacobi_eigen([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
        self.assertAlmostEqual(vals[0], 1.0)
        self.assertAlmostEqual(vals[2], 3.0)
        self.assertAlmostEqual(abs(vecs[0][1]), 1.0)

    def test_solve_linear(self):
        x = sf.solve_linear([[2.0, 1.0], [1.0, 3.0]], [5.0, 10.0])
        self.assertAlmostEqual(x[0], 1.0)
        self.assertAlmostEqual(x[1], 3.0)

    def test_solve_linear_singular(self):
        with self.assertRaises(ValueError):
            sf.solve_linear([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])


class TestFitPlane(unittest.TestCase):
    def test_recovers_analytic_plane(self):
        normal = sf._normalize((1.0, 2.0, 3.0))
        pts = _plane_points(normal, 1.7)
        (n, off), rms = sf.fit_plane(pts)
        self.assertLess(rms, 1e-9)
        # normal is recovered up to sign
        self.assertAlmostEqual(abs(sf._dot(n, normal)), 1.0, places=9)
        self.assertAlmostEqual(abs(off), 1.7, places=8)

    def test_axis_aligned_plane(self):
        pts = [(x, y, 4.0) for x in range(3) for y in range(3)]
        (n, off), rms = sf.fit_plane(pts)
        self.assertLess(rms, 1e-9)
        self.assertAlmostEqual(abs(n[2]), 1.0)
        self.assertAlmostEqual(abs(off), 4.0)

    def test_too_few_points(self):
        with self.assertRaises(ValueError):
            sf.fit_plane([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])

    def test_distance_to_plane(self):
        self.assertAlmostEqual(sf.distance_to_plane((0.0, 0.0, 0.0), ((0.0, 0.0, 1.0), 2.0)), 2.0)


class TestFitSphere(unittest.TestCase):
    def test_recovers_analytic_sphere(self):
        centre = (1.0, -2.0, 0.5)
        pts = _sphere_points(centre, 2.25)
        (c, r), rms = sf.fit_sphere(pts)
        self.assertLess(rms, 1e-8)
        self.assertAlmostEqual(r, 2.25, places=8)
        for a, b in zip(c, centre):
            self.assertAlmostEqual(a, b, places=8)

    def test_too_few_points(self):
        with self.assertRaises(ValueError):
            sf.fit_sphere([(0.0, 0.0, 0.0)])

    def test_distance_to_sphere(self):
        d = sf.distance_to_sphere((3.0, 0.0, 0.0), ((0.0, 0.0, 0.0), 1.0))
        self.assertAlmostEqual(d, 2.0)


class TestFitCylinder(unittest.TestCase):
    def test_recovers_axis_with_normals(self):
        axis = sf._normalize((0.0, 1.0, 1.0))
        axis_point = (2.0, 0.0, 0.0)
        pts, normals = _cylinder_points(axis_point, axis, 1.3)
        (ap, ax, r), rms = sf.fit_cylinder(pts, normals)
        self.assertLess(rms, 1e-7)
        self.assertAlmostEqual(r, 1.3, places=7)
        self.assertAlmostEqual(abs(sf._dot(ax, axis)), 1.0, places=7)
        # the recovered axis line passes through the true axis point
        d = sf._sub(axis_point, ap)
        perp = sf._sub(d, sf._scale(ax, sf._dot(d, ax)))
        self.assertLess(sf._norm(perp), 1e-6)

    def test_recovers_axis_without_normals(self):
        axis = sf._normalize((0.0, 0.0, 1.0))
        pts, _ = _cylinder_points((0.5, -0.25, 0.0), axis, 0.8)
        (_, ax, r), rms = sf.fit_cylinder(pts)
        self.assertLess(rms, 1e-3)
        self.assertAlmostEqual(r, 0.8, places=3)
        self.assertGreater(abs(sf._dot(ax, axis)), 0.999)

    def test_oblique_axis_without_normals(self):
        axis = sf._normalize((1.0, 1.0, 2.0))
        pts, _ = _cylinder_points((0.0, 0.0, 0.0), axis, 1.0)
        (_, ax, r), rms = sf.fit_cylinder(pts)
        self.assertLess(rms, 5e-3)
        self.assertAlmostEqual(r, 1.0, places=2)
        self.assertGreater(abs(sf._dot(ax, axis)), 0.999)

    def test_distance_to_cylinder(self):
        params = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0)
        self.assertAlmostEqual(sf.distance_to_cylinder((3.0, 0.0, 7.0), params), 2.0)


class TestFitCone(unittest.TestCase):
    def test_recovers_analytic_cone(self):
        apex = (0.3, -0.2, 1.0)
        axis = sf._normalize((0.0, 0.0, 1.0))
        half = math.radians(25.0)
        pts, normals = _cone_points(apex, axis, half)
        (a, ax, alpha), rms = sf.fit_cone(pts, normals)
        self.assertLess(rms, 1e-7)
        self.assertAlmostEqual(alpha, half, places=7)
        self.assertAlmostEqual(abs(sf._dot(ax, axis)), 1.0, places=7)
        for u, v in zip(a, apex):
            self.assertAlmostEqual(u, v, places=6)

    def test_oblique_cone(self):
        apex = (1.0, 1.0, 1.0)
        axis = sf._normalize((1.0, 0.0, 1.0))
        half = math.radians(40.0)
        pts, normals = _cone_points(apex, axis, half)
        (a, _, alpha), rms = sf.fit_cone(pts, normals)
        self.assertLess(rms, 1e-6)
        self.assertAlmostEqual(alpha, half, places=6)
        for u, v in zip(a, apex):
            self.assertAlmostEqual(u, v, places=5)

    def test_requires_normals(self):
        with self.assertRaises(ValueError):
            sf.fit_cone([(0.0, 0.0, 0.0)] * 6, None)


class TestDistanceToTorus(unittest.TestCase):
    def test_point_on_torus(self):
        params = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.0, 0.5)
        self.assertAlmostEqual(sf.distance_to_torus((2.5, 0.0, 0.0), params), 0.0)
        self.assertAlmostEqual(sf.distance_to_torus((2.0, 0.0, 0.5), params), 0.0)
        self.assertAlmostEqual(sf.distance_to_torus((4.0, 0.0, 0.0), params), 1.5)


class TestFitBest(unittest.TestCase):
    def test_plane_points_select_plane(self):
        pts = _plane_points((0.0, 0.0, 1.0), 0.0)
        kind, _, rms = sf.fit_best(pts)
        self.assertEqual(kind, sf.PLANE)
        self.assertLess(rms, 1e-9)

    def test_sphere_points_select_sphere(self):
        pts = _sphere_points((0.0, 0.0, 0.0), 1.0)
        kind, params, rms = sf.fit_best(pts)
        self.assertEqual(kind, sf.SPHERE)
        self.assertAlmostEqual(params[1], 1.0, places=8)
        self.assertLess(rms, 1e-8)

    def test_cylinder_points_select_cylinder(self):
        pts, normals = _cylinder_points((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0)
        kind, _, rms = sf.fit_best(pts, normals)
        self.assertEqual(kind, sf.CYLINDER)
        self.assertLess(rms, 1e-7)

    def test_cone_points_select_cone(self):
        pts, normals = _cone_points((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), math.radians(30.0))
        kind, _, rms = sf.fit_best(pts, normals)
        self.assertEqual(kind, sf.CONE)
        self.assertLess(rms, 1e-7)

    def test_deterministic(self):
        pts, normals = _cylinder_points((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), 0.7)
        first = sf.fit_best(pts, normals)
        second = sf.fit_best(pts, normals)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
