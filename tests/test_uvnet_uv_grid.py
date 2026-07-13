"""Tests for UV-Net face UV-grid sampling (validated against analytic surfaces)."""

import math
import unittest

from harnesscad.domain.geometry.parametric import complexgen_surface_fit as sf
from harnesscad.domain.geometry.parametric import uvnet_uv_grid as uvg


class LinspaceGridTest(unittest.TestCase):
    def test_linspace_inclusive(self):
        self.assertEqual(uvg.linspace(0.0, 1.0, 5),
                         [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(uvg.linspace(2.0, 4.0, 1), [3.0])
        with self.assertRaises(ValueError):
            uvg.linspace(0.0, 1.0, 0)

    def test_grid_parameters_shape_and_corners(self):
        params = uvg.grid_parameters(((0.0, 1.0), (0.0, 2.0)), 3, 4)
        self.assertEqual(len(params), 3)
        self.assertEqual(len(params[0]), 4)
        self.assertEqual(params[0][0], (0.0, 0.0))
        self.assertEqual(params[2][3], (1.0, 2.0))


class PlaneGridTest(unittest.TestCase):
    def test_points_lie_on_plane_and_normals_constant(self):
        plane = uvg.Plane(origin=(0.0, 0.0, 3.0), axis=(0.0, 0.0, 1.0),
                          u_range=(-1.0, 1.0), v_range=(-2.0, 2.0))
        pts = uvg.uv_grid(plane, 5, 6, method=uvg.POINT)
        nrm = uvg.uv_grid(plane, 5, 6, method=uvg.NORMAL)
        for row_p, row_n in zip(pts, nrm):
            for p, n in zip(row_p, row_n):
                self.assertAlmostEqual(p[2], 3.0, places=12)
                self.assertAlmostEqual(sf.distance_to_plane(p, ((0, 0, 1), 3.0)),
                                       0.0, places=12)
                self.assertAlmostEqual(n[2], 1.0, places=12)

    def test_reverse_flips_normal(self):
        plane = uvg.Plane(origin=(0, 0, 0), axis=(0, 0, 1), reverse=True)
        n = plane.normal(0.3, 0.4)
        self.assertAlmostEqual(n[2], -1.0, places=12)

    def test_determinism(self):
        plane = uvg.Plane(origin=(1.0, 2.0, 3.0), axis=(1.0, 1.0, 0.0))
        a = uvg.face_feature_grid(plane, 4, 4)
        b = uvg.face_feature_grid(plane, 4, 4)
        self.assertEqual(a, b)


class CylinderGridTest(unittest.TestCase):
    def test_points_at_constant_radius(self):
        cyl = uvg.Cylinder(origin=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0),
                           radius=2.0, v_range=(0.0, 5.0))
        grid = uvg.uv_grid(cyl, 8, 4, method=uvg.POINT)
        for row in grid:
            for p in row:
                self.assertAlmostEqual(math.hypot(p[0], p[1]), 2.0, places=10)
                self.assertAlmostEqual(
                    sf.distance_to_cylinder(p, ((0, 0, 0), (0, 0, 1), 2.0)),
                    0.0, places=10)

    def test_normal_is_radial_and_unit(self):
        cyl = uvg.Cylinder(origin=(1.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0),
                           radius=3.0)
        for u in (0.0, 1.0, 2.5):
            n = cyl.normal(u, 0.7)
            self.assertAlmostEqual(uvg._norm(n), 1.0, places=12)
            self.assertAlmostEqual(n[2], 0.0, places=12)
            p = cyl.point(u, 0.7)
            radial = uvg._normalize((p[0] - 1.0, p[1], 0.0))
            for a, b in zip(n, radial):
                self.assertAlmostEqual(a, b, places=10)


class ConeGridTest(unittest.TestCase):
    def test_points_on_cone_surface(self):
        half = math.radians(30.0)
        cone = uvg.Cone(origin=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0),
                        radius=0.0, half_angle=half, v_range=(0.5, 3.0))
        for row in uvg.uv_grid(cone, 6, 5, method=uvg.POINT):
            for p in row:
                d = sf.distance_to_cone(p, ((0, 0, 0), (0, 0, 1), half))
                self.assertAlmostEqual(d, 0.0, places=10)

    def test_normal_orthogonal_to_ruling(self):
        half = math.radians(20.0)
        cone = uvg.Cone(origin=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0),
                        radius=1.0, half_angle=half)
        u, v = 0.9, 1.5
        p0 = cone.point(u, 0.0)
        p1 = cone.point(u, v)
        ruling = uvg._sub(p1, p0)
        n = cone.normal(u, v)
        self.assertAlmostEqual(uvg._norm(n), 1.0, places=12)
        self.assertAlmostEqual(uvg._dot(n, ruling), 0.0, places=10)


class SphereGridTest(unittest.TestCase):
    def test_points_at_radius_and_normals_radial(self):
        sph = uvg.Sphere(centre=(1.0, -2.0, 0.5), radius=4.0)
        for row in uvg.face_feature_grid(sph, 7, 5):
            for c in row:
                p = (c[0], c[1], c[2])
                n = (c[3], c[4], c[5])
                self.assertAlmostEqual(
                    sf.distance_to_sphere(p, ((1.0, -2.0, 0.5), 4.0)), 0.0,
                    places=10)
                radial = uvg._normalize(uvg._sub(p, (1.0, -2.0, 0.5)))
                for a, b in zip(n, radial):
                    self.assertAlmostEqual(a, b, places=10)

    def test_poles_are_on_axis(self):
        sph = uvg.Sphere(centre=(0.0, 0.0, 0.0), radius=1.0)
        top = sph.point(0.0, math.pi / 2.0)
        bot = sph.point(1.3, -math.pi / 2.0)
        self.assertAlmostEqual(top[2], 1.0, places=12)
        self.assertAlmostEqual(bot[2], -1.0, places=12)


class TorusGridTest(unittest.TestCase):
    def test_points_on_torus(self):
        tor = uvg.Torus(centre=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0),
                        major_radius=5.0, minor_radius=1.5)
        for row in uvg.uv_grid(tor, 6, 6, method=uvg.POINT):
            for p in row:
                d = sf.distance_to_torus(p, ((0, 0, 0), (0, 0, 1), 5.0, 1.5))
                self.assertAlmostEqual(d, 0.0, places=10)

    def test_normal_unit_and_matches_finite_difference(self):
        tor = uvg.Torus(centre=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0),
                        major_radius=3.0, minor_radius=1.0)
        u, v, h = 0.6, 1.1, 1e-6
        du = uvg._scale(uvg._sub(tor.point(u + h, v), tor.point(u - h, v)),
                        1.0 / (2 * h))
        dv = uvg._scale(uvg._sub(tor.point(u, v + h), tor.point(u, v - h)),
                        1.0 / (2 * h))
        n = tor.normal(u, v)
        self.assertAlmostEqual(uvg._norm(n), 1.0, places=12)
        self.assertAlmostEqual(uvg._dot(n, du), 0.0, places=6)
        self.assertAlmostEqual(uvg._dot(n, dv), 0.0, places=6)


class BSplineGridTest(unittest.TestCase):
    def test_bilinear_patch_is_a_plane(self):
        poles = [[(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
                 [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
        weights = [[1.0, 1.0], [1.0, 1.0]]
        knots = [0.0, 0.0, 1.0, 1.0]
        surf = uvg.BSplineSurface(poles, weights, 1, 1, knots, knots)
        self.assertEqual(surf.domain(), ((0.0, 1.0), (0.0, 1.0)))
        grid = uvg.face_feature_grid(surf, 4, 4)
        self.assertEqual(uvg.grid_shape(grid), (4, 4, 7))
        for row in grid:
            for c in row:
                self.assertAlmostEqual(c[2], 0.0, places=12)
                self.assertAlmostEqual(abs(c[5]), 1.0, places=10)


class TrimmingMaskTest(unittest.TestCase):
    def test_square_hole_even_odd(self):
        outer = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        hole = [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)]
        self.assertEqual(uvg.visibility_status((0.2, 0.2), [outer, hole]), uvg.IN)
        self.assertEqual(uvg.visibility_status((0.5, 0.5), [outer, hole]), uvg.OUT)
        self.assertEqual(uvg.visibility_status((1.5, 0.5), [outer, hole]), uvg.OUT)
        self.assertEqual(uvg.visibility_status((0.0, 0.5), [outer, hole]), uvg.ON)
        self.assertEqual(uvg.trimming_mask((0.0, 0.5), [outer, hole]), 1)
        self.assertEqual(uvg.trimming_mask((0.5, 0.5), [outer, hole]), 0)

    def test_no_loops_means_everything_inside(self):
        self.assertEqual(uvg.visibility_status((9.0, 9.0), None), uvg.IN)
        self.assertEqual(uvg.trimming_mask((9.0, 9.0), []), 1)

    def test_mask_channel_in_face_grid(self):
        plane = uvg.Plane(origin=(0, 0, 0), axis=(0, 0, 1),
                          u_range=(0.0, 1.0), v_range=(0.0, 1.0))
        hole = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]
        outer = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        grid = uvg.face_feature_grid(plane, 5, 5, trim_loops=[outer, hole])
        # 5x5 inclusive grid: interior nodes at 0.25/0.5/0.75 -> the hole
        # boundary nodes count as ON (mask 1), only the centre is masked out.
        self.assertEqual(grid[2][2][6], 0.0)
        self.assertEqual(grid[0][0][6], 1.0)
        self.assertAlmostEqual(uvg.mask_ratio(grid), 24.0 / 25.0, places=12)
        self.assertEqual(len(uvg.masked_points(grid)), 24)


class SurfaceFromFitTest(unittest.TestCase):
    def test_round_trip_cylinder_fit_to_grid(self):
        cyl = uvg.Cylinder(origin=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0),
                           radius=2.0, v_range=(0.0, 3.0))
        grid = uvg.face_feature_grid(cyl, 12, 6)
        pts = uvg.masked_points(grid)
        normals = [(c[3], c[4], c[5]) for row in grid for c in row]
        kind, params, rms = sf.fit_best(pts, normals)
        self.assertEqual(kind, "cylinder")
        self.assertLess(rms, 1e-6)
        rebuilt = uvg.surface_from_fit(kind, params, v_range=(0.0, 3.0))
        self.assertIsInstance(rebuilt, uvg.Cylinder)
        for p in uvg.masked_points(uvg.face_feature_grid(rebuilt, 5, 3)):
            self.assertAlmostEqual(
                sf.distance_to_cylinder(p, ((0, 0, 0), (0, 0, 1), 2.0)),
                0.0, places=6)

    def test_plane_from_fit(self):
        surf = uvg.surface_from_fit("plane", ((0.0, 0.0, 1.0), 2.0))
        p = surf.point(0.3, 0.4)
        self.assertAlmostEqual(p[2], 2.0, places=12)

    def test_unknown_kind(self):
        with self.assertRaises(ValueError):
            uvg.surface_from_fit("bezier", ())

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            uvg.uv_grid(uvg.Plane(origin=(0, 0, 0)), 2, 2, method="colour")


if __name__ == "__main__":
    unittest.main()
