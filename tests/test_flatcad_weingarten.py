"""Tests for geometry.flatcad_weingarten (Weingarten map & curvature)."""
import math
import random
import unittest

from geometry.flatcad_weingarten import (
    orthonormal_tangent_frame, rotate_frame, random_tangent_frame,
    shape_operator, off_diagonal_weingarten, s12_from_principal,
    off_diagonal_weingarten_fd,
    gaussian_curvature, mean_curvature, principal_curvatures,
    odw_loss_l1, odw_loss_l2, expected_s12_squared, expected_abs_s12,
    classify_curvature,
)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


# Analytic SDF differential quantities on a sphere |x|-r at point p (|p|=r):
#   g = p/r (unit outward),  H = (I - nn^T)/r
def sphere_grad_hess(p, r):
    n = tuple(c / r for c in p)
    g = n
    H = tuple(tuple((1.0 if i == j else 0.0) - n[i] * n[j] for j in range(3))
              for i in range(3))
    H = tuple(tuple(H[i][j] / r for j in range(3)) for i in range(3))
    return g, H


class TangentFrameTests(unittest.TestCase):
    def test_frame_orthonormal_and_tangent(self):
        n = (0.3, -0.6, 0.74)
        u, v = orthonormal_tangent_frame(n)
        self.assertAlmostEqual(_dot(u, u), 1.0, places=12)
        self.assertAlmostEqual(_dot(v, v), 1.0, places=12)
        self.assertAlmostEqual(_dot(u, v), 0.0, places=12)
        self.assertAlmostEqual(_dot(u, n), 0.0, places=12)
        self.assertAlmostEqual(_dot(v, n), 0.0, places=12)

    def test_rotate_preserves_orthonormality(self):
        u, v = orthonormal_tangent_frame((0, 0, 1))
        ur, vr = rotate_frame(u, v, 0.7)
        self.assertAlmostEqual(_dot(ur, ur), 1.0, places=12)
        self.assertAlmostEqual(_dot(ur, vr), 0.0, places=12)

    def test_random_frame_deterministic(self):
        n = (1.0, 2.0, 3.0)
        a = random_tangent_frame(n, random.Random(42))
        b = random_tangent_frame(n, random.Random(42))
        self.assertEqual(a, b)


class CurvatureTests(unittest.TestCase):
    def test_sphere_mean_and_gaussian(self):
        r = 2.5
        g, H = sphere_grad_hess((r, 0.0, 0.0), r)
        self.assertAlmostEqual(mean_curvature(g, H), 1.0 / r, places=10)
        self.assertAlmostEqual(gaussian_curvature(g, H), 1.0 / (r * r), places=10)

    def test_sphere_principal_curvatures_equal(self):
        r = 4.0
        g, H = sphere_grad_hess((0.0, r, 0.0), r)
        k1, k2 = principal_curvatures(g, H)
        self.assertAlmostEqual(k1, 1.0 / r, places=8)
        self.assertAlmostEqual(k2, 1.0 / r, places=8)

    def test_plane_zero_curvature(self):
        # Plane z = 0, SDF f = z: grad = (0,0,1), Hessian = 0.
        g = (0.0, 0.0, 1.0)
        H = ((0.0, 0.0, 0.0),) * 3
        self.assertAlmostEqual(mean_curvature(g, H), 0.0, places=12)
        self.assertAlmostEqual(gaussian_curvature(g, H), 0.0, places=12)
        self.assertEqual(classify_curvature(*principal_curvatures(g, H)), "planar")

    def test_cylinder_curvature(self):
        # Cylinder radius R about z-axis, SDF f = sqrt(x^2+y^2) - R.
        # At (R,0,0): g=(1,0,0); H = diag over tangent: circumferential 1/R,
        # axial 0. Gaussian = 0 (developable), mean = 1/(2R).
        R = 3.0
        g = (1.0, 0.0, 0.0)
        H = ((0.0, 0.0, 0.0), (0.0, 1.0 / R, 0.0), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(gaussian_curvature(g, H), 0.0, places=12)
        self.assertAlmostEqual(mean_curvature(g, H), 1.0 / (2.0 * R), places=10)
        k1, k2 = principal_curvatures(g, H)
        self.assertEqual(classify_curvature(k1, k2), "parabolic")

    def test_saddle_is_hyperbolic(self):
        # grad along x, tangential principal curvatures +2 and -2.
        g = (1.0, 0.0, 0.0)
        H = ((0.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, -2.0))
        self.assertLess(gaussian_curvature(g, H), 0.0)
        k1, k2 = principal_curvatures(g, H)
        self.assertEqual(classify_curvature(k1, k2), "hyperbolic")


class ShapeOperatorTests(unittest.TestCase):
    def test_sphere_shape_operator_is_umbilic(self):
        r = 2.0
        p = (r, 0.0, 0.0)
        g, H = sphere_grad_hess(p, r)
        u, v = orthonormal_tangent_frame(g)
        S = shape_operator(g, H, u, v)
        self.assertAlmostEqual(S[0][0], 1.0 / r, places=10)
        self.assertAlmostEqual(S[1][1], 1.0 / r, places=10)
        self.assertAlmostEqual(S[0][1], 0.0, places=10)  # umbilic -> no warp

    def test_off_diagonal_matches_analytic_rotation(self):
        # Diagonal Hessian in tangent plane with k1=1, k2=3 (grad along x).
        g = (1.0, 0.0, 0.0)
        k1, k2 = 1.0, 3.0
        H = ((0.0, 0.0, 0.0), (0.0, k1, 0.0), (0.0, 0.0, k2))
        u0, v0 = orthonormal_tangent_frame(g)  # principal-ish frame in y,z
        theta = 0.4
        u, v = rotate_frame(u0, v0, theta)
        s12 = off_diagonal_weingarten(g, H, u, v)
        # magnitude must equal |1/2 (k2-k1) sin 2theta| regardless of frame sign
        self.assertAlmostEqual(abs(s12),
                               abs(s12_from_principal(k1, k2, theta)), places=8)

    def test_umbilic_off_diagonal_vanishes_any_frame(self):
        r = 5.0
        g, H = sphere_grad_hess((0, 0, r), r)
        rng = random.Random(7)
        for _ in range(5):
            u, v = random_tangent_frame(g, rng)
            self.assertAlmostEqual(off_diagonal_weingarten(g, H, u, v), 0.0,
                                   places=10)


class ExpectationTests(unittest.TestCase):
    def test_monte_carlo_matches_closed_form(self):
        k1, k2 = 1.0, 4.0
        rng = random.Random(123)
        n = 40000
        acc_sq = acc_abs = 0.0
        for _ in range(n):
            th = rng.uniform(0.0, 2.0 * math.pi)
            s = s12_from_principal(k1, k2, th)
            acc_sq += s * s
            acc_abs += abs(s)
        self.assertAlmostEqual(acc_sq / n, expected_s12_squared(k1, k2), places=2)
        self.assertAlmostEqual(acc_abs / n, expected_abs_s12(k1, k2), places=2)

    def test_expectation_zero_iff_equal(self):
        self.assertEqual(expected_s12_squared(2.0, 2.0), 0.0)
        self.assertEqual(expected_abs_s12(2.0, 2.0), 0.0)


class ODWLossTests(unittest.TestCase):
    def test_loss_zero_on_sphere_flat(self):
        r = 2.0
        g, H = sphere_grad_hess((r, 0, 0), r)
        rng = random.Random(1)
        samples = []
        for _ in range(10):
            u, v = random_tangent_frame(g, rng)
            samples.append((g, H, u, v))
        self.assertAlmostEqual(odw_loss_l1(samples), 0.0, places=10)
        self.assertAlmostEqual(odw_loss_l2(samples), 0.0, places=12)

    def test_loss_positive_on_saddle(self):
        g = (1.0, 0.0, 0.0)
        H = ((0.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, -2.0))
        u0, v0 = orthonormal_tangent_frame(g)
        u, v = rotate_frame(u0, v0, math.pi / 4)  # max warp frame
        self.assertGreater(odw_loss_l1([(g, H, u, v)]), 0.5)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            odw_loss_l1([])


class FiniteDifferenceRouteTests(unittest.TestCase):
    def test_fd_s12_matches_autodiff_style_on_quadratic(self):
        # SDF with exact constant Hessian; grad varies but |grad|~1 near origin
        # is not guaranteed, so compare fd S12 to analytic u^T H v / |grad|.
        def f(x, y, z):
            return x + 0.5 * (y * y + z * z) + 0.6 * y * z
        x = (0.0, 0.1, 0.05)
        # grad = (1, y+0.6z, z+0.6y); H = [[0,0,0],[0,1,0.6],[0,0.6,1]]
        gx = (1.0, x[1] + 0.6 * x[2], x[2] + 0.6 * x[1])
        H = ((0.0, 0.0, 0.0), (0.0, 1.0, 0.6), (0.0, 0.6, 1.0))
        u, v = orthonormal_tangent_frame(gx)
        analytic = off_diagonal_weingarten(gx, H, u, v)
        fd = off_diagonal_weingarten_fd(f, x, u, v, h=1e-3)
        self.assertAlmostEqual(fd, analytic, places=4)


if __name__ == "__main__":
    unittest.main()
