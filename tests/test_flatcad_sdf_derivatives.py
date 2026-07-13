"""Tests for numeric.flatcad_sdf_derivatives (FlatCAD FD samplers)."""
import math
import unittest

from harnesscad.domain.numeric.flatcad_sdf_derivatives import (
    central_gradient, central_hessian, mixed_stencil_uv,
    forward_mixed_uv, backward_mixed_uv,
)


def sphere_sdf(r):
    return lambda x, y, z: math.sqrt(x * x + y * y + z * z) - r


def quadratic_sdf(a, b, c, d, e, f):
    # f = 1/2 (a x^2 + d y^2 + f z^2) + b xy + c xz + e yz
    # exact constant Hessian [[a,b,c],[b,d,e],[c,e,f]]
    return lambda x, y, z: (0.5 * (a * x * x + d * y * y + f * z * z)
                            + b * x * y + c * x * z + e * y * z)


class GradientTests(unittest.TestCase):
    def test_sphere_gradient_is_unit_outward(self):
        f = sphere_sdf(2.0)
        g = central_gradient(f, (2.0, 0.0, 0.0), h=1e-4)
        self.assertAlmostEqual(g[0], 1.0, places=5)
        self.assertAlmostEqual(g[1], 0.0, places=6)
        self.assertAlmostEqual(g[2], 0.0, places=6)

    def test_rejects_nonpositive_step(self):
        with self.assertRaises(ValueError):
            central_gradient(sphere_sdf(1.0), (1, 0, 0), h=0.0)


class HessianTests(unittest.TestCase):
    def test_recovers_constant_hessian(self):
        # a,b,c,d,e,f -> Hessian [[a,b,c],[b,d,e],[c,e,f]]
        f = quadratic_sdf(1.5, -0.7, 0.4, 2.1, 0.9, -1.3)
        M = central_hessian(f, (0.3, -0.2, 0.5), h=1e-2)
        expect = ((1.5, -0.7, 0.4), (-0.7, 2.1, 0.9), (0.4, 0.9, -1.3))
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(M[i][j], expect[i][j], places=6)

    def test_symmetric(self):
        f = quadratic_sdf(1.0, 0.3, -0.2, 2.0, 0.1, 0.5)
        M = central_hessian(f, (0.1, 0.2, 0.3), h=1e-2)
        for i in range(3):
            for j in range(3):
                self.assertEqual(M[i][j], M[j][i])

    def test_sphere_tangential_hessian(self):
        # On sphere |x|-r at (r,0,0): H = (I - nn^T)/r, so H_yy = H_zz = 1/r.
        r = 3.0
        M = central_hessian(sphere_sdf(r), (r, 0.0, 0.0), h=1e-3)
        self.assertAlmostEqual(M[1][1], 1.0 / r, places=4)
        self.assertAlmostEqual(M[2][2], 1.0 / r, places=4)


class MixedStencilTests(unittest.TestCase):
    def test_symmetric_stencil_matches_uHv(self):
        f = quadratic_sdf(1.0, 0.6, -0.3, 2.0, 0.2, 0.5)
        u = (1.0, 0.0, 0.0)
        v = (0.0, 1.0, 0.0)
        dc = mixed_stencil_uv(f, (0.4, -0.1, 0.2), u, v, h=1e-2)
        self.assertAlmostEqual(dc, 0.6, places=6)  # u^T H v = H_xy = 0.6

    def test_forward_backward_average(self):
        f = quadratic_sdf(1.0, 0.5, 0.0, 1.0, 0.0, 0.0)
        u = (1.0, 0.0, 0.0)
        v = (0.0, 1.0, 0.0)
        x = (0.2, 0.3, 0.1)
        fp = forward_mixed_uv(f, x, u, v, h=1e-2)
        fm = backward_mixed_uv(f, x, u, v, h=1e-2)
        dc = mixed_stencil_uv(f, x, u, v, h=1e-2)
        self.assertAlmostEqual(dc, 0.5 * (fp + fm), places=12)

    def test_second_order_convergence(self):
        # Non-quadratic SDF: symmetric stencil error should shrink ~ h^2.
        f = lambda x, y, z: math.sin(x) * math.cos(y) + 0.5 * z * z
        x = (0.5, 0.3, 0.2)
        u = (1.0, 0.0, 0.0)
        v = (0.0, 1.0, 0.0)
        exact = -math.cos(x[0]) * math.sin(x[1])  # d2/dxdy
        e1 = abs(mixed_stencil_uv(f, x, u, v, h=1e-1) - exact)
        e2 = abs(mixed_stencil_uv(f, x, u, v, h=5e-2) - exact)
        # halving h should cut error by ~4x for O(h^2); allow slack
        self.assertLess(e2, e1 / 3.0)


if __name__ == "__main__":
    unittest.main()
