"""Tests for numeric.libfive_forward_ad.

The gradient produced by forward-mode AD must match a central finite-difference
estimate, and must equal the closed-form analytic gradient for known shapes.
"""

from __future__ import annotations

import math
import unittest

from harnesscad.domain.geometry import libfive_frep_ir as ir
from harnesscad.domain.numeric import libfive_forward_ad as ad


def _fd_gradient(node, x, y, z, h=1e-6):
    fx = (ir.eval_point(node, x + h, y, z) - ir.eval_point(node, x - h, y, z)) / (2 * h)
    fy = (ir.eval_point(node, x, y + h, z) - ir.eval_point(node, x, y - h, z)) / (2 * h)
    fz = (ir.eval_point(node, x, y, z + h) - ir.eval_point(node, x, y, z - h)) / (2 * h)
    return (fx, fy, fz)


class TestGradientVsFiniteDifference(unittest.TestCase):
    def _check(self, node, pts):
        for (x, y, z) in pts:
            g = ad.gradient(node, x, y, z)
            fd = _fd_gradient(node, x, y, z)
            for a, b in zip(g, fd):
                self.assertAlmostEqual(a, b, places=5)

    def test_polynomial(self):
        g = ir.Graph()
        expr = g.x() * g.x() * g.y() + g.z() * 3.0 - g.y() / 2.0
        self._check(expr, [(1.0, 2.0, 3.0), (-1.5, 0.7, 2.0), (0.3, -0.4, 1.1)])

    def test_transcendental(self):
        g = ir.Graph()
        expr = g.sin(g.x()) * g.exp(g.y()) + g.log(g.z())
        self._check(expr, [(0.5, 0.3, 1.2), (1.0, -0.5, 2.0)])

    def test_sphere_field(self):
        g = ir.Graph()
        expr = ir.sphere(g, 0.0, 0.0, 0.0, 1.0)
        self._check(expr, [(1.0, 2.0, 3.0), (0.5, -0.5, 0.5)])

    def test_csg_min_max_branches(self):
        g = ir.Graph()
        a = ir.circle(g, -0.5, 0.0, 1.0)
        b = ir.circle(g, 0.5, 0.0, 1.0)
        expr = ir.union(g, a, b)
        # away from the seam the min picks a single branch; FD should agree
        self._check(expr, [(-1.3, 0.2, 0.0), (1.3, 0.2, 0.0)])


class TestAnalyticGradient(unittest.TestCase):
    def test_sphere_normal_is_radial(self):
        g = ir.Graph()
        expr = ir.sphere(g, 0.0, 0.0, 0.0, 1.0)
        # gradient of |p| - r is the unit radial direction
        x, y, z = 2.0, 0.0, 0.0
        self.assertAlmostEqual(ad.normal(expr, x, y, z)[0], 1.0, places=9)
        x, y, z = 0.0, 3.0, 0.0
        n = ad.normal(expr, x, y, z)
        self.assertAlmostEqual(n[0], 0.0, places=9)
        self.assertAlmostEqual(n[1], 1.0, places=9)

    def test_square_derivative(self):
        g = ir.Graph()
        expr = g.square(g.x())  # d/dx x^2 = 2x
        self.assertAlmostEqual(ad.gradient(expr, 3.0, 0.0, 0.0)[0], 6.0, places=9)

    def test_normal_unit_length(self):
        g = ir.Graph()
        expr = ir.circle(g, 0.0, 0.0, 1.0)
        n = ad.normal(expr, 0.7, 0.7, 0.0)
        self.assertAlmostEqual(math.hypot(n[0], n[1]), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
