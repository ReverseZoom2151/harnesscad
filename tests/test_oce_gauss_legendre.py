"""Tests for numeric.oce_gauss_legendre (OCCT math Gauss-Legendre table)."""

from __future__ import annotations

import math
import unittest

from numeric.oce_gauss_legendre import (
    gauss_points_max,
    integrate,
    integrate_2d,
    legendre_p,
    nodes_and_weights,
)


class TestGaussPointsMax(unittest.TestCase):
    def test_occt_cap(self):
        # OCCT math::GaussPointsMax() returns 61.
        self.assertEqual(gauss_points_max(), 61)


class TestLegendrePolynomial(unittest.TestCase):
    def test_known_values(self):
        # P_2(x) = (3x^2 - 1)/2, P_3(x) = (5x^3 - 3x)/2.
        p2, _ = legendre_p(2, 0.5)
        self.assertAlmostEqual(p2, (3 * 0.25 - 1) / 2, places=14)
        p3, _ = legendre_p(3, 0.3)
        self.assertAlmostEqual(p3, (5 * 0.027 - 0.9) / 2, places=14)

    def test_endpoints(self):
        # P_n(1) == 1, P_n(-1) == (-1)^n.
        for n in range(0, 8):
            pn_pos, _ = legendre_p(n, 1.0)
            pn_neg, _ = legendre_p(n, -1.0)
            self.assertAlmostEqual(pn_pos, 1.0, places=13)
            self.assertAlmostEqual(pn_neg, (-1.0) ** n, places=13)

    def test_derivative_matches_finite_difference(self):
        n, x, h = 5, 0.37, 1e-6
        _, dp = legendre_p(n, x)
        fp, _ = legendre_p(n, x + h)
        fm, _ = legendre_p(n, x - h)
        self.assertAlmostEqual(dp, (fp - fm) / (2 * h), places=6)


class TestNodesAndWeights(unittest.TestCase):
    """Verify generated nodes/weights against OCCT's tabulated math.cxx values."""

    def test_n1(self):
        nodes, weights = nodes_and_weights(1)
        self.assertEqual(nodes, [0.0])
        self.assertEqual(weights, [2.0])

    def test_n2_matches_occt(self):
        nodes, weights = nodes_and_weights(2)
        # OCCT Point[N=2] = 0.577350269189625764..., Weight = 1.0
        self.assertAlmostEqual(nodes[1], 0.577350269189625764, places=14)
        self.assertAlmostEqual(nodes[0], -0.577350269189625764, places=14)
        self.assertAlmostEqual(weights[0], 1.0, places=14)
        self.assertAlmostEqual(weights[1], 1.0, places=14)

    def test_n3_matches_occt(self):
        nodes, weights = nodes_and_weights(3)
        # OCCT: abscissa 0.774596669241483377, 0.0
        #       weights  0.555555555555555556, 0.888888888888888889
        self.assertAlmostEqual(nodes[0], -0.774596669241483377, places=14)
        self.assertAlmostEqual(nodes[1], 0.0, places=14)
        self.assertAlmostEqual(nodes[2], 0.774596669241483377, places=14)
        self.assertAlmostEqual(weights[1], 0.888888888888888889, places=14)
        self.assertAlmostEqual(weights[0], 0.555555555555555556, places=14)
        self.assertAlmostEqual(weights[2], 0.555555555555555556, places=14)

    def test_n4_matches_occt(self):
        nodes, weights = nodes_and_weights(4)
        # OCCT abscissae: 0.861136311594052575, 0.339981043584856265
        #      weights:   0.347854845137453857, 0.652145154862546143
        self.assertAlmostEqual(nodes[3], 0.861136311594052575, places=14)
        self.assertAlmostEqual(nodes[2], 0.339981043584856265, places=14)
        self.assertAlmostEqual(nodes[0], -0.861136311594052575, places=14)
        self.assertAlmostEqual(weights[3], 0.347854845137453857, places=14)
        self.assertAlmostEqual(weights[2], 0.652145154862546143, places=14)

    def test_n5_matches_occt(self):
        nodes, weights = nodes_and_weights(5)
        # OCCT abscissae: 0.906179845938663993, 0.538469310105683091, 0.0
        #      weights:   0.236926885056189088, 0.478628670499366468,
        #                 0.568888888888888889
        self.assertAlmostEqual(nodes[4], 0.906179845938663993, places=14)
        self.assertAlmostEqual(nodes[3], 0.538469310105683091, places=14)
        self.assertAlmostEqual(nodes[2], 0.0, places=14)
        self.assertAlmostEqual(weights[4], 0.236926885056189088, places=14)
        self.assertAlmostEqual(weights[3], 0.478628670499366468, places=14)
        self.assertAlmostEqual(weights[2], 0.568888888888888889, places=14)

    def test_symmetry_and_weight_sum(self):
        # Nodes symmetric, weights sum to 2 (measure of [-1,1]).
        for n in range(1, 40):
            nodes, weights = nodes_and_weights(n)
            self.assertEqual(len(nodes), n)
            self.assertAlmostEqual(sum(weights), 2.0, places=12)
            for i in range(n):
                self.assertAlmostEqual(nodes[i], -nodes[n - 1 - i], places=12)
            # sorted ascending
            self.assertEqual(nodes, sorted(nodes))

    def test_roots_are_legendre_roots(self):
        for n in (7, 12, 20):
            nodes, _ = nodes_and_weights(n)
            for x in nodes:
                p, _ = legendre_p(n, x)
                self.assertAlmostEqual(p, 0.0, places=11)


class TestIntegrate(unittest.TestCase):
    def test_exactness_for_polynomials(self):
        # n-point rule is exact for degree <= 2n-1. Use n=4 -> exact to deg 7.
        # integral of x^7 - 2x^3 + 5 over [0,2].
        f = lambda x: x ** 7 - 2 * x ** 3 + 5
        exact = (2 ** 8) / 8 - 2 * (2 ** 4) / 4 + 5 * 2
        self.assertAlmostEqual(integrate(f, 0.0, 2.0, 4), exact, places=10)

    def test_sine_convergence(self):
        # integral of sin(x) on [0, pi] == 2.
        self.assertAlmostEqual(integrate(math.sin, 0.0, math.pi, 10), 2.0, places=12)

    def test_exp(self):
        # integral of e^x on [-1, 1] == e - 1/e.
        val = integrate(math.exp, -1.0, 1.0, 12)
        self.assertAlmostEqual(val, math.e - 1.0 / math.e, places=12)

    def test_constant_and_linear(self):
        self.assertAlmostEqual(integrate(lambda x: 3.0, 1.0, 4.0, 1), 9.0, places=12)
        self.assertAlmostEqual(integrate(lambda x: x, 0.0, 10.0, 2), 50.0, places=12)

    def test_reversed_bounds_negate(self):
        fwd = integrate(math.cos, 0.0, 1.0, 6)
        rev = integrate(math.cos, 1.0, 0.0, 6)
        self.assertAlmostEqual(fwd, -rev, places=13)


class TestIntegrate2D(unittest.TestCase):
    def test_area(self):
        # area of [0,2]x[0,3] == 6.
        val = integrate_2d(lambda x, y: 1.0, 0.0, 2.0, 0.0, 3.0, 2, 2)
        self.assertAlmostEqual(val, 6.0, places=12)

    def test_separable_polynomial(self):
        # integral over [0,1]^2 of x^2 * y == (1/3)*(1/2) = 1/6.
        val = integrate_2d(lambda x, y: x * x * y, 0.0, 1.0, 0.0, 1.0, 3, 3)
        self.assertAlmostEqual(val, 1.0 / 6.0, places=12)


if __name__ == "__main__":
    unittest.main()
