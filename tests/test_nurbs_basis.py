"""Tests for numeric.nurbs_basis (Cox-de Boor basis, knot machinery)."""

import unittest

from numeric import nurbs_basis as nb


class TestKnotHelpers(unittest.TestCase):
    def test_uniform_clamped_length_and_clamp(self):
        # n = 4 (5 control points), p = 2 -> length n + p + 2 = 8.
        U = nb.uniform_clamped_knots(4, 2)
        self.assertEqual(len(U), 8)
        self.assertEqual(U[:3], [0.0, 0.0, 0.0])
        self.assertEqual(U[-3:], [1.0, 1.0, 1.0])
        nb.validate_knot_vector(U, 4, 2)  # must not raise

    def test_multiplicity_roundtrip(self):
        U = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        pairs = nb.knot_multiplicities(U)
        self.assertEqual(pairs, [(0.0, 3), (0.5, 1), (1.0, 3)])
        vals = [v for v, _ in pairs]
        mults = [m for _, m in pairs]
        self.assertEqual(nb.expand_multiplicities(vals, mults), U)

    def test_validate_rejects_bad_length_and_order(self):
        with self.assertRaises(ValueError):
            nb.validate_knot_vector([0.0, 1.0], 4, 2)
        with self.assertRaises(ValueError):
            nb.validate_knot_vector([0, 0, 0, 1, 0.5, 1, 1, 1], 4, 2)


class TestPartitionOfUnity(unittest.TestCase):
    def test_basis_sums_to_one(self):
        # For any valid u in the domain, sum_i N_{i,p}(u) == 1.
        n, p = 5, 3
        U = nb.uniform_clamped_knots(n, p)
        for k in range(21):
            u = k / 20.0
            vec = nb.all_basis(n, p, u, U)
            self.assertAlmostEqual(sum(vec), 1.0, places=12)
            # All basis functions are non-negative.
            self.assertTrue(all(x >= -1e-12 for x in vec))

    def test_nonzero_support_width(self):
        # At most p + 1 basis functions are non-zero anywhere.
        n, p = 6, 2
        U = nb.uniform_clamped_knots(n, p)
        vec = nb.all_basis(n, p, 0.37, U)
        nonzero = [x for x in vec if abs(x) > 1e-12]
        self.assertLessEqual(len(nonzero), p + 1)


class TestCoxDeBoorVsFast(unittest.TestCase):
    def test_recursion_matches_a22(self):
        # The literal Cox-de Boor recursion must agree with the A2.2 form.
        n, p = 5, 3
        U = nb.uniform_clamped_knots(n, p)
        for k in range(1, 20):
            u = k / 20.0
            span = nb.find_span(n, p, u, U)
            fast = nb.basis_functions(span, u, p, U)
            for local in range(p + 1):
                i = span - p + local
                slow = nb.cox_de_boor(i, p, u, U)
                self.assertAlmostEqual(slow, fast[local], places=12)

    def test_degree0_indicator(self):
        U = [0.0, 1.0, 2.0, 3.0]
        self.assertEqual(nb.cox_de_boor(1, 0, 1.5, U), 1.0)
        self.assertEqual(nb.cox_de_boor(0, 0, 1.5, U), 0.0)


class TestFindSpan(unittest.TestCase):
    def test_endpoints_and_interior(self):
        n, p = 4, 2
        U = nb.uniform_clamped_knots(n, p)
        self.assertEqual(nb.find_span(n, p, 0.0, U), p)
        self.assertEqual(nb.find_span(n, p, 1.0, U), n)
        span = nb.find_span(n, p, 0.5, U)
        self.assertTrue(U[span] <= 0.5 < U[span + 1] or abs(U[span] - 0.5) < 1e-9)


class TestBasisDerivatives(unittest.TestCase):
    def test_zeroth_derivative_matches_basis(self):
        n, p = 5, 3
        U = nb.uniform_clamped_knots(n, p)
        u = 0.42
        span = nb.find_span(n, p, u, U)
        ders = nb.basis_derivatives(span, u, p, U, 2)
        base = nb.basis_functions(span, u, p, U)
        for j in range(p + 1):
            self.assertAlmostEqual(ders[0][j], base[j], places=12)

    def test_first_derivatives_sum_to_zero(self):
        # d/du (sum_i N_i) = d/du (1) = 0.
        n, p = 5, 3
        U = nb.uniform_clamped_knots(n, p)
        u = 0.31
        span = nb.find_span(n, p, u, U)
        ders = nb.basis_derivatives(span, u, p, U, 1)
        self.assertAlmostEqual(sum(ders[1]), 0.0, places=10)

    def test_derivative_matches_finite_difference(self):
        n, p = 5, 3
        U = nb.uniform_clamped_knots(n, p)
        u = 0.4
        h = 1e-6
        span = nb.find_span(n, p, u, U)
        ders = nb.basis_derivatives(span, u, p, U, 1)
        # Compare local basis j against central finite difference.
        for local in range(p + 1):
            i = span - p + local
            fd = (nb.cox_de_boor(i, p, u + h, U)
                  - nb.cox_de_boor(i, p, u - h, U)) / (2 * h)
            self.assertAlmostEqual(ders[1][local], fd, places=5)


if __name__ == "__main__":
    unittest.main()
