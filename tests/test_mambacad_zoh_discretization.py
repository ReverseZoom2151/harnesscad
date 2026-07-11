import math
import unittest

from numeric.mambacad_zoh_discretization import (
    zoh_abar, zoh_bbar, zoh_bbar_simplified, discretize,
    discrete_scan, analytic_state,
)


class TestAbar(unittest.TestCase):
    def test_exp_of_delta_a(self):
        a = (-1.0, -2.0, 0.5)
        delta = 0.25
        abar = zoh_abar(a, delta)
        for ai, bi in zip(a, abar):
            self.assertAlmostEqual(bi, math.exp(delta * ai))

    def test_delta_zero_gives_identity(self):
        a = (-1.0, 3.0, -0.7)
        self.assertEqual(zoh_abar(a, 0.0), (1.0, 1.0, 1.0))

    def test_negative_delta_raises(self):
        with self.assertRaises(ValueError):
            zoh_abar((1.0,), -0.1)


class TestBbar(unittest.TestCase):
    def test_exact_formula(self):
        a = (-2.0,)
        b = (3.0,)
        delta = 0.5
        bbar = zoh_bbar(a, b, delta)
        expected = (math.exp(delta * a[0]) - 1.0) / a[0] * b[0]
        self.assertAlmostEqual(bbar[0], expected)

    def test_zero_a_limit_is_delta_b(self):
        # As a -> 0, Bbar -> delta * b.
        a = (0.0,)
        b = (5.0,)
        delta = 0.3
        bbar = zoh_bbar(a, b, delta)
        self.assertAlmostEqual(bbar[0], delta * b[0])

    def test_tiny_a_matches_delta_b(self):
        a = (1e-12,)
        b = (7.0,)
        delta = 0.4
        self.assertAlmostEqual(zoh_bbar(a, b, delta)[0], delta * b[0])

    def test_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            zoh_bbar((1.0, 2.0), (1.0,), 0.1)


class TestSimplified(unittest.TestCase):
    def test_delta_b(self):
        b = (1.0, -2.0, 3.0)
        delta = 0.6
        self.assertEqual(zoh_bbar_simplified(b, delta),
                         tuple(delta * x for x in b))

    def test_exact_approaches_simplified_as_delta_small(self):
        # For small delta, exact Bbar ~= delta*B (they agree to O(delta^2)).
        a = (-1.5,)
        b = (2.0,)
        delta = 1e-4
        exact = zoh_bbar(a, b, delta)[0]
        simp = zoh_bbar_simplified(b, delta)[0]
        self.assertAlmostEqual(exact, simp, places=6)


class TestDiscretize(unittest.TestCase):
    def test_returns_abar_bbar(self):
        a = (-1.0, -3.0)
        b = (2.0, 4.0)
        delta = 0.2
        abar, bbar = discretize(a, b, delta)
        self.assertEqual(abar, zoh_abar(a, delta))
        self.assertEqual(bbar, zoh_bbar(a, b, delta))

    def test_simplified_uses_delta_b_but_exact_abar(self):
        a = (-1.0,)
        b = (2.0,)
        delta = 0.5
        abar, bbar = discretize(a, b, delta, simplified=True)
        self.assertEqual(abar, zoh_abar(a, delta))
        self.assertEqual(bbar, zoh_bbar_simplified(b, delta))

    def test_delta_to_zero_recovers_continuous(self):
        # Abar -> I, Bbar -> 0 as delta -> 0 (continuous system limit).
        a = (-2.0, 1.0)
        b = (3.0, -1.0)
        abar, bbar = discretize(a, b, 1e-9)
        for x in abar:
            self.assertAlmostEqual(x, 1.0)
        for x in bbar:
            self.assertAlmostEqual(x, 0.0)


class TestDiscreteScanExactness(unittest.TestCase):
    def test_matches_analytic_scalar_system(self):
        # Constant input u; ZOH is exact for piecewise-constant input, so the
        # discrete scan reproduces the analytic continuous state at t = n*delta.
        a, b, u = -1.3, 2.0, 0.75
        delta = 0.1
        n = 20
        abar, bbar = discretize((a,), (b,), delta)
        x_seq = tuple((u,) for _ in range(n))
        states, hfin = discrete_scan(x_seq, abar, bbar)
        for k in range(n):
            t = (k + 1) * delta
            self.assertAlmostEqual(states[k][0], analytic_state(a, b, u, t),
                                   places=9)
        self.assertAlmostEqual(hfin[0],
                               analytic_state(a, b, u, n * delta), places=9)

    def test_matches_analytic_with_nonzero_h0(self):
        a, b, u = -0.5, 1.0, 1.0
        delta = 0.05
        n = 10
        h0 = 2.0
        abar, bbar = discretize((a,), (b,), delta)
        x_seq = tuple((u,) for _ in range(n))
        states, _ = discrete_scan(x_seq, abar, bbar, h0=(h0,))
        for k in range(n):
            t = (k + 1) * delta
            self.assertAlmostEqual(states[k][0],
                                   analytic_state(a, b, u, t, h0=h0), places=9)

    def test_zero_a_scan_is_accumulation(self):
        # a = 0 => H' = b u => H(t) = b u t ; discrete scan accumulates delta*b*u.
        a, b, u = 0.0, 2.0, 3.0
        delta = 0.5
        n = 4
        abar, bbar = discretize((a,), (b,), delta)
        x_seq = tuple((u,) for _ in range(n))
        states, _ = discrete_scan(x_seq, abar, bbar)
        for k in range(n):
            self.assertAlmostEqual(states[k][0], b * u * (k + 1) * delta)

    def test_empty_sequence(self):
        states, h = discrete_scan((), (0.9,), (0.1,))
        self.assertEqual(states, ())
        self.assertEqual(h, (0.0,))

    def test_width_mismatch_raises(self):
        with self.assertRaises(ValueError):
            discrete_scan(((1.0, 2.0),), (0.5,), (0.5,))


class TestAnalytic(unittest.TestCase):
    def test_zero_a_linear(self):
        self.assertAlmostEqual(analytic_state(0.0, 2.0, 3.0, 4.0), 24.0)

    def test_decay_to_steady_state(self):
        # Stable system a<0 with constant input converges to -b u / a.
        a, b, u = -2.0, 4.0, 1.0
        val = analytic_state(a, b, u, 50.0)
        self.assertAlmostEqual(val, -b * u / a, places=6)


if __name__ == "__main__":
    unittest.main()
