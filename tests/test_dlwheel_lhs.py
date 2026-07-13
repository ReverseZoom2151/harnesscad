"""Tests for exploration.dlwheel_lhs (paper 112 LHSnorm sampler)."""

import math
import unittest

from harnesscad.agents.exploration import dlwheel_lhs as lhs


class ProbitTests(unittest.TestCase):
    def test_median(self):
        self.assertAlmostEqual(lhs.inverse_normal_cdf(0.5), 0.0, places=9)

    def test_known_quantiles(self):
        # ~1.6449 at 0.95; ~-2.3263 at 0.01
        self.assertAlmostEqual(lhs.inverse_normal_cdf(0.95), 1.6448536, places=5)
        self.assertAlmostEqual(lhs.inverse_normal_cdf(0.01), -2.3263479, places=5)

    def test_symmetry(self):
        self.assertAlmostEqual(
            lhs.inverse_normal_cdf(0.2), -lhs.inverse_normal_cdf(0.8), places=6
        )

    def test_domain(self):
        with self.assertRaises(ValueError):
            lhs.inverse_normal_cdf(0.0)
        with self.assertRaises(ValueError):
            lhs.inverse_normal_cdf(1.0)


class LhsNormalTests(unittest.TestCase):
    def test_shape(self):
        s = lhs.lhs_normal(20, [0.0, 5.0], [1.0, 2.0], seed=1)
        self.assertEqual(len(s), 20)
        self.assertTrue(all(len(r) == 2 for r in s))

    def test_determinism(self):
        a = lhs.lhs_normal(15, [0.0], [1.0], seed=7)
        b = lhs.lhs_normal(15, [0.0], [1.0], seed=7)
        self.assertEqual(a, b)

    def test_seed_changes_output(self):
        a = lhs.lhs_normal(15, [0.0], [1.0], seed=1)
        b = lhs.lhs_normal(15, [0.0], [1.0], seed=2)
        self.assertNotEqual(a, b)

    def test_marginal_recovers_params(self):
        # A large LHS sample should closely match the requested mean/std.
        s = lhs.lhs_normal(2000, [3.0, -1.0], [2.0, 0.5], seed=42)
        stats = lhs.column_stats(s)
        self.assertAlmostEqual(stats[0][0], 3.0, delta=0.1)
        self.assertAlmostEqual(stats[0][1], 2.0, delta=0.1)
        self.assertAlmostEqual(stats[1][0], -1.0, delta=0.05)
        self.assertAlmostEqual(stats[1][1], 0.5, delta=0.05)

    def test_stratification_spread(self):
        # With n samples the probability strata are distinct; values sorted
        # should be strictly increasing (no collisions) for 1D.
        s = lhs.lhs_standard_normal(50, 1, seed=3)
        vals = sorted(r[0] for r in s)
        for a, b in zip(vals, vals[1:]):
            self.assertLess(a, b)

    def test_empty(self):
        self.assertEqual(lhs.lhs_normal(0, [0.0], [1.0], seed=1), [])

    def test_validation(self):
        with self.assertRaises(ValueError):
            lhs.lhs_normal(-1, [0.0], [1.0], seed=1)
        with self.assertRaises(ValueError):
            lhs.lhs_normal(5, [0.0], [1.0, 2.0], seed=1)
        with self.assertRaises(ValueError):
            lhs.lhs_normal(5, [], [], seed=1)
        with self.assertRaises(ValueError):
            lhs.lhs_normal(5, [0.0], [0.0], seed=1)


class ColumnStatsTests(unittest.TestCase):
    def test_stats(self):
        stats = lhs.column_stats([[1.0], [3.0]])
        self.assertAlmostEqual(stats[0][0], 2.0)
        self.assertAlmostEqual(stats[0][1], 1.0)

    def test_empty(self):
        with self.assertRaises(ValueError):
            lhs.column_stats([])


if __name__ == "__main__":
    unittest.main()
