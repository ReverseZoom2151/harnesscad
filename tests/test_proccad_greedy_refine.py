"""Tests for exploration.proccad_greedy_refine."""

import unittest

from harnesscad.agents.exploration.proccad_greedy_refine import (
    gradient_descent,
    match_target,
    multistart_local_optima,
)


class GradientDescentTest(unittest.TestCase):
    def test_finds_minimum_of_parabola(self):
        f = lambda x: (x[0] - 3.0) ** 2 + (x[1] + 1.0) ** 2
        x, fx, _ = gradient_descent(f, [0.0, 0.0], step=0.2)
        self.assertAlmostEqual(x[0], 3.0, places=3)
        self.assertAlmostEqual(x[1], -1.0, places=3)
        self.assertLess(fx, 1e-5)

    def test_respects_bounds(self):
        f = lambda x: (x[0] - 10.0) ** 2  # min wants x=10
        x, _, _ = gradient_descent(f, [0.0], step=0.3, bounds=[(-1.0, 2.0)])
        self.assertLessEqual(x[0], 2.0 + 1e-9)
        self.assertAlmostEqual(x[0], 2.0, places=3)

    def test_deterministic(self):
        f = lambda x: (x[0] - 1.0) ** 2
        a = gradient_descent(f, [0.0], step=0.1)
        b = gradient_descent(f, [0.0], step=0.1)
        self.assertEqual(a, b)


class MatchTargetTest(unittest.TestCase):
    def test_hits_nonnegotiable_target(self):
        # response g(x) = 2*x ; want g = 7 -> x = 3.5
        g = lambda x: 2.0 * x[0]
        x, achieved, _ = match_target(g, [0.0], target=7.0, step=0.2)
        self.assertAlmostEqual(achieved, 7.0, places=3)
        self.assertAlmostEqual(x[0], 3.5, places=3)


class MultistartTest(unittest.TestCase):
    def test_finds_both_basins(self):
        # double well with minima near x=-2 and x=+2
        f = lambda x: (x[0] ** 2 - 4.0) ** 2
        optima = multistart_local_optima(
            f, bounds=[(-4.0, 4.0)], n_starts=12, seed=1, step=0.05, dedup_tol=1e-2
        )
        xs = sorted(round(o[0][0], 1) for o in optima)
        self.assertIn(-2.0, xs)
        self.assertIn(2.0, xs)

    def test_sorted_best_first(self):
        f = lambda x: (x[0] - 1.0) ** 2
        optima = multistart_local_optima(f, bounds=[(-3.0, 3.0)], n_starts=5, seed=2)
        vals = [o[1] for o in optima]
        self.assertEqual(vals, sorted(vals))

    def test_deterministic_seed(self):
        f = lambda x: (x[0] - 1.0) ** 2 + (x[1] + 2.0) ** 2
        a = multistart_local_optima(f, bounds=[(-3, 3), (-3, 3)], n_starts=6, seed=7)
        b = multistart_local_optima(f, bounds=[(-3, 3), (-3, 3)], n_starts=6, seed=7)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
