"""Tests for the self-adaptive (mu,lambda)/(mu+lambda) evolution strategy."""
import unittest

from harnesscad.agents.exploration.llmdesopt_es_optimizer import optimise, ESResult, Individual


def sphere(x):
    return sum(v * v for v in x)


class ConvergenceTests(unittest.TestCase):
    def test_minimises_sphere(self):
        res = optimise(sphere, [5.0, -5.0, 5.0], seed=1, max_generations=200)
        self.assertIsInstance(res, ESResult)
        # Should get close to the origin.
        self.assertLess(res.best.fitness, 1e-2)
        self.assertLess(sphere([3.0, 3.0, 3.0]), 100.0)  # sanity

    def test_history_lengths_match_generations(self):
        res = optimise(sphere, [2.0, 2.0], seed=7, max_generations=30)
        self.assertEqual(len(res.history_best), res.generations)
        self.assertEqual(len(res.history_mean), res.generations)
        self.assertEqual(len(res.history_sigma), res.generations)

    def test_best_is_non_increasing(self):
        res = optimise(sphere, [4.0, 4.0], seed=3, max_generations=50)
        for a, b in zip(res.history_best, res.history_best[1:]):
            self.assertLessEqual(b, a + 1e-12)


class DeterminismTests(unittest.TestCase):
    def test_same_seed_same_result(self):
        r1 = optimise(sphere, [1.0, 2.0, 3.0], seed=42, max_generations=25)
        r2 = optimise(sphere, [1.0, 2.0, 3.0], seed=42, max_generations=25)
        self.assertEqual(r1.best.genome, r2.best.genome)
        self.assertEqual(r1.history_best, r2.history_best)

    def test_different_seed_differs(self):
        r1 = optimise(sphere, [1.0, 2.0, 3.0], seed=1, max_generations=25)
        r2 = optimise(sphere, [1.0, 2.0, 3.0], seed=2, max_generations=25)
        self.assertNotEqual(r1.best.genome, r2.best.genome)


class SelectionSchemeTests(unittest.TestCase):
    def test_plus_selection_never_loses_best(self):
        # In (mu+lambda) the recorded generation-best equals the running best.
        res = optimise(sphere, [6.0, 6.0], seed=5, max_generations=40,
                       plus_selection=True)
        # history_best strictly non-increasing already checked; here ensure
        # elitist run reaches at least as good a value as the comma run.
        comma = optimise(sphere, [6.0, 6.0], seed=5, max_generations=40,
                         plus_selection=False)
        self.assertLessEqual(res.best.fitness, comma.best.fitness + 1e-6)


class TokenisationTests(unittest.TestCase):
    def test_integer_bounds_clamp_and_round(self):
        # Objective wants tokens near 100; genome must stay integer in [0,32768).
        def obj(x):
            return sum((v - 100.0) ** 2 for v in x)

        res = optimise(obj, [0.0, 0.0], seed=11, max_generations=100,
                       bounds=(0.0, 32767.0), integer=True, sigma0=50.0)
        for v in res.best.genome:
            self.assertEqual(v, float(round(v)))
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 32767.0)

    def test_bounds_respected_at_extremes(self):
        def obj(x):
            return -sum(x)  # maximise -> pushes toward upper bound
        res = optimise(obj, [0.0], seed=2, max_generations=60,
                       bounds=(0.0, 10.0), integer=True, sigma0=5.0)
        self.assertLessEqual(res.best.genome[0], 10.0)
        self.assertGreaterEqual(res.best.genome[0], 0.0)


class ValidationTests(unittest.TestCase):
    def test_bad_mu_lambda(self):
        with self.assertRaises(ValueError):
            optimise(sphere, [1.0], seed=1, mu=5, lam=3)

    def test_empty_x0(self):
        with self.assertRaises(ValueError):
            optimise(sphere, [], seed=1)


if __name__ == "__main__":
    unittest.main()
