"""Tests for optimisation convergence and diversity metrics."""
import math
import unittest

from harnesscad.eval.bench.harness.optimization_convergence import (
    std,
    generation_stats,
    trajectory_stats,
    running_minimum,
    oscillation_index,
    is_monotonic_non_increasing,
    parameter_variance_trajectory,
    has_converged,
    Z_95,
)


class GenerationStatsTests(unittest.TestCase):
    def test_basic_stats(self):
        gs = generation_stats(0, [1.0, 2.0, 3.0])
        self.assertAlmostEqual(gs.mean, 2.0)
        self.assertEqual(gs.minimum, 1.0)
        self.assertEqual(gs.maximum, 3.0)
        self.assertAlmostEqual(gs.std, math.sqrt(2.0 / 3.0))

    def test_ci_symmetric_around_mean(self):
        gs = generation_stats(1, [1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual((gs.ci95_low + gs.ci95_high) / 2.0, gs.mean)
        half = Z_95 * gs.std / math.sqrt(4)
        self.assertAlmostEqual(gs.ci95_high - gs.mean, half)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            generation_stats(0, [])

    def test_trajectory_indexes_generations(self):
        traj = trajectory_stats([[1.0, 2.0], [0.5, 1.5]])
        self.assertEqual([g.generation for g in traj], [0, 1])


class RunningMinimumTests(unittest.TestCase):
    def test_running_minimum_monotone(self):
        pops = [[0.5, 0.9], [0.6, 0.7], [0.3, 0.8]]
        rm = running_minimum(pops)
        self.assertEqual(rm, [0.5, 0.5, 0.3])
        self.assertTrue(is_monotonic_non_increasing(rm))

    def test_empty_generation_raises(self):
        with self.assertRaises(ValueError):
            running_minimum([[]])


class OscillationTests(unittest.TestCase):
    def test_monotone_curve_scores_one(self):
        self.assertAlmostEqual(oscillation_index([0.0, 1.0, 2.0, 3.0]), 1.0)

    def test_oscillating_curve_scores_high(self):
        smooth = oscillation_index([0.0, 1.0, 2.0])
        noisy = oscillation_index([0.0, 2.0, 0.0, 2.0, 0.0])
        self.assertGreater(noisy, smooth)

    def test_flat_curve_zero(self):
        self.assertEqual(oscillation_index([1.0, 1.0, 1.0]), 0.0)

    def test_single_point_zero(self):
        self.assertEqual(oscillation_index([1.0]), 0.0)


class MonotoneTests(unittest.TestCase):
    def test_non_increasing_true(self):
        self.assertTrue(is_monotonic_non_increasing([3.0, 3.0, 2.0, 1.0]))

    def test_non_increasing_false(self):
        self.assertFalse(is_monotonic_non_increasing([3.0, 4.0, 2.0]))


class ParameterConvergenceTests(unittest.TestCase):
    def test_variance_trajectory_decreases(self):
        pops = [[0.0, 1.0, 2.0], [0.9, 1.0, 1.1], [1.0, 1.0, 1.0]]
        var = parameter_variance_trajectory(pops)
        self.assertTrue(var[0] > var[1] > var[2] or var[0] > var[1] >= var[2])
        self.assertAlmostEqual(var[2], 0.0)

    def test_has_converged_true(self):
        var = [1.0, 0.5, 0.01, 0.005, 0.001]
        self.assertTrue(has_converged(var, tol=0.02, window=3))

    def test_has_converged_false_when_recent_high(self):
        var = [0.001, 0.001, 0.5]
        self.assertFalse(has_converged(var, tol=0.02, window=3))

    def test_has_converged_false_when_too_short(self):
        self.assertFalse(has_converged([0.001], tol=0.02, window=3))

    def test_bad_window(self):
        with self.assertRaises(ValueError):
            has_converged([0.1, 0.2], tol=0.1, window=0)


class StdTests(unittest.TestCase):
    def test_std_of_constant_is_zero(self):
        self.assertEqual(std([5.0, 5.0, 5.0]), 0.0)


if __name__ == "__main__":
    unittest.main()
