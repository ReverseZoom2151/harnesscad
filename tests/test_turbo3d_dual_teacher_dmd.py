"""Tests for numeric.turbo3d_dual_teacher_dmd."""

import unittest

from harnesscad.domain.numeric.turbo3d_dual_teacher_dmd import (
    compounding_collapse_indicator,
    dmd_gradient,
    dual_teacher_gradient,
    few_step_timesteps,
    rms,
    step_reduction_factor,
)


class FewStepTimestepsTest(unittest.TestCase):
    def test_four_step_over_thousand(self):
        steps = few_step_timesteps(1000, 4)
        self.assertEqual(steps[0], 1000)
        self.assertEqual(steps[-1], 1)
        self.assertEqual(len(steps), 4)
        # strictly descending
        self.assertTrue(all(steps[i] > steps[i + 1] for i in range(len(steps) - 1)))

    def test_single_step_is_terminal(self):
        self.assertEqual(few_step_timesteps(1000, 1), [1000])

    def test_all_steps(self):
        self.assertEqual(few_step_timesteps(4, 4), [4, 3, 2, 1])

    def test_evenly_spaced(self):
        # 1000, 4 steps -> 1000, 667, 334, 1
        self.assertEqual(few_step_timesteps(1000, 4), [1000, 667, 334, 1])

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            few_step_timesteps(0, 1)
        with self.assertRaises(ValueError):
            few_step_timesteps(10, 20)


class StepReductionFactorTest(unittest.TestCase):
    def test_ratio(self):
        self.assertAlmostEqual(step_reduction_factor(200, 4), 50.0)

    def test_bad(self):
        with self.assertRaises(ValueError):
            step_reduction_factor(0, 4)


class DmdGradientTest(unittest.TestCase):
    def test_difference(self):
        g = dmd_gradient([1.0, 2.0, 3.0], [0.5, 1.0, 4.0])
        self.assertEqual(g, [0.5, 1.0, -1.0])

    def test_weight(self):
        g = dmd_gradient([2.0], [0.0], weight=0.25)
        self.assertAlmostEqual(g[0], 0.5)

    def test_zero_when_scores_match(self):
        g = dmd_gradient([1.0, 1.0], [1.0, 1.0])
        self.assertEqual(g, [0.0, 0.0])

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            dmd_gradient([1.0], [1.0, 2.0])


class DualTeacherGradientTest(unittest.TestCase):
    def test_lambda_one_mean_of_views(self):
        mv = [1.0, 1.0]
        sv = [[2.0, 0.0], [0.0, 2.0]]
        # mv + 1 * mean(sv) = [1,1] + [1,1] = [2,2]
        self.assertEqual(dual_teacher_gradient(mv, sv, lam=1.0), [2.0, 2.0])

    def test_lambda_zero_is_mv_only(self):
        mv = [1.0, 2.0]
        sv = [[9.0, 9.0]]
        self.assertEqual(dual_teacher_gradient(mv, sv, lam=0.0), [1.0, 2.0])

    def test_lambda_scaling(self):
        mv = [0.0]
        sv = [[4.0], [4.0]]
        # 0 + 0.5 * mean(4,4)=4 = 2
        self.assertAlmostEqual(dual_teacher_gradient(mv, sv, lam=0.5)[0], 2.0)

    def test_four_views(self):
        mv = [0.0]
        sv = [[1.0], [2.0], [3.0], [4.0]]  # K=4 like the paper
        # mean = 2.5
        self.assertAlmostEqual(dual_teacher_gradient(mv, sv, lam=1.0)[0], 2.5)

    def test_errors(self):
        with self.assertRaises(ValueError):
            dual_teacher_gradient([1.0], [], lam=1.0)
        with self.assertRaises(ValueError):
            dual_teacher_gradient([1.0], [[1.0, 2.0]], lam=1.0)
        with self.assertRaises(ValueError):
            dual_teacher_gradient([1.0], [[1.0]], lam=-1.0)


class CompoundingCollapseTest(unittest.TestCase):
    def test_balanced_is_one(self):
        self.assertAlmostEqual(compounding_collapse_indicator([2.0, 2.0, 2.0]), 1.0)

    def test_collapsed_near_zero(self):
        val = compounding_collapse_indicator([10.0, 0.01])
        self.assertLess(val, 0.01)

    def test_all_zero(self):
        self.assertEqual(compounding_collapse_indicator([0.0, 0.0]), 0.0)

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            compounding_collapse_indicator([-1.0])


class RmsTest(unittest.TestCase):
    def test_value(self):
        # rms(3,4) = sqrt((9+16)/2) = sqrt(12.5)
        self.assertAlmostEqual(rms([3.0, 4.0]), 12.5 ** 0.5)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            rms([])


if __name__ == "__main__":
    unittest.main()
