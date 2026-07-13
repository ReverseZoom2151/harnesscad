"""Tests for Dr. CPPO advantage estimation and clipped PPO objective."""

import unittest

from harnesscad.data.dataengine.cadrille_drcppo import (
    advantages,
    select_strongest,
    clip,
    ppo_clip_objective,
    drcppo_step,
    DEFAULT_EPSILON,
)


class AdvantageTest(unittest.TestCase):
    def test_advantages_std_free(self):
        # A_g = r_g - mean(r); mean of [1,2,3] is 2
        self.assertEqual(advantages([1.0, 2.0, 3.0]), [-1.0, 0.0, 1.0])

    def test_advantages_sum_to_zero(self):
        advs = advantages([4.0, 1.0, 7.0, 2.0])
        self.assertAlmostEqual(sum(advs), 0.0)

    def test_advantages_empty(self):
        with self.assertRaises(ValueError):
            advantages([])


class SelectionTest(unittest.TestCase):
    def test_select_top_abs(self):
        advs = [-1.0, 0.0, 1.0, -3.0]
        # largest |A|: index 3 (3.0), then indices 0 and 2 (1.0) -> tie by index
        self.assertEqual(select_strongest(advs, 2), [0, 3])

    def test_select_returns_sorted_indices(self):
        advs = [5.0, -4.0, 3.0, -2.0]
        sel = select_strongest(advs, 3)
        self.assertEqual(sel, sorted(sel))

    def test_select_caps_at_len(self):
        self.assertEqual(select_strongest([1.0, -2.0], 10), [0, 1])

    def test_select_invalid_n(self):
        with self.assertRaises(ValueError):
            select_strongest([1.0], 0)


class ClipObjectiveTest(unittest.TestCase):
    def test_clip(self):
        self.assertEqual(clip(1.5, 0.9, 1.1), 1.1)
        self.assertEqual(clip(0.5, 0.9, 1.1), 0.9)
        self.assertEqual(clip(1.0, 0.9, 1.1), 1.0)

    def test_positive_advantage_clipped(self):
        # ratio 1.5, adv 2, eps 0.1 -> min(3.0, 1.1*2=2.2) = 2.2
        self.assertAlmostEqual(ppo_clip_objective(1.5, 2.0, 0.1), 2.2)

    def test_negative_advantage(self):
        # ratio 0.5, adv -2, eps 0.1 -> min(-1.0, 0.9*-2=-1.8) = -1.8
        self.assertAlmostEqual(ppo_clip_objective(0.5, -2.0, 0.1), -1.8)

    def test_unclipped_in_band(self):
        self.assertAlmostEqual(ppo_clip_objective(1.05, 2.0, 0.1), 2.1)


class DrCppoStepTest(unittest.TestCase):
    def test_step_selects_and_averages(self):
        rewards = [1.0, 2.0, 3.0, 10.0]
        ratios = [1.0, 1.0, 1.0, 1.0]
        out = drcppo_step(rewards, ratios, n=2, epsilon=DEFAULT_EPSILON)
        # mean reward 4.0 -> advs [-3,-2,-1,6]; |A| top 2 -> indices 3(6),0(3)
        self.assertEqual(out["selected"], [0, 3])
        # ratio 1.0 -> objective == advantage
        self.assertAlmostEqual(out["surrogate"], (-3.0 + 6.0) / 2.0)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            drcppo_step([1.0, 2.0], [1.0], n=1)


if __name__ == "__main__":
    unittest.main()
