"""Tests for data.dataengine.reward.group_relative_advantage."""

import unittest

from harnesscad.data.dataengine.reward.group_relative_advantage import (
    cad_reward,
    geometric_reward,
    group_relative_advantage,
)


class GeometricRewardTest(unittest.TestCase):
    def test_zero_cd_is_one(self):
        self.assertAlmostEqual(geometric_reward(0.0), 1.0)

    def test_monotone_decreasing(self):
        self.assertGreater(geometric_reward(0.1), geometric_reward(5.0))

    def test_negative_cd(self):
        with self.assertRaises(ValueError):
            geometric_reward(-1.0)


class CadRewardTest(unittest.TestCase):
    def test_non_executing_is_zero(self):
        self.assertEqual(cad_reward(0.0, executes=False), 0.0)

    def test_perfect_executing(self):
        r = cad_reward(0.0, executes=True, w_geometric=0.8, w_format=0.2)
        self.assertAlmostEqual(r, 1.0)

    def test_format_component(self):
        # CD huge -> geo ~ 0, so reward ~ w_format.
        r = cad_reward(1e6, executes=True, w_geometric=0.8, w_format=0.2)
        self.assertAlmostEqual(r, 0.2, places=4)


class AdvantageTest(unittest.TestCase):
    def test_zero_mean(self):
        adv = group_relative_advantage([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(adv), 0.0, places=6)

    def test_ordering_preserved(self):
        adv = group_relative_advantage([1.0, 3.0])
        self.assertLess(adv[0], adv[1])

    def test_degenerate_group(self):
        self.assertEqual(group_relative_advantage([5.0, 5.0, 5.0]), [0.0, 0.0, 0.0])

    def test_singleton(self):
        self.assertEqual(group_relative_advantage([1.0]), [0.0])


if __name__ == "__main__":
    unittest.main()
