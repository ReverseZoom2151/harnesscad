import unittest

from dataengine.intent2exec_trs import (
    DEFAULT_EPS_HIGH,
    DEFAULT_EPS_LOW,
    clip_ratio,
    is_clipped,
    trs_objective,
    trs_token_objective,
)


class TestClip(unittest.TestCase):
    def test_inside(self):
        self.assertEqual(clip_ratio(1.0), 1.0)

    def test_below(self):
        self.assertEqual(clip_ratio(0.1), DEFAULT_EPS_LOW)

    def test_above(self):
        self.assertEqual(clip_ratio(5.0), DEFAULT_EPS_HIGH)

    def test_bad_bounds(self):
        with self.assertRaises(ValueError):
            clip_ratio(1.0, eps_low=1.2, eps_high=1.5)


class TestTokenObjective(unittest.TestCase):
    def test_positive_advantage_uses_clip(self):
        # ratio above eps_high, A>0 -> min picks clipped (smaller) value
        val = trs_token_objective(5.0, 2.0)
        self.assertAlmostEqual(val, DEFAULT_EPS_HIGH * 2.0)

    def test_negative_advantage_uses_unclipped(self):
        # ratio above eps_high, A<0 -> unclipped r*A is more negative -> min
        val = trs_token_objective(5.0, -2.0)
        self.assertAlmostEqual(val, 5.0 * -2.0)

    def test_wider_than_ppo(self):
        # A ratio of 1.5 is clipped by symmetric PPO(0.8,1.2) but not by TRS.
        self.assertFalse(is_clipped(1.5))
        self.assertAlmostEqual(trs_token_objective(1.5, 1.0), 1.5)


class TestBatch(unittest.TestCase):
    def test_mean(self):
        val = trs_objective([1.0, 1.0], [2.0, 4.0])
        self.assertAlmostEqual(val, 3.0)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            trs_objective([1.0], [1.0, 2.0])

    def test_empty(self):
        with self.assertRaises(ValueError):
            trs_objective([], [])


class TestIsClipped(unittest.TestCase):
    def test_flags(self):
        self.assertTrue(is_clipped(0.3))
        self.assertTrue(is_clipped(2.0))
        self.assertFalse(is_clipped(1.0))


if __name__ == "__main__":
    unittest.main()
