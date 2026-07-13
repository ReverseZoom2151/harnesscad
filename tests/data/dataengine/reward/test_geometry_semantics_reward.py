import unittest

from harnesscad.data.dataengine.reward.geometry_semantics_reward import (
    DEFAULT_LAMBDA_1,
    DEFAULT_LAMBDA_2,
    DEFAULT_TAU,
    format_reward,
    geometry_semantics_term,
    phi,
    unified_reward,
)


class TestPhi(unittest.TestCase):
    def test_at_threshold_is_zero(self):
        self.assertEqual(phi(DEFAULT_TAU), 0.0)

    def test_below_threshold_is_zero(self):
        self.assertEqual(phi(0.3, tau=0.55), 0.0)

    def test_one_maps_to_one(self):
        self.assertEqual(phi(1.0, tau=0.55), 1.0)

    def test_linear_midpoint(self):
        # halfway between tau and 1 -> 0.5
        mid = (0.55 + 1.0) / 2
        self.assertAlmostEqual(phi(mid, tau=0.55), 0.5)

    def test_clamps_above_one(self):
        self.assertEqual(phi(1.7, tau=0.55), 1.0)

    def test_invalid_tau_raises(self):
        with self.assertRaises(ValueError):
            phi(0.9, tau=1.0)
        with self.assertRaises(ValueError):
            phi(0.9, tau=-0.1)


class TestFormatReward(unittest.TestCase):
    def test_valid_think_block(self):
        self.assertEqual(format_reward("<think>reason</think> answer"), 1.0)

    def test_leading_whitespace_tolerated(self):
        self.assertEqual(format_reward("\n  <think>x</think>ok"), 1.0)

    def test_missing_block(self):
        self.assertEqual(format_reward("just an answer"), 0.0)

    def test_not_at_start(self):
        self.assertEqual(format_reward("prefix <think>x</think>"), 0.0)

    def test_unclosed_block(self):
        self.assertEqual(format_reward("<think>never closed"), 0.0)

    def test_empty(self):
        self.assertEqual(format_reward(""), 0.0)


class TestGeometrySemanticsTerm(unittest.TestCase):
    def test_min_takes_weaker(self):
        # iou 0.8, phi(0.775, 0.55) = 0.5 -> min = 0.5
        self.assertAlmostEqual(
            geometry_semantics_term(0.8, 0.775, tau=0.55), 0.5)

    def test_geometry_can_be_limiter(self):
        self.assertAlmostEqual(
            geometry_semantics_term(0.2, 1.0, tau=0.55), 0.2)

    def test_low_similarity_zeroes_term(self):
        self.assertEqual(geometry_semantics_term(0.9, 0.5, tau=0.55), 0.0)

    def test_invalid_iou_raises(self):
        with self.assertRaises(ValueError):
            geometry_semantics_term(1.5, 0.9)


class TestUnifiedReward(unittest.TestCase):
    def test_full_composition(self):
        # iou 0.8, phi(0.775)=0.5 -> min 0.5; think present -> 1.0
        r = unified_reward(0.8, 0.775, "<think>t</think>c", tau=0.55,
                           lambda_1=0.1, lambda_2=0.9)
        self.assertAlmostEqual(r, 0.1 * 0.5 + 0.9 * 1.0)

    def test_default_weights(self):
        self.assertAlmostEqual(DEFAULT_LAMBDA_1 + DEFAULT_LAMBDA_2, 1.0)

    def test_no_format_no_format_credit(self):
        r = unified_reward(1.0, 1.0, "no think block")
        self.assertAlmostEqual(r, DEFAULT_LAMBDA_1 * 1.0)

    def test_negative_weight_raises(self):
        with self.assertRaises(ValueError):
            unified_reward(0.5, 0.9, "<think>x</think>", lambda_1=-1.0)

    def test_deterministic(self):
        a = unified_reward(0.6, 0.8, "<think>x</think>y")
        b = unified_reward(0.6, 0.8, "<think>x</think>y")
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
