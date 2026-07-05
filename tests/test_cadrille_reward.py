"""Tests for cadrille verifiable reward shaping and hard-example mining."""

import unittest

from dataengine.cadrille_reward import (
    IOU_SCALE,
    INVALID_PENALTY,
    HARD_MINING_THRESHOLD,
    r_iou,
    r_invalid,
    cadrille_reward,
    reward_components,
    mean_reward,
    is_hard_example,
    mine_hard_examples,
    mining_report,
)


class RewardTermsTest(unittest.TestCase):
    def test_iou_term_scaled(self):
        self.assertEqual(r_iou(1.0), IOU_SCALE)
        self.assertEqual(r_iou(0.0), 0.0)
        self.assertAlmostEqual(r_iou(0.5), 5.0)

    def test_iou_out_of_range(self):
        with self.assertRaises(ValueError):
            r_iou(1.5)
        with self.assertRaises(ValueError):
            r_iou(-0.1)

    def test_invalid_term(self):
        self.assertEqual(r_invalid(True), 0.0)
        self.assertEqual(r_invalid(False), INVALID_PENALTY)

    def test_total_reward_valid(self):
        self.assertAlmostEqual(cadrille_reward(0.9, True), 9.0)

    def test_total_reward_invalid_ignores_iou(self):
        self.assertEqual(cadrille_reward(0.9, False), INVALID_PENALTY)

    def test_components(self):
        c = reward_components(0.8, True)
        self.assertAlmostEqual(c["r_iou"], 8.0)
        self.assertEqual(c["r_invalid"], 0.0)
        self.assertAlmostEqual(c["total"], 8.0)
        bad = reward_components(0.8, False)
        self.assertEqual(bad["total"], INVALID_PENALTY)


class MiningTest(unittest.TestCase):
    def test_mean_reward(self):
        self.assertAlmostEqual(mean_reward([1.0, 2.0, 3.0]), 2.0)
        with self.assertRaises(ValueError):
            mean_reward([])

    def test_is_hard_example_threshold(self):
        # mean below 7.5 -> hard
        self.assertTrue(is_hard_example([5.0, 6.0, 7.0]))
        # mean at/above 7.5 -> not hard (easy)
        self.assertFalse(is_hard_example([8.0, 8.0, 8.0]))
        self.assertFalse(is_hard_example([7.5, 7.5, 7.5]))

    def test_mine_hard_examples(self):
        samples = [
            ("easy", [9.0, 9.0, 9.0]),
            ("hard", [5.0, 5.0, 5.0]),
            ("borderline", [7.0, 8.0, 6.0]),  # mean 7.0 -> hard
        ]
        kept = mine_hard_examples(samples)
        self.assertEqual(kept, ["hard", "borderline"])

    def test_mining_report(self):
        samples = [("a", [9.0]), ("b", [1.0]), ("c", [2.0])]
        report = mining_report(samples)
        self.assertEqual(report["total"], 3)
        self.assertEqual(report["kept"], 2)
        self.assertAlmostEqual(report["retained_fraction"], 2 / 3)
        self.assertEqual(report["threshold"], HARD_MINING_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
