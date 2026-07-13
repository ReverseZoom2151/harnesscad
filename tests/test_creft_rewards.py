"""Tests for dataengine.creft_rewards."""

import unittest

from harnesscad.data.dataengine.creft_rewards import (
    EASY,
    HARD,
    MEDIUM,
    all_parameters_correct,
    attribute_reward,
    classify_difficulty,
    curriculum_reward,
    difficulty_map,
    reward_p1,
    reward_p2,
    reward_p3,
    reward_p3_from_accuracies,
)


class Task1Test(unittest.TestCase):
    def test_all_correct(self):
        gt = {"a": 1, "b": 2}
        self.assertTrue(all_parameters_correct({"a": 1, "b": 2}, gt))
        self.assertEqual(reward_p1({"a": 1, "b": 2}, gt), 1.0)

    def test_one_wrong(self):
        gt = {"a": 1, "b": 2}
        self.assertFalse(all_parameters_correct({"a": 1, "b": 9}, gt))
        self.assertEqual(reward_p1({"a": 1, "b": 9}, gt), 0.0)

    def test_missing_key(self):
        self.assertEqual(reward_p1({"a": 1}, {"a": 1, "b": 2}), 0.0)


class Task2Test(unittest.TestCase):
    def test_exact_correct_set(self):
        self.assertEqual(reward_p2({"c0", "c1"}, {"c0", "c1"}, {"c2"}), 1.0)

    def test_partial_subset(self):
        self.assertEqual(reward_p2({"c0"}, {"c0", "c1"}, {"c2"}), 0.2)

    def test_mixed_selection(self):
        self.assertEqual(reward_p2({"c0", "c2"}, {"c0", "c1"}, {"c2"}), 0.0)

    def test_all_incorrect(self):
        self.assertEqual(reward_p2({"c2"}, {"c0", "c1"}, {"c2"}), 0.0)

    def test_empty_selection(self):
        self.assertEqual(reward_p2(set(), {"c0"}, {"c1"}), 0.0)


class DifficultyTest(unittest.TestCase):
    def test_classification_boundaries(self):
        self.assertEqual(classify_difficulty(0.9), EASY)
        self.assertEqual(classify_difficulty(0.81), EASY)
        self.assertEqual(classify_difficulty(0.8), MEDIUM)  # not > 0.8
        self.assertEqual(classify_difficulty(0.5), MEDIUM)
        self.assertEqual(classify_difficulty(0.2), MEDIUM)  # not < 0.2
        self.assertEqual(classify_difficulty(0.1), HARD)

    def test_out_of_range(self):
        with self.assertRaises(ValueError):
            classify_difficulty(1.5)

    def test_difficulty_map(self):
        m = difficulty_map({"a": 0.9, "b": 0.5, "c": 0.05})
        self.assertEqual(m, {"a": EASY, "b": MEDIUM, "c": HARD})

    def test_attribute_reward(self):
        self.assertEqual(attribute_reward(EASY), 1.0)
        self.assertEqual(attribute_reward(MEDIUM), 1.5)
        self.assertEqual(attribute_reward(HARD), 2.0)
        with self.assertRaises(ValueError):
            attribute_reward("trivial")


class Task3Test(unittest.TestCase):
    def test_weighted_sum(self):
        gt = {"a": 1, "b": 2, "c": 3}
        pred = {"a": 1, "b": 2, "c": 99}  # a,b right; c wrong
        diffs = {"a": EASY, "b": HARD, "c": MEDIUM}
        # 1 (easy a) + 2 (hard b) + 0 (c wrong) = 3
        self.assertEqual(reward_p3(pred, gt, diffs), 3.0)

    def test_default_easy(self):
        gt = {"a": 1}
        self.assertEqual(reward_p3({"a": 1}, gt, {}), 1.0)

    def test_from_accuracies(self):
        gt = {"a": 1, "b": 2}
        pred = {"a": 1, "b": 2}
        accs = {"a": 0.9, "b": 0.1}  # easy + hard = 1 + 2 = 3
        self.assertEqual(reward_p3_from_accuracies(pred, gt, accs), 3.0)


class CurriculumTest(unittest.TestCase):
    def test_all_three(self):
        gt = {"a": 1, "b": 2}
        out = curriculum_reward(
            {"a": 1, "b": 2}, gt, {"a": EASY, "b": MEDIUM},
            choice_selected={"c0"}, choice_correct={"c0"}, choice_incorrect={"c1"},
        )
        self.assertEqual(out["p1"], 1.0)
        self.assertEqual(out["p2"], 1.0)
        self.assertEqual(out["p3"], 2.5)


if __name__ == "__main__":
    unittest.main()
