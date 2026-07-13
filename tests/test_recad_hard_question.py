import unittest

from harnesscad.data.dataengine.reward.recad_hard_question import (
    DEFAULT_TAU_H,
    OBJ_GRPO,
    OBJ_GUIDED,
    identify_hard_questions,
    is_hard,
    max_group_reward,
    objective_value,
    partition_questions,
    select_objective,
)


class TestMaxGroupReward(unittest.TestCase):
    def test_max(self):
        self.assertEqual(max_group_reward([0.1, 0.9, 0.3]), 0.9)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            max_group_reward([])


class TestIsHard(unittest.TestCase):
    def test_hard_when_all_below(self):
        self.assertTrue(is_hard([0.1, 0.5, 0.79], tau_h=0.8))

    def test_easy_when_one_reaches(self):
        self.assertFalse(is_hard([0.1, 0.85, 0.2], tau_h=0.8))

    def test_boundary_equal_is_easy(self):
        # strict < : max exactly at tau_h is NOT hard
        self.assertFalse(is_hard([0.8], tau_h=0.8))

    def test_default_threshold(self):
        self.assertEqual(DEFAULT_TAU_H, 0.8)


class TestSelectObjective(unittest.TestCase):
    def test_hard_routes_guided(self):
        self.assertEqual(select_objective([0.1, 0.2]), OBJ_GUIDED)

    def test_easy_routes_grpo(self):
        self.assertEqual(select_objective([0.1, 0.95]), OBJ_GRPO)


class TestObjectiveValue(unittest.TestCase):
    def test_hard_picks_guided(self):
        self.assertEqual(objective_value([0.1], 5.0, 9.0), 5.0)

    def test_easy_picks_grpo(self):
        self.assertEqual(objective_value([0.9], 5.0, 9.0), 9.0)


class TestPartition(unittest.TestCase):
    def test_partition_preserves_order(self):
        qr = {"a": [0.1], "b": [0.9], "c": [0.2], "d": [0.85]}
        out = partition_questions(qr, tau_h=0.8)
        self.assertEqual(out["hard"], ["a", "c"])
        self.assertEqual(out["easy"], ["b", "d"])

    def test_all_easy(self):
        out = partition_questions({"x": [1.0], "y": [0.9]})
        self.assertEqual(out["hard"], [])
        self.assertEqual(out["easy"], ["x", "y"])


class TestIdentify(unittest.TestCase):
    def test_end_to_end(self):
        # deterministic sampler keyed on question
        table = {1: [0.1, 0.2], 2: [0.3, 0.99], 3: [0.5, 0.7]}
        out = identify_hard_questions([1, 2, 3], lambda q: table[q], tau_h=0.8)
        self.assertEqual(out["hard"], [1, 3])
        self.assertEqual(out["easy"], [2])

    def test_deterministic(self):
        s = lambda q: [0.4, 0.6]
        a = identify_hard_questions([1, 2], s)
        b = identify_hard_questions([1, 2], s)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
