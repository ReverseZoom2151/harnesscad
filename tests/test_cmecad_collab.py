import math
import unittest

from harnesscad.data.dataengine import cmecad_collab as c


class TestAverageReward(unittest.TestCase):
    def test_absolute(self):
        self.assertAlmostEqual(c.average_absolute_reward([-1.0, 3.0]), 2.0)

    def test_empty(self):
        with self.assertRaises(ValueError):
            c.average_absolute_reward([])


class TestBestWorst(unittest.TestCase):
    def setUp(self):
        self.rewards = {"e1": [0.4, 0.5], "e2": [0.9, 0.7], "e3": [0.1, 0.2]}

    def test_best(self):
        self.assertEqual(c.best_expert(self.rewards), "e2")

    def test_worst(self):
        self.assertEqual(c.worst_expert(self.rewards), "e3")

    def test_route(self):
        self.assertEqual(c.route_best_expert(self.rewards), "e2")

    def test_tie_breaks_lowest_id(self):
        rewards = {"b": [1.0], "a": [1.0]}
        self.assertEqual(c.best_expert(rewards), "a")
        self.assertEqual(c.worst_expert(rewards), "a")

    def test_numeric_ids(self):
        rewards = {1: [0.1], 2: [0.9], 3: [0.5]}
        self.assertEqual(c.best_expert(rewards), 2)
        self.assertEqual(c.worst_expert(rewards), 1)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            c.best_expert({})


class TestCreditAssignment(unittest.TestCase):
    def test_credit(self):
        info = c.credit_assignment({"e1": [0.4], "e2": [0.9], "e3": [0.1]})
        self.assertEqual(info["best_expert"], "e2")
        self.assertEqual(info["worst_expert"], "e3")
        self.assertEqual(info["kl_teacher"], "e2")
        self.assertEqual(info["kl_student"], "e3")
        self.assertEqual(info["ranking"], ["e2", "e1", "e3"])

    def test_pairs_input(self):
        info = c.credit_assignment([("a", [0.2]), ("b", [0.8])])
        self.assertEqual(info["best_expert"], "b")


class TestKL(unittest.TestCase):
    def test_identical_zero(self):
        self.assertAlmostEqual(c.kl_divergence([0.5, 0.5], [0.5, 0.5]), 0.0)

    def test_known_value(self):
        # KL([1,0]||[0.5,0.5]) = 1*log(1/0.5) = log 2
        self.assertAlmostEqual(c.kl_divergence([1.0, 0.0], [0.5, 0.5]), math.log(2))

    def test_zero_p_skipped(self):
        # p=0 term ignored even though q=0 there
        self.assertAlmostEqual(c.kl_divergence([1.0, 0.0], [0.5, 0.0]), math.log(2))

    def test_q_zero_where_p_positive_raises(self):
        with self.assertRaises(ValueError):
            c.kl_divergence([0.5, 0.5], [1.0, 0.0])

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            c.kl_divergence([1.0], [0.5, 0.5])

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            c.kl_divergence([-0.1, 1.1], [0.5, 0.5])

    def test_penalty_direction(self):
        student = [0.7, 0.3]
        teacher = [0.5, 0.5]
        self.assertAlmostEqual(c.collaborative_kl_penalty(student, teacher),
                               c.kl_divergence(student, teacher))


if __name__ == "__main__":
    unittest.main()
