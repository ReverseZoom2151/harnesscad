import unittest

from harnesscad.data.dataengine.reward import expert_advantage as a


class TestBaselineAndAdvantages(unittest.TestCase):
    def test_baseline(self):
        self.assertAlmostEqual(a.group_baseline([1.0, 2.0, 3.0]), 2.0)

    def test_baseline_empty(self):
        with self.assertRaises(ValueError):
            a.group_baseline([])

    def test_advantages_sum_to_zero(self):
        advs = a.expert_advantages([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(sum(advs), 0.0)
        self.assertEqual(advs, [-1.5, -0.5, 0.5, 1.5])

    def test_advantages_empty(self):
        with self.assertRaises(ValueError):
            a.expert_advantages([])


class TestTruncation(unittest.TestCase):
    def test_truncate_nonneg(self):
        self.assertEqual(a.truncate_nonneg([-1.0, 0.0, 2.0, -0.5]), [0.0, 0.0, 2.0, 0.0])

    def test_expert_advantages_truncated(self):
        # rewards [0,0,0,4] -> mean 1 -> advs [-1,-1,-1,3] -> trunc [0,0,0,3]
        self.assertEqual(a.expert_advantages_truncated([0, 0, 0, 4]),
                         [0.0, 0.0, 0.0, 3.0])


class TestEstimate(unittest.TestCase):
    def test_estimate_dict(self):
        out = a.estimate({"e1": [1, 2, 3], "e2": [0, 0, 6]})
        self.assertAlmostEqual(out["e1"]["baseline"], 2.0)
        self.assertEqual(out["e1"]["truncated"], [0.0, 0.0, 1.0])
        self.assertAlmostEqual(out["e2"]["baseline"], 2.0)
        self.assertEqual(out["e2"]["truncated"], [0.0, 0.0, 4.0])

    def test_estimate_pairs(self):
        out = a.estimate([("x", [1.0, 3.0])])
        self.assertEqual(out["x"]["advantages"], [-1.0, 1.0])

    def test_per_expert_baseline_independent(self):
        # Pooling would give a different baseline; verify per-expert grouping.
        out = a.estimate({"hi": [10, 10], "lo": [0, 0]})
        # each expert's advantages are zero within its own group
        self.assertEqual(out["hi"]["advantages"], [0.0, 0.0])
        self.assertEqual(out["lo"]["advantages"], [0.0, 0.0])


class TestGrpoLoss(unittest.TestCase):
    def test_surrogate_term_truncates(self):
        self.assertEqual(a.grpo_surrogate_term(2.0, -1.0), 0.0)
        self.assertEqual(a.grpo_surrogate_term(2.0, 3.0), 6.0)

    def test_expert_grpo_loss(self):
        # rewards [0,4] -> advs [-2,2] -> trunc [0,2]; log_probs [-1,-0.5]
        # terms = [0, -0.5*2=-1.0]; mean = -0.5; loss = 0.5
        loss = a.expert_grpo_loss([-1.0, -0.5], [0.0, 4.0])
        self.assertAlmostEqual(loss, 0.5)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            a.expert_grpo_loss([1.0], [1.0, 2.0])

    def test_determinism(self):
        r1 = a.estimate({"e": [1, 5, 2]})
        r2 = a.estimate({"e": [1, 5, 2]})
        self.assertEqual(r1, r2)


if __name__ == "__main__":
    unittest.main()
