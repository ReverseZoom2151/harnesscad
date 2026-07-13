import unittest

from harnesscad.data.dataengine.selftrain.pllm_confidence_score import (
    DEFAULT_WEIGHTS, agreement_margin, conciseness, confidence_score,
    fidelity_from_chamfer, rank_candidates,
)


class TestFidelity(unittest.TestCase):
    def test_zero_distance_is_one(self):
        self.assertAlmostEqual(fidelity_from_chamfer(0.0), 1.0)

    def test_half_at_scale(self):
        self.assertAlmostEqual(fidelity_from_chamfer(2.0, scale=2.0), 0.5)

    def test_monotone_decreasing(self):
        self.assertGreater(fidelity_from_chamfer(0.1), fidelity_from_chamfer(0.5))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            fidelity_from_chamfer(-1.0)
        with self.assertRaises(ValueError):
            fidelity_from_chamfer(0.1, scale=0)


class TestAgreement(unittest.TestCase):
    def test_no_runner_up_is_confident(self):
        self.assertEqual(agreement_margin(0.1, None), 1.0)

    def test_equal_is_zero(self):
        self.assertEqual(agreement_margin(0.1, 0.1), 0.0)

    def test_larger_gap_more_confident(self):
        self.assertGreater(agreement_margin(0.1, 0.9), agreement_margin(0.1, 0.2))

    def test_worse_runner_up_positive(self):
        self.assertGreater(agreement_margin(0.1, 0.5), 0.0)


class TestConciseness(unittest.TestCase):
    def test_short_full_score(self):
        self.assertEqual(conciseness(50, 100), 1.0)

    def test_long_penalised(self):
        self.assertAlmostEqual(conciseness(200, 100), 0.5)

    def test_zero_length(self):
        self.assertEqual(conciseness(0, 100), 1.0)

    def test_bad_ref(self):
        with self.assertRaises(ValueError):
            conciseness(10, 0)


class TestConfidenceScore(unittest.TestCase):
    def test_non_executable_zero(self):
        self.assertEqual(confidence_score(0.0, executable=False), 0.0)

    def test_perfect_label_high(self):
        s = confidence_score(0.0, True, runner_up_chamfer=None, length=10,
                             ref_length=100)
        self.assertAlmostEqual(s, 1.0)

    def test_in_unit_interval(self):
        s = confidence_score(0.4, True, runner_up_chamfer=0.6, length=150,
                             ref_length=100)
        self.assertTrue(0.0 <= s <= 1.0)

    def test_lower_chamfer_scores_higher(self):
        a = confidence_score(0.1, True, length=10)
        b = confidence_score(0.8, True, length=10)
        self.assertGreater(a, b)

    def test_weights_sum_guard(self):
        with self.assertRaises(ValueError):
            confidence_score(0.1, True, weights={"fidelity": 0.0, "validity": 0.0,
                                                 "agreement": 0.0,
                                                 "conciseness": 0.0})

    def test_default_weights_sum_one(self):
        self.assertAlmostEqual(sum(DEFAULT_WEIGHTS.values()), 1.0)


class TestRank(unittest.TestCase):
    def test_orders_best_first(self):
        recs = [
            {"program": "hi", "chamfer": 0.05, "executable": True, "length": 10},
            {"program": "bad", "chamfer": 0.9, "executable": True, "length": 10},
            {"program": "dead", "chamfer": 0.01, "executable": False, "length": 5},
        ]
        ranked = rank_candidates(recs)
        self.assertEqual(ranked[0][0]["program"], "hi")
        self.assertEqual(ranked[-1][0]["program"], "dead")  # non-exec -> 0
        # scores are descending
        scores = [s for _, s in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
