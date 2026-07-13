import unittest

from harnesscad.eval.bench.protocols.qa_scoring import (
    majority_correct, qa_scorecard, spatial_run_scores, run_comparison,
)


class MajorityTest(unittest.TestCase):
    def test_two_of_three(self):
        self.assertTrue(majority_correct([True, True, False]))
        self.assertFalse(majority_correct([True, False, False]))

    def test_custom_threshold(self):
        self.assertTrue(majority_correct([True, False, False], threshold=1))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            majority_correct([])


class QaScorecardTest(unittest.TestCase):
    def test_grouping_and_errors(self):
        questions = [
            {"id": "Q1", "image_type": "diagram", "format": "free text",
             "repeats": [False, False, False], "error": "imprecise"},
            {"id": "Q3", "image_type": "3d-model", "format": "free text",
             "repeats": [True, True, True]},
            {"id": "Q9", "image_type": "table", "format": "numerical",
             "repeats": [True, False, True]},
        ]
        card = qa_scorecard(questions)
        self.assertEqual(card["correct"], 2)
        self.assertEqual(card["total"], 3)
        self.assertEqual(card["by_image_type"]["table"]["accuracy"], 1.0)
        self.assertEqual(card["by_image_type"]["diagram"]["accuracy"], 0.0)
        self.assertEqual(card["errors"]["imprecise"], 1)

    def test_unknown_error_raises(self):
        with self.assertRaises(ValueError):
            qa_scorecard([{"id": "Q", "image_type": "d", "format": "f",
                           "repeats": [False], "error": "bogus"}])


class SpatialTest(unittest.TestCase):
    def test_packing_run_scores(self):
        # Two questions, correct key, 4-option test.
        runs = {"Run1": ["A", "B"], "Run2": ["A", "C"]}
        key = ["A", "B"]
        res = spatial_run_scores(runs, key, num_options=4)
        self.assertEqual(res["per_run"]["Run1"]["accuracy"], 1.0)
        self.assertEqual(res["per_run"]["Run2"]["accuracy"], 0.5)
        self.assertEqual(res["average_accuracy"], 0.75)
        self.assertEqual(res["random_baseline"], 0.25)
        # Q0 correct in both runs -> consistent; Q1 only in one run.
        self.assertEqual(res["consistent_questions"], (0,))

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            spatial_run_scores({"R": ["A"]}, ["A", "B"], 4)

    def test_run_comparison(self):
        a = {"average_accuracy": 0.16}
        b = {"average_accuracy": 0.20}
        cmp = run_comparison(a, b)
        self.assertAlmostEqual(cmp["delta"], 0.04)
        self.assertEqual(cmp["better"], "b")


if __name__ == "__main__":
    unittest.main()
