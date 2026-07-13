import unittest

from harnesscad.eval.bench.protocols.engdesign_description_match import (
    score_trial, random_baseline, case_scorecard, match_scorecard,
)


class ScoreTrialTest(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(score_trial("BAABBCCBB", "BAABBCCBB"), (9, 9))

    def test_partial(self):
        # Table 2 no-text trial-1 was 6/10.
        pred = ["B", "A", "D", "A", "B", "D", "C", "D", "A", "B"]
        key = ["B", "A", "C", "A", "B", "C", "C", "C", "B", "B"]
        self.assertEqual(score_trial(pred, key), (6, 10))

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            score_trial("AB", "ABC")


class BaselineTest(unittest.TestCase):
    def test_baseline(self):
        # 10 questions, 4 options -> 2.5 expected (paper's random 2.5/10).
        self.assertEqual(random_baseline(4, 10), 2.5)

    def test_invalid_options(self):
        with self.assertRaises(ValueError):
            random_baseline(0, 10)


class ScorecardTest(unittest.TestCase):
    def test_case_average(self):
        trials = [("BB", "BB"), ("BC", "BB"), ("CC", "BB")]
        card = case_scorecard(trials, num_options=4)
        # correct counts 2, 1, 0 -> avg 1.0
        self.assertEqual(card["avg_correct"], 1.0)
        self.assertEqual(card["questions"], 2)
        self.assertEqual(card["random_baseline"], 0.5)

    def test_match_scorecard(self):
        cases = {
            "with_text": {"trials": [("AB", "AB"), ("AB", "AB")],
                          "num_options": 4},
            "no_text": {"trials": [("AB", "AC")], "num_options": 3},
        }
        out = match_scorecard(cases)
        self.assertEqual(out["with_text"]["avg_accuracy"], 1.0)
        self.assertEqual(out["no_text"]["avg_correct"], 1.0)


if __name__ == "__main__":
    unittest.main()
