import unittest

from harnesscad.eval.bench.engdesign_defect_confusion import (
    DEFECT_CLASSES, confusion_metrics, f1_from_matrix, defect_scorecard,
    perfect_prediction_tally,
)


class ConfusionTest(unittest.TestCase):
    def test_crack_matrix_matches_paper(self):
        # Table 10 crack: TP=11, FP=24, FN=3, TN=19. F1=0.45, recall 0.79,
        # specificity 0.44.
        m = confusion_metrics(tp=11, fp=24, fn=3, tn=19)
        self.assertAlmostEqual(m["f1"], 0.4489795918, places=6)
        self.assertAlmostEqual(m["recall"], 0.7857142857, places=6)
        self.assertAlmostEqual(m["specificity"], 0.4418604651, places=6)

    def test_spallation_f1(self):
        # Table 11 spallation: TP=11, FP=3, FN=8, TN=35 -> F1 0.67.
        self.assertAlmostEqual(f1_from_matrix(11, 3, 8, 35), 0.6666666667,
                               places=6)

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            confusion_metrics(-1, 0, 0, 0)


class ScorecardTest(unittest.TestCase):
    def test_five_classes(self):
        self.assertEqual(len(DEFECT_CLASSES), 5)

    def test_macro(self):
        matrices = {
            "crack": {"tp": 11, "fp": 24, "fn": 3, "tn": 19},
            "spallation": {"tp": 11, "fp": 3, "fn": 8, "tn": 35},
        }
        card = defect_scorecard(matrices)
        self.assertEqual(card["n_classes"], 2)
        expected = (0.4489795918 + 0.6666666667) / 2
        self.assertAlmostEqual(card["macro"]["f1"], expected, places=6)


class PerfectPredictionTest(unittest.TestCase):
    def test_tally_split(self):
        queries = [
            {"predicted": {"crack"}, "truth": {"crack"}},          # present ok
            {"predicted": set(), "truth": set()},                  # absent ok
            {"predicted": {"crack"}, "truth": {"spallation"}},     # wrong
            {"predicted": set(), "truth": set()},                  # absent ok
        ]
        res = perfect_prediction_tally(queries)
        self.assertEqual(res["perfect"], 3)
        self.assertEqual(res["perfect_absent"], 2)
        self.assertEqual(res["perfect_present"], 1)
        self.assertEqual(res["present"], 2)
        self.assertEqual(res["absent"], 2)


if __name__ == "__main__":
    unittest.main()
