import unittest

from harnesscad.eval.bench.protocols.engdesign_dfm_scoring import (
    MACHINING_FEATURES, score_additive_design, additive_scorecard,
    feature_recognition, feature_scorecard,
)


class AdditiveTest(unittest.TestCase):
    def test_design_a_trial1(self):
        # Table 9 Design A trial 1: manuf=1, rule=0, incorrect=-4 -> score -3.
        res = score_additive_design(True, False, 4)
        self.assertEqual(res["score"], -3)

    def test_design_b_trial1(self):
        # manuf=1, rule=1, incorrect=-1 -> score 1.
        res = score_additive_design(True, True, 1)
        self.assertEqual(res["score"], 1)

    def test_negative_incorrect_raises(self):
        with self.assertRaises(ValueError):
            score_additive_design(True, True, -1)

    def test_scorecard(self):
        trials = [
            {"manufacturable_correct": True, "correct_rule": False,
             "num_incorrect_rules": 4},
            {"manufacturable_correct": True, "correct_rule": True,
             "num_incorrect_rules": 1},
        ]
        card = additive_scorecard(trials)
        self.assertEqual(card["total_score"], -2)
        self.assertEqual(card["manufacturable_rate"], 1.0)
        self.assertEqual(card["correct_rule_rate"], 0.5)


class FeatureRecognitionTest(unittest.TestCase):
    def test_taxonomy_size(self):
        self.assertEqual(len(MACHINING_FEATURES), 15)

    def test_partial_overlap(self):
        # Figure 21 sample: GPT predicts triangular through slot + rectangular
        # passage + 2 sided through step; GT is triangular + rectangular slot.
        pred = ["triangular through slot", "rectangular passage",
                "2 sided through step"]
        gt = ["triangular through slot", "rectangular through slot"]
        res = feature_recognition(pred, gt)
        self.assertTrue(res["at_least_one_correct"])
        self.assertFalse(res["exact"])
        self.assertAlmostEqual(res["precision"], 1 / 3)
        self.assertAlmostEqual(res["recall"], 1 / 2)

    def test_exact_match(self):
        res = feature_recognition(["chamfer"], ["chamfer"])
        self.assertTrue(res["exact"])
        self.assertEqual(res["f1"], 1.0)

    def test_unknown_feature_raises(self):
        with self.assertRaises(ValueError):
            feature_recognition(["banana"], ["chamfer"])

    def test_scorecard(self):
        samples = [
            (["chamfer"], ["chamfer"]),
            (["rectangular pocket"], ["triangular pocket"]),
        ]
        card = feature_scorecard(samples)
        self.assertEqual(card["n"], 2)
        self.assertEqual(card["at_least_one_rate"], 0.5)
        self.assertEqual(card["exact_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
