"""Tests for bench/mfgfeat_afr_metrics.py."""

from __future__ import annotations

import unittest

from harnesscad.eval.bench.protocols import manufacturing_feature_recognition as m


class TestFNA(unittest.TestCase):
    def test_perfect(self):
        gt = {"hole": 3, "slot": 1}
        pred = {"hole": 5, "slot": 2}  # quantity ignored for FNA
        self.assertEqual(m.feature_name_accuracy(pred, gt), 1.0)

    def test_partial(self):
        gt = {"hole": 3, "slot": 1, "pocket": 2}
        pred = {"hole": 1}
        self.assertAlmostEqual(m.feature_name_accuracy(pred, gt), 1 / 3)

    def test_extra_name_does_not_help(self):
        gt = {"hole": 1}
        pred = {"hole": 1, "slot": 9}
        self.assertEqual(m.feature_name_accuracy(pred, gt), 1.0)

    def test_empty_gt(self):
        self.assertEqual(m.feature_name_accuracy({}, {}), 1.0)
        self.assertEqual(m.feature_name_accuracy({"hole": 1}, {}), 0.0)


class TestFQA(unittest.TestCase):
    def test_perfect(self):
        gt = {"hole": 4, "slot": 2}
        self.assertEqual(m.feature_quantity_accuracy(dict(gt), gt), 1.0)

    def test_undercount(self):
        gt = {"hole": 10}
        pred = {"hole": 7}
        self.assertAlmostEqual(m.feature_quantity_accuracy(pred, gt), 0.7)

    def test_overcount_capped(self):
        # true positive quantity cannot exceed ground truth
        gt = {"hole": 4}
        pred = {"hole": 100}
        self.assertEqual(m.feature_quantity_accuracy(pred, gt), 1.0)

    def test_mixed(self):
        gt = {"hole": 4, "slot": 2}  # total 6
        pred = {"hole": 4, "slot": 1}  # tp = 4 + 1 = 5
        self.assertAlmostEqual(m.feature_quantity_accuracy(pred, gt), 5 / 6)


class TestHR(unittest.TestCase):
    def test_no_hallucination(self):
        gt = {"hole": 4}
        self.assertEqual(m.hallucination_rate({"hole": 4}, gt), 0.0)

    def test_pure_hallucination(self):
        gt = {"hole": 1}
        pred = {"pocket": 3}  # none real -> all hallucinated
        self.assertEqual(m.hallucination_rate(pred, gt), 1.0)

    def test_overcount_hallucination(self):
        gt = {"hole": 2}
        pred = {"hole": 5}  # predicted 5, tp 2 -> (5-2)/5
        self.assertAlmostEqual(m.hallucination_rate(pred, gt), 3 / 5)

    def test_empty_prediction(self):
        self.assertEqual(m.hallucination_rate({}, {"hole": 3}), 0.0)


class TestMAE(unittest.TestCase):
    def test_zero(self):
        gt = {"hole": 4, "slot": 2}
        self.assertEqual(m.mean_absolute_error(dict(gt), gt), 0.0)

    def test_union_average(self):
        gt = {"hole": 4, "slot": 2}
        pred = {"hole": 2, "pocket": 3}
        # features: hole|2, slot|2, pocket|3 -> (2+2+3)/3
        self.assertAlmostEqual(m.mean_absolute_error(pred, gt), 7 / 3)

    def test_feature_space_fixed(self):
        gt = {"hole": 4}
        pred = {"hole": 2}
        # score over 3 fixed features, 2 absent => errors 2,0,0 over n=3
        mae = m.mean_absolute_error(pred, gt,
                                    feature_space=["hole", "slot", "pocket"])
        self.assertAlmostEqual(mae, 2 / 3)

    def test_empty(self):
        self.assertEqual(m.mean_absolute_error({}, {}), 0.0)


class TestNormalisationAndValidation(unittest.TestCase):
    def test_normalize_folds_aliases(self):
        gt = {"hole": 5}
        pred = {"blind hole": 3, "through hole": 2}  # both -> hole, sum 5
        self.assertEqual(
            m.feature_quantity_accuracy(pred, gt, normalize=True), 1.0)
        self.assertEqual(m.hallucination_rate(pred, gt, normalize=True), 0.0)

    def test_negative_quantity_raises(self):
        with self.assertRaises(ValueError):
            m.feature_name_accuracy({"hole": -1}, {"hole": 1})

    def test_zero_dropped(self):
        # a zero-count entry is ignored entirely
        self.assertEqual(m.feature_name_accuracy({"hole": 0}, {"hole": 1}), 0.0)


class TestScorecard(unittest.TestCase):
    def test_aggregate(self):
        samples = [
            ({"hole": 4}, {"hole": 4}),   # perfect
            ({"hole": 2}, {"hole": 4}),   # under
        ]
        card = m.afr_scorecard(samples)
        self.assertEqual(card["n"], 2)
        self.assertAlmostEqual(card["mean_fqa"], (1.0 + 0.5) / 2)
        self.assertAlmostEqual(card["mean_fna"], 1.0)
        self.assertAlmostEqual(card["mean_hr"], 0.0)
        self.assertAlmostEqual(card["mean_mae"], (0 + 2) / 2)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            m.afr_scorecard([])


if __name__ == "__main__":
    unittest.main()
