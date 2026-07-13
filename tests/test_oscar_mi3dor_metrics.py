"""Tests for bench.oscar_mi3dor_metrics -- MI3DOR shape-retrieval criteria."""

from __future__ import annotations

import unittest

from harnesscad.eval.bench.retrieval.oscar_mi3dor_metrics import (
    nearest_neighbour,
    first_tier,
    second_tier,
    f_measure_at_k,
    dcg,
    anmrr,
    mi3dor_report,
)


class TestNN(unittest.TestCase):
    def test_top1_relevant(self):
        self.assertEqual(nearest_neighbour([1, 0, 0]), 1.0)

    def test_top1_irrelevant(self):
        self.assertEqual(nearest_neighbour([0, 1, 1]), 0.0)

    def test_empty(self):
        self.assertEqual(nearest_neighbour([]), 0.0)


class TestTiers(unittest.TestCase):
    def test_first_tier_perfect(self):
        # C=4 -> tier size 3; top-3 all relevant
        self.assertAlmostEqual(first_tier([1, 1, 1, 0, 0], 4), 1.0)

    def test_first_tier_partial(self):
        # tier size 3, 2 hits in top-3
        self.assertAlmostEqual(first_tier([1, 1, 0, 1, 0], 4), 2.0 / 3.0)

    def test_first_tier_singleton_class(self):
        self.assertEqual(first_tier([1, 0], 1), 0.0)

    def test_second_tier_window_double(self):
        # C=3 -> tier 2, window top-4; hits within top-4 = 2 -> 2/2 = 1.0
        self.assertAlmostEqual(second_tier([0, 1, 0, 1, 0], 3), 1.0)

    def test_second_tier_partial(self):
        # C=4 -> tier 3, window top-6; 2 hits -> 2/3
        self.assertAlmostEqual(second_tier([1, 0, 0, 0, 0, 1, 0], 4), 2.0 / 3.0)

    def test_second_tier_caps_at_one(self):
        # more hits than tier size cannot exceed 1.0
        self.assertLessEqual(second_tier([1, 1, 1, 1, 1, 1], 3), 1.0)


class TestFMeasure(unittest.TestCase):
    def test_basic(self):
        # C=3 -> recall denom 2; k=2; top-2 has 2 hits
        # precision = 2/2 = 1.0, recall = 2/2 = 1.0 -> F1 1.0
        self.assertAlmostEqual(f_measure_at_k([1, 1, 0], 3, k=2), 1.0)

    def test_precision_recall_mix(self):
        # C=3 -> recall denom 2; k=4; hits in top-4 = 1
        # precision 1/4, recall 1/2 -> F1 = 2*.25*.5/.75
        self.assertAlmostEqual(f_measure_at_k([1, 0, 0, 0], 3, k=4),
                               2 * 0.25 * 0.5 / 0.75)

    def test_singleton_class(self):
        self.assertEqual(f_measure_at_k([1, 0], 1, k=5), 0.0)

    def test_no_hits(self):
        self.assertEqual(f_measure_at_k([0, 0, 0], 3, k=2), 0.0)

    def test_bad_k(self):
        with self.assertRaises(ValueError):
            f_measure_at_k([1], 2, k=0)


class TestDCG(unittest.TestCase):
    def test_perfect_ranking_is_one(self):
        self.assertAlmostEqual(dcg([1, 1, 0, 0]), 1.0)

    def test_reversed_less_than_one(self):
        self.assertLess(dcg([0, 0, 1, 1]), 1.0)


class TestANMRR(unittest.TestCase):
    def test_perfect_retrieval_is_zero(self):
        # NG=3, all relevant at ranks 1,2,3 -> best possible -> 0.0
        self.assertAlmostEqual(anmrr([1, 1, 1, 0, 0], 3), 0.0)

    def test_worst_retrieval_near_one(self):
        # NG=2, none within window (window K=4) -> both penalised -> ~1.0
        self.assertAlmostEqual(anmrr([0, 0, 0, 0, 0, 0], 2), 1.0)

    def test_monotonic_worse_when_ranked_later(self):
        good = anmrr([1, 1, 0, 0, 0, 0], 2)
        bad = anmrr([0, 0, 1, 1, 0, 0], 2)
        self.assertLess(good, bad)

    def test_in_unit_interval(self):
        v = anmrr([0, 1, 0, 1, 0], 3)
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0)

    def test_no_relevant(self):
        self.assertEqual(anmrr([0, 0], 0), 0.0)


class TestReport(unittest.TestCase):
    def test_aggregate_means(self):
        queries = [
            {"relevances": [1, 1, 0, 0], "num_relevant": 3},
            {"relevances": [0, 1, 0, 0], "num_relevant": 2},
        ]
        rep = mi3dor_report(queries, f_k=2)
        # NN: 1.0 and 0.0 -> mean 0.5
        self.assertAlmostEqual(rep["NN"], 0.5)
        for key in ("NN", "FT", "ST", "F", "DCG", "ANMRR"):
            self.assertIn(key, rep)
            self.assertGreaterEqual(rep[key], 0.0)
            self.assertLessEqual(rep[key], 1.0)

    def test_empty_batch(self):
        rep = mi3dor_report([])
        self.assertEqual(rep["NN"], 0.0)
        self.assertEqual(rep["ANMRR"], 0.0)


if __name__ == "__main__":
    unittest.main()
