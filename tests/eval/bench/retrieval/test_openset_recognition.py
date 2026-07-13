"""Tests for bench.oscar_openset_metrics -- open-set recognition metrics."""

from __future__ import annotations

import unittest

from harnesscad.eval.bench.retrieval.openset_recognition import (
    auroc,
    openset_confusion,
    openset_f_measure,
    rejection_accuracy,
    balanced_rejection_accuracy,
    best_threshold,
    novelty_score,
)


class TestAuroc(unittest.TestCase):
    def test_perfect_separation(self):
        self.assertAlmostEqual(auroc([0.8, 0.9, 1.0], [0.1, 0.2, 0.3]), 1.0)

    def test_inverted_separation(self):
        self.assertAlmostEqual(auroc([0.1, 0.2], [0.8, 0.9]), 0.0)

    def test_chance_overlap(self):
        # symmetric: knowns and unknowns interleave equally
        self.assertAlmostEqual(auroc([0.0, 1.0], [0.0, 1.0]), 0.5)

    def test_ties_count_half(self):
        # pairs: 0.5=0.5 tie(.5), 0.5>0.0, 1.0>0.5, 1.0>0.0 -> (3 + 0.5)/4
        self.assertAlmostEqual(auroc([0.5, 1.0], [0.5, 0.0]), 0.875)

    def test_empty_group_returns_half(self):
        self.assertEqual(auroc([], [0.1]), 0.5)
        self.assertEqual(auroc([0.1], []), 0.5)


class TestConfusion(unittest.TestCase):
    def test_counts(self):
        c = openset_confusion([0.9, 0.4], [0.6, 0.1], tau=0.5)
        # known: 0.9>=0.5 tp, 0.4<0.5 fn -> tp1 fn1
        # unknown: 0.6>=0.5 fp, 0.1<0.5 tn -> fp1 tn1
        self.assertEqual((c.tp, c.fn, c.fp, c.tn), (1, 1, 1, 1))


class TestFMeasure(unittest.TestCase):
    def test_perfect(self):
        p, r, f = openset_f_measure([0.9, 0.8], [0.1, 0.2], tau=0.5)
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(f, 1.0)

    def test_precision_recall_tradeoff(self):
        p, r, f = openset_f_measure([0.9, 0.4], [0.6, 0.1], tau=0.5)
        # tp1 fp1 fn1 -> precision .5 recall .5 f1 .5
        self.assertAlmostEqual(p, 0.5)
        self.assertAlmostEqual(r, 0.5)
        self.assertAlmostEqual(f, 0.5)

    def test_nothing_accepted(self):
        p, r, f = openset_f_measure([0.1], [0.2], tau=0.9)
        self.assertEqual((p, r, f), (0.0, 0.0, 0.0))

    def test_beta_weights_recall(self):
        # beta>1 emphasises recall
        p, r, fb = openset_f_measure([0.9, 0.4], [0.1], tau=0.5, beta=2.0)
        # p=1.0 (tp1 fp0), r=0.5 (tp1 fn1); F2 = 5*1*.5/(4*1+.5)=2.5/4.5
        self.assertAlmostEqual(fb, 2.5 / 4.5)

    def test_bad_beta(self):
        with self.assertRaises(ValueError):
            openset_f_measure([0.1], [0.2], tau=0.5, beta=0.0)


class TestAccuracy(unittest.TestCase):
    def test_rejection_accuracy(self):
        acc = rejection_accuracy([0.9, 0.8], [0.1, 0.2], tau=0.5)
        self.assertAlmostEqual(acc, 1.0)

    def test_rejection_accuracy_empty(self):
        self.assertEqual(rejection_accuracy([], [], 0.5), 0.0)

    def test_balanced_accuracy(self):
        # imbalanced: 1 known accepted, 3 unknown all rejected
        ba = balanced_rejection_accuracy([0.9], [0.1, 0.2, 0.3], tau=0.5)
        self.assertAlmostEqual(ba, 1.0)

    def test_balanced_one_group_empty(self):
        ba = balanced_rejection_accuracy([0.9, 0.1], [], tau=0.5)
        # only known-recall rate counts: 1 of 2 accepted -> 0.5
        self.assertAlmostEqual(ba, 0.5)


class TestBestThreshold(unittest.TestCase):
    def test_finds_separating_threshold(self):
        tau, val = best_threshold([0.8, 0.9], [0.1, 0.2], objective="f1")
        self.assertAlmostEqual(val, 1.0)
        # a tau in (0.2, 0.8] separates perfectly; lowest such candidate is 0.8
        self.assertLessEqual(tau, 0.8)

    def test_balanced_objective(self):
        tau, val = best_threshold([0.8, 0.9], [0.1, 0.2], objective="balanced")
        self.assertAlmostEqual(val, 1.0)

    def test_bad_objective(self):
        with self.assertRaises(ValueError):
            best_threshold([0.1], [0.2], objective="nope")

    def test_no_scores(self):
        with self.assertRaises(ValueError):
            best_threshold([], [], objective="f1")


class TestNovelty(unittest.TestCase):
    def test_confident_match_large_gap(self):
        # top1 far above neighbours -> high novelty (known) score
        s = novelty_score([0.95, 0.30, 0.28, 0.25], m=3)
        self.assertAlmostEqual(s, 0.95 - (0.30 + 0.28 + 0.25) / 3)

    def test_flat_profile_near_zero(self):
        s = novelty_score([0.5, 0.5, 0.5], m=2)
        self.assertAlmostEqual(s, 0.0)

    def test_too_few(self):
        self.assertEqual(novelty_score([0.9]), 0.0)

    def test_m_clamped_to_available(self):
        s = novelty_score([1.0, 0.0], m=10)
        self.assertAlmostEqual(s, 1.0)

    def test_nonpositive_m(self):
        self.assertEqual(novelty_score([1.0, 0.5], m=0), 0.0)


if __name__ == "__main__":
    unittest.main()
