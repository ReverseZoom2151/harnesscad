"""Tests for bench.deepclustering_ensemble."""

import unittest

from harnesscad.eval.bench.retrieval.deepclustering_ensemble import (
    ensemble_by_majority_vote,
    ensemble_human_balanced_accuracy,
    human_ensemble,
    kendall_tau,
    rank_methods,
    ranking_agreement,
)


class MajorityVoteTests(unittest.TestCase):
    def test_three_matrices(self):
        a = {(0, 1): 1, (0, 2): -1}
        b = {(0, 1): 1, (0, 2): 1}
        c = {(0, 1): -1, (0, 2): 1}
        # (0,1): 2 pos of 3 -> threshold ceil(4/2)=2 -> +1
        # (0,2): 2 pos of 3 -> +1
        result = ensemble_by_majority_vote([a, b, c])
        self.assertEqual(result[(0, 1)], 1)
        self.assertEqual(result[(0, 2)], 1)

    def test_even_tie_is_negative(self):
        # N=2, threshold = ceil(3/2) = 2; one positive -> -1
        a = {(0, 1): 1}
        b = {(0, 1): -1}
        self.assertEqual(ensemble_by_majority_vote([a, b])[(0, 1)], -1)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            ensemble_by_majority_vote([])


class HumanEnsembleTests(unittest.TestCase):
    def test_tie_is_unknown(self):
        sets = [{(0, 1): 1}, {(0, 1): 1}, {(0, 1): -1}, {(0, 1): -1}]
        self.assertEqual(human_ensemble(sets)[(0, 1)], 0)

    def test_majority_positive(self):
        sets = [{(0, 1): 1}, {(0, 1): 1}, {(0, 1): -1}]
        self.assertEqual(human_ensemble(sets)[(0, 1)], 1)

    def test_sparse_absent_counts_nothing(self):
        sets = [{(0, 1): 1}, {}, {(0, 1): -1}, {(0, 1): -1}]
        self.assertEqual(human_ensemble(sets)[(0, 1)], -1)


class EnsembleHumanTests(unittest.TestCase):
    def test_average(self):
        pred = {(0, 1): 1, (0, 2): -1}
        ensemble = {(0, 1): 1, (0, 2): -1}       # perfect -> 1.0
        human = {(0, 1): -1, (0, 2): -1}         # tp=0/1, tn=1/1 -> 0.5
        val = ensemble_human_balanced_accuracy(pred, ensemble, human)
        self.assertAlmostEqual(val, 0.5 * (1.0 + 0.5))


class RankingTests(unittest.TestCase):
    def test_rank_descending(self):
        scores = {"a": 0.9, "b": 0.5, "c": 0.7}
        self.assertEqual(rank_methods(scores), ["a", "c", "b"])

    def test_rank_tie_breaks_by_name(self):
        scores = {"b": 0.5, "a": 0.5}
        self.assertEqual(rank_methods(scores), ["a", "b"])

    def test_agreement_identical(self):
        r = ["a", "b", "c", "d"]
        self.assertAlmostEqual(ranking_agreement(r, r), 1.0)

    def test_agreement_reversed(self):
        self.assertAlmostEqual(
            ranking_agreement(["a", "b", "c"], ["c", "b", "a"]), 0.0)

    def test_agreement_partial(self):
        # swap one adjacent pair out of 3 pairs -> 2/3 concordant
        self.assertAlmostEqual(
            ranking_agreement(["a", "b", "c"], ["b", "a", "c"]), 2.0 / 3.0)

    def test_agreement_mismatched_sets(self):
        with self.assertRaises(ValueError):
            ranking_agreement(["a", "b"], ["a", "c"])

    def test_kendall_tau(self):
        self.assertAlmostEqual(kendall_tau(["a", "b", "c"], ["a", "b", "c"]), 1.0)
        self.assertAlmostEqual(kendall_tau(["a", "b", "c"], ["c", "b", "a"]), -1.0)


if __name__ == "__main__":
    unittest.main()
