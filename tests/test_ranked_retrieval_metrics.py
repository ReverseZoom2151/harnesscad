"""Tests for bench.ranked_retrieval_metrics -- DCG/NDCG, MRR, success rate, enrichment factor."""

from __future__ import annotations

import math
import unittest

from bench.ranked_retrieval_metrics import (
    dcg_at_k,
    ndcg_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    success_at_k,
    success_rate_at_k,
    enrichment_factor,
)


class TestDCG(unittest.TestCase):
    def test_dcg_single_relevant_first(self):
        # gain 1 at rank 0 -> 1 / log2(2) = 1.0
        self.assertAlmostEqual(dcg_at_k([1, 0, 0]), 1.0)

    def test_dcg_discount_by_rank(self):
        # gain 1 at rank 1 -> 1 / log2(3)
        self.assertAlmostEqual(dcg_at_k([0, 1, 0]), 1.0 / math.log2(3))

    def test_dcg_graded_gains_sum(self):
        expected = 3.0 / math.log2(2) + 2.0 / math.log2(3) + 1.0 / math.log2(4)
        self.assertAlmostEqual(dcg_at_k([3, 2, 1]), expected)

    def test_dcg_k_truncates(self):
        self.assertAlmostEqual(dcg_at_k([1, 1, 1], k=1), 1.0)

    def test_dcg_k_zero_is_zero(self):
        self.assertEqual(dcg_at_k([1, 1], k=0), 0.0)

    def test_dcg_negative_k_raises(self):
        with self.assertRaises(ValueError):
            dcg_at_k([1], k=-1)


class TestNDCG(unittest.TestCase):
    def test_ideal_ranking_is_one(self):
        self.assertAlmostEqual(ndcg_at_k([3, 2, 1]), 1.0)

    def test_worst_order_below_one(self):
        self.assertLess(ndcg_at_k([1, 2, 3]), 1.0)

    def test_all_zero_gains_is_zero(self):
        self.assertEqual(ndcg_at_k([0, 0, 0]), 0.0)

    def test_ndcg_in_unit_interval(self):
        val = ndcg_at_k([0, 3, 1, 2])
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)


class TestReciprocalRank(unittest.TestCase):
    def test_first_relevant_rank_one(self):
        self.assertEqual(reciprocal_rank([1, 0, 0]), 1.0)

    def test_first_relevant_rank_three(self):
        self.assertAlmostEqual(reciprocal_rank([0, 0, 1]), 1.0 / 3.0)

    def test_none_relevant_is_zero(self):
        self.assertEqual(reciprocal_rank([0, 0, 0]), 0.0)

    def test_mrr_averages(self):
        rankings = [[1, 0], [0, 1], [0, 0]]
        self.assertAlmostEqual(mean_reciprocal_rank(rankings), (1.0 + 0.5 + 0.0) / 3.0)

    def test_mrr_empty(self):
        self.assertEqual(mean_reciprocal_rank([]), 0.0)


class TestSuccessRate(unittest.TestCase):
    def test_hit_in_top_k(self):
        self.assertEqual(success_at_k([0, 1, 0], 2), 1.0)

    def test_miss_outside_top_k(self):
        self.assertEqual(success_at_k([0, 0, 1], 2), 0.0)

    def test_negative_k_raises(self):
        with self.assertRaises(ValueError):
            success_at_k([1], -1)

    def test_success_rate_averages(self):
        rankings = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
        # top-1 hits: query0 yes, query1 no, query2 no -> 1/3
        self.assertAlmostEqual(success_rate_at_k(rankings, 1), 1.0 / 3.0)

    def test_success_rate_empty(self):
        self.assertEqual(success_rate_at_k([], 5), 0.0)


class TestEnrichmentFactor(unittest.TestCase):
    def test_perfect_enrichment(self):
        # 10 items, 2 relevant both at the very top; fraction 0.2 -> top 2 items.
        rel = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        # hit rate top = 2/2 = 1.0; baseline = 2/10 = 0.2 -> EF = 5.0
        self.assertAlmostEqual(enrichment_factor(rel, 0.2), 5.0)

    def test_random_enrichment_near_one(self):
        # evenly spread relevants -> EF around 1.0
        rel = [1, 0, 0, 0, 0] * 2  # 10 items, 2 relevant, one in top 20%
        self.assertAlmostEqual(enrichment_factor(rel, 0.2), (1.0 / 2.0) / (2.0 / 10.0))

    def test_no_relevant_is_zero(self):
        self.assertEqual(enrichment_factor([0, 0, 0], 0.5), 0.0)

    def test_external_total_relevant(self):
        # ranking is a shortlist of 4; the full candidate pool had 20 relevant of 100.
        rel = [1, 0, 1, 0]
        n_top = max(1, math.ceil(0.5 * 4))  # 2
        expected = (1.0 / n_top) / (20.0 / 4)
        self.assertAlmostEqual(enrichment_factor(rel, 0.5, total_relevant=20), expected)

    def test_bad_fraction_raises(self):
        with self.assertRaises(ValueError):
            enrichment_factor([1, 0], 0.0)
        with self.assertRaises(ValueError):
            enrichment_factor([1, 0], 1.5)

    def test_empty_ranking_is_zero(self):
        self.assertEqual(enrichment_factor([], 0.5), 0.0)


if __name__ == "__main__":
    unittest.main()
