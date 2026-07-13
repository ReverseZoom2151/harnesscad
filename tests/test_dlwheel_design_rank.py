"""Tests for quality.dlwheel_design_rank (paper 112 dedup + stiffness ranking)."""

import math
import unittest

from harnesscad.eval.quality import dlwheel_design_rank as dr


class L1DedupTests(unittest.TestCase):
    def test_l1_distance(self):
        self.assertAlmostEqual(dr.l1_distance([1.0, 2.0], [1.0, 5.0]), 3.0)

    def test_l1_length_mismatch(self):
        with self.assertRaises(ValueError):
            dr.l1_distance([1.0], [1.0, 2.0])

    def test_flatten(self):
        self.assertEqual(dr.flatten([[1, 2], [3, 4]]), [1.0, 2.0, 3.0, 4.0])

    def test_dedup_removes_near_duplicates(self):
        designs = [
            [0.0, 0.0],
            [0.0, 0.5],   # within threshold 1 of first -> dropped
            [5.0, 5.0],   # distinct -> kept
            [5.0, 5.2],   # within threshold of third -> dropped
        ]
        kept = dr.deduplicate_l1(designs, threshold=1.0)
        self.assertEqual(kept, [0, 2])

    def test_dedup_all_distinct(self):
        designs = [[0.0], [10.0], [20.0]]
        self.assertEqual(dr.deduplicate_l1(designs, threshold=1.0), [0, 1, 2])

    def test_dedup_bad_threshold(self):
        with self.assertRaises(ValueError):
            dr.deduplicate_l1([[0.0]], threshold=-1.0)

    def test_mean_pairwise(self):
        # distances: |0-2|=2, |0-4|=4, |2-4|=2 -> mean 8/3
        self.assertAlmostEqual(dr.mean_pairwise_l1([[0.0], [2.0], [4.0]]), 8.0 / 3.0)
        self.assertEqual(dr.mean_pairwise_l1([[1.0]]), 0.0)


class RankingTests(unittest.TestCase):
    def test_stiffness_recovery(self):
        ranked = dr.rank_by_stiffness([(2.0, 10.0)], stiffness_standard=0.0)
        expected = 2.0 * (2.0 * math.pi * 10.0) ** 2
        self.assertAlmostEqual(ranked[0].stiffness, expected)

    def test_ranking_order(self):
        # same mass, higher frequency -> higher stiffness first
        concepts = [(2.0, 10.0), (2.0, 30.0), (2.0, 20.0)]
        ranked = dr.rank_by_stiffness(concepts, stiffness_standard=0.0)
        self.assertEqual([c.index for c in ranked], [1, 2, 0])

    def test_standard_flag_and_elimination(self):
        concepts = [(2.0, 10.0), (2.0, 30.0)]
        std = dr._stiffness(20.0, 2.0)
        ranked = dr.rank_by_stiffness(concepts, stiffness_standard=std)
        kept = dr.eliminate_below_standard(ranked)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].index, 1)
        self.assertTrue(kept[0].meets_standard)

    def test_top_k(self):
        concepts = [(1.0, 10.0), (1.0, 20.0), (1.0, 30.0)]
        ranked = dr.rank_by_stiffness(concepts, stiffness_standard=0.0)
        top = dr.top_k(ranked, 2)
        self.assertEqual([c.index for c in top], [2, 1])
        with self.assertRaises(ValueError):
            dr.top_k(ranked, -1)

    def test_invalid_concept(self):
        with self.assertRaises(ValueError):
            dr.rank_by_stiffness([(0.0, 10.0)], stiffness_standard=0.0)


if __name__ == "__main__":
    unittest.main()
