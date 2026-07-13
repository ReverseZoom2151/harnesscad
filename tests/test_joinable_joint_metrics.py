"""Tests for bench.joinable_joint_metrics."""

import unittest

from harnesscad.eval.bench.joinable_joint_metrics import (
    DEFAULT_K_SEQUENCE,
    aggregate_precision_at_k,
    flatten,
    hit_at_top_k,
    joint_axis_error_stats,
    joint_axis_hit,
    k_sequence,
    mean_reciprocal_rank,
    precision_at_k_sequence,
    rank_of_first_hit,
    ranked_indices,
)


class HelperTests(unittest.TestCase):
    def test_k_sequence(self):
        ks = k_sequence()
        self.assertEqual(ks[:5], [1, 2, 3, 4, 5])
        self.assertEqual(ks[-1], 100)
        self.assertEqual(tuple(ks), DEFAULT_K_SEQUENCE)

    def test_flatten_matrix_and_flat(self):
        self.assertEqual(flatten([[1, 2], [3, 4]]), [1, 2, 3, 4])
        self.assertEqual(flatten([1, 2, 3]), [1, 2, 3])
        self.assertEqual(flatten([]), [])

    def test_ranked_indices_tie_break_by_index(self):
        self.assertEqual(ranked_indices([0.5, 0.9, 0.9, 0.1]), [1, 2, 0, 3])


class TopKTests(unittest.TestCase):
    def test_top1_hit(self):
        scores = [[0.1, 0.9], [0.2, 0.3]]
        labels = [[0, 1], [0, 0]]
        self.assertTrue(hit_at_top_k(scores, labels, k=1))

    def test_top1_miss_but_top3_hit(self):
        scores = [0.9, 0.8, 0.7, 0.6]
        labels = [0, 0, 1, 0]
        self.assertFalse(hit_at_top_k(scores, labels, k=1))
        self.assertFalse(hit_at_top_k(scores, labels, k=2))
        self.assertTrue(hit_at_top_k(scores, labels, k=3))

    def test_k_clamped_to_candidate_count(self):
        self.assertTrue(hit_at_top_k([0.1, 0.2], [1, 0], k=100))

    def test_no_positive_label_is_never_a_hit(self):
        self.assertFalse(hit_at_top_k([0.1, 0.2], [0, 0], k=50))

    def test_multiple_equivalents_count_as_hit(self):
        scores = [0.4, 0.9, 0.5]
        labels = [1, 0, 1]
        self.assertFalse(hit_at_top_k(scores, labels, k=1))
        self.assertTrue(hit_at_top_k(scores, labels, k=2))

    def test_bad_shapes(self):
        with self.assertRaises(ValueError):
            hit_at_top_k([0.1, 0.2], [1])
        with self.assertRaises(ValueError):
            hit_at_top_k([], [])
        with self.assertRaises(ValueError):
            hit_at_top_k([0.1], [1], k=0)


class PrecisionAtKTests(unittest.TestCase):
    def test_sequence_for_single_sample(self):
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        labels = [0, 0, 1, 0, 0]
        hits = precision_at_k_sequence(scores, labels, ks=[1, 2, 3, 5])
        self.assertEqual(hits, [0, 0, 1, 1])

    def test_default_ks_length(self):
        hits = precision_at_k_sequence([0.5], [1])
        self.assertEqual(len(hits), len(DEFAULT_K_SEQUENCE))
        self.assertTrue(all(h == 1 for h in hits))

    def test_aggregate_percent(self):
        curve = aggregate_precision_at_k([[0, 1, 1], [1, 1, 1]])
        self.assertAlmostEqual(curve[0], 50.0)
        self.assertAlmostEqual(curve[1], 100.0)

    def test_aggregate_fraction(self):
        curve = aggregate_precision_at_k([[0, 1], [1, 1]], use_percent=False)
        self.assertAlmostEqual(curve[0], 0.5)

    def test_aggregate_bad_input(self):
        with self.assertRaises(ValueError):
            aggregate_precision_at_k([])
        with self.assertRaises(ValueError):
            aggregate_precision_at_k([[1, 0], [1]])

    def test_precision_curve_is_monotone(self):
        scores = [0.9, 0.1, 0.5, 0.4]
        labels = [0, 1, 0, 0]
        hits = precision_at_k_sequence(scores, labels, ks=[1, 2, 3, 4])
        self.assertEqual(hits, [0, 0, 0, 1])
        self.assertTrue(all(b >= a for a, b in zip(hits, hits[1:])))


class RankTests(unittest.TestCase):
    def test_rank_of_first_hit(self):
        self.assertEqual(rank_of_first_hit([0.1, 0.9, 0.5], [1, 0, 0]), 3)
        self.assertEqual(rank_of_first_hit([0.1, 0.9, 0.5], [0, 1, 0]), 1)
        self.assertIsNone(rank_of_first_hit([0.1, 0.9], [0, 0]))

    def test_mrr(self):
        samples = [
            ([0.9, 0.1], [1, 0]),   # rank 1
            ([0.9, 0.1], [0, 1]),   # rank 2
            ([0.9, 0.1], [0, 0]),   # miss
        ]
        self.assertAlmostEqual(mean_reciprocal_rank(samples),
                               (1.0 + 0.5 + 0.0) / 3.0)

    def test_mrr_empty(self):
        with self.assertRaises(ValueError):
            mean_reciprocal_rank([])


class AxisMetricTests(unittest.TestCase):
    def test_axis_hit_against_equivalents(self):
        predicted = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        equivalents = [
            ((5.0, 0.0, 0.0), (0.0, 0.0, 1.0)),   # parallel but offset
            ((0.0, 0.0, 3.0), (0.0, 0.0, -1.0)),  # same line, reversed
        ]
        self.assertTrue(joint_axis_hit(predicted, equivalents))

    def test_axis_miss(self):
        predicted = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        equivalents = [((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))]
        self.assertFalse(joint_axis_hit(predicted, equivalents))

    def test_axis_stats_single_ground_truth(self):
        pairs = [
            (((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
             ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
            (((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
             ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
        ]
        stats = joint_axis_error_stats(pairs)
        self.assertEqual(stats["count"], 2)
        self.assertEqual(stats["hit_count"], 1)
        self.assertAlmostEqual(stats["hit_rate"], 0.5)
        self.assertAlmostEqual(stats["mean_angle_degs"], 45.0)
        self.assertAlmostEqual(stats["median_angle_degs"], 45.0)

    def test_axis_stats_takes_best_equivalent(self):
        pairs = [
            (((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
             [((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
              ((0.0, 0.0, 2.0), (0.0, 0.0, 1.0))]),
        ]
        stats = joint_axis_error_stats(pairs)
        self.assertAlmostEqual(stats["mean_angle_degs"], 0.0)
        self.assertAlmostEqual(stats["mean_distance"], 0.0)
        self.assertEqual(stats["hit_count"], 1)

    def test_axis_stats_std_and_empty(self):
        pairs = [(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
                  ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)))]
        stats = joint_axis_error_stats(pairs)
        self.assertAlmostEqual(stats["std_angle_degs"], 0.0)
        with self.assertRaises(ValueError):
            joint_axis_error_stats([])


if __name__ == "__main__":
    unittest.main()
