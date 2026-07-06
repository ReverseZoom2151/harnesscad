"""Tests for bench.gencad3d_seqlen_norm (sequence-length normalized metrics)."""

import unittest

from bench.gencad3d_seqlen_norm import (
    NormalizedMetric,
    compare_normalized,
    per_length_means,
    relative_error_improvement,
    relative_reduction,
    sequence_length_normalized,
)


class PerLengthMeansTests(unittest.TestCase):
    def test_grouping(self):
        vals = [1.0, 0.0, 1.0, 1.0]
        lens = [5, 5, 9, 9]
        means = per_length_means(vals, lens)
        self.assertAlmostEqual(means[5], 0.5)
        self.assertAlmostEqual(means[9], 1.0)

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            per_length_means([1.0], [5, 9])


class SequenceLengthNormalizedTests(unittest.TestCase):
    def test_unnormalized_is_plain_mean(self):
        vals = [1.0, 1.0, 1.0, 0.0]  # 3 short good, 1 long bad
        lens = [5, 5, 5, 40]
        res = sequence_length_normalized(vals, lens)
        self.assertAlmostEqual(res.unnormalized, 0.75)

    def test_normalization_equalizes_buckets(self):
        # 3 items length 5 (all correct) + 1 item length 40 (wrong).
        vals = [1.0, 1.0, 1.0, 0.0]
        lens = [5, 5, 5, 40]
        res = sequence_length_normalized(vals, lens)
        # buckets: len5 -> 1.0, len40 -> 0.0 ; mean = 0.5
        self.assertAlmostEqual(res.normalized, 0.5)
        # normalization reveals the hidden failure: much lower than 0.75
        self.assertLess(res.normalized, res.unnormalized)

    def test_bias(self):
        vals = [1.0, 1.0, 1.0, 0.0]
        lens = [5, 5, 5, 40]
        res = sequence_length_normalized(vals, lens)
        self.assertAlmostEqual(res.bias, 0.75 - 0.5)

    def test_balanced_matches_unnormalized(self):
        vals = [1.0, 0.0, 1.0, 0.0]
        lens = [5, 5, 40, 40]
        res = sequence_length_normalized(vals, lens)
        self.assertAlmostEqual(res.normalized, res.unnormalized)

    def test_worst_length_higher_better(self):
        vals = [1.0, 1.0, 0.0]
        lens = [5, 5, 40]
        res = sequence_length_normalized(vals, lens, higher_is_better=True)
        self.assertEqual(res.worst_length(), 40)

    def test_worst_length_lower_better(self):
        # lower is better (e.g. invalid ratio): worst = highest value
        vals = [0.0, 0.0, 1.0]
        lens = [5, 5, 40]
        res = sequence_length_normalized(vals, lens, higher_is_better=False)
        self.assertEqual(res.worst_length(), 40)

    def test_to_dict(self):
        res = sequence_length_normalized([1.0, 0.0], [5, 40])
        d = res.to_dict()
        self.assertIn("normalized", d)
        self.assertIn("per_length", d)
        self.assertEqual(d["worst_length"], 40)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            sequence_length_normalized([], [])

    def test_counts(self):
        res = sequence_length_normalized([1.0, 1.0, 0.0], [5, 5, 40])
        self.assertEqual(res.counts[5], 2)
        self.assertEqual(res.counts[40], 1)


class RelativeImprovementTests(unittest.TestCase):
    def test_paper_examples(self):
        # 0.995 upon 0.990 and 0.95 upon 0.90 both == 50%
        self.assertAlmostEqual(relative_error_improvement(0.995, 0.990), 0.5)
        self.assertAlmostEqual(relative_error_improvement(0.95, 0.90), 0.5)

    def test_no_change(self):
        self.assertAlmostEqual(relative_error_improvement(0.9, 0.9), 0.0)

    def test_baseline_one_raises(self):
        with self.assertRaises(ValueError):
            relative_error_improvement(1.0, 1.0)

    def test_out_of_range(self):
        with self.assertRaises(ValueError):
            relative_error_improvement(1.2, 0.5)

    def test_relative_reduction(self):
        # error halved -> 50% reduction
        self.assertAlmostEqual(relative_reduction(0.5, 1.0), 0.5)

    def test_relative_reduction_bad_baseline(self):
        with self.assertRaises(ValueError):
            relative_reduction(0.1, 0.0)


class CompareNormalizedTests(unittest.TestCase):
    def test_higher_is_better(self):
        cand = sequence_length_normalized([1.0, 1.0], [5, 40])       # both 1.0
        base = sequence_length_normalized([1.0, 0.0], [5, 40])       # norm 0.5
        cmp = compare_normalized(cand, base)
        # normalized: (1.0 - 0.5)/(1 - 0.5) = 1.0
        self.assertAlmostEqual(cmp["normalized"], 1.0)

    def test_lower_is_better(self):
        cand = sequence_length_normalized([0.0, 0.0], [5, 40], higher_is_better=False)
        base = sequence_length_normalized([1.0, 1.0], [5, 40], higher_is_better=False)
        cmp = compare_normalized(cand, base)
        self.assertAlmostEqual(cmp["normalized"], 1.0)  # error fully removed

    def test_direction_mismatch(self):
        cand = sequence_length_normalized([1.0], [5], higher_is_better=True)
        base = sequence_length_normalized([1.0], [5], higher_is_better=False)
        with self.assertRaises(ValueError):
            compare_normalized(cand, base)


if __name__ == "__main__":
    unittest.main()
