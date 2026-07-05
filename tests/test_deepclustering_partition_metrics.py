"""Tests for bench.deepclustering_partition_metrics."""

import unittest

from bench.deepclustering_partition_metrics import (
    adjusted_rand_index,
    clustering_accuracy,
    contingency_table,
    entropy,
    mutual_information,
    normalized_mutual_information,
    purity,
    rand_index,
)


class ContingencyTests(unittest.TestCase):
    def test_counts(self):
        matrix, rows, cols = contingency_table([0, 0, 1, 1], ["a", "b", "a", "a"])
        self.assertEqual(rows, [0, 1])
        self.assertEqual(cols, ["a", "b"])
        # row 0 (label 0): one 'a', one 'b'; row 1 (label 1): two 'a'
        self.assertEqual(matrix, [[1, 1], [2, 0]])

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            contingency_table([0, 1], [0])


class EntropyMITests(unittest.TestCase):
    def test_entropy_uniform(self):
        # two equally sized clusters -> ln 2
        import math
        self.assertAlmostEqual(entropy([0, 0, 1, 1]), math.log(2))

    def test_entropy_single_cluster_zero(self):
        self.assertEqual(entropy([5, 5, 5]), 0.0)

    def test_mutual_information_identical(self):
        labels = [0, 0, 1, 1, 2, 2]
        self.assertAlmostEqual(mutual_information(labels, labels), entropy(labels))

    def test_mutual_information_independent_is_zero(self):
        # perfectly balanced independent labellings
        a = [0, 0, 1, 1]
        b = [0, 1, 0, 1]
        self.assertAlmostEqual(mutual_information(a, b), 0.0)


class NMITests(unittest.TestCase):
    def test_identical_is_one(self):
        labels = [0, 1, 2, 0, 1, 2]
        for avg in ("arithmetic", "geometric", "min", "max"):
            self.assertAlmostEqual(
                normalized_mutual_information(labels, labels, avg), 1.0)

    def test_relabel_invariant(self):
        a = [0, 0, 1, 1, 2, 2]
        b = ["x", "x", "y", "y", "z", "z"]
        self.assertAlmostEqual(normalized_mutual_information(a, b), 1.0)

    def test_both_trivial_single_cluster(self):
        self.assertEqual(normalized_mutual_information([0, 0, 0], [1, 1, 1]), 1.0)

    def test_range(self):
        a = [0, 0, 1, 1, 2, 3]
        b = [0, 1, 1, 1, 2, 2]
        v = normalized_mutual_information(a, b)
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0)

    def test_bad_average(self):
        with self.assertRaises(ValueError):
            normalized_mutual_information([0, 1], [0, 1], "harmonic")


class RandTests(unittest.TestCase):
    def test_rand_identical(self):
        labels = [0, 0, 1, 1]
        self.assertAlmostEqual(rand_index(labels, labels), 1.0)

    def test_ari_identical(self):
        labels = [0, 0, 1, 1, 2, 2]
        self.assertAlmostEqual(adjusted_rand_index(labels, labels), 1.0)

    def test_ari_relabel_invariant(self):
        a = [0, 0, 1, 1, 2, 2]
        b = [2, 2, 0, 0, 1, 1]
        self.assertAlmostEqual(adjusted_rand_index(a, b), 1.0)

    def test_ari_known_value(self):
        # Classic worked example (Hubert & Arabie style small case).
        a = [0, 0, 0, 1, 1, 1]
        b = [0, 0, 1, 1, 2, 2]
        val = adjusted_rand_index(a, b)
        self.assertGreater(val, 0.0)
        self.assertLess(val, 1.0)

    def test_ari_trivial_partitions(self):
        self.assertEqual(adjusted_rand_index([0, 0, 0], [1, 1, 1]), 1.0)


class ClusteringAccuracyTests(unittest.TestCase):
    def test_perfect_after_permutation(self):
        pred = [2, 2, 0, 0, 1, 1]
        truth = [0, 0, 1, 1, 2, 2]
        self.assertAlmostEqual(clustering_accuracy(pred, truth), 1.0)

    def test_partial(self):
        pred = [0, 0, 1, 1]
        truth = [0, 1, 1, 1]
        # best mapping: pred0->truth0 (1 correct), pred1->truth1 (2 correct) = 3/4
        self.assertAlmostEqual(clustering_accuracy(pred, truth), 0.75)

    def test_more_pred_clusters(self):
        pred = [0, 1, 2, 3]
        truth = [0, 0, 1, 1]
        acc = clustering_accuracy(pred, truth)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)


class PurityTests(unittest.TestCase):
    def test_pure(self):
        self.assertAlmostEqual(purity([0, 0, 1, 1], [0, 0, 1, 1]), 1.0)

    def test_impure(self):
        # cluster 0 has truth {0,0,1} majority 0 (2), cluster 1 has {1} -> (2+1)/4
        self.assertAlmostEqual(purity([0, 0, 0, 1], [0, 0, 1, 1]), 0.75)


if __name__ == "__main__":
    unittest.main()
