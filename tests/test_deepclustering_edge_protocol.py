"""Tests for bench.deepclustering_edge_protocol."""

import unittest

from harnesscad.eval.bench.retrieval.deepclustering_edge_protocol import (
    balanced_accuracy,
    edge_accuracy,
    edge_confusion_matrix,
    known_edges,
    partition_to_edges,
    partition_to_similarity_matrix,
)


class PartitionToEdgesTests(unittest.TestCase):
    def test_edges(self):
        edges = partition_to_edges([0, 0, 1])
        self.assertEqual(edges[(0, 1)], 1)   # same cluster
        self.assertEqual(edges[(0, 2)], -1)  # different
        self.assertEqual(edges[(1, 2)], -1)

    def test_matrix_symmetric(self):
        m = partition_to_similarity_matrix([0, 1, 0])
        self.assertEqual(m[0][2], 1)
        self.assertEqual(m[0][1], -1)
        self.assertEqual(m[0][0], 1)
        for i in range(3):
            for j in range(3):
                self.assertEqual(m[i][j], m[j][i])


class KnownEdgesTests(unittest.TestCase):
    def test_drops_unknown(self):
        ref = {(0, 1): 1, (0, 2): 0, (1, 2): -1}
        self.assertEqual(known_edges(ref), {(0, 1): 1, (1, 2): -1})


class EdgeAccuracyTests(unittest.TestCase):
    def test_perfect(self):
        labels = [0, 0, 1, 1]
        edges = partition_to_edges(labels)
        self.assertAlmostEqual(edge_accuracy(edges, edges), 1.0)

    def test_only_known_edges_scored(self):
        pred = partition_to_edges([0, 0, 1, 1])
        ref = {(0, 1): 1, (2, 3): 1, (0, 2): 0}  # (0,2) unknown -> ignored
        self.assertAlmostEqual(edge_accuracy(pred, ref), 1.0)

    def test_half_wrong(self):
        pred = {(0, 1): 1, (0, 2): 1}
        ref = {(0, 1): 1, (0, 2): -1}
        self.assertAlmostEqual(edge_accuracy(pred, ref), 0.5)

    def test_no_known_raises(self):
        with self.assertRaises(ValueError):
            edge_accuracy({(0, 1): 1}, {(0, 1): 0})

    def test_missing_prediction_raises(self):
        with self.assertRaises(ValueError):
            edge_accuracy({}, {(0, 1): 1})


class ConfusionBalancedTests(unittest.TestCase):
    def test_confusion(self):
        pred = {(0, 1): 1, (0, 2): 1, (1, 2): -1, (0, 3): -1}
        ref = {(0, 1): 1, (0, 2): -1, (1, 2): -1, (0, 3): 1}
        cm = edge_confusion_matrix(pred, ref)
        self.assertEqual(cm, {"tp": 1, "fp": 1, "fn": 1, "tn": 1})

    def test_balanced_accuracy_perfect(self):
        pred = {(0, 1): 1, (0, 2): -1}
        ref = {(0, 1): 1, (0, 2): -1}
        self.assertAlmostEqual(balanced_accuracy(pred, ref), 1.0)

    def test_balanced_accuracy_imbalance(self):
        # 1 positive (correct), 3 negatives (2 correct, 1 wrong).
        pred = {(0, 1): 1, (0, 2): -1, (0, 3): -1, (0, 4): 1}
        ref = {(0, 1): 1, (0, 2): -1, (0, 3): -1, (0, 4): -1}
        # TPR = 1/1 = 1.0 ; TNR = 2/3 ; balanced = 0.5*(1 + 2/3)
        self.assertAlmostEqual(balanced_accuracy(pred, ref), 0.5 * (1.0 + 2.0 / 3.0))

    def test_only_negatives(self):
        pred = {(0, 1): -1, (0, 2): 1}
        ref = {(0, 1): -1, (0, 2): -1}
        # no positives -> returns TNR = 1/2
        self.assertAlmostEqual(balanced_accuracy(pred, ref), 0.5)


if __name__ == "__main__":
    unittest.main()
