"""Tests for the SSRL representation-quality evaluation protocol."""

from __future__ import annotations

import math
import unittest

from harnesscad.eval.bench.retrieval.representation_quality import (
    alignment,
    uniformity,
    knn_classify,
    knn_accuracy,
    LinearProbe,
    linear_probe_accuracy,
)

# Two well-separated clusters in 2D -> a good, linearly separable embedding.
CLUSTER_A = [(0.0, 0.0), (0.1, 0.1), (-0.1, 0.05), (0.05, -0.1)]
CLUSTER_B = [(5.0, 5.0), (5.1, 4.9), (4.9, 5.1), (5.05, 5.0)]
GOOD_X = CLUSTER_A + CLUSTER_B
GOOD_Y = ["a"] * 4 + ["b"] * 4


class AlignmentTests(unittest.TestCase):
    def test_identical_pairs_zero_alignment(self):
        pairs = [((1.0, 0.0), (1.0, 0.0)), ((0.0, 2.0), (0.0, 3.0))]
        # After L2-normalisation both members of each pair coincide.
        self.assertAlmostEqual(alignment(pairs), 0.0)

    def test_orthogonal_pair_alignment(self):
        # Normalised orthogonal unit vectors: sq-dist = 2, alpha=2 -> 2.
        pairs = [((1.0, 0.0), (0.0, 1.0))]
        self.assertAlmostEqual(alignment(pairs), 2.0)

    def test_closer_pairs_lower_alignment(self):
        close = [((1.0, 0.0), (1.0, 0.1))]
        far = [((1.0, 0.0), (0.0, 1.0))]
        self.assertLess(alignment(close), alignment(far))

    def test_requires_pairs(self):
        with self.assertRaises(ValueError):
            alignment([])


class UniformityTests(unittest.TestCase):
    def test_spread_lower_than_collapsed(self):
        # Four evenly spread directions vs four near-identical directions.
        spread = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
        collapsed = [(1.0, 0.0), (1.0, 0.01), (1.0, -0.01), (1.0, 0.005)]
        self.assertLess(uniformity(spread), uniformity(collapsed))

    def test_deterministic(self):
        vs = [(1.0, 0.0), (0.0, 1.0), (-1.0, 1.0)]
        self.assertEqual(uniformity(vs), uniformity(vs))

    def test_requires_two(self):
        with self.assertRaises(ValueError):
            uniformity([(1.0, 0.0)])


class KnnTests(unittest.TestCase):
    def test_perfect_on_separated_clusters(self):
        acc = knn_accuracy(GOOD_X, GOOD_Y, GOOD_X, GOOD_Y, k=3)
        self.assertEqual(acc, 1.0)

    def test_query_label(self):
        preds = knn_classify(GOOD_X, GOOD_Y, [(0.02, 0.02), (5.0, 5.02)], k=3)
        self.assertEqual(preds, ["a", "b"])

    def test_k_clamped_to_train_size(self):
        preds = knn_classify(CLUSTER_A, ["a"] * 4, [(0.0, 0.0)], k=99)
        self.assertEqual(preds, ["a"])

    def test_deterministic(self):
        p1 = knn_classify(GOOD_X, GOOD_Y, GOOD_X, k=5)
        p2 = knn_classify(GOOD_X, GOOD_Y, GOOD_X, k=5)
        self.assertEqual(p1, p2)

    def test_validation(self):
        with self.assertRaises(ValueError):
            knn_classify([], [], [(0.0, 0.0)])
        with self.assertRaises(ValueError):
            knn_accuracy(GOOD_X, GOOD_Y, [], [])


class LinearProbeTests(unittest.TestCase):
    def test_fit_predict_separable(self):
        probe = LinearProbe.fit(GOOD_X, GOOD_Y, ridge=0.1)
        self.assertEqual(probe.classes, ["a", "b"])
        preds = probe.predict([(0.0, 0.0), (5.0, 5.0)])
        self.assertEqual(preds, ["a", "b"])

    def test_accuracy_perfect_on_separable(self):
        acc = linear_probe_accuracy(GOOD_X, GOOD_Y, GOOD_X, GOOD_Y, ridge=0.1)
        self.assertEqual(acc, 1.0)

    def test_accuracy_better_than_chance_on_three_classes(self):
        xs = [(0.0, 0.0), (0.1, 0.0), (5.0, 0.0), (5.1, 0.0),
              (0.0, 5.0), (0.1, 5.0)]
        ys = ["p", "p", "q", "q", "r", "r"]
        acc = linear_probe_accuracy(xs, ys, xs, ys, ridge=0.01)
        self.assertGreaterEqual(acc, 0.5)

    def test_deterministic(self):
        p1 = LinearProbe.fit(GOOD_X, GOOD_Y).predict(GOOD_X)
        p2 = LinearProbe.fit(GOOD_X, GOOD_Y).predict(GOOD_X)
        self.assertEqual(p1, p2)

    def test_validation(self):
        with self.assertRaises(ValueError):
            LinearProbe.fit([], [])
        with self.assertRaises(ValueError):
            LinearProbe.fit(GOOD_X, GOOD_Y, ridge=-1.0)
        with self.assertRaises(ValueError):
            LinearProbe.fit(GOOD_X, GOOD_Y[:3])


class ProbeVsCollapseTests(unittest.TestCase):
    """A better representation should probe higher -- the protocol's purpose."""

    def test_separable_beats_collapsed(self):
        good_acc = linear_probe_accuracy(GOOD_X, GOOD_Y, GOOD_X, GOOD_Y, ridge=0.1)
        # Collapsed embedding: both classes on top of each other -> not separable.
        collapsed = [(0.0, 0.0)] * 8
        bad_acc = linear_probe_accuracy(collapsed, GOOD_Y, collapsed, GOOD_Y,
                                        ridge=0.1)
        self.assertGreater(good_acc, bad_acc)


if __name__ == "__main__":
    unittest.main()
