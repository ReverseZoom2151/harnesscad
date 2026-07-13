"""Tests for bench/contrastcad_latent_metrics.py — latent-space quality metrics."""

import math
import unittest

from harnesscad.eval.bench.contrastcad_latent_metrics import (
    average_set_distance,
    distance_matrix,
    euclidean_distance,
    kmeans,
    silhouette_coefficient,
    sse,
)


class TestEuclideanDistance(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(euclidean_distance([0, 0], [3, 4]), 5.0)

    def test_zero(self):
        self.assertAlmostEqual(euclidean_distance([1, 2, 3], [1, 2, 3]), 0.0)

    def test_dim_mismatch(self):
        with self.assertRaises(ValueError):
            euclidean_distance([1], [1, 2])


class TestDistanceMatrix(unittest.TestCase):
    def test_symmetric_zero_diagonal(self):
        pts = [[0, 0], [3, 4], [6, 8]]
        m = distance_matrix(pts)
        for i in range(3):
            self.assertEqual(m[i][i], 0.0)
            for j in range(3):
                self.assertAlmostEqual(m[i][j], m[j][i])


class TestAverageSetDistance(unittest.TestCase):
    def test_mean(self):
        d = average_set_distance([0, 0], [[3, 4], [6, 8]])
        self.assertAlmostEqual(d, (5.0 + 10.0) / 2)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            average_set_distance([0, 0], [])


class TestSSE(unittest.TestCase):
    def test_single_cluster_centroid(self):
        # points symmetric about centroid (1,0): sse = 1+1 = 2
        pts = [[0, 0], [2, 0]]
        self.assertAlmostEqual(sse(pts, [0, 0]), 2.0)

    def test_tighter_clusters_lower_sse(self):
        pts = [[0, 0], [0.1, 0], [10, 0], [10.1, 0]]
        tight = sse(pts, [0, 0, 1, 1])
        loose = sse(pts, [0, 1, 0, 1])
        self.assertLess(tight, loose)


class TestSilhouette(unittest.TestCase):
    def test_well_separated_high(self):
        pts = [[0, 0], [0, 1], [10, 0], [10, 1]]
        labels = [0, 0, 1, 1]
        self.assertGreater(silhouette_coefficient(pts, labels), 0.8)

    def test_range(self):
        pts = [[0, 0], [1, 1], [2, 2], [3, 3]]
        labels = [0, 1, 0, 1]
        s = silhouette_coefficient(pts, labels)
        self.assertTrue(-1.0 <= s <= 1.0)

    def test_needs_two_clusters(self):
        with self.assertRaises(ValueError):
            silhouette_coefficient([[0, 0], [1, 1]], [0, 0])


class TestKMeans(unittest.TestCase):
    def setUp(self):
        self.pts = [[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]]

    def test_deterministic(self):
        a, _ = kmeans(self.pts, 2, seed=1)
        b, _ = kmeans(self.pts, 2, seed=1)
        self.assertEqual(a, b)

    def test_recovers_two_clusters(self):
        labels, _ = kmeans(self.pts, 2, seed=1)
        # first three points share a label distinct from last three
        self.assertEqual(len(set(labels[:3])), 1)
        self.assertEqual(len(set(labels[3:])), 1)
        self.assertNotEqual(labels[0], labels[3])

    def test_k_too_large(self):
        with self.assertRaises(ValueError):
            kmeans([[0, 0]], 2, seed=1)

    def test_labels_feed_silhouette(self):
        labels, _ = kmeans(self.pts, 2, seed=1)
        s = silhouette_coefficient(self.pts, labels)
        self.assertGreater(s, 0.5)


if __name__ == "__main__":
    unittest.main()
