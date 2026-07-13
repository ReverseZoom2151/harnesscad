"""Tests for bench.deepclustering_algorithms."""

import unittest

from harnesscad.eval.bench.retrieval.clustering_algorithms import (
    agglomerative_clustering,
    jacobi_eigen,
    kmeans_plus_plus,
    spectral_clustering,
)
from harnesscad.eval.bench.retrieval.clustering_external_indices import clustering_accuracy


# Three well-separated blobs in 2D.
BLOBS = [
    (0.0, 0.0), (0.2, 0.1), (0.1, 0.2),
    (10.0, 0.0), (10.2, 0.1), (10.1, 0.2),
    (5.0, 10.0), (5.2, 10.1), (5.1, 10.2),
]
TRUTH = [0, 0, 0, 1, 1, 1, 2, 2, 2]


class KMeansPlusPlusTests(unittest.TestCase):
    def test_recovers_blobs(self):
        labels, centroids = kmeans_plus_plus(BLOBS, 3, seed=0)
        self.assertEqual(len(centroids), 3)
        self.assertAlmostEqual(clustering_accuracy(labels, TRUTH), 1.0)

    def test_deterministic(self):
        a, _ = kmeans_plus_plus(BLOBS, 3, seed=7)
        b, _ = kmeans_plus_plus(BLOBS, 3, seed=7)
        self.assertEqual(a, b)

    def test_bad_k(self):
        with self.assertRaises(ValueError):
            kmeans_plus_plus(BLOBS, 0, seed=0)
        with self.assertRaises(ValueError):
            kmeans_plus_plus(BLOBS, 100, seed=0)


class AgglomerativeTests(unittest.TestCase):
    def test_recovers_blobs_average(self):
        labels = agglomerative_clustering(3, points=BLOBS, linkage="average")
        self.assertAlmostEqual(clustering_accuracy(labels, TRUTH), 1.0)

    def test_single_and_complete(self):
        for linkage in ("single", "complete"):
            labels = agglomerative_clustering(3, points=BLOBS, linkage=linkage)
            self.assertEqual(len(set(labels)), 3)

    def test_from_distance_matrix(self):
        dist = [[0.0, 1.0, 9.0], [1.0, 0.0, 9.0], [9.0, 9.0, 0.0]]
        labels = agglomerative_clustering(2, distances=dist, linkage="single")
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])

    def test_requires_one_input(self):
        with self.assertRaises(ValueError):
            agglomerative_clustering(2, points=BLOBS, distances=[[0.0]])
        with self.assertRaises(ValueError):
            agglomerative_clustering(2)


class JacobiEigenTests(unittest.TestCase):
    def test_diagonal(self):
        vals, _ = jacobi_eigen([[2.0, 0.0], [0.0, 5.0]])
        self.assertAlmostEqual(vals[0], 2.0)
        self.assertAlmostEqual(vals[1], 5.0)

    def test_known_symmetric(self):
        # [[2,1],[1,2]] has eigenvalues 1 and 3.
        vals, _ = jacobi_eigen([[2.0, 1.0], [1.0, 2.0]])
        self.assertAlmostEqual(vals[0], 1.0, places=6)
        self.assertAlmostEqual(vals[1], 3.0, places=6)

    def test_eigenvector_relation(self):
        m = [[2.0, 1.0], [1.0, 2.0]]
        vals, vecs = jacobi_eigen(m)
        # A v = lambda v for the first eigenpair
        v = [vecs[0][0], vecs[1][0]]
        av = [m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]]
        self.assertAlmostEqual(av[0], vals[0] * v[0], places=6)
        self.assertAlmostEqual(av[1], vals[0] * v[1], places=6)


class SpectralTests(unittest.TestCase):
    def test_two_block_affinity(self):
        # Block-diagonal affinity: {0,1,2} strongly linked, {3,4,5} strongly linked.
        n = 6
        aff = [[0.0] * n for _ in range(n)]
        blocks = [[0, 1, 2], [3, 4, 5]]
        for block in blocks:
            for i in block:
                for j in block:
                    if i != j:
                        aff[i][j] = 1.0
        labels = spectral_clustering(aff, 2, seed=0)
        # Nodes in the same block share a label.
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[1], labels[2])
        self.assertEqual(labels[3], labels[4])
        self.assertNotEqual(labels[0], labels[3])

    def test_deterministic(self):
        n = 4
        aff = [[0.0, 1.0, 0.0, 0.0],
               [1.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 1.0],
               [0.0, 0.0, 1.0, 0.0]]
        self.assertEqual(spectral_clustering(aff, 2, seed=1),
                         spectral_clustering(aff, 2, seed=1))

    def test_bad_k(self):
        with self.assertRaises(ValueError):
            spectral_clustering([[0.0]], 5, seed=0)


if __name__ == "__main__":
    unittest.main()
