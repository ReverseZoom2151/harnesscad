"""Tests for bench.deepclustering_init_protocol."""

import unittest

from harnesscad.eval.bench.retrieval.deepclustering_init_protocol import (
    annotation_budget_fraction,
    induced_known_edges,
    oversegment,
)


class OversegmentTests(unittest.TestCase):
    def test_all_clusters_within_max(self):
        points = [(float(i), 0.0) for i in range(20)]
        clusters = oversegment(points, max_size=4, seed=0)
        self.assertTrue(all(len(c) <= 4 for c in clusters))

    def test_partition_covers_all(self):
        points = [(float(i), float(i % 3)) for i in range(15)]
        clusters = oversegment(points, max_size=3, seed=1)
        covered = sorted(i for c in clusters for i in c)
        self.assertEqual(covered, list(range(15)))
        # No item appears twice.
        flat = [i for c in clusters for i in c]
        self.assertEqual(len(flat), len(set(flat)))

    def test_deterministic(self):
        points = [(float(i), float((i * 7) % 5)) for i in range(18)]
        a = oversegment(points, max_size=4, seed=3)
        b = oversegment(points, max_size=4, seed=3)
        self.assertEqual(a, b)

    def test_small_input_single_cluster(self):
        points = [(0.0, 0.0), (1.0, 1.0)]
        clusters = oversegment(points, max_size=12, seed=0)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(sorted(clusters[0]), [0, 1])

    def test_coincident_points_still_split(self):
        # All identical -> degenerate k-means; positional halving must terminate.
        points = [(0.0, 0.0)] * 10
        clusters = oversegment(points, max_size=3, seed=0)
        self.assertTrue(all(len(c) <= 3 for c in clusters))
        covered = sorted(i for c in clusters for i in c)
        self.assertEqual(covered, list(range(10)))

    def test_bad_params(self):
        with self.assertRaises(ValueError):
            oversegment([(0.0, 0.0)], max_size=0, seed=0)
        with self.assertRaises(ValueError):
            oversegment([(0.0, 0.0)], max_size=2, seed=0, split_k=1)


class InducedEdgesTests(unittest.TestCase):
    def test_intra_cluster_only(self):
        clusters = [[0, 1, 2], [3, 4]]
        edges = induced_known_edges(clusters)
        self.assertIn((0, 1), edges)
        self.assertIn((3, 4), edges)
        self.assertNotIn((2, 3), edges)  # cross-cluster -> unknown

    def test_with_truth(self):
        clusters = [[0, 1, 2]]
        truth = [5, 5, 9]
        edges = induced_known_edges(clusters, truth)
        self.assertEqual(edges[(0, 1)], 1)   # same truth label
        self.assertEqual(edges[(0, 2)], -1)  # different truth label


class BudgetTests(unittest.TestCase):
    def test_fraction(self):
        # two clusters of size 12 out of 100 items.
        clusters = [list(range(12)), list(range(12, 24))]
        frac = annotation_budget_fraction(clusters, 100)
        expected = (2 * (12 * 11 // 2)) / (100 * 99 // 2)
        self.assertAlmostEqual(frac, expected)

    def test_full_annotation_is_one(self):
        clusters = [[0, 1, 2, 3]]
        self.assertAlmostEqual(annotation_budget_fraction(clusters, 4), 1.0)

    def test_bad_total(self):
        with self.assertRaises(ValueError):
            annotation_budget_fraction([[0]], 1)


if __name__ == "__main__":
    unittest.main()
