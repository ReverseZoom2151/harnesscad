"""Tests for bench.deepclustering_internal_indices."""

import math
import unittest

from harnesscad.eval.bench.deepclustering_internal_indices import (
    calinski_harabasz_index,
    davies_bouldin_index,
    dunn_index,
)


# Two tight, well-separated clusters in 2D.
GOOD_POINTS = [(0.0, 0.0), (0.1, 0.0), (0.0, 0.1),
               (10.0, 10.0), (10.1, 10.0), (10.0, 10.1)]
GOOD_LABELS = [0, 0, 0, 1, 1, 1]

# Same points but scrambled labels -> worse clustering.
BAD_LABELS = [0, 1, 0, 1, 0, 1]


class DaviesBouldinTests(unittest.TestCase):
    def test_lower_for_good_clustering(self):
        good = davies_bouldin_index(GOOD_POINTS, GOOD_LABELS)
        bad = davies_bouldin_index(GOOD_POINTS, BAD_LABELS)
        self.assertLess(good, bad)

    def test_positive(self):
        self.assertGreater(davies_bouldin_index(GOOD_POINTS, GOOD_LABELS), 0.0)

    def test_needs_two_clusters(self):
        with self.assertRaises(ValueError):
            davies_bouldin_index([(0.0, 0.0), (1.0, 1.0)], [0, 0])

    def test_coincident_centroids_raise(self):
        pts = [(0.0, 0.0), (0.0, 0.0)]
        with self.assertRaises(ValueError):
            davies_bouldin_index(pts, [0, 1])


class CalinskiHarabaszTests(unittest.TestCase):
    def test_higher_for_good_clustering(self):
        good = calinski_harabasz_index(GOOD_POINTS, GOOD_LABELS)
        bad = calinski_harabasz_index(GOOD_POINTS, BAD_LABELS)
        self.assertGreater(good, bad)

    def test_perfect_separation_infinite(self):
        pts = [(0.0, 0.0), (0.0, 0.0), (5.0, 5.0), (5.0, 5.0)]
        self.assertEqual(calinski_harabasz_index(pts, [0, 0, 1, 1]), math.inf)

    def test_needs_more_points_than_clusters(self):
        with self.assertRaises(ValueError):
            calinski_harabasz_index([(0.0, 0.0), (1.0, 0.0)], [0, 1])


class DunnTests(unittest.TestCase):
    def test_higher_for_good_clustering(self):
        good = dunn_index(GOOD_POINTS, GOOD_LABELS)
        bad = dunn_index(GOOD_POINTS, BAD_LABELS)
        self.assertGreater(good, bad)

    def test_all_singletons_infinite(self):
        pts = [(0.0, 0.0), (1.0, 0.0)]
        self.assertEqual(dunn_index(pts, [0, 1]), math.inf)

    def test_value(self):
        # two clusters: within-diameter 1, between-min 10 -> dunn = 10
        pts = [(0.0, 0.0), (1.0, 0.0), (11.0, 0.0), (12.0, 0.0)]
        self.assertAlmostEqual(dunn_index(pts, [0, 0, 1, 1]), 10.0)


if __name__ == "__main__":
    unittest.main()
