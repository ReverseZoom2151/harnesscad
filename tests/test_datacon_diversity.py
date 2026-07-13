"""Tests for bench/datacon_diversity.py numeric diversity/novelty/coverage metrics."""
import unittest

from harnesscad.eval.bench.generative.datacon_diversity import (
    euclidean,
    pairwise_diversity,
    nearest_neighbor_distances,
    coverage,
    grid_occupancy,
    novelty_vs_reference,
    novelty_score,
    is_novel,
    diversity_report,
    simpson_diversity,
    shannon_entropy,
)


class TestDataconDiversity(unittest.TestCase):
    def test_euclidean_dim_mismatch(self):
        with self.assertRaises(ValueError):
            euclidean((0.0, 0.0), (1.0,))

    def test_pairwise_diversity_known(self):
        # points (0,0),(0,2),(2,0): distances 2, 2, sqrt(8) -> mean.
        pts = [(0.0, 0.0), (0.0, 2.0), (2.0, 0.0)]
        expected = (2.0 + 2.0 + (8.0 ** 0.5)) / 3.0
        self.assertAlmostEqual(pairwise_diversity(pts), expected)

    def test_pairwise_diversity_single(self):
        self.assertEqual(pairwise_diversity([(1.0, 2.0)]), 0.0)

    def test_nearest_neighbor_distances(self):
        pts = [(0.0, 0.0), (0.0, 2.0), (2.0, 0.0)]
        nn = nearest_neighbor_distances(pts)
        self.assertAlmostEqual(nn[0], 2.0)
        self.assertAlmostEqual(nn[1], 2.0)
        self.assertAlmostEqual(nn[2], 2.0)

    def test_coverage_identical_low(self):
        pts = [(1.0, 1.0)] * 4
        # all land in one cell -> 1/4
        self.assertAlmostEqual(coverage(pts, bins=8), 0.25)

    def test_coverage_spread_high(self):
        pts = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)]
        self.assertAlmostEqual(coverage(pts, bins=8), 1.0)

    def test_grid_occupancy(self):
        pts = [(0.0, 0.0), (1.0, 1.0)]
        occ, total = grid_occupancy(pts, [(0.0, 1.0), (0.0, 1.0)], bins=4)
        self.assertEqual(occ, 2)
        self.assertEqual(total, 16)

    def test_grid_occupancy_too_large(self):
        with self.assertRaises(ValueError):
            grid_occupancy([(0.0,) * 10], [(0.0, 1.0)] * 10, bins=8)

    def test_novelty_vs_reference(self):
        reference = [(0.0, 0.0), (1.0, 0.0)]
        near = (0.0, 0.5)
        far = (100.0, 100.0)
        nov = novelty_vs_reference([near, far], reference, k=1)
        self.assertLess(nov[0], nov[1])

    def test_novelty_empty_reference(self):
        with self.assertRaises(ValueError):
            novelty_vs_reference([(1.0,)], [])

    def test_is_novel_threshold(self):
        reference = [(0.0, 0.0)]
        far = (10.0, 0.0)
        near = (0.1, 0.0)
        self.assertTrue(is_novel(far, reference, threshold=5.0))
        self.assertFalse(is_novel(near, reference, threshold=5.0))
        self.assertAlmostEqual(novelty_score(far, reference), 10.0)

    def test_simpson_and_shannon(self):
        uniform = ["five-spoke", "multispoke", "mesh", "minimalist"]
        skewed = ["mesh", "mesh", "mesh", "minimalist"]
        single = ["mesh", "mesh", "mesh"]
        self.assertGreater(simpson_diversity(uniform), simpson_diversity(skewed))
        self.assertGreater(shannon_entropy(uniform), shannon_entropy(skewed))
        self.assertAlmostEqual(shannon_entropy(single), 0.0)
        self.assertAlmostEqual(simpson_diversity(single), 0.0)
        self.assertAlmostEqual(shannon_entropy(uniform), 2.0)

    def test_diversity_report_keys(self):
        pts = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
        rep = diversity_report(pts)
        for key in ("n", "pairwise_diversity", "mean_nn_distance",
                    "min_nn_distance", "coverage"):
            self.assertIn(key, rep)
        self.assertNotIn("mean_novelty", rep)
        rep2 = diversity_report(pts, reference=[(0.0, 0.0)])
        self.assertIn("mean_novelty", rep2)
        self.assertIn("max_novelty", rep2)
        self.assertEqual(rep2["n"], 3)


if __name__ == "__main__":
    unittest.main()
