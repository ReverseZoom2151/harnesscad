"""Tests for reconstruction.recognize.uv_grid_style."""

import math
import unittest

from harnesscad.domain.reconstruction.recognize.uv_grid_style import (
    gram_vector,
    style_fingerprint,
    style_distance,
)


class TestGram(unittest.TestCase):
    def test_length_is_upper_triangle(self):
        grid = [[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]]
        g = gram_vector(grid)
        self.assertEqual(len(g), 3 * 4 // 2)  # C=3 -> 6

    def test_permutation_invariance(self):
        grid = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]]
        shuffled = [grid[i] for i in (3, 1, 0, 2)]
        self.assertEqual(gram_vector(grid), gram_vector(shuffled))

    def test_known_values(self):
        # single sample (2,0): G = [[4,0],[0,0]] -> upper tri [4,0,0]
        g = gram_vector([[2.0, 0.0]])
        self.assertEqual(g, [4.0, 0.0, 0.0])

    def test_feature_norm_unit_diagonal(self):
        grid = [[3.0, 0.0], [4.0, 0.0], [0.0, 5.0]]
        g = gram_vector(grid, normalize="feature")
        n_ch = 2
        # diagonal entries are indices 0 and 2 in the upper-tri packing
        # each channel L2-normalised then mean of squares == 1/N * (sum of
        # normalised squares) == 1/N * 1  -> but summed over all samples == 1
        self.assertAlmostEqual(g[0] * len(grid), 1.0)

    def test_instance_norm_is_covariance(self):
        grid = [[1.0, 1.0], [3.0, 3.0]]
        g = gram_vector(grid, normalize="instance")
        # centred: [-1,-1],[1,1]; cov entries all == 1
        self.assertAlmostEqual(g[0], 1.0)
        self.assertAlmostEqual(g[1], 1.0)
        self.assertAlmostEqual(g[2], 1.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            gram_vector([])

    def test_ragged_raises(self):
        with self.assertRaises(ValueError):
            gram_vector([[1.0, 2.0], [1.0]])


class TestDistance(unittest.TestCase):
    def _cube_like(self):
        return [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # layer 0
            [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]],                    # layer 1
        ]

    def test_identical_zero(self):
        fp = style_fingerprint(self._cube_like())
        self.assertAlmostEqual(style_distance(fp, fp), 0.0)

    def test_symmetric(self):
        a = style_fingerprint(self._cube_like())
        b = style_fingerprint([
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ])
        self.assertAlmostEqual(style_distance(a, b), style_distance(b, a))

    def test_different_shapes_positive(self):
        a = style_fingerprint(self._cube_like())
        b = style_fingerprint([
            [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
            [[9.0, 0.0, 0.0], [0.0, 0.0, 9.0]],
        ])
        self.assertGreater(style_distance(a, b), 0.0)

    def test_weights_applied(self):
        a = style_fingerprint(self._cube_like())
        b = style_fingerprint([
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ])
        d_all = style_distance(a, b, weights=[1.0, 1.0])
        d_l0 = style_distance(a, b, weights=[1.0, 0.0])
        self.assertLess(d_l0, d_all)

    def test_cosine_metric(self):
        a = style_fingerprint(self._cube_like())
        # scaled copy: cosine distance should be ~0 (same direction)
        scaled = [[[2 * x for x in s] for s in layer] for layer in self._cube_like()]
        b = style_fingerprint(scaled)
        self.assertAlmostEqual(style_distance(a, b, metric="cosine"), 0.0, places=6)

    def test_layer_mismatch_raises(self):
        a = style_fingerprint(self._cube_like())
        b = [a[0]]
        with self.assertRaises(ValueError):
            style_distance(a, b)


if __name__ == "__main__":
    unittest.main()
