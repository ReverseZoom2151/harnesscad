"""Tests for METRO-style hierarchical mesh coarsening / upsampling + positional encoding."""

import math
import unittest

from harnesscad.domain.geometry.mesh import hierarchical_sampler as hs


# A unit square split into two triangles (4 vertices).
SQUARE_V = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
SQUARE_F = [(0, 1, 2), (0, 2, 3)]


class CoarsenTest(unittest.TestCase):
    def test_reaches_target_cluster_count(self):
        h = hs.coarsen(SQUARE_V, SQUARE_F, target=2)
        self.assertEqual(h.num_coarse, 2)
        self.assertEqual(h.num_fine, 4)
        # every fine vertex mapped to a valid coarse id
        self.assertTrue(all(0 <= c < 2 for c in h.vertex_map))

    def test_down_operator_averages_clusters(self):
        h = hs.coarsen(SQUARE_V, SQUARE_F, target=2)
        for row in h.down:
            self.assertAlmostEqual(sum(w for _, w in row), 1.0)
        # coarse vertex is the centroid of its cluster members
        for c, members in enumerate(h.clusters):
            cx = sum(SQUARE_V[m][0] for m in members) / len(members)
            self.assertAlmostEqual(h.coarse_vertices[c][0], cx)

    def test_deterministic(self):
        a = hs.coarsen(SQUARE_V, SQUARE_F, target=2)
        b = hs.coarsen(SQUARE_V, SQUARE_F, target=2)
        self.assertEqual(a.vertex_map, b.vertex_map)
        self.assertEqual(a.coarse_vertices, b.coarse_vertices)

    def test_target_bounds(self):
        with self.assertRaises(ValueError):
            hs.coarsen(SQUARE_V, SQUARE_F, target=0)
        with self.assertRaises(ValueError):
            hs.coarsen(SQUARE_V, SQUARE_F, target=99)

    def test_up_weights_normalised(self):
        h = hs.coarsen(SQUARE_V, SQUARE_F, target=2)
        for row in h.up:
            self.assertAlmostEqual(sum(w for _, w in row), 1.0)


class ApplyOperatorTest(unittest.TestCase):
    def test_downsample_signal(self):
        h = hs.coarsen(SQUARE_V, SQUARE_F, target=2)
        # push the vertex coordinates down: should equal coarse vertices
        down = hs.apply_operator(h.down, SQUARE_V)
        for c in range(h.num_coarse):
            for d in range(3):
                self.assertAlmostEqual(down[c][d], h.coarse_vertices[c][d])

    def test_up_then_shapes(self):
        h = hs.coarsen(SQUARE_V, SQUARE_F, target=2)
        up = hs.apply_operator(h.up, list(h.coarse_vertices))
        self.assertEqual(len(up), h.num_fine)
        self.assertEqual(len(up[0]), 3)


class PositionalEncodingTest(unittest.TestCase):
    def test_template_encoding_is_coordinate(self):
        pe = hs.template_positional_encoding(SQUARE_V)
        self.assertEqual(pe[1], (1.0, 0.0, 0.0))

    def test_sinusoidal_dimensions(self):
        pe = hs.sinusoidal_positional_encoding(SQUARE_V, num_freqs=4, include_input=True)
        # per coord: 1 raw + 2 per freq -> 3 * (1 + 2*4) = 27
        self.assertEqual(len(pe[0]), 3 * (1 + 2 * 4))

    def test_sinusoidal_values(self):
        pe = hs.sinusoidal_positional_encoding([(0.5, 0.0, 0.0)], num_freqs=1, include_input=False)
        # coord 0.5: sin(0.5), cos(0.5); coord 0: sin0, cos0; coord 0: sin0, cos0
        self.assertAlmostEqual(pe[0][0], math.sin(0.5))
        self.assertAlmostEqual(pe[0][1], math.cos(0.5))

    def test_bad_num_freqs(self):
        with self.assertRaises(ValueError):
            hs.sinusoidal_positional_encoding(SQUARE_V, num_freqs=0)


if __name__ == "__main__":
    unittest.main()
