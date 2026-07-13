"""Tests for bench.ssgnn_graph_contrastive (graph-level NT-Xent pretext)."""

from __future__ import annotations

import math
import unittest

from harnesscad.eval.bench.retrieval.ssgnn_graph_contrastive import (
    build_pretext_views,
    graph_nt_xent,
    graph_nt_xent_anchor,
    pretext_loss,
    similarity_matrix,
)
from harnesscad.domain.reconstruction.recognize.ssgnn_graph_augment import build_graph
from harnesscad.domain.reconstruction.recognize.ssgnn_graph_descriptors import descriptor_vector


def _ring(n: int):
    nodes = [(1.0, 1.0, 1.0)] * n
    edges = [(i, (i + 1) % n, (1.0,)) for i in range(n)]
    return build_graph(nodes, edges)


class AnchorLossTests(unittest.TestCase):
    def test_perfect_alignment_low_loss(self):
        # z'_n == z''_n and orthogonal-ish negatives -> small loss.
        first = [[1.0, 0.0], [0.0, 1.0]]
        second = [[1.0, 0.0], [0.0, 1.0]]
        loss = graph_nt_xent_anchor(first, second, 0, temperature=0.5)
        self.assertGreater(loss, 0.0)

    def test_positive_lowers_loss_vs_confusable(self):
        aligned_first = [[1.0, 0.0], [0.0, 1.0]]
        aligned_second = [[1.0, 0.0], [0.0, 1.0]]
        confusable_second = [[1.0, 0.0], [1.0, 0.0]]
        good = graph_nt_xent_anchor(aligned_first, aligned_second, 0, 0.5)
        bad = graph_nt_xent_anchor(aligned_first, confusable_second, 0, 0.5)
        self.assertLess(good, bad)

    def test_include_positive_flag_changes_value(self):
        first = [[1.0, 0.0], [0.0, 1.0]]
        second = [[1.0, 0.0], [0.0, 1.0]]
        with_pos = graph_nt_xent_anchor(first, second, 0, 0.5, include_positive=True)
        without = graph_nt_xent_anchor(first, second, 0, 0.5, include_positive=False)
        self.assertNotAlmostEqual(with_pos, without)

    def test_bad_temperature(self):
        with self.assertRaises(ValueError):
            graph_nt_xent_anchor([[1.0]], [[1.0]], 0, temperature=0.0)

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            graph_nt_xent_anchor([[1.0]], [[1.0], [2.0]], 0)

    def test_index_out_of_range(self):
        with self.assertRaises(ValueError):
            graph_nt_xent_anchor([[1.0]], [[1.0]], 5)


class BatchLossTests(unittest.TestCase):
    def test_mean_over_batch(self):
        first = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        second = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        each = [graph_nt_xent_anchor(first, second, i, 0.5) for i in range(3)]
        self.assertAlmostEqual(graph_nt_xent(first, second, 0.5),
                               sum(each) / 3, places=9)

    def test_empty_batch(self):
        with self.assertRaises(ValueError):
            graph_nt_xent([], [])

    def test_symmetric_averages_directions(self):
        first = [[1.0, 0.0], [0.2, 1.0]]
        second = [[0.9, 0.1], [0.0, 1.0]]
        fwd = graph_nt_xent(first, second, 0.5)
        bwd = graph_nt_xent(second, first, 0.5)
        sym = graph_nt_xent(first, second, 0.5, symmetric=True)
        self.assertAlmostEqual(sym, (fwd + bwd) / 2, places=9)

    def test_similarity_matrix_shape(self):
        m = similarity_matrix([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(len(m), 2)
        self.assertEqual(len(m[0]), 2)
        self.assertAlmostEqual(m[0][0], 1.0, places=9)


class PretextIntegrationTests(unittest.TestCase):
    def test_build_views_deterministic(self):
        graphs = [_ring(5), _ring(6), _ring(7)]
        a1, a2 = build_pretext_views(graphs, descriptor_vector, seed=3)
        b1, b2 = build_pretext_views(graphs, descriptor_vector, seed=3)
        self.assertEqual(a1, b1)
        self.assertEqual(a2, b2)

    def test_view_counts_match_batch(self):
        graphs = [_ring(5), _ring(6)]
        first, second = build_pretext_views(graphs, descriptor_vector, seed=1)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 2)

    def test_pretext_loss_finite_and_deterministic(self):
        graphs = [_ring(5), _ring(6), _ring(8)]
        l1 = pretext_loss(graphs, descriptor_vector, seed=0)
        l2 = pretext_loss(graphs, descriptor_vector, seed=0)
        self.assertTrue(math.isfinite(l1))
        self.assertAlmostEqual(l1, l2, places=12)


if __name__ == "__main__":
    unittest.main()
