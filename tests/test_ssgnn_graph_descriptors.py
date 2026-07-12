"""Tests for reconstruction.ssgnn_graph_descriptors (structural graph descriptors)."""

from __future__ import annotations

import unittest

from reconstruction.ssgnn_graph_augment import build_graph
from reconstruction.ssgnn_graph_descriptors import (
    degree_histogram,
    descriptor_vector,
    wl_kernel,
    wl_label_histogram,
    wl_refine,
    wl_similarity,
)


def _ring(n: int):
    nodes = [(float(i),) for i in range(n)]
    edges = [(i, (i + 1) % n, (1.0,)) for i in range(n)]
    return build_graph(nodes, edges)


def _path(n: int):
    nodes = [(float(i),) for i in range(n)]
    edges = [(i, i + 1, (1.0,)) for i in range(n - 1)]
    return build_graph(nodes, edges)


def _relabelled_ring(n: int):
    # Same ring topology, node ids rotated -> isomorphic graph.
    nodes = [(float(i),) for i in range(n)]
    edges = [((i + 2) % n, (i + 3) % n, (1.0,)) for i in range(n)]
    return build_graph(nodes, edges)


class DegreeHistogramTests(unittest.TestCase):
    def test_ring_all_degree_two(self):
        h = degree_histogram(_ring(5))
        self.assertAlmostEqual(sum(h), 1.0, places=9)
        # degree 2 bucket (edges (0,1,2,...): value 2 -> bucket index 2)
        self.assertAlmostEqual(h[2], 1.0, places=9)

    def test_empty_graph(self):
        g = build_graph([], [])
        h = degree_histogram(g)
        self.assertEqual(sum(h), 0.0)

    def test_length_fixed(self):
        self.assertEqual(len(degree_histogram(_ring(4))),
                         len(degree_histogram(_path(7))))


class WLRefineTests(unittest.TestCase):
    def test_history_length(self):
        hist = wl_refine(_ring(4), iterations=3)
        self.assertEqual(len(hist), 4)  # iteration 0 + 3 updates

    def test_permutation_invariant_histogram(self):
        a = wl_label_histogram(_ring(6))
        b = wl_label_histogram(_relabelled_ring(6))
        self.assertEqual(a, b)

    def test_different_topology_differs(self):
        a = wl_label_histogram(_ring(6))
        b = wl_label_histogram(_path(6))
        self.assertNotEqual(a, b)

    def test_negative_iterations(self):
        with self.assertRaises(ValueError):
            wl_refine(_ring(4), iterations=-1)

    def test_deterministic_across_calls(self):
        self.assertEqual(wl_label_histogram(_ring(5)),
                         wl_label_histogram(_ring(5)))


class WLKernelTests(unittest.TestCase):
    def test_self_kernel_positive(self):
        g = _ring(5)
        self.assertGreater(wl_kernel(g, g), 0)

    def test_isomorphic_similarity_one(self):
        s = wl_similarity(_ring(6), _relabelled_ring(6))
        self.assertAlmostEqual(s, 1.0, places=9)

    def test_similarity_bounded(self):
        s = wl_similarity(_ring(6), _path(6))
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)

    def test_ring_more_similar_to_ring(self):
        base = _ring(6)
        self.assertGreater(wl_similarity(base, _ring(6)),
                           wl_similarity(base, _path(6)))

    def test_empty_similarity_zero(self):
        empty = build_graph([], [])
        self.assertEqual(wl_similarity(empty, _ring(4)), 0.0)

    def test_kernel_symmetric(self):
        a, b = _ring(5), _path(5)
        self.assertEqual(wl_kernel(a, b), wl_kernel(b, a))


class DescriptorVectorTests(unittest.TestCase):
    def test_fixed_length(self):
        v1 = descriptor_vector(_ring(4))
        v2 = descriptor_vector(_path(9))
        self.assertEqual(len(v1), len(v2))

    def test_isomorphic_equal_vectors(self):
        self.assertEqual(descriptor_vector(_ring(6)),
                         descriptor_vector(_relabelled_ring(6)))

    def test_scalars_present(self):
        v = descriptor_vector(_ring(4))
        # last three entries are the log-scaled counts + density; all finite
        self.assertTrue(all(isinstance(x, float) for x in v[-3:]))


if __name__ == "__main__":
    unittest.main()
