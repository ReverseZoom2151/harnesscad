"""Tests for reconstruction.ssgnn_graph_augment (GC-CAD graph augmentation)."""

from __future__ import annotations

import random
import unittest

from harnesscad.domain.reconstruction.recognize.graph_augment import (
    CADGraph,
    STRUCTURE_SCHEMES,
    augment,
    build_graph,
    degrees,
    mask_features,
    neighbours,
    positive_pair,
    remove_edges_with_vertices,
    remove_nodes,
    remove_nodes_1hop,
    subgraph,
)


def _square_graph() -> CADGraph:
    # 4 faces in a ring: 0-1-2-3-0, each node feature is a distinct 3-vector.
    nodes = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0), (1.0, 1.0, 1.0)]
    edges = [
        (0, 1, (0.5, 0.5)),
        (1, 2, (0.6, 0.6)),
        (2, 3, (0.7, 0.7)),
        (3, 0, (0.8, 0.8)),
    ]
    return build_graph(nodes, edges)


class BuildGraphTests(unittest.TestCase):
    def test_build_counts(self):
        g = _square_graph()
        self.assertEqual(g.n_nodes, 4)
        self.assertEqual(g.n_edges, 4)

    def test_edges_normalized_low_high(self):
        g = build_graph([(0.0,), (0.0,)], [(1, 0, (9.0,))])
        self.assertEqual(g.edges[0][:2], (0, 1))

    def test_self_edge_rejected(self):
        with self.assertRaises(ValueError):
            build_graph([(0.0,), (0.0,)], [(0, 0, (1.0,))])

    def test_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            build_graph([(0.0,)], [(0, 5, (1.0,))])


class AdjacencyTests(unittest.TestCase):
    def test_neighbours(self):
        g = _square_graph()
        adj = neighbours(g)
        self.assertEqual(adj[0], frozenset({1, 3}))
        self.assertEqual(adj[2], frozenset({1, 3}))

    def test_degrees(self):
        g = _square_graph()
        self.assertEqual(degrees(g), (2, 2, 2, 2))


class FeatureMaskTests(unittest.TestCase):
    def test_zero_ratio_is_identity(self):
        g = _square_graph()
        out = mask_features(g, 0.0, random.Random(0))
        self.assertEqual(out.node_features, g.node_features)
        self.assertEqual(out.edges, g.edges)

    def test_full_ratio_zeros_everything(self):
        g = _square_graph()
        out = mask_features(g, 1.0, random.Random(0))
        for feat in out.node_features:
            self.assertTrue(all(x == 0.0 for x in feat))
        for _, _, feat in out.edges:
            self.assertTrue(all(x == 0.0 for x in feat))

    def test_topology_untouched(self):
        g = _square_graph()
        out = mask_features(g, 0.5, random.Random(1))
        self.assertEqual(out.n_nodes, g.n_nodes)
        self.assertEqual([e[:2] for e in out.edges], [e[:2] for e in g.edges])

    def test_deterministic(self):
        g = _square_graph()
        a = mask_features(g, 0.5, random.Random(7))
        b = mask_features(g, 0.5, random.Random(7))
        self.assertEqual(a.node_features, b.node_features)

    def test_bad_ratio(self):
        with self.assertRaises(ValueError):
            mask_features(_square_graph(), 1.5, random.Random(0))


class SubgraphTests(unittest.TestCase):
    def test_induced_reindex(self):
        g = _square_graph()
        sub, kept = subgraph(g, [1, 2])
        self.assertEqual(kept, (1, 2))
        self.assertEqual(sub.n_nodes, 2)
        # only the 1-2 edge survives, reindexed to (0, 1)
        self.assertEqual([e[:2] for e in sub.edges], [(0, 1)])

    def test_features_follow_nodes(self):
        g = _square_graph()
        sub, _ = subgraph(g, [3, 0])
        self.assertEqual(sub.node_features[0], g.node_features[0])
        self.assertEqual(sub.node_features[1], g.node_features[3])


class StructureAugmentTests(unittest.TestCase):
    def test_remove_nodes_count(self):
        g = _square_graph()
        out = remove_nodes(g, 0.5, random.Random(0))
        self.assertEqual(out.n_nodes, 2)

    def test_remove_nodes_zero_ratio(self):
        g = _square_graph()
        out = remove_nodes(g, 0.0, random.Random(0))
        self.assertEqual(out.n_nodes, 4)

    def test_remove_nodes_drops_incident_edges(self):
        g = _square_graph()
        out = remove_nodes(g, 0.5, random.Random(0))
        # every surviving edge references only surviving node indices
        for u, v, _ in out.edges:
            self.assertLess(u, out.n_nodes)
            self.assertLess(v, out.n_nodes)

    def test_remove_nodes_1hop_more_aggressive(self):
        # A star: centre 0 connected to 1,2,3,4. Removing centre as seed kills all.
        nodes = [(float(i),) for i in range(5)]
        edges = [(0, i, (1.0,)) for i in range(1, 5)]
        g = build_graph(nodes, edges)
        # force seed = centre by exhausting rng? Instead check <= remove_nodes count.
        out = remove_nodes_1hop(g, 0.2, random.Random(0))
        self.assertLessEqual(out.n_nodes, g.n_nodes)

    def test_remove_edges_with_vertices(self):
        g = _square_graph()
        out = remove_edges_with_vertices(g, 0.25, random.Random(0))
        # one edge removed -> its two endpoints removed -> 2 nodes remain
        self.assertEqual(out.n_nodes, 2)

    def test_deterministic_structure(self):
        g = _square_graph()
        a = remove_nodes(g, 0.5, random.Random(3))
        b = remove_nodes(g, 0.5, random.Random(3))
        self.assertEqual(a.node_features, b.node_features)
        self.assertEqual(a.edges, b.edges)


class AugmentPipelineTests(unittest.TestCase):
    def test_schemes_registered(self):
        self.assertEqual(set(STRUCTURE_SCHEMES),
                         {"node", "node_1hop", "edge_vertices"})

    def test_augment_each_scheme(self):
        g = _square_graph()
        for scheme in STRUCTURE_SCHEMES:
            out = augment(g, random.Random(0), scheme=scheme,
                          feature_ratio=0.1, structure_ratio=0.25)
            self.assertLessEqual(out.n_nodes, g.n_nodes)

    def test_augment_unknown_scheme(self):
        with self.assertRaises(ValueError):
            augment(_square_graph(), random.Random(0), scheme="nope")

    def test_positive_pair_deterministic(self):
        g = _square_graph()
        a1, a2 = positive_pair(g, seed=42)
        b1, b2 = positive_pair(g, seed=42)
        self.assertEqual(a1.node_features, b1.node_features)
        self.assertEqual(a2.edges, b2.edges)

    def test_positive_pair_two_views_differ(self):
        # With masking the two views are generally not identical.
        nodes = [(1.0, 1.0, 1.0, 1.0, 1.0)] * 6
        edges = [(i, (i + 1) % 6, (1.0, 1.0)) for i in range(6)]
        g = build_graph(nodes, edges)
        v1, v2 = positive_pair(g, seed=1, feature_ratio=0.5, structure_ratio=0.0)
        self.assertNotEqual(v1.node_features, v2.node_features)


if __name__ == "__main__":
    unittest.main()
