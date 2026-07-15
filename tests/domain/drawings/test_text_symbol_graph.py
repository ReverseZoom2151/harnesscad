import unittest

from harnesscad.domain.drawings import text_symbol_graph as tsg


class TestAnnotationFilter(unittest.TestCase):
    def test_counts(self):
        c = tsg.annotation_counts(["a", "a", "b"])
        self.assertEqual(c["a"], 2)
        self.assertEqual(c["b"], 1)

    def test_filter_low_frequency(self):
        labels = ["door", "door", "door", "xq7z"]
        kept = tsg.filter_annotations(labels, min_count=2)
        self.assertEqual(kept, ["door", "door", "door"])
        self.assertNotIn("xq7z", kept)


class TestBuildNodes(unittest.TestCase):
    def test_geom_and_text_nodes(self):
        nodes = tsg.build_nodes(
            geom_centers=[(0.0, 0.0), (1.0, 1.0, 0.5)],
            text_items=[(0.5, 0.5, "kitchen"), (2.0, 2.0, "rare")],
            min_count=1,
        )
        kinds = [n.kind for n in nodes]
        self.assertEqual(kinds.count("geom"), 2)
        self.assertEqual(kinds.count("text"), 2)
        self.assertEqual(nodes[1].orient, 0.5)

    def test_rare_text_dropped(self):
        nodes = tsg.build_nodes(
            geom_centers=[(0.0, 0.0)],
            text_items=[(0.5, 0.5, "kitchen"), (0.5, 0.5, "kitchen"),
                        (2.0, 2.0, "rare")],
            min_count=2,
        )
        labels = {n.label for n in nodes if n.kind == "text"}
        self.assertIn("kitchen", labels)
        self.assertNotIn("rare", labels)


class TestKnn(unittest.TestCase):
    def test_nearest_first(self):
        nodes = [
            tsg.Node("geom", 0.0, 0.0),
            tsg.Node("geom", 1.0, 0.0),
            tsg.Node("geom", 5.0, 0.0),
        ]
        nbr = tsg.knn_neighbors(nodes, k=2)
        self.assertEqual(nbr[0][0], 1)  # closest to node0 is node1
        self.assertEqual(nbr[0], [1, 2])


class TestEdgeFeatures(unittest.TestCase):
    def test_type_onehot(self):
        g = tsg.Node("geom", 0.0, 0.0)
        t = tsg.Node("text", 1.0, 0.0)
        self.assertEqual(tsg.edge_type_onehot(g, g), (1.0, 0.0, 0.0))
        self.assertEqual(tsg.edge_type_onehot(g, t), (0.0, 1.0, 0.0))
        self.assertEqual(tsg.edge_type_onehot(t, t), (0.0, 0.0, 1.0))

    def test_relation_length(self):
        a = tsg.Node("geom", 0.0, 0.0)
        b = tsg.Node("geom", 3.0, 4.0)
        e = tsg.edge_relation(a, b, diag=1.0)
        self.assertEqual(len(e), 7)
        self.assertAlmostEqual(e[0], 5.0)  # distance
        self.assertAlmostEqual(e[1], 3.0)  # dx
        self.assertAlmostEqual(e[2], 4.0)  # dy

    def test_full_feature_is_10d(self):
        nodes = [
            tsg.Node("geom", 0.0, 0.0),
            tsg.Node("text", 1.0, 0.0, label="x"),
            tsg.Node("geom", 0.0, 2.0),
        ]
        feats = tsg.type_aware_edge_features(nodes, k=2, diag=10.0)
        self.assertEqual(len(feats), 3)
        for per_node in feats:
            for j, vec in per_node:
                self.assertEqual(len(vec), 10)  # 3 type + 7 relation

    def test_deterministic(self):
        nodes = [tsg.Node("geom", float(i), 0.0) for i in range(5)]
        a = tsg.type_aware_edge_features(nodes, k=3)
        b = tsg.type_aware_edge_features(nodes, k=3)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
