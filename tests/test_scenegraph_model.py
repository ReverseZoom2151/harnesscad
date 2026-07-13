"""Tests for reconstruction.scenegraph_model."""

import unittest

from harnesscad.domain.reconstruction.scene.scenegraph_model import (
    AABB,
    RelationEdge,
    RelationType,
    SceneGraph,
    SceneNode,
    inverse_relation,
    is_symmetric,
)


def _box(lo, hi):
    return AABB(lo, hi)


class TestAABB(unittest.TestCase):
    def test_centroid_extent_volume(self):
        b = _box((0.0, 0.0, 0.0), (2.0, 4.0, 6.0))
        self.assertEqual(b.centroid, (1.0, 2.0, 3.0))
        self.assertEqual(b.extent, (2.0, 4.0, 6.0))
        self.assertEqual(b.volume, 48.0)

    def test_invalid_box_raises(self):
        with self.assertRaises(ValueError):
            _box((1.0, 0.0, 0.0), (0.0, 1.0, 1.0))

    def test_contains(self):
        outer = _box((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
        inner = _box((2.0, 2.0, 2.0), (5.0, 5.0, 5.0))
        self.assertTrue(outer.contains(inner))
        self.assertFalse(inner.contains(outer))

    def test_overlaps_and_touching(self):
        a = _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        b = _box((1.0, 0.0, 0.0), (2.0, 1.0, 1.0))  # shares a face
        c = _box((5.0, 5.0, 5.0), (6.0, 6.0, 6.0))
        self.assertTrue(a.overlaps(b))
        self.assertFalse(a.overlaps(c))

    def test_gap(self):
        a = _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        b = _box((4.0, 0.0, 0.0), (5.0, 1.0, 1.0))
        self.assertAlmostEqual(a.gap(b), 3.0)
        touching = _box((1.0, 0.0, 0.0), (2.0, 1.0, 1.0))
        self.assertAlmostEqual(a.gap(touching), 0.0)

    def test_contains_point(self):
        b = _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        self.assertTrue(b.contains_point((0.5, 0.5, 0.5)))
        self.assertFalse(b.contains_point((2.0, 0.5, 0.5)))


class TestRelationVocab(unittest.TestCase):
    def test_inverse_pairs(self):
        self.assertIs(inverse_relation(RelationType.ON_TOP_OF), RelationType.SUPPORTS)
        self.assertIs(inverse_relation(RelationType.SUPPORTS), RelationType.ON_TOP_OF)
        self.assertIs(inverse_relation(RelationType.CONTAINS), RelationType.CONTAINED_BY)
        self.assertIs(inverse_relation(RelationType.ABOVE), RelationType.BELOW)

    def test_symmetric(self):
        self.assertTrue(is_symmetric(RelationType.ADJACENT_TO))
        self.assertTrue(is_symmetric(RelationType.CONNECTED_TO))
        self.assertTrue(is_symmetric(RelationType.TOUCHING))
        self.assertFalse(is_symmetric(RelationType.ON_TOP_OF))

    def test_double_inverse_identity(self):
        for rel in RelationType:
            self.assertIs(inverse_relation(inverse_relation(rel)), rel)


class TestSceneGraph(unittest.TestCase):
    def _graph(self):
        g = SceneGraph()
        g.add_node(SceneNode("a", "table", _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))))
        g.add_node(SceneNode("b", "box", _box((0.0, 0.0, 1.0), (1.0, 1.0, 2.0))))
        return g

    def test_add_node_and_lookup(self):
        g = self._graph()
        self.assertEqual(len(g), 2)
        self.assertIn("a", g)
        self.assertEqual(g.get_node("a").obj_type, "table")
        self.assertEqual(g.node_ids, ["a", "b"])

    def test_duplicate_node_raises(self):
        g = self._graph()
        with self.assertRaises(ValueError):
            g.add_node(SceneNode("a", "x", _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))))

    def test_add_edge_with_inverse(self):
        g = self._graph()
        g.add_edge("b", RelationType.ON_TOP_OF, "a", add_inverse=True)
        self.assertTrue(g.has_edge("b", RelationType.ON_TOP_OF, "a"))
        self.assertTrue(g.has_edge("a", RelationType.SUPPORTS, "b"))
        self.assertEqual(len(g.edges), 2)

    def test_duplicate_edge_ignored(self):
        g = self._graph()
        g.add_edge("b", RelationType.ON_TOP_OF, "a")
        self.assertIsNone(g.add_edge("b", RelationType.ON_TOP_OF, "a"))
        self.assertEqual(len(g.edges), 1)

    def test_unknown_endpoint_raises(self):
        g = self._graph()
        with self.assertRaises(KeyError):
            g.add_edge("a", RelationType.ADJACENT_TO, "zzz")

    def test_neighbor_queries(self):
        g = self._graph()
        g.add_edge("b", RelationType.ON_TOP_OF, "a", add_inverse=True)
        self.assertEqual(g.neighbors("b", RelationType.ON_TOP_OF), ["a"])
        self.assertEqual(g.neighbors("a", RelationType.SUPPORTS), ["b"])
        self.assertEqual(sorted(g.undirected_neighbors("a")), ["b"])
        self.assertEqual(g.degree("a"), 2)

    def test_out_in_edge_filter(self):
        g = self._graph()
        g.add_edge("a", RelationType.ADJACENT_TO, "b")
        g.add_edge("a", RelationType.SUPPORTS, "b")
        self.assertEqual(len(g.out_edges("a")), 2)
        self.assertEqual(len(g.out_edges("a", RelationType.ADJACENT_TO)), 1)
        self.assertEqual(len(g.in_edges("b", RelationType.SUPPORTS)), 1)

    def test_edge_tuple_and_inverse(self):
        e = RelationEdge("x", RelationType.CONTAINS, "y")
        self.assertEqual(e.as_tuple(), ("x", "contains", "y"))
        self.assertEqual(e.inverse(), RelationEdge("y", RelationType.CONTAINED_BY, "x"))


if __name__ == "__main__":
    unittest.main()
