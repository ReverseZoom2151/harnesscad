"""Tests for reconstruction.scenegraph_query."""

import unittest

from harnesscad.domain.reconstruction.scene.scenegraph_model import AABB, RelationType, SceneGraph, SceneNode
from harnesscad.domain.reconstruction.scene.scenegraph_query import (
    connected_component,
    connected_components,
    count_by_type,
    objects_by_affordance,
    objects_of_type,
    objects_with_attribute,
    path_exists,
    related,
    relation_between,
    shortest_path,
)


def _box(i):
    return AABB((float(i), 0.0, 0.0), (float(i) + 1.0, 1.0, 1.0))


class TestQuery(unittest.TestCase):
    def _graph(self):
        g = SceneGraph()
        for nid, typ in [("p1", "pipe"), ("p2", "pipe"), ("v1", "valve"),
                         ("g1", "gauge"), ("p3", "pipe")]:
            g.add_node(SceneNode(nid, typ, _box(0)))
        # chain p1 - p2 - v1 - g1 ; p3 isolated
        g.add_edge("p1", RelationType.CONNECTED_TO, "p2", add_inverse=True)
        g.add_edge("p2", RelationType.CONNECTED_TO, "v1", add_inverse=True)
        g.add_edge("v1", RelationType.CONNECTED_TO, "g1", add_inverse=True)
        g.get_node("v1").attributes["affordance"] = "turn"
        g.get_node("g1").attributes["material"] = "brass"
        return g

    def test_objects_of_type(self):
        g = self._graph()
        self.assertEqual([n.node_id for n in objects_of_type(g, "pipe")], ["p1", "p2", "p3"])
        self.assertEqual([n.node_id for n in objects_of_type(g, "valve")], ["v1"])

    def test_objects_with_attribute(self):
        g = self._graph()
        self.assertEqual([n.node_id for n in objects_with_attribute(g, "material")], ["g1"])
        self.assertEqual(
            [n.node_id for n in objects_with_attribute(g, "material", "brass")], ["g1"]
        )
        self.assertEqual(objects_with_attribute(g, "material", "steel"), [])

    def test_objects_by_affordance(self):
        g = self._graph()
        self.assertEqual([n.node_id for n in objects_by_affordance(g, "turn")], ["v1"])

    def test_count_by_type(self):
        g = self._graph()
        self.assertEqual(count_by_type(g), {"pipe": 3, "valve": 1, "gauge": 1})

    def test_related(self):
        g = self._graph()
        self.assertEqual(sorted(related(g, "p2", RelationType.CONNECTED_TO)), ["p1", "v1"])

    def test_relation_between(self):
        g = self._graph()
        self.assertEqual(relation_between(g, "p1", "p2"), [RelationType.CONNECTED_TO])
        self.assertEqual(relation_between(g, "p1", "g1"), [])

    def test_shortest_path(self):
        g = self._graph()
        self.assertEqual(shortest_path(g, "p1", "g1"), ["p1", "p2", "v1", "g1"])
        self.assertEqual(shortest_path(g, "p1", "p1"), ["p1"])

    def test_no_path(self):
        g = self._graph()
        self.assertIsNone(shortest_path(g, "p1", "p3"))
        self.assertFalse(path_exists(g, "p1", "p3"))
        self.assertTrue(path_exists(g, "p1", "g1"))

    def test_missing_node_path(self):
        g = self._graph()
        self.assertIsNone(shortest_path(g, "p1", "zzz"))

    def test_shortest_path_relation_filter(self):
        g = self._graph()
        g.add_node(SceneNode("x", "pipe", _box(0)))
        g.add_edge("p1", RelationType.ADJACENT_TO, "x", add_inverse=True)
        # no CONNECTED_TO path to x
        self.assertIsNone(shortest_path(g, "p2", "x", RelationType.CONNECTED_TO))
        self.assertIsNotNone(shortest_path(g, "p2", "x"))

    def test_connected_component(self):
        g = self._graph()
        self.assertEqual(connected_component(g, "p1"), ["p1", "p2", "v1", "g1"])
        self.assertEqual(connected_component(g, "p3"), ["p3"])

    def test_connected_components(self):
        g = self._graph()
        comps = connected_components(g)
        self.assertEqual(comps, [["p1", "p2", "v1", "g1"], ["p3"]])


if __name__ == "__main__":
    unittest.main()
