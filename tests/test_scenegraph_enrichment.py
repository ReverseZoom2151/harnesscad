"""Tests for reconstruction.scenegraph_enrichment."""

import unittest

from harnesscad.domain.reconstruction.scene.scenegraph_model import AABB, SceneGraph, SceneNode
from harnesscad.domain.reconstruction.scene.scenegraph_enrichment import (
    DEFAULT_DIMENSION_RULES,
    Vocabulary,
    affordance_for,
    classify_by_dimensions,
    coverage,
    enrich_graph,
    enrich_node,
    flatness,
)


class TestVocabulary(unittest.TestCase):
    def test_add_and_lookup(self):
        v = Vocabulary({"valve": ["ball_valve", "wheel_valve"], "gauge": ["pressure_gauge"]})
        self.assertTrue(v.has_group("valve"))
        self.assertTrue(v.has_name("wheel_valve"))
        self.assertEqual(v.group_of("pressure_gauge"), "gauge")
        self.assertEqual(sorted(v.names("valve")), ["ball_valve", "wheel_valve"])

    def test_validate(self):
        v = Vocabulary({"valve": ["ball_valve"]})
        self.assertTrue(v.validate("valve", "ball_valve"))
        self.assertTrue(v.validate("valve", None))
        self.assertFalse(v.validate("valve", "unknown"))
        self.assertFalse(v.validate("nope", None))

    def test_name_collision_raises(self):
        v = Vocabulary({"valve": ["x"]})
        with self.assertRaises(ValueError):
            v.add_name("gauge", "x")

    def test_roundtrip(self):
        data = {"valve": ["a", "b"], "gauge": ["c"]}
        v = Vocabulary.from_dict(data)
        self.assertEqual(v.to_dict(), data)


class TestDimensions(unittest.TestCase):
    def test_flatness(self):
        thin = AABB((0.0, 0.0, 0.0), (10.0, 10.0, 1.0))
        self.assertAlmostEqual(flatness(thin), 0.1)
        cube = AABB((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
        self.assertAlmostEqual(flatness(cube), 1.0)

    def test_gasket_vs_flange(self):
        gasket = AABB((0.0, 0.0, 0.0), (10.0, 10.0, 0.5))  # flatness 0.05
        flange = AABB((0.0, 0.0, 0.0), (10.0, 10.0, 5.0))  # flatness 0.5
        self.assertEqual(classify_by_dimensions(gasket), "gasket")
        self.assertEqual(classify_by_dimensions(flange), "flange")

    def test_degenerate_box(self):
        pt = AABB((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        self.assertEqual(flatness(pt), 0.0)


class TestAffordance(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(affordance_for("valve"), "turn")
        self.assertEqual(affordance_for("gauge"), "read")
        self.assertIsNone(affordance_for("mystery"))

    def test_custom_table(self):
        self.assertEqual(affordance_for("x", {"x": "poke"}), "poke")


class TestEnrichment(unittest.TestCase):
    def _graph(self):
        g = SceneGraph()
        g.add_node(SceneNode("n1", "unknown", AABB((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))))
        g.add_node(SceneNode("n2", "unknown", AABB((2.0, 0.0, 0.0), (3.0, 1.0, 1.0))))
        return g

    def test_enrich_node_sets_attrs(self):
        g = self._graph()
        node = g.get_node("n1")
        enrich_node(node, "valve", name="wheel_valve", material="steel", usd_path="/env/n1")
        self.assertEqual(node.obj_type, "valve")
        self.assertEqual(node.attributes["group"], "valve")
        self.assertEqual(node.attributes["name"], "wheel_valve")
        self.assertEqual(node.attributes["material"], "steel")
        self.assertEqual(node.attributes["affordance"], "turn")
        self.assertEqual(node.attributes["usd_path"], "/env/n1")

    def test_enrich_validates_vocab(self):
        g = self._graph()
        v = Vocabulary({"valve": ["wheel_valve"]})
        with self.assertRaises(ValueError):
            enrich_node(g.get_node("n1"), "valve", name="bad", vocabulary=v)

    def test_enrich_graph_and_coverage(self):
        g = self._graph()
        n = enrich_graph(g, {
            "n1": {"group": "valve", "name": "wheel_valve"},
            "n2": {"group": "gauge"},
        })
        self.assertEqual(n, 2)
        self.assertAlmostEqual(coverage(g, "group"), 1.0)
        self.assertAlmostEqual(coverage(g, "name"), 0.5)

    def test_coverage_empty(self):
        self.assertEqual(coverage(SceneGraph()), 0.0)


if __name__ == "__main__":
    unittest.main()
