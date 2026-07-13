"""Tests for reconstruction.scenegraph_construction."""

import unittest

from harnesscad.domain.reconstruction.scenegraph_model import AABB, RelationType
from harnesscad.domain.reconstruction.scenegraph_construction import (
    ConstructionConfig,
    Primitive,
    build_scene_graph,
    connect_by_proximity,
    directional_relation,
    is_on_top_of,
)


def _p(pid, typ, lo, hi):
    return Primitive(pid, typ, AABB(lo, hi))


class TestPredicates(unittest.TestCase):
    def test_on_top_of(self):
        cfg = ConstructionConfig()
        lower = AABB((0.0, 0.0, 0.0), (2.0, 2.0, 1.0))
        upper = AABB((0.0, 0.0, 1.0), (2.0, 2.0, 2.0))
        self.assertTrue(is_on_top_of(upper, lower, cfg))
        self.assertFalse(is_on_top_of(lower, upper, cfg))

    def test_on_top_of_requires_footprint(self):
        cfg = ConstructionConfig()
        lower = AABB((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        upper = AABB((5.0, 5.0, 1.0), (6.0, 6.0, 2.0))  # no xy overlap
        self.assertFalse(is_on_top_of(upper, lower, cfg))

    def test_directional_axis_selection(self):
        a = AABB((10.0, 0.0, 0.0), (11.0, 1.0, 1.0))
        b = AABB((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        self.assertIs(directional_relation(a, b), RelationType.RIGHT_OF)
        self.assertIs(directional_relation(b, a), RelationType.LEFT_OF)
        above = AABB((0.0, 0.0, 10.0), (1.0, 1.0, 11.0))
        self.assertIs(directional_relation(above, b), RelationType.ABOVE)


class TestBuild(unittest.TestCase):
    def test_stacked_support(self):
        prims = [
            _p("table", "table", (0.0, 0.0, 0.0), (2.0, 2.0, 1.0)),
            _p("box", "box", (0.0, 0.0, 1.0), (1.0, 1.0, 2.0)),
        ]
        g = build_scene_graph(prims)
        self.assertTrue(g.has_edge("box", RelationType.ON_TOP_OF, "table"))
        self.assertTrue(g.has_edge("table", RelationType.SUPPORTS, "box"))

    def test_containment(self):
        prims = [
            _p("tank", "tank", (0.0, 0.0, 0.0), (10.0, 10.0, 10.0)),
            _p("bolt", "bolt", (2.0, 2.0, 2.0), (3.0, 3.0, 3.0)),
        ]
        g = build_scene_graph(prims)
        self.assertTrue(g.has_edge("tank", RelationType.CONTAINS, "bolt"))
        self.assertTrue(g.has_edge("bolt", RelationType.CONTAINED_BY, "tank"))
        # containment short-circuits adjacency
        self.assertFalse(g.has_edge("tank", RelationType.ADJACENT_TO, "bolt"))

    def test_adjacency_vs_far(self):
        prims = [
            _p("a", "pipe", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            _p("b", "pipe", (1.005, 0.0, 0.0), (2.0, 1.0, 1.0)),  # gap 0.005 <= 0.01
            _p("c", "pipe", (5.0, 0.0, 0.0), (6.0, 1.0, 1.0)),  # far
        ]
        g = build_scene_graph(prims)
        self.assertTrue(g.has_edge("a", RelationType.TOUCHING, "b"))
        self.assertTrue(g.has_edge("b", RelationType.TOUCHING, "a"))
        self.assertEqual(g.neighbors("c"), [])

    def test_determinism(self):
        prims = [
            _p("a", "pipe", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            _p("b", "pipe", (1.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
            _p("c", "pipe", (2.0, 0.0, 0.0), (3.0, 1.0, 1.0)),
        ]
        g1 = build_scene_graph(prims)
        g2 = build_scene_graph(prims)
        self.assertEqual([e.as_tuple() for e in g1.edges], [e.as_tuple() for e in g2.edges])

    def test_disable_directional(self):
        cfg = ConstructionConfig(emit_directional=False)
        prims = [
            _p("a", "pipe", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            _p("b", "pipe", (1.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
        ]
        g = build_scene_graph(prims, cfg)
        dirs = {RelationType.LEFT_OF, RelationType.RIGHT_OF, RelationType.ABOVE,
                RelationType.BELOW, RelationType.FRONT_OF, RelationType.BEHIND}
        self.assertFalse(any(e.relation in dirs for e in g.edges))

    def test_connect_by_proximity(self):
        prims = [
            _p("a", "pipe", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            _p("b", "pipe", (1.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
            _p("c", "pipe", (9.0, 0.0, 0.0), (10.0, 1.0, 1.0)),
        ]
        g = build_scene_graph(prims, ConstructionConfig(emit_adjacency=False,
                                                        emit_directional=False,
                                                        emit_support=False,
                                                        emit_containment=False))
        added = connect_by_proximity(g, prims)
        self.assertEqual(added, 1)
        self.assertTrue(g.has_edge("a", RelationType.CONNECTED_TO, "b"))
        self.assertFalse(g.has_edge("a", RelationType.CONNECTED_TO, "c"))


if __name__ == "__main__":
    unittest.main()
