"""Tests for reconstruction.scenegraph_validity."""

import unittest

from harnesscad.domain.reconstruction.scenegraph_model import (
    AABB,
    RelationEdge,
    RelationType,
    SceneGraph,
    SceneNode,
)
from harnesscad.domain.reconstruction.scenegraph_validity import (
    check_acyclic,
    check_inverse_consistency,
    check_scene_graph,
)


def _box(i):
    return AABB((float(i), 0.0, 0.0), (float(i) + 1.0, 1.0, 1.0))


def _graph(ids):
    g = SceneGraph()
    for i, nid in enumerate(ids):
        g.add_node(SceneNode(nid, "pipe", _box(i)))
    return g


class TestValidity(unittest.TestCase):
    def test_clean_graph_ok(self):
        g = _graph(["a", "b"])
        g.add_edge("a", RelationType.ON_TOP_OF, "b", add_inverse=True)
        report = check_scene_graph(g)
        self.assertTrue(report.ok)
        self.assertEqual(report.errors, [])

    def test_missing_inverse_warns(self):
        g = _graph(["a", "b"])
        g.add_edge("a", RelationType.ON_TOP_OF, "b")  # no inverse
        report = check_scene_graph(g)
        self.assertTrue(report.ok)  # warning, not error
        self.assertIn("missing_inverse", report.codes())
        self.assertEqual(len(report.warnings), 1)

    def test_symmetric_missing_mirror(self):
        g = _graph(["a", "b"])
        # add adjacency one way only by bypassing helper
        g.add_edge("a", RelationType.ADJACENT_TO, "b")
        issues = check_inverse_consistency(g)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "missing_inverse")

    def test_self_loop_error(self):
        g = _graph(["a"])
        # force a self loop through the internal insert path
        g._insert(RelationEdge("a", RelationType.ADJACENT_TO, "a"))
        report = check_scene_graph(g)
        self.assertFalse(report.ok)
        self.assertIn("self_loop", report.codes())

    def test_containment_cycle_detected(self):
        g = _graph(["a", "b", "c"])
        g.add_edge("a", RelationType.CONTAINS, "b")
        g.add_edge("b", RelationType.CONTAINS, "c")
        g.add_edge("c", RelationType.CONTAINS, "a")  # cycle
        issues = check_acyclic(g, RelationType.CONTAINS, "containment_cycle")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "containment_cycle")

    def test_containment_dag_ok(self):
        g = _graph(["a", "b", "c"])
        g.add_edge("a", RelationType.CONTAINS, "b")
        g.add_edge("b", RelationType.CONTAINS, "c")
        self.assertEqual(check_acyclic(g, RelationType.CONTAINS, "containment_cycle"), [])

    def test_support_cycle_is_error(self):
        g = _graph(["a", "b"])
        g._insert(RelationEdge("a", RelationType.SUPPORTS, "b"))
        g._insert(RelationEdge("b", RelationType.SUPPORTS, "a"))
        report = check_scene_graph(g, require_inverses=False)
        self.assertFalse(report.ok)
        self.assertIn("support_cycle", report.codes())

    def test_isolated_node_info(self):
        g = _graph(["a", "lonely"])
        g.add_edge("a", RelationType.ADJACENT_TO, "a") if False else None
        report = check_scene_graph(g, require_inverses=False)
        self.assertIn("isolated_node", report.codes())
        # info severity => still ok
        self.assertTrue(report.ok)

    def test_report_isolated_toggle(self):
        g = _graph(["a"])
        report = check_scene_graph(g, report_isolated=False)
        self.assertNotIn("isolated_node", report.codes())


if __name__ == "__main__":
    unittest.main()
