"""Tests for reconstruction.scenegraph_functional (Algorithm 1)."""

import unittest

from reconstruction.scenegraph_model import AABB, RelationType, SceneGraph, SceneNode
from reconstruction.scenegraph_functional import (
    FunctionalGraph,
    extract_functional_relations,
    find_functional_units,
)


def _box(i):
    return AABB((float(i), 0.0, 0.0), (float(i) + 1.0, 1.0, 1.0))


def _chain(specs):
    """Build a linear graph from [(id, group), ...] with CONNECTED_TO edges."""
    g = SceneGraph()
    for i, (nid, grp) in enumerate(specs):
        g.add_node(SceneNode(nid, grp, _box(i)))
    for (a, _), (b, _) in zip(specs, specs[1:]):
        g.add_edge(a, RelationType.CONNECTED_TO, b, add_inverse=True)
    return g


class TestFindUnits(unittest.TestCase):
    def test_same_group_cluster(self):
        # two valve nodes adjacent form one unit; separated valve is its own unit
        g = _chain([("v1", "valve"), ("v2", "valve"), ("p1", "pipe_assembly"), ("v3", "valve")])
        units = find_functional_units(g, ["valve"])
        self.assertEqual(len(units), 2)
        self.assertEqual(units[0], {"v1", "v2"})
        self.assertEqual(units[1], {"v3"})

    def test_gauge_units(self):
        g = _chain([("g1", "gauge"), ("p1", "pipe_assembly"), ("g2", "gauge")])
        units = find_functional_units(g, ["gauge"])
        self.assertEqual([sorted(u) for u in units], [["g1"], ["g2"]])


class TestAlgorithm1(unittest.TestCase):
    def test_two_valves_linked_by_pipe(self):
        # valve -- pipe -- pipe -- valve : both grow through pipes and connect
        g = _chain([
            ("v1", "valve"),
            ("p1", "pipe_assembly"),
            ("p2", "pipe_assembly"),
            ("v2", "valve"),
        ])
        units = find_functional_units(g, ["valve"])
        fg = extract_functional_relations(g, units, ["pipe_assembly"])
        self.assertEqual(len(fg.units), 2)
        # units grew to claim the pipe nodes between them
        total_claimed = set().union(*fg.units)
        self.assertEqual(total_claimed, {"v1", "p1", "p2", "v2"})
        # they are functionally related
        self.assertEqual(fg.edges, {(0, 1)})
        self.assertEqual(sorted(fg.neighbors(0)), [1])

    def test_structural_support_ignored(self):
        # a 'support' mesh is not a connector, so it must not bridge units
        g = SceneGraph()
        for nid, grp in [("v1", "valve"), ("s", "support"), ("v2", "valve")]:
            g.add_node(SceneNode(nid, grp, _box(0)))
        g.add_edge("v1", RelationType.CONNECTED_TO, "s", add_inverse=True)
        g.add_edge("s", RelationType.CONNECTED_TO, "v2", add_inverse=True)
        units = find_functional_units(g, ["valve"])
        fg = extract_functional_relations(g, units, ["pipe_assembly"])
        # support never claimed, no functional edge
        self.assertEqual(fg.edges, set())
        for u in fg.units:
            self.assertNotIn("s", u)

    def test_central_tank_links_multiple(self):
        # three valves each linked to a central pipe network -> all related
        g = SceneGraph()
        for nid, grp in [
            ("v1", "valve"), ("v2", "valve"), ("v3", "valve"),
            ("t", "pipe_assembly"),
        ]:
            g.add_node(SceneNode(nid, grp, _box(0)))
        for v in ("v1", "v2", "v3"):
            g.add_edge(v, RelationType.CONNECTED_TO, "t", add_inverse=True)
        units = find_functional_units(g, ["valve"])
        fg = extract_functional_relations(g, units, ["pipe_assembly"])
        self.assertEqual(len(fg.units), 3)
        # central pipe claimed by exactly one unit (global marked set)
        claims = sum(1 for u in fg.units if "t" in u)
        self.assertEqual(claims, 1)
        # the claiming unit is related to the other two
        owner = fg.unit_of("t")
        self.assertEqual(len(fg.neighbors(owner)), 2)

    def test_labels_default(self):
        g = _chain([("v1", "valve"), ("p1", "pipe_assembly"), ("g1", "gauge")])
        units = find_functional_units(g, ["valve", "gauge"])
        fg = extract_functional_relations(g, units, ["pipe_assembly"])
        self.assertIn("valve", fg.labels)
        self.assertIn("gauge", fg.labels)

    def test_determinism(self):
        g = _chain([
            ("v1", "valve"), ("p1", "pipe_assembly"),
            ("p2", "pipe_assembly"), ("v2", "valve"),
        ])
        units = find_functional_units(g, ["valve"])
        f1 = extract_functional_relations(g, units, ["pipe_assembly"])
        units2 = find_functional_units(g, ["valve"])
        f2 = extract_functional_relations(g, units2, ["pipe_assembly"])
        self.assertEqual(f1.edges, f2.edges)
        self.assertEqual([sorted(u) for u in f1.units], [sorted(u) for u in f2.units])

    def test_input_units_not_mutated(self):
        g = _chain([("v1", "valve"), ("p1", "pipe_assembly"), ("v2", "valve")])
        units = find_functional_units(g, ["valve"])
        snapshot = [set(u) for u in units]
        extract_functional_relations(g, units, ["pipe_assembly"])
        self.assertEqual(units, snapshot)


if __name__ == "__main__":
    unittest.main()
