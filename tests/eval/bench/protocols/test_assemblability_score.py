"""Tests for bench.muse_assemblability."""

import unittest

from harnesscad.eval.bench.protocols.assemblability_score import (
    graphs_isomorphic,
    hub_node,
    joint_constrained_dof,
    muse_assemblability,
    normalize_joint,
    score_assembly_ready,
    score_connectable,
)


def _chair_graph():
    nodes = ["seat", "fl", "fr", "rl", "rr", "back"]
    edges = [("seat", "fl"), ("seat", "fr"), ("seat", "rl"),
             ("seat", "rr"), ("seat", "back")]
    return (nodes, edges)


class JointTests(unittest.TestCase):
    def test_normalize_aliases(self):
        self.assertEqual(normalize_joint("Mortise & Tenon"), "Interlocking")
        self.assertEqual(normalize_joint("dowel joint"), "Nailing")
        self.assertEqual(normalize_joint("hinge"), "Pivot")
        self.assertEqual(normalize_joint("N/A"), "None")

    def test_constrained_dof(self):
        self.assertEqual(joint_constrained_dof("Interlocking"), 6)
        self.assertEqual(joint_constrained_dof("Pivot"), 5)
        self.assertEqual(joint_constrained_dof("Nailing"), 4)

    def test_unknown_joint_raises(self):
        with self.assertRaises(ValueError):
            normalize_joint("magnetic levitation")


class GraphTests(unittest.TestCase):
    def test_isomorphic_star(self):
        g = _chair_graph()
        # relabelled but same topology
        g2 = (["a", "b", "c", "d", "e", "f"],
              [("a", "b"), ("a", "c"), ("a", "d"), ("a", "e"), ("a", "f")])
        self.assertTrue(graphs_isomorphic(g, g2))

    def test_non_isomorphic_missing_edge(self):
        g = _chair_graph()
        g2 = (["a", "b", "c", "d", "e", "f"],
              [("a", "b"), ("a", "c"), ("a", "d"), ("a", "e")])
        self.assertFalse(graphs_isomorphic(g, g2))

    def test_non_isomorphic_wrong_shape(self):
        g = _chair_graph()
        # path instead of star, same node & edge count
        g2 = (["a", "b", "c", "d", "e", "f"],
              [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f")])
        self.assertFalse(graphs_isomorphic(g, g2))

    def test_hub_detection(self):
        nodes, edges = _chair_graph()
        self.assertEqual(hub_node(nodes, edges), "seat")

    def test_no_unique_hub(self):
        self.assertIsNone(hub_node(["a", "b"], [("a", "b")]))


class AssemblyReadyTests(unittest.TestCase):
    def test_matching_graph_passes(self):
        design = {"target_graph": _chair_graph(),
                  "inferred_graph": _chair_graph()}
        self.assertEqual(score_assembly_ready(design)["assembly_ready"], 1)

    def test_missing_node_fails(self):
        target = _chair_graph()
        inferred = (["seat", "fl", "fr", "rl", "back"],
                    [("seat", "fl"), ("seat", "fr"), ("seat", "rl"),
                     ("seat", "back")])
        r = score_assembly_ready({"target_graph": target,
                                  "inferred_graph": inferred})
        self.assertEqual(r["assembly_ready"], 0)
        self.assertIn("node_count_mismatch", r["reasons"])

    def test_wrong_topology_fails(self):
        target = _chair_graph()
        # backrest attached to a leg instead of the seat hub
        inferred = (["seat", "fl", "fr", "rl", "rr", "back"],
                    [("seat", "fl"), ("seat", "fr"), ("seat", "rl"),
                     ("seat", "rr"), ("fl", "back")])
        r = score_assembly_ready({"target_graph": target,
                                  "inferred_graph": inferred})
        self.assertEqual(r["assembly_ready"], 0)
        self.assertIn("graph_not_isomorphic", r["reasons"])

    def test_standalone_vase(self):
        design = {"target_graph": (["vase"], []),
                  "inferred_graph": (["vase"], [])}
        self.assertEqual(score_assembly_ready(design)["assembly_ready"], 1)


class ConnectableTests(unittest.TestCase):
    def test_good_interlocking(self):
        design = {"process": "CNC Milling", "interfaces": [
            {"name": "leg1", "required_joint": "Mortise & Tenon",
             "actual_joint": "Interlocking",
             "required_direction": (0, 0, 1), "actual_direction": (0, 0, 1),
             "clearance": 0.08}]}
        self.assertEqual(score_connectable(design)["connectable"], 1)

    def test_joint_type_mismatch(self):
        design = {"process": "CNC Milling", "interfaces": [
            {"name": "leg1", "required_joint": "Interlocking",
             "actual_joint": "Bonding", "clearance": 0.08}]}
        r = score_connectable(design)
        self.assertEqual(r["connectable"], 0)
        self.assertIn("joint_type_mismatch:leg1", r["reasons"])

    def test_interpenetration(self):
        design = {"process": "CNC Milling", "interfaces": [
            {"name": "leg1", "required_joint": "Interlocking",
             "actual_joint": "Interlocking", "clearance": -0.5}]}
        r = score_connectable(design)
        self.assertIn("interpenetration:leg1", r["reasons"])

    def test_floating_gap(self):
        design = {"process": "CNC Milling", "interfaces": [
            {"name": "leg1", "required_joint": "Interlocking",
             "actual_joint": "Interlocking", "clearance": 0.5}]}
        r = score_connectable(design)
        self.assertIn("floating_gap:leg1", r["reasons"])

    def test_illegal_fusion(self):
        design = {"process": "CNC Milling", "interfaces": [
            {"name": "leg1", "required_joint": "Interlocking",
             "actual_joint": "Interlocking", "clearance": 0.0}]}
        r = score_connectable(design)
        self.assertIn("illegal_fusion:leg1", r["reasons"])

    def test_wrong_direction(self):
        design = {"process": "3D Printing", "interfaces": [
            {"name": "cap", "required_joint": "Snap-fit",
             "actual_joint": "Snap-fit",
             "required_direction": (0, 0, 1), "actual_direction": (1, 0, 0),
             "clearance": 0.2}]}
        r = score_connectable(design)
        self.assertIn("wrong_assembly_direction:cap", r["reasons"])

    def test_standalone_no_joint(self):
        design = {"process": "3D Printing", "interfaces": [
            {"name": "body", "required_joint": "None", "actual_joint": "N/A"}]}
        self.assertEqual(score_connectable(design)["connectable"], 1)


class PillarTests(unittest.TestCase):
    def test_full_pillar(self):
        design = {
            "target_graph": _chair_graph(), "inferred_graph": _chair_graph(),
            "process": "CNC Milling",
            "interfaces": [
                {"name": "leg1", "required_joint": "Interlocking",
                 "actual_joint": "Interlocking", "clearance": 0.08}],
        }
        self.assertEqual(muse_assemblability(design)["average"], 1.0)

    def test_half_pillar(self):
        design = {
            "target_graph": _chair_graph(), "inferred_graph": _chair_graph(),
            "process": "CNC Milling",
            "interfaces": [
                {"name": "leg1", "required_joint": "Interlocking",
                 "actual_joint": "Bonding", "clearance": 0.08}],
        }
        r = muse_assemblability(design)
        self.assertEqual(r["assembly_ready"], 1)
        self.assertEqual(r["connectable"], 0)
        self.assertEqual(r["average"], 0.5)


if __name__ == "__main__":
    unittest.main()
