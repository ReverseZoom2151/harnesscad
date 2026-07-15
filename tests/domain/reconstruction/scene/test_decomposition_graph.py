"""Tests for domain.reconstruction.scene.decomposition_graph."""

import unittest

from harnesscad.domain.reconstruction.scene.decomposition_graph import (
    Constraint,
    DecompositionGraph,
    Node,
    geometric_constraint_satisfaction,
    hierarchy_accuracy,
    node_accuracy,
    parse,
    serialize,
)


def _ref():
    return DecompositionGraph(nodes=[
        Node("microwave_oven", 0, "Composite of Door + Body"),
        Node("Body", 1, "Composite of Body_shell + Turntable"),
        Node("Door", 1, "Composite of Door_window + Control_panel",
             (Constraint("Align(XYZ)", "Door.back_face", "Body.front_face"),)),
        Node("Door_window", 2, "Cuboid 0.560x0.028x0.420 m"),
    ])


class RoundTripTest(unittest.TestCase):
    def test_serialize_has_layers(self):
        text = serialize(_ref())
        self.assertIn("# Layer 0", text)
        self.assertIn("# Layer 2", text)
        self.assertIn("|| Door | Composite of Door_window + Control_panel |", text)

    def test_parse_inverts_serialize(self):
        g = _ref()
        g2 = parse(serialize(g))
        self.assertEqual(len(g2.nodes), len(g.nodes))
        self.assertEqual(node_accuracy(g2, g), 1.0)
        self.assertEqual(hierarchy_accuracy(g2, g), 1.0)
        self.assertEqual(geometric_constraint_satisfaction(g2, g), 1.0)


class MetricsTest(unittest.TestCase):
    def test_node_accuracy_partial(self):
        ref = _ref()
        pred = DecompositionGraph(nodes=[
            Node("microwave_oven", 0, "Composite of Door + Body"),
            Node("Body", 1, "wrong descriptor"),
        ])
        # 1 of 4 reference nodes correctly reproduced
        self.assertAlmostEqual(node_accuracy(pred, ref), 0.25)

    def test_hierarchy_accuracy_wrong_layer(self):
        ref = _ref()
        pred = DecompositionGraph(nodes=[
            Node("microwave_oven", 0, "Composite of Door + Body"),
            Node("Body", 2, "Composite of Body_shell + Turntable"),  # wrong layer
            Node("Door", 1, "Composite of Door_window + Control_panel"),
            Node("Door_window", 2, "Cuboid 0.560x0.028x0.420 m"),
        ])
        self.assertAlmostEqual(hierarchy_accuracy(pred, ref), 0.75)

    def test_gcs_missing_constraint(self):
        ref = _ref()
        pred = DecompositionGraph(nodes=[
            Node("Door", 1, "Composite of Door_window + Control_panel"),  # no constraint
        ])
        self.assertAlmostEqual(geometric_constraint_satisfaction(pred, ref), 0.0)

    def test_gcs_full_when_no_ref_constraints(self):
        ref = DecompositionGraph(nodes=[Node("a", 0, "x")])
        self.assertEqual(geometric_constraint_satisfaction(ref, ref), 1.0)

    def test_empty_ref_raises(self):
        with self.assertRaises(ValueError):
            node_accuracy(_ref(), DecompositionGraph())


if __name__ == "__main__":
    unittest.main()
