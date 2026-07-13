import unittest

from harnesscad.eval.quality.intentgraph import (
    IntentGraph,
    IntentNode,
    IntentRelation,
    RelationKind,
)


class IntentGraphTests(unittest.TestCase):
    def graph(self):
        nodes = [
            IntentNode("brief", "carry radial load"),
            IntentNode("boss", "locate bearing", "f2"),
            IntentNode("hole", "accept shaft", "f3"),
        ]
        return IntentGraph(nodes, [
            IntentRelation("brief", "boss", RelationKind.CAUSAL),
            IntentRelation("boss", "hole", RelationKind.SPATIAL, "concentric"),
            IntentRelation("hole", "boss", RelationKind.FUNCTIONAL, "bearing-seat"),
        ])

    def test_typed_adjacency_and_causal_order(self):
        graph = self.graph()
        self.assertEqual(graph.causal_order()[0], "brief")
        spatial = graph.adjacent("hole", RelationKind.SPATIAL)
        self.assertEqual(spatial[0].label, "concentric")

    def test_deterministic_serialization(self):
        self.assertEqual(self.graph().to_json(), self.graph().to_json())

    def test_rejects_missing_endpoints_and_causal_cycles(self):
        graph = IntentGraph([IntentNode("a", "a"), IntentNode("b", "b")])
        with self.assertRaises(ValueError):
            graph.add_relation(IntentRelation("a", "missing", RelationKind.CAUSAL))
        graph.add_relation(IntentRelation("a", "b", RelationKind.CAUSAL))
        graph.add_relation(IntentRelation("b", "a", RelationKind.CAUSAL))
        with self.assertRaises(ValueError):
            graph.causal_order()
