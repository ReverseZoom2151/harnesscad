import json
import unittest

from harnesscad.core.cisp.ops import AddRectangle, Extrude, NewSketch
from harnesscad.eval.quality.graph.featuregraph import FeatureEdge, FeatureGraph, FeatureNode
from harnesscad.core.state.opdag import OpDAG
from harnesscad.io.surfaces.graphview import build_graph_view
from harnesscad.eval.verifiers.verify import Diagnostic, Severity


class GraphViewTests(unittest.TestCase):
    def make_dag(self):
        dag = OpDAG()
        dag.append(NewSketch("XY"))
        dag.branch("variant")
        dag.append(AddRectangle("sk1", 0, 0, 10, 5))
        dag.checkout("variant")
        dag.append(Extrude("sk1", 3))
        return dag

    def test_json_is_deterministic_and_canonical(self):
        graph = FeatureGraph(
            [FeatureNode("b<1", "body", {"z": 2, "a": 1}), FeatureNode("h1", "hole")],
            [FeatureEdge("h1", "b<1", "cuts")],
        )
        diagnostics = [
            Diagnostic(Severity.WARNING, "z", "later"),
            Diagnostic(Severity.ERROR, "a", "first", "b<1"),
        ]
        first = build_graph_view(self.make_dag(), graph, diagnostics).to_json()
        second = build_graph_view(self.make_dag(), graph, reversed(diagnostics)).to_json()
        self.assertEqual(first, second)
        parsed = json.loads(first)
        self.assertEqual(1, parsed["version"])
        self.assertEqual("variant", parsed["active_branch"])
        self.assertEqual(["a", "z"], [d["code"] for d in parsed["diagnostics"]])
        self.assertEqual(parsed, json.loads(json.dumps(parsed)))

    def test_shared_history_is_one_node(self):
        view = build_graph_view(self.make_dag())
        operations = [n for n in view.nodes if n.kind == "operation"]
        self.assertEqual(3, len(operations))
        self.assertEqual(2, len([n for n in view.nodes if n.kind == "branch"]))

    def test_svg_escapes_all_untrusted_text_and_has_no_script(self):
        graph = FeatureGraph(
            [FeatureNode('x"><script>alert(1)</script>', "body&part")], []
        )
        diagnostic = Diagnostic(
            Severity.ERROR, "<bad>", 'failure </text><script>x</script> & "quoted"'
        )
        svg = build_graph_view(feature_graph=graph, diagnostics=[diagnostic]).to_svg()
        self.assertNotIn("<script>", svg)
        self.assertNotIn("</text><script>", svg)
        self.assertIn("&lt;script&gt;", svg)
        self.assertIn("&amp;", svg)
        self.assertTrue(svg.startswith("<svg "))
        self.assertTrue(svg.endswith("</svg>"))

    def test_feature_edges_are_namespaced(self):
        graph = FeatureGraph(
            [FeatureNode("a", "body"), FeatureNode("b", "hole")],
            [FeatureEdge("b", "a", "cuts")],
        )
        data = build_graph_view(feature_graph=graph).to_dict()
        self.assertIn(
            {"source": "feature:b", "target": "feature:a", "relation": "cuts"},
            data["edges"],
        )


if __name__ == "__main__":
    unittest.main()
