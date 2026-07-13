import unittest

from harnesscad.agents.agent.cad_observation import CADObservation
from harnesscad.agents.agent.termination import TerminationDecision, gate_termination
from harnesscad.eval.bench.agent_cost import agent_cost
from harnesscad.eval.bench.cad_qa import grade_answer, qa_accuracy
from harnesscad.eval.bench.capability_retention import capability_retention
from harnesscad.eval.bench.code_execution import validate_cad_code, valid_syntax_rate
from harnesscad.eval.bench.image_conditioning import ImageCondition, evaluate_conditions
from harnesscad.eval.bench.sketch_metrics import sketch_f1
from harnesscad.eval.bench.solid_iou import best_solid_iou, inertia_scale, proper_axis_alignments
from harnesscad.eval.bench.tool_retrieval import evaluate_tool_retrieval
from harnesscad.eval.bench.tool_trajectory import audit_tool_trajectory
from harnesscad.data.dataengine.code_complexity import analyze_code, overflow_route
from harnesscad.data.dataengine.generation_manifest import GenerationManifest
from harnesscad.data.datagen.cadquery_codegen import emit_cadquery
from harnesscad.data.datagen.image_code_manifest import ImageCodeManifest, audit_manifests
from harnesscad.io.ingest.cross_section import cross_section, triangle_plane_segment
from harnesscad.agents.llm.generation_contract import assess_generation
from harnesscad.eval.quality.cad_code_normalize import normalize_cad_code
from harnesscad.eval.quality.constraint_impact import analyze_constraint
from harnesscad.eval.quality.sketch_serialization import (
    serialize_circle, serialize_line, serialize_sketch, validate_redundancy,
)
from harnesscad.io.surfaces.id_overlay import overlay_svg, place_labels


class AgentInfrastructureTests(unittest.TestCase):
    def test_termination_is_verifier_gated(self):
        denied = gate_termination(TerminationDecision("complete"), False)
        self.assertFalse(denied.accepted)
        self.assertFalse(denied.terminal)
        self.assertTrue(gate_termination(TerminationDecision("complete"), True).terminal)

    def test_observation_digest_and_staleness(self):
        observation = CADObservation("state-1", {"faces": 6},
                                     {"front": b"svg"}, ("f2", "f1"))
        self.assertEqual(observation.canonical_json(), observation.canonical_json())
        observation.require_current("state-1")
        with self.assertRaises(RuntimeError):
            observation.require_current("state-2")

    def test_tool_retrieval(self):
        class Card:
            def __init__(self, name): self.name, self.summary = name, name
        cards = {"hole": Card("hole"), "fillet": Card("fillet")}
        report = evaluate_tool_retrieval(
            ({"task": "make hole", "required": ("hole",)},),
            lambda task, k: (cards["hole"], cards["fillet"]), k=2)
        self.assertEqual(report["recall_at_k"], 1)
        self.assertEqual(report["rows"][0]["irrelevant"], 1)

    def test_grounded_qa_and_cost(self):
        row = grade_answer({"answer": "6", "evidence": ("faces",)}, "6",
                           observation_fields=("faces",))
        self.assertTrue(row["correct"])
        self.assertEqual(qa_accuracy((row,)), 1)
        cost = agent_cost(({"model": "m", "input_tokens": 1000,
                            "output_tokens": 500, "tool_tokens": 100,
                            "latency_seconds": 2},),
                          {"m": {"input_per_1k": 1, "output_per_1k": 2}})
        self.assertAlmostEqual(cost["cost"], 2.1)


class SketchAndGeometryTests(unittest.TestCase):
    def test_redundant_sketch_schema(self):
        line = serialize_line("l", (0, 0), (3, 4))
        circle = serialize_circle("c", (1, 1), 2)
        self.assertEqual(line["length"], 5)
        self.assertEqual(validate_redundancy(line), ())
        bad = dict(line, length=6)
        self.assertEqual(validate_redundancy(bad), ("inconsistent-length",))
        sketch = serialize_sketch((circle, line),
                                  ({"id": "k", "type": "equal",
                                    "primitives": ("l", "c")},))
        self.assertEqual([item["id"] for item in sketch["primitives"]], ["c", "l"])

    def test_safe_id_overlay_collision(self):
        svg, placements = overlay_svg({"<a>": (0, 0), "b": (0, 0)})
        self.assertIn("&lt;a&gt;", svg)
        self.assertNotEqual(placements[0].bbox, placements[1].bbox)

    def test_constraint_impact(self):
        before = {"geometry": {"a": (0, 0), "b": (1, 0)}, "dof": 2}
        result = analyze_constraint(
            before, {"type": "vertical"},
            lambda state, constraint: {
                "geometry": {"a": (0, 0), "b": (0, 1)}, "dof": 1, "valid": True})
        self.assertEqual(result.moved, ("b",))
        self.assertEqual(result.dof_after, 1)

    def test_cross_section_and_stitch(self):
        triangle = ((-1, 0, -1), (1, 0, 1), (0, 1, 1))
        segment = triangle_plane_segment(triangle, (0, 0, 0), (0, 0, 1))
        self.assertIsNotNone(segment)
        section = cross_section((triangle,), (0, 0, 0), (0, 0, 1))
        self.assertEqual(len(section), 1)

    def test_continuous_sketch_f1(self):
        expected = ({"id": "e", "type": "line", "params": (0, 0, 1, 0)},)
        actual = ({"id": "a", "type": "line", "params": (0, 0, 1.0001, 0)},)
        ec = ({"type": "horizontal", "primitives": ("e",),
               "subreferences": ("whole",)},)
        ac = ({"type": "horizontal", "primitives": ("a",),
               "subreferences": ("whole",)},)
        result = sketch_f1(actual, expected, ac, ec, tolerance=.001)
        self.assertEqual(result["primitive"]["f1"], 1)
        self.assertEqual(result["constraint"]["f1"], 1)


class CodeDataTests(unittest.TestCase):
    def test_restricted_cadquery_emitter(self):
        code = emit_cadquery((
            {"op": "new_sketch", "name": "s", "plane": "XY"},
            {"op": "add_rectangle", "sketch": "s", "w": 4, "h": 5},
            {"op": "extrude", "sketch": "s", "distance": 6},
            {"op": "fillet", "radius": .5},
        ))
        self.assertIn(".rect(4.0, 5.0)", code)
        self.assertIn("solid = solid.edges().fillet(0.5)", code)
        with self.assertRaises(ValueError):
            emit_cadquery(({"op": "shell"},))

    def test_ast_normalization_and_static_safety(self):
        normalized = normalize_cad_code("import cadquery as cq\nimport cadquery as cq\nsolid=1\n")
        self.assertEqual(normalized.count("import cadquery"), 1)
        valid = validate_cad_code(normalized)
        unsafe = validate_cad_code("import os\nsolid = 1\n")
        missing = validate_cad_code("x = 1\n")
        self.assertTrue(valid["valid"])
        self.assertEqual(unsafe["category"], "unsafe-import")
        self.assertEqual(missing["category"], "output-contract")
        self.assertEqual(valid_syntax_rate((valid, unsafe)), .5)

    def test_manifest_leakage_and_complexity(self):
        a = ImageCodeManifest.create("a", shape=b"s", code="solid=1",
                                     image=b"a", view="iso", renderer="r",
                                     split="train", source="x")
        b = ImageCodeManifest.create("b", shape=b"s", code="solid=2",
                                     image=b"b", view="iso", renderer="r",
                                     split="test", source="x")
        self.assertEqual(a.shape_digest, b.shape_digest)
        self.assertEqual(audit_manifests((a, b))[0][1],
                         "cross-split-shape-leakage")
        value = analyze_code("x = f(1)\nsolid=x\n")
        self.assertEqual(value.calls, 1)
        self.assertEqual(overflow_route(4097, 4096), "reject")
        self.assertEqual(overflow_route(4096, 4096), "accept")

    def test_generation_contract_and_manifest(self):
        status = assess_generation("solid = foo(", finish_reason="length",
                                   output_tokens=100, maximum_tokens=100,
                                   require_solid=True)
        self.assertTrue(status.truncated)
        self.assertIn("incomplete-syntax", status.issues)
        manifest = GenerationManifest.create(
            model="m", checkpoint=b"c", prompt="p", image=b"i",
            temperature=0, top_p=1, seed=1, maximum_tokens=100,
            provider_version="v", finish_reason="stop")
        self.assertEqual(manifest.digest, manifest.digest)


class EvaluationTests(unittest.TestCase):
    def test_tool_trajectory_audit(self):
        class Tool:
            def validate_args(self, args):
                if "x" not in args: raise ValueError()
        valid = audit_tool_trajectory(
            ({"tool": "t", "arguments": {"x": 1}, "result_id": "r",
              "state_digest": "a", "current_digest": "a"},),
            {"t": Tool()}, final_verified=True)
        self.assertTrue(valid["valid"])
        stale = audit_tool_trajectory(
            ({"tool": "t", "arguments": {}, "state_digest": "a",
              "current_digest": "b"},), {"t": Tool()})
        self.assertIn((0, "stale-state"), stale["issues"])

    def test_image_conditioning_and_retention(self):
        conditions = (ImageCondition("base", "iso", "gray", "white"),
                      ImageCondition("photo", "perspective", "metal", "wood", "real"))
        result = evaluate_conditions(conditions, lambda item: item.camera,
                                     lambda a, b: 1 if a == b else 0)
        self.assertEqual(result["consistency"], .5)
        cases = ({"id": "f", "operation": "fillet"},)
        retained = capability_retention(cases, lambda _: 1, lambda _: .5)
        self.assertEqual(retained["retention_rate"], 0)

    def test_inertia_alignment_with_injected_adapter(self):
        self.assertEqual(len(proper_axis_alignments()), 24)
        self.assertEqual(inertia_scale(1, 2), 1)
        class Adapter:
            def properties(self, solid):
                return {"volume": 1, "inertia_trace": 2,
                        "centroid": (0, 0, 0), "repeated_eigenvalues": solid == "g"}
            def normalize(self, solid, centroid, scale): return solid
            def align(self, solid, matrix): return solid, matrix
            def iou(self, left, right):
                return 1 if left[1] == proper_axis_alignments()[-1] else .5
        result = best_solid_iou("g", "t", Adapter())
        self.assertEqual(result["iou"], 1)
        self.assertTrue(result["degenerate"])


if __name__ == "__main__":
    unittest.main()
