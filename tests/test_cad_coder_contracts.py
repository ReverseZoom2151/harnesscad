import unittest

from harnesscad.eval.bench.code_metrics import (
    annotation_accuracy, function_accuracy, parameter_accuracy, parsing_rate,
)
from harnesscad.eval.bench.code_passk import estimate_pass_at_k, macro_pass_at_k
from harnesscad.eval.bench.cross_platform import evaluate_platforms
from harnesscad.eval.bench.geometry_distance import sampled_distance, symmetric_chamfer
from harnesscad.core.cisp.annotations import Linear, SurfaceRoughness, Tolerance, remap_annotations
from harnesscad.data.dataengine.cfsc_record import CFSCRecord, audit_leakage
from harnesscad.data.datagen.code_comments import (
    ambiguous, inherit_comments, intent_comments, lint_intent_comments,
)
from harnesscad.data.datagen.script_family import ParentTemplate, expand
from harnesscad.io.formats.dxf_contract import (
    DraftAnnotation, DxfDocument, Entity, Layer,
)
from harnesscad.governance.research.ablation_matrix import compare_ablation


class CadCoderContractsTests(unittest.TestCase):
    def test_neutral_document_and_annotations(self):
        document = DxfDocument(
            "mm", (Layer("GEOMETRY"),),
            {"e1": Entity("line", {"start": (0, 0), "end": (10, 0)}, "GEOMETRY")},
            (DraftAnnotation("linear", ("e1",), 10, "mm"),),
        )
        self.assertEqual(document.annotations[0].value, 10)
        with self.assertRaises(ValueError):
            DxfDocument("pixel", (), {})

    def test_annotation_validation_and_persistence(self):
        items = (
            Linear("d1", ("e1",), 10.0, "mm"),
            Tolerance("t1", ("e1",), 10.0, "mm", -.1, .1),
            SurfaceRoughness("r1", ("f1",), 3.2, "um"),
        )
        mapped = remap_annotations(items, {"e1": "e9", "f1": "f9"})
        self.assertEqual(mapped[0].entity_ids, ("e9",))
        with self.assertRaises(ValueError):
            Tolerance("bad", ("e",), 1.0, "mm", 1, -1)

    def test_seeded_legal_script_family_replay(self):
        template = ParentTemplate(
            "rings", "import cad", "def build(): return ({outer}, {inner})",
            "result = build()", {"outer": (5, 10), "inner": (2, 8)})
        legal = lambda p: (p["inner"] < p["outer"], "inner-not-smaller")
        a = expand(template, 2, 4, legal, max_attempts=30)
        b = expand(template, 2, 4, legal, max_attempts=30)
        self.assertEqual(a, b)
        self.assertTrue(all(v.parameters["inner"] < v.parameters["outer"]
                            for v in a.variants))
        self.assertGreaterEqual(a.attempts, 2)

    def test_comment_ambiguity_lint_and_inheritance(self):
        a = "x = circle(c, r1)\ny = 35\n"
        b = "x = circle(c, r1)\ny = 36\n"
        self.assertTrue(ambiguous(a, b, threshold=.7))
        parent = "# intent: tangent circles\ndef build(): pass\n"
        child = "def build(): return 1\n"
        inherited = inherit_comments(parent, child)
        self.assertEqual(intent_comments(inherited), ("tangent circles",))
        self.assertEqual(lint_intent_comments(inherited, ("tangent",)), ())

    def test_safe_ast_metrics_and_unbiased_passk(self):
        expected = "make(3, radius=2)\nfinish()\n"
        actual = "make(3, radius=4)\nfinish()\n"
        self.assertEqual(parsing_rate((expected, "not ( valid")), .5)
        self.assertTrue(function_accuracy(expected, actual)["exact"])
        self.assertEqual(parameter_accuracy(expected, actual)["accuracy"], .5)
        note = annotation_accuracy((("radius", 2),), (("radius", 3),))
        self.assertEqual(note["type_error_rate"], 0)
        self.assertEqual(note["data_error_rate"], 1)
        self.assertAlmostEqual(estimate_pass_at_k(5, 2, 2), .7)
        self.assertAlmostEqual(macro_pass_at_k(((5, 2), (5, 2)), 2), .7)

    def test_geometry_and_platform_evidence(self):
        self.assertEqual(symmetric_chamfer(((0, 0),), ((1, 0),)), 1)
        sampler = lambda shape, count, seed: shape
        self.assertEqual(sampled_distance(((0, 0),), ((1, 0),), sampler,
                                          count=1, scale=2), 2)
        matrix = evaluate_platforms(b"x", (
            ("B", "1", lambda p: {"opened": True, "reexported": True,
                                  "geometry_fidelity": .9,
                                  "annotation_retention": .8}),
            ("A", "2", lambda p: (_ for _ in ()).throw(RuntimeError("offline"))),
        ))
        self.assertEqual([x.platform for x in matrix], ["A", "B"])
        self.assertFalse(matrix[0].opened)

    def test_cfsc_leakage_and_paired_ablation(self):
        def record(i, split):
            return CFSCRecord(i, "same prompt", "x=1", i, "part", "parent",
                              (("x", 1),), "2d", "none", True, True, True, split)
        self.assertTrue(audit_leakage((record("a", "train"), record("b", "test"))))
        report = compare_ablation((
            {"stratum": "2d", "pair_id": "a", "variant": "control", "score": .5},
            {"stratum": "2d", "pair_id": "a", "variant": "treatment", "score": .8},
        ), metric="score")
        self.assertAlmostEqual(report["2d"]["mean_delta"], .3)
        self.assertEqual(report["2d"]["wins"], 1)


if __name__ == "__main__":
    unittest.main()
