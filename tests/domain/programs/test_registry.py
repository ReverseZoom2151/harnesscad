"""The program surface: language-dispatched parse / validate / emit / review."""

import unittest

from harnesscad.domain.programs import registry as P


# The same part, expressed as the neutral operation IR: a 20 x 10 plate,
# extruded 5 -- the vocabulary comes from validate.operation_schema.
PLATE_OPS = [
    {"operation": "rectangle",
     "args": {"center": [0.0, 0.0, 0.0], "width": 20.0, "height": 10.0},
     "result": "profile"},
    {"operation": "extrude",
     "args": {"profile": "profile", "height": 5.0},
     "result": "plate"},
]

CQ_SOURCE = (
    'import cadquery as cq\n'
    'plate = cq.Workplane("XY").rect(20.0, 10.0).extrude(5.0)\n'
)

SCAD_SOURCE = (
    'width = 20; // the plate width\n'
    'linear_extrude(height = 5) { square(size = [20, 10], center = true); }\n'
)


class DiscoveryTests(unittest.TestCase):
    def test_registry_discovers_many_real_modules(self):
        langs = P.languages()
        modules = {m for name in langs for m in P.language(name).modules}
        self.assertGreater(len(modules), 10)
        self.assertGreaterEqual(len(langs), 5)

        from harnesscad import registry as capability_registry

        indexed = {e.dotted for e in capability_registry.index()}
        for dotted in modules:
            self.assertIn(dotted, indexed)

    def test_every_language_declares_its_capabilities(self):
        for name in P.languages():
            caps = P.capabilities(name)
            self.assertTrue(caps)
            for cap in caps:
                self.assertIn(cap, P.CAPABILITIES)

    def test_unknown_language_raises(self):
        with self.assertRaises(P.UnknownLanguage):
            P.language("solidworks")
        with self.assertRaises(P.UnknownLanguage):
            P.parse("cube(1);", "solidworks")

    def test_discovery_is_deterministic(self):
        self.assertEqual(P.languages(), P.languages())
        self.assertEqual(list(P.languages()), sorted(P.languages()))


class CadQueryTests(unittest.TestCase):
    def test_parse_validate_emit_round_trip(self):
        program = P.parse(CQ_SOURCE, "cadquery")
        self.assertEqual(program.lang, "cadquery")
        self.assertEqual(P.validate(program), ())
        self.assertIn('cq.Workplane("XY")', P.serialize(program))

        source = P.emit(PLATE_OPS, "cadquery")
        self.assertIn("import cadquery as cq", source)
        self.assertIn(".rect(20.0, 10.0)", source)
        self.assertIn(".extrude(5.0)", source)
        # what we emit, we can parse and validate again.
        self.assertEqual(P.validate(P.parse(source, "cadquery")), ())

    def test_review_is_static_and_finds_the_api_profile(self):
        findings = P.review(CQ_SOURCE, "cadquery")
        codes = [f.code for f in findings]
        self.assertIn("api-profile", codes)
        self.assertFalse([f for f in findings if f.severity == P.ERROR])

    def test_review_reports_an_invalid_call(self):
        bad = 'import cadquery as cq\nr = cq.Workplane("XY").rect(1.0).extrude(2.0)\n'
        findings = P.review(bad, "cadquery")
        self.assertTrue([f for f in findings if f.severity == P.ERROR])

    def test_extract_recovers_source_from_an_llm_reply(self):
        reply = ("### Response:\n```python\nimport cadquery as cq\n"
                 "r = cq.Workplane('XY').box(1, 2, 3)\n"
                 "cq.exporters.export(r, 'out.stl')\n```\n<|endoftext|> and more prose")
        code = P.extract(reply, "cadquery", export_path="part.stl")
        self.assertIn("cadquery", code)
        self.assertNotIn("### Response", code)
        self.assertNotIn("```", code)
        self.assertNotIn("more prose", code)      # truncated at the EOS token
        self.assertIn('cq.exporters.export(r, "part.stl")', code)


class OpenSCADTests(unittest.TestCase):
    def test_parse_validate_emit_round_trip(self):
        program = P.parse(SCAD_SOURCE, "openscad")
        self.assertEqual(program.lang, "openscad")
        self.assertEqual([f for f in P.validate(program) if f.severity == P.ERROR], [])
        self.assertIn("linear_extrude", P.serialize(program))

        source = P.emit(PLATE_OPS, "openscad")
        self.assertIn("linear_extrude(height = 5)", source)
        self.assertIn("square(center = true, size = [20, 10])", source)
        # what we emit, we can parse again.
        reparsed = P.parse(source, "openscad")
        self.assertIn("linear_extrude", P.serialize(reparsed))

    def test_validate_flags_broken_source(self):
        findings = P.validate("cube(10;\n", lang="openscad")
        self.assertTrue([f for f in findings if f.severity == P.ERROR])

    def test_review_segments_the_program_into_blocks(self):
        findings = P.review(SCAD_SOURCE, "openscad")
        self.assertIn("blocks", [f.code for f in findings])

    def test_params_uses_the_unified_cross_language_schema(self):
        params = P.params(SCAD_SOURCE, "openscad")
        self.assertEqual([p.name for p in params], ["width"])
        self.assertEqual(params[0].type, "number")
        self.assertEqual(params[0].initial, 20)

    def test_annotate_repair_and_quantize_are_openscad_capabilities(self):
        self.assertIn("TBC", P.annotate(SCAD_SOURCE, "openscad"))
        self.assertTrue(P.repair("cube(10);", "openscad"))
        self.assertTrue(P.quantize(SCAD_SOURCE, "openscad"))


class LanguageDispatchTests(unittest.TestCase):
    """LANGUAGE IS THE KEY: languages are dispatched, never blended or guessed."""

    def test_cadquery_and_openscad_emit_the_same_part_differently(self):
        cq = P.emit(PLATE_OPS, "cadquery")
        scad = P.emit(PLATE_OPS, "openscad")
        self.assertNotEqual(cq, scad)
        self.assertIn("cq.Workplane", cq)
        self.assertNotIn("cq.Workplane", scad)
        self.assertIn("linear_extrude", scad)
        self.assertNotIn("linear_extrude", cq)

    def test_the_typed_csg_emitter_type_checks_instead_of_emitting_text(self):
        tree = P.emit(PLATE_OPS, "typed_csg")
        self.assertEqual(P.validate(P.Program("typed_csg", tree)), ())

    def test_parsing_one_language_as_another_raises_rather_than_guessing(self):
        # OpenSCAD source is not valid CadQuery, and vice versa. Neither parser
        # "mostly works" on the other's text -- both refuse.
        with self.assertRaises(Exception) as cq_ctx:
            P.parse(SCAD_SOURCE, "cadquery")
        self.assertNotIsInstance(cq_ctx.exception, AssertionError)

        with self.assertRaises(Exception) as scad_ctx:
            P.parse(CQ_SOURCE, "openscad")
        self.assertNotIsInstance(scad_ctx.exception, AssertionError)

    def test_validating_a_program_with_another_languages_validator_raises(self):
        scad = P.parse(SCAD_SOURCE, "openscad")
        with self.assertRaises(P.LanguageMismatch) as ctx:
            P.validate(scad, lang="cadquery")
        self.assertIn("openscad", str(ctx.exception))
        self.assertIn("cadquery", str(ctx.exception))

        cq = P.parse(CQ_SOURCE, "cadquery")
        with self.assertRaises(P.LanguageMismatch):
            P.serialize(cq, lang="openscad")

    def test_a_missing_capability_is_refused_not_delegated(self):
        # bpy has no emitter; the surface says so rather than emitting OpenSCAD.
        with self.assertRaises(P.Unsupported):
            P.emit(PLATE_OPS, "bpy")
        # typed CSG deliberately has no source parser.
        with self.assertRaises(P.Unsupported):
            P.parse("cube(1);", "typed_csg")
        # CadQuery has no CADTalk annotator.
        with self.assertRaises(P.Unsupported):
            P.annotate(CQ_SOURCE, "cadquery")

    def test_an_op_a_language_cannot_express_raises_unsupported(self):
        ops = list(PLATE_OPS) + [
            {"operation": "revolve",
             "args": {"profile": "profile", "axis": "profile", "angle": 90.0},
             "result": "r"}]
        with self.assertRaises(P.Unsupported):
            P.emit(ops, "cadquery")

    def test_the_neutral_op_ir_is_validated_before_any_lowering(self):
        bad = [{"operation": "rectangle",
                "args": {"center": [0.0, 0.0], "width": 1.0, "height": 1.0},
                "result": "p"}]
        self.assertTrue(P.validate_ops(bad))       # a 2-element point is not a point
        for lang in ("cadquery", "openscad", "typed_csg"):
            with self.assertRaises(P.ProgramError):
                P.emit(bad, lang)
        self.assertIn("extrude", P.operations())


if __name__ == "__main__":
    unittest.main()
