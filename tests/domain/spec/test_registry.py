"""The spec surface: brief -> checked spec -> constraints, and EXPRESS/Part-21."""

import unittest

from harnesscad.core.loop import HarnessSession
from harnesscad.domain.spec import registry as S
from harnesscad.io.backends.stub import StubBackend


BRIEF = ("A rectangular aluminium mounting plate 100 mm long, 60 mm wide and "
         "8 mm thick with 4 mounting holes, tolerance +/- 0.1 mm")

# A tiny but real EXPRESS schema: `plate` inherits `name` from `shape`, so a
# part-21 PLATE record must supply TWO attributes (the flattened arity), not one.
SCHEMA = """
SCHEMA demo;
ENTITY shape;
  name : STRING;
END_ENTITY;
ENTITY plate
  SUBTYPE OF (shape);
  thickness : REAL;
END_ENTITY;
END_SCHEMA;
"""

GOOD_P21 = (
    "ISO-10303-21;\n"
    "HEADER;\nENDSEC;\n"
    "DATA;\n"
    "#1=PLATE('p1',8.0);\n"
    "ENDSEC;\n"
    "END-ISO-10303-21;\n"
)

# Two independent violations: PLATE is given one attribute where the flattened
# arity is two, and WIDGET is not in the schema at all.
BAD_P21 = (
    "ISO-10303-21;\n"
    "HEADER;\nENDSEC;\n"
    "DATA;\n"
    "#1=PLATE('p1');\n"
    "#2=WIDGET('x');\n"
    "ENDSEC;\n"
    "END-ISO-10303-21;\n"
)


class TestDiscovery(unittest.TestCase):
    def test_discovers_more_than_five_real_modules(self):
        routed = S.routed_modules()
        self.assertGreater(len(routed), 5, routed)
        for dotted in routed:
            self.assertTrue(dotted.startswith("harnesscad.domain.spec."), dotted)

    def test_every_registered_route_is_present(self):
        rows = S.discover()
        self.assertGreater(len(rows), 5)
        for row in rows:
            self.assertTrue(row["present"], row)

    def test_discovery_is_deterministic(self):
        self.assertEqual(S.discover(), S.discover())
        self.assertEqual(S.routed_modules(), S.routed_modules())

    def test_unadapted_modules_state_a_reason(self):
        for dotted, reason in S.unadapted():
            self.assertNotEqual(reason, "no route yet",
                                "%s has no stated reason" % dotted)


class TestRivals(unittest.TestCase):
    def test_interpreters_are_exposed_by_name(self):
        names = S.interpreters()
        for expected in ("formalize", "case_frame", "command_recovery",
                         "parse_states", "dialogue_state"):
            self.assertIn(expected, names)

    def test_rival_families_are_declared(self):
        families = {f for f, _doc, _members in S.RIVAL_FAMILIES}
        self.assertEqual(families, {"brief-interpreter", "clarifier"})

    def test_rivals_are_selected_never_blended(self):
        """Two interpreters over one brief give DIFFERENT answers, kept apart."""
        text = "add a 10 mm hole to the top face"
        formal = S.interpret(text, "formalize")
        frame = S.interpret(text, "case_frame")
        self.assertEqual(formal.name, "formalize")
        self.assertEqual(frame.name, "case_frame")
        # The case-frame parse recovers the feature; formalize does not fire on
        # an imperative command. Nothing averages the two.
        self.assertNotEqual(formal.requirements, frame.requirements)

    def test_unknown_interpreter_raises_rather_than_falling_back(self):
        with self.assertRaises(S.UnknownRoute):
            S.interpret("a plate", "no_such_interpreter")

    def test_unknown_clarifier_raises(self):
        with self.assertRaises(S.UnknownRoute):
            S.clarify("a plate", None, "no_such_clarifier")


class TestBriefToCheckedSpec(unittest.TestCase):
    def test_end_to_end_brief_to_constraints(self):
        result = S.compile_brief(BRIEF)
        self.assertTrue(result.ok)
        self.assertEqual(result.interpreter, "formalize")

        kinds = {c.kind for c in result.constraints}
        self.assertIn("dimension", kinds)
        self.assertIn("count", kinds)
        self.assertIn("material", kinds)

        params = result.parameters
        self.assertEqual(params["length"], 100.0)
        self.assertEqual(params["width"], 60.0)
        self.assertEqual(params["thickness"], 8.0)
        self.assertEqual(params["n_hole"], 4)
        self.assertEqual(params["material"], "aluminium")
        self.assertEqual(params["tolerance"], 0.1)
        # the axis aliases the geometry side speaks
        self.assertEqual((params["x"], params["y"], params["z"]),
                         (100.0, 60.0, 8.0))

    def test_contract_is_seeded_from_the_brief(self):
        contract = S.compile_brief(BRIEF).contract
        self.assertEqual(contract["bbox"]["x"], {"target": 100.0, "tol": 0.1})
        self.assertEqual(contract["hole_count"], 4)

    def test_verifiers_check_a_built_model_against_the_spec(self):
        result = S.compile_brief(BRIEF)
        checks = S.verifiers(result)
        self.assertGreaterEqual(len(checks), 2)
        names = {c.name for c in checks}
        self.assertEqual(names, {"contract", "requirements"})

        session = HarnessSession(StubBackend())
        session.apply_ops(S.to_ops(result))
        for check in checks:
            report = check.check(session.backend, session.opdag)
            # The stub carries no kernel, so measurable asks INFO-skip rather
            # than erroring -- what matters is that the route runs end to end.
            self.assertIsNotNone(report.diagnostics)

    def test_spec_drives_the_master_layout_and_the_ops_apply(self):
        result = S.compile_brief(BRIEF)
        skeleton = S.skeleton(result)
        self.assertEqual(skeleton.envelope.depth, 8.0)
        self.assertIn("origin", skeleton.datum_names())

        ops = S.to_ops(result)
        self.assertGreater(len(ops), 0)
        session = HarnessSession(StubBackend())
        applied = session.apply_ops(ops)
        self.assertTrue(applied.ok, [d.message for d in applied.diagnostics])
        self.assertEqual(applied.applied, len(ops))

    def test_clarifier_asks_for_what_the_brief_left_out(self):
        result = S.compile_brief("a plate 100 mm long")
        # The brief names no count, no tolerance and no load: three real gaps.
        self.assertIn("quantity", result.missing)
        self.assertIn("tolerance", result.missing)
        self.assertIn("load", result.missing)
        self.assertEqual(len(result.questions), len(result.missing))

    def test_a_complete_brief_leaves_fewer_gaps_than_a_bare_one(self):
        full = S.compile_brief(BRIEF)
        bare = S.compile_brief("a plate 100 mm long")
        self.assertLess(len(full.missing), len(bare.missing))

    def test_the_two_clarifiers_are_not_merged(self):
        interp = S.interpret(BRIEF)
        interview_q, interview_missing = S.clarify(BRIEF, interp, "interview")
        ambiguity_q, ambiguity_missing = S.clarify(BRIEF, interp, "ambiguity")
        # Different question spaces over the same brief -- selected, never fused.
        self.assertNotEqual(interview_q, ambiguity_q)
        self.assertNotEqual(interview_missing, ambiguity_missing)

    def test_code_leakage_in_a_brief_is_blocking(self):
        result = S.compile_brief(
            'cq.Workplane("XY").rect(20, 10).extrude(5)')
        self.assertFalse(result.ok)
        self.assertTrue(any(i.startswith("leakage:") for i in result.issues),
                        result.issues)


class TestBlockingIsDeclaredNotHardcoded(unittest.TestCase):
    """``_blocking`` must read the DECLARED blocking flags, not one linter's name.

    Leakage is the only blocking linter today, so a ``startswith('leakage:')``
    hardcode passes every other test in this file while quietly ignoring the
    flag. These tests register a SECOND blocking linter, which is the only way
    to tell the two implementations apart.
    """

    def _register(self, name, fn, blocking):
        S._LINTERS[name] = (fn, "harnesscad.domain.spec.clarify_leakage",
                            "test-only linter", blocking)
        blocking_set = S._BLOCKING
        S._BLOCKING = frozenset(
            n for n, v in S._LINTERS.items() if v[3])
        self.addCleanup(setattr, S, "_BLOCKING", blocking_set)
        self.addCleanup(S._LINTERS.pop, name, None)

    def test_a_second_blocking_linter_actually_blocks(self):
        self._register("banned", lambda b: ["banned: brief names a vendor"], True)
        result = S.compile_brief("a plate 100 mm long")
        self.assertTrue(any(i.startswith("banned:") for i in result.issues),
                        result.issues)
        # Fails under the `startswith("leakage:")` hardcode: the declared flag
        # is ignored and the brief compiles ok.
        self.assertFalse(result.ok, result.issues)

    def test_a_non_blocking_linters_findings_do_not_block(self):
        self._register("advice", lambda b: ["advice: consider a fillet"], False)
        result = S.compile_brief("a plate 100 mm long")
        self.assertTrue(any(i.startswith("advice:") for i in result.issues),
                        result.issues)
        self.assertTrue(result.ok, result.issues)

    def test_style_findings_from_the_blocking_leakage_linter_do_not_block(self):
        # `leakage` is blocking, but it also emits advisory `style:` lines that
        # are not code leakage. Blocking is attributed by the finding's own
        # prefix, so a style-only brief stays ok.
        self.assertIn("leakage", S._BLOCKING)
        self.assertFalse(S._blocking(["style: brief is very long"]))
        self.assertFalse(S._blocking(["lint-error: leakage raised OSError: x"]))
        self.assertTrue(S._blocking(["leakage: code in a brief: cq.Workplane"]))

    def test_compile_is_deterministic(self):
        self.assertEqual(S.compile_brief(BRIEF).to_dict(),
                         S.compile_brief(BRIEF).to_dict())


class TestExpressPart21(unittest.TestCase):
    def test_schema_parses_with_its_inheritance_graph(self):
        schema, graph = S.parse_schema(SCHEMA)
        self.assertIn("plate", schema.entities)
        self.assertTrue(graph.is_subtype_of("plate", "shape"))

    def test_a_conforming_part21_file_validates(self):
        report = S.validate_part21(GOOD_P21, SCHEMA)
        self.assertTrue(report.ok, [str(i) for i in report.issues])
        self.assertEqual(report.checked, 1)

    def test_a_nonconforming_part21_file_is_REJECTED(self):
        report = S.validate_part21(BAD_P21, SCHEMA)
        self.assertFalse(report.ok)
        details = " ".join(str(i) for i in report.issues)
        # wrong arity: PLATE inherits `name`, so it takes TWO attributes
        self.assertIn("PLATE", details)
        # and an entity the schema never declared
        self.assertIn("unknown entity type", details)

    def test_validation_is_deterministic(self):
        first = [str(i) for i in S.validate_part21(BAD_P21, SCHEMA).issues]
        second = [str(i) for i in S.validate_part21(BAD_P21, SCHEMA).issues]
        self.assertEqual(first, second)


class TestStructuredFormats(unittest.TestCase):
    def test_formats_are_keyed_by_name_never_sniffed(self):
        for expected in ("urdf", "srdf", "plate", "rim", "express"):
            self.assertIn(expected, S.formats())

    def test_rim_spec_format(self):
        parsed = S.parse_format("rim", "17 4H PCD 114.3 7J ET34 C/B:73")
        self.assertEqual(parsed["spec"].diameter_code, 17)

    def test_unknown_format_raises(self):
        with self.assertRaises(S.UnknownRoute):
            S.parse_format("nope", "x")

    def test_srdf_without_its_urdf_refuses(self):
        with self.assertRaises(S.SpecError):
            S.parse_format("srdf", "<robot name='r'/>")


class TestCoverageAndMetrics(unittest.TestCase):
    def test_coverage_of_a_spec_against_itself_is_total(self):
        report = S.coverage(BRIEF, [BRIEF])
        self.assertEqual(report.global_ratio, 1.0)

    def test_clarifier_efficiency_is_scored(self):
        score = S.score_clarification(["material", "tolerance"],
                                      ["material", "load"])
        self.assertGreater(score.f1, 0.0)
        self.assertLess(score.f1, 1.0)


if __name__ == "__main__":
    unittest.main()
