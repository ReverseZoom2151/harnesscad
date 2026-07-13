"""Tests for the evolutionary data engine (lineage, QD archive, trace slicing,
cube rotations, template collapse, staged validation) and the host-neutral
intent-resolution / confirmation lifecycle.

Rewritten from bare pytest-style module functions (never collected by
``python -m unittest``) into unittest.TestCase classes.
"""

import unittest

from harnesscad.io.adapters.rhino_contract import (HostCapabilities, HostResult, HostScript,
                                     validate_script)
from harnesscad.agents.agent.host_feedback import HostProposal, confirm, execute, preview, refine
from harnesscad.agents.agent.intent_resolution import resolve_intent
from harnesscad.eval.bench.harness.evolution_dynamics import evolution_dynamics, lineage_stats
from harnesscad.eval.bench.data.nl_cad_casebook import evaluate_case, paper_casebook
from harnesscad.eval.bench.data.operator_profile import operator_profile
from harnesscad.data.dataengine.audit.template_collapse import identifier_leakage, template_collapse
from harnesscad.data.datagen.cube_rotations import (apply_rotation, cube_rotations,
                                    inverse_rotation, rewrite_calls)
from harnesscad.data.datagen.evolution import (GeneratorRecord, sample_parents, termination,
                               validate_lineage)
from harnesscad.data.datagen.evolution_validation import canonical_seven_views, validate_candidate
from harnesscad.data.datagen.parameter_qd import fill_archive
from harnesscad.data.datagen.trace_slice import slice_trace, verify_slice


class EvolutionLineageTest(unittest.TestCase):
    RECORDS = (GeneratorRecord("a", "a", "", "", ""),
               GeneratorRecord("b", "b", "", "", "", ("a",)))

    def test_well_formed_lineage_reports_no_violations(self):
        self.assertFalse(validate_lineage(self.RECORDS))

    def test_parent_sampling_is_seeded_and_reproducible(self):
        self.assertEqual(sample_parents(self.RECORDS, 1, seed=4),
                         sample_parents(self.RECORDS, 1, seed=4))

    def test_zero_novelty_generations_terminate_on_saturation(self):
        self.assertEqual(termination(({"novelty_ratio": 0},) * 3, budget=9),
                         "novelty-saturation")

    def test_lineage_stats_report_the_maximum_depth(self):
        self.assertEqual(lineage_stats(self.RECORDS)["max_depth"], 1)


class QualityDiversityArchiveTest(unittest.TestCase):
    @staticmethod
    def _evaluate(params):
        return {"valid": True, "solid_count": 1, "watertight": True,
                "bounds": (-40, -30, -30, 40, 30, 30),
                "descriptor": (params[0],)}

    def test_archive_stops_on_budget_with_only_novel_entries(self):
        report = fill_archive(lambda i: (i % 2,), self._evaluate,
                              target=3, budget=4, epsilon=.1)
        self.assertEqual(len(report["entries"]), 2)
        self.assertEqual(report["termination"], "budget")

    def test_duplicate_descriptors_are_rejected_as_not_novel(self):
        report = fill_archive(lambda i: (i % 2,), self._evaluate,
                              target=3, budget=4, epsilon=.1)
        self.assertTrue(any(attempt["reason"] == "not-novel"
                            for attempt in report["attempts"]))


class TraceSliceTest(unittest.TestCase):
    TRACE = (
        {"kind": "log", "statement": "print('x')"},
        {"kind": "extrude",
         "statement": "shape = cq.Workplane('XY').box(width, 2, 3)",
         "output": "shape"},
    )

    def test_slice_drops_side_effects_and_ends_with_the_result_binding(self):
        source = slice_trace((("width", 4),), self.TRACE)
        self.assertNotIn("print", source)
        self.assertTrue(source.endswith("result = shape\n"))

    def test_slice_is_gated_on_geometric_equivalence(self):
        source = slice_trace((("width", 4),), self.TRACE)
        verified = verify_slice(source, lambda s: "shape",
                                lambda value: {"equivalent": value == "shape"})
        self.assertTrue(verified["accepted"])


class CubeRotationsTest(unittest.TestCase):
    def test_the_rotation_group_has_exactly_24_elements(self):
        self.assertEqual(len(cube_rotations()), 24)

    def test_every_rotation_is_reversible_by_its_inverse(self):
        vector = (1, 2, 3)
        for matrix in cube_rotations():
            self.assertEqual(
                apply_rotation(inverse_rotation(matrix),
                               apply_rotation(matrix, vector)),
                vector)

    def test_rewrite_touches_global_calls_but_not_local_ones(self):
        vector = (1, 2, 3)
        matrix = cube_rotations()[7]
        calls = ({"kind": "line", "args": {"point": vector}},
                 {"kind": "translate_global", "args": {"vector": vector}})
        rewritten = rewrite_calls(calls, matrix)
        self.assertEqual(rewritten[0], calls[0])
        self.assertNotEqual(rewritten[1]["args"]["vector"], vector)


class TemplateCollapseTest(unittest.TestCase):
    RECORDS = (
        {"family": "a", "code": "x=box(1)",
         "operations": ("box", "extrude"), "face_count": 6},
        {"family": "a", "code": "y=box(2)",
         "operations": ("box",), "face_count": 5},
    )

    def test_single_family_corpus_is_fully_concentrated(self):
        report = template_collapse(self.RECORDS)
        self.assertEqual(report["families"][0]["concentration"], 1)

    def test_identifier_leakage_surfaces_reused_identifiers(self):
        self.assertIn("x", identifier_leakage(self.RECORDS))

    def test_operator_profile_reports_rates_and_reference_delta(self):
        profile = operator_profile(self.RECORDS, reference={"box": .5})
        self.assertEqual(profile["operation_rates"]["box"], 1)
        self.assertEqual(profile["operation_delta"]["box"], .5)


class EvolutionValidationTest(unittest.TestCase):
    def test_validation_stops_at_the_first_failed_stage(self):
        ok = lambda value: {"accepted": True, "output": value}
        admission = validate_candidate(
            "candidate", execute=ok, integrity=ok, render=ok,
            semantic=lambda value: {"accepted": False, "reason": "mismatch"})
        self.assertEqual(admission.stage, "semantic")
        self.assertEqual(admission.repair_packet["reason"], "mismatch")

    def test_canonical_view_set_has_seven_views(self):
        self.assertEqual(len(canonical_seven_views()), 7)

    def test_falling_accept_rate_is_reported_as_diminishing_returns(self):
        dynamics = evolution_dynamics(
            ({"proposed": 10, "invalid": 1, "novel": 8, "accepted": 7},
             {"proposed": 10, "invalid": 3, "novel": 4, "accepted": 3}))
        self.assertTrue(dynamics["diminishing"])


class IntentResolutionTest(unittest.TestCase):
    def test_unseeded_random_intent_needs_clarification(self):
        unresolved = resolve_intent(
            "Create a 100 mm box at a random edge and union it")
        self.assertTrue(unresolved.needs_clarification)

    def test_seeded_random_intent_resolves_and_records_the_assumption(self):
        intent = resolve_intent(
            "Create a 100 mm box at a random edge, union and bake", seed=7)
        self.assertFalse(intent.needs_clarification)
        self.assertIn("random-choice-seed:7", intent.assumptions)

    def test_resolved_intent_fully_covers_the_first_casebook_case(self):
        intent = resolve_intent(
            "Create a 100 mm box at a random edge, union and bake", seed=7)
        case = paper_casebook()[0]
        result = evaluate_case(intent,
                               required_operations=case["operations"],
                               required_capabilities=case["capabilities"])
        self.assertEqual(result["intent_coverage"], 1)


class HostContractTest(unittest.TestCase):
    SCRIPT = HostScript("s", "rhinoscript", "Box()", ("box",), True)

    def test_script_within_host_capabilities_validates(self):
        self.assertFalse(
            validate_script(self.SCRIPT, HostCapabilities(frozenset({"box"}))))

    def test_preview_confirm_execute_then_refine_restarts_the_lifecycle(self):
        script = self.SCRIPT

        class Host:
            def preview(self, script):
                return {"safe": True}

            def execute(self, script):
                return HostResult(True, "done", "rollback")

        proposal = HostProposal("p", script, "make box")
        staged = preview(proposal, Host())
        done = execute(confirm(staged), Host())
        self.assertEqual(done.status, "executed")
        revised = refine(done, script)
        self.assertEqual(revised.status, "proposed")
        self.assertEqual(revised.lineage, ("p",))


if __name__ == "__main__":
    unittest.main()
