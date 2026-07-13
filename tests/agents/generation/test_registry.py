"""The generation surface: discovery, strategies that build a solid with the stub
planner, the rival correction loops, and the rag/memory/context layer that feeds them.

Everything here runs with NO model and NO network: the planner is deterministic.
"""

import json
import tempfile
import unittest

from harnesscad.agents.generation import registry as G
from harnesscad.agents.memory.error_notebook import ErrorNotebook
from harnesscad.agents.memory.store import MemoryStore
from harnesscad.io.surfaces.server import CISPServer

BRIEF = "a 60 x 40 x 12 mm mounting plate"

#: An op stream the backend WILL reject (a zero-width rectangle). The correction
#: loops must repair it; the direct strategy must not pretend it worked.
BROKEN_OPS = [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0, "w": 0.0, "h": 40.0},
    {"op": "extrude", "sketch": "sk1", "distance": 12.0},
]


class TestDiscovery(unittest.TestCase):
    def test_discovers_more_than_five_real_generation_modules(self):
        from harnesscad import registry as capability_registry

        indexed = {e.dotted for e in capability_registry.find(package="generation")}
        bound = set()
        for name in G.strategies():
            bound.update(G.strategy(name).modules)
        self.assertGreater(len(bound), 5)
        for dotted in bound:
            self.assertIn(dotted, indexed)      # nothing invented

    def test_every_strategy_is_described(self):
        self.assertGreaterEqual(len(G.strategies()), 6)
        for name in G.strategies():
            self.assertTrue(G.strategy(name).description)

    def test_unadapted_modules_are_reported_not_hidden(self):
        # These need a renderer / a VLM / a trainer, none of which exist here.
        for dotted in ("harnesscad.agents.generation.three_view",
                       "harnesscad.agents.generation.stepwise_visual_feedback",
                       "harnesscad.agents.generation.training_schedule",
                       "harnesscad.agents.generation.caption_feedback"):
            self.assertIn(dotted, G.unadapted())

    def test_unknown_strategy_raises(self):
        with self.assertRaises(G.UnknownStrategy):
            G.strategy("no-such-generator")


class TestStubPlanner(unittest.TestCase):
    def test_the_brief_is_read_not_invented(self):
        b = G.parse_brief(BRIEF)
        self.assertEqual((b.width, b.height, b.depth), (60.0, 40.0, 12.0))
        self.assertEqual(b.profile, "rectangle")
        # An unstated dimension falls back to the DOCUMENTED default, not a guess.
        self.assertEqual(G.parse_brief("a plate").width, 20.0)

    def test_the_stub_planner_needs_no_model_and_is_deterministic(self):
        planner = G.StubPlanner()
        first = planner.plan(BRIEF, None, None)
        second = planner.plan(BRIEF, None, None)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["op"], "new_sketch")
        self.assertTrue(any(op["op"] == "extrude" for op in first))

    def test_the_planner_repairs_a_diagnostic_rather_than_re_emitting_it(self):
        repaired = G.repair(BROKEN_OPS, BRIEF,
                            ["bad-value: rectangle w and h must be > 0"])
        self.assertEqual(repaired[1]["w"], 60.0)


class TestStrategiesDriveASession(unittest.TestCase):
    def _assert_built_a_solid(self, result):
        self.assertTrue(result.ok, result.diagnostics)
        self.assertTrue(result.summary["solid_present"])
        self.assertTrue(result.digest)
        self.assertIsNotNone(result.session)
        # The reported ops really rebuild the same model.
        server = CISPServer(backend="stub")
        replay = server.applyOps([dict(op) for op in result.ops])
        self.assertTrue(replay["ok"], replay)
        self.assertEqual(replay["digest"], result.digest)

    def test_direct_builds_a_solid_with_the_stub_planner(self):
        result = G.generate("direct", BRIEF)
        self._assert_built_a_solid(result)
        from harnesscad.domain.editing import registry as editing

        self.assertEqual(editing.shape_of(result.session).extents,
                         (60.0, 40.0, 12.0))

    def test_dual_loop_builds_a_solid_and_hits_the_design_plan(self):
        result = G.generate("dual_loop", BRIEF)
        self._assert_built_a_solid(result)
        self.assertTrue(result.detail["passed"])
        self.assertEqual(result.detail["plan"]["target_bbox_mm"], [60.0, 40.0, 12.0])
        self.assertTrue(result.detail["tier"])

    def test_verify_loop_builds_a_solid_and_answers_its_own_questions(self):
        result = G.generate("verify_loop", BRIEF)
        self._assert_built_a_solid(result)
        self.assertTrue(all(a == "Yes" for a in result.detail["answers"]))

    def test_prompt_evolution_builds_a_solid(self):
        result = G.generate("prompt_evolution", BRIEF)
        self._assert_built_a_solid(result)
        self.assertTrue(result.detail["converged"])

    def test_tiled_composes_one_program_from_several_tiles(self):
        result = G.generate(
            "tiled", "a 60 x 40 x 12 mm base plate. a 30 x 20 x 8 mm boss on top.")
        self._assert_built_a_solid(result)
        self.assertEqual(len(result.detail["tiles"]), 2)
        self.assertEqual(result.summary["sketch_count"], 2)
        code = result.detail["code"]
        self.assertIn("import cadquery", code)
        self.assertIn("result_0", code)          # the tile fragments were merged
        self.assertIn("result_1", code)

    def test_freecad_macro_lowers_a_macro_onto_ops(self):
        result = G.generate("freecad_macro", BRIEF)
        self._assert_built_a_solid(result)
        self.assertIn("Part", result.detail["source"])

    def test_worldcraft_places_real_assembly_instances(self):
        result = G.generate("worldcraft", BRIEF, objects=[
            {"id": "a", "category": "box", "half_extent": (2, 2, 2),
             "position": (0, 0, 2)},
            {"id": "b", "category": "box", "half_extent": (2, 2, 2),
             "position": (1, 1, 2)}])
        self._assert_built_a_solid(result)
        self.assertLess(result.detail["final_cost"], result.detail["initial_cost"])
        self.assertTrue(result.detail["satisfied"])       # the solver really solved
        self.assertEqual(sorted(result.detail["instances"]), ["a", "b"])
        self.assertEqual(
            len([op for op in result.ops if op["op"] == "add_instance"]), 2)

    def test_building_stacks_plates_into_a_real_model(self):
        result = G.generate("building", "a two-plate tower", plates=[
            {"name": "p1", "category": "vertex", "thickness": 2.0,
             "vertices": [(0, 0), (10, 0), (10, 6), (0, 6)]},
            {"name": "p2", "category": "vertex", "thickness": 3.0,
             "vertices": [(1, 1), (9, 1), (9, 5), (1, 5)]}])
        self._assert_built_a_solid(result)
        self.assertEqual(result.detail["height"], 5.0)
        self.assertEqual(result.detail["solids"], 2)

    def test_a_strategy_that_needs_input_it_was_not_given_refuses(self):
        result = G.generate("building", BRIEF)          # no plates
        self.assertFalse(result.ok)
        self.assertIn("Unsupported", result.diagnostics[0])


class TestCorrectionLoopsActuallyCorrect(unittest.TestCase):
    """Handed a program the backend rejects, each loop must repair it."""

    def test_direct_does_not_pretend_a_broken_program_worked(self):
        result = G._finish("direct", BROKEN_OPS, 1, {})
        self.assertFalse(result.ok)

    def test_dual_loop_repairs_a_rejected_program(self):
        result = G.generate("dual_loop", BRIEF, code=BROKEN_OPS)
        self.assertTrue(result.ok, result.diagnostics)
        self.assertTrue(result.summary["solid_present"])
        rect = [op for op in result.ops if op["op"] == "add_rectangle"][0]
        self.assertEqual(rect["w"], 60.0)
        inner = [t for t in result.detail["trace"] if t["stage"] == "inner"]
        self.assertTrue(inner)                               # the inner loop ran
        self.assertIn("pattern", inner[0])                   # KB2 was consulted

    def test_the_error_knowledge_base_matches_a_pattern_it_really_knows(self):
        broken_extrude = [
            {"op": "new_sketch", "plane": "XY"},
            {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0,
             "w": 60.0, "h": 40.0},
            {"op": "extrude", "sketch": "sk1", "distance": 0.0},
        ]
        result = G.generate("dual_loop", BRIEF, code=broken_extrude)
        self.assertTrue(result.ok, result.diagnostics)
        inner = [t for t in result.detail["trace"] if t["stage"] == "inner"]
        self.assertIsNotNone(inner[0]["pattern"])
        extrude = [op for op in result.ops if op["op"] == "extrude"][0]
        self.assertEqual(extrude["distance"], 12.0)

    def test_verify_loop_repairs_a_rejected_program(self):
        result = G.generate("verify_loop", BRIEF, code=BROKEN_OPS)
        self.assertTrue(result.ok, result.diagnostics)
        self.assertGreater(result.detail["repair_attempts"], 0)
        rect = [op for op in result.ops if op["op"] == "add_rectangle"][0]
        self.assertEqual(rect["w"], 60.0)


class TestRivals(unittest.TestCase):
    def test_the_rival_families_are_exposed_by_name(self):
        families = G.rivals()
        self.assertEqual(families["correction"],
                         ("dual_loop", "verify_loop", "prompt_evolution"))
        self.assertEqual(families["retrieval"], ("hybrid", "sphere_knn"))
        for name in families["correction"]:
            self.assertIn(name, G.strategies())

    def test_the_correction_rivals_report_their_own_work_and_are_never_averaged(self):
        results = {name: G.generate(name, BRIEF, code=BROKEN_OPS)
                   if name != "prompt_evolution" else G.generate(name, BRIEF)
                   for name in G.rivals()["correction"]}
        for name, result in results.items():
            self.assertTrue(result.ok, (name, result.diagnostics))
        # Each keeps its OWN trace/vocabulary -- there is no merged number.
        self.assertIn("trace", results["dual_loop"].detail)
        self.assertIn("answers", results["verify_loop"].detail)
        self.assertIn("constraints", results["prompt_evolution"].detail)
        self.assertFalse(hasattr(G, "run_all"))
        self.assertFalse(hasattr(G, "ensemble"))

    def test_the_two_retrieval_backends_are_different_and_never_merged(self):
        query = "evolutionary search over cad programs"
        hybrid = [d for d, _ in G.retrieve(query, k=5)]
        knn = [d for d, _ in G.retrieve(query, k=5, backend="sphere_knn")]
        self.assertEqual(len(hybrid), 5)
        self.assertEqual(len(knn), 5)
        self.assertNotEqual(hybrid, knn)          # they genuinely disagree
        with self.assertRaises(G.RivalBlend):
            G.retrieve(query, backend="hybrid+sphere_knn")


class TestFailureIsCaptured(unittest.TestCase):
    def test_a_raising_strategy_is_captured_not_fatal(self):
        class Exploding:
            def plan(self, *_a, **_kw):
                raise RuntimeError("the planner blew up")

        result = G.generate("dual_loop", BRIEF, planner=Exploding())
        self.assertFalse(result.ok)
        self.assertIn("RuntimeError", result.diagnostics[0])
        # The surface still works afterwards.
        self.assertTrue(G.generate("direct", BRIEF).ok)

    def test_an_unlowerable_macro_primitive_is_refused_not_approximated(self):
        from harnesscad.agents.generation import freecad_macro as fm

        macro = fm.FreeCADMacro(primitives=[fm.Primitive(
            name="S", kind="sphere", params={"radius": 5.0})])
        result = G.generate("freecad_macro", BRIEF, macro=macro)
        self.assertFalse(result.ok)
        self.assertIn("Unsupported", result.diagnostics[0])


class TestRetrievalMemoryContext(unittest.TestCase):
    def test_retrieval_runs_over_this_repo_s_own_capability_index(self):
        hits = G.retrieve("evolutionary cad program search", k=5)
        self.assertEqual(len(hits), 5)
        for dotted, _score in hits:
            self.assertTrue(dotted.startswith("harnesscad."))

    def test_reranking_through_the_error_notebook_is_available(self):
        hits = G.retrieve("sketch constraint solver", k=3, notebook=ErrorNotebook())
        self.assertEqual(len(hits), 3)

    def test_the_api_context_is_built_from_the_real_op_vocabulary(self):
        cards = G.api_cards()
        methods = {c.method for c in cards}
        self.assertIn("add_rectangle", methods)
        self.assertIn("extrude", methods)
        context = G.api_context("add a rectangle to a sketch", top_k=3)
        self.assertIn("cisp.add_rectangle", context)

    def test_memory_records_the_run_and_ages_the_store(self):
        store = MemoryStore()
        result = G.generate("direct", BRIEF)
        report = G.remember(store, BRIEF, result)
        self.assertEqual(report["episodes"], 1)
        self.assertEqual(store.recall_episodic(BRIEF, k=1)[0].outcome, "ok")
        self.assertEqual(report["saliences"][0].recall_count, 1)

    def test_the_assembled_context_is_budgeted_and_inspectable(self):
        store = MemoryStore()
        result = G.generate("direct", BRIEF)
        G.remember(store, BRIEF, result)
        with tempfile.TemporaryDirectory() as tmp:
            ctx = G.assemble_context(BRIEF, session=result.session, store=store,
                                     stage_dir=tmp)
        self.assertTrue(ctx["report"]["ok"])
        self.assertLessEqual(ctx["report"]["total"], ctx["report"]["budget"])
        self.assertTrue(ctx["retrieved"])
        self.assertTrue(ctx["memory_map"])
        self.assertIn(BRIEF, ctx["episodes"])
        self.assertIn("# BRIEF", ctx["staged"])
        self.assertGreater(ctx["read_plan"]["total_tokens"], 0)

    def test_an_over_tight_budget_reports_overflow_instead_of_lying(self):
        ctx = G.assemble_context(BRIEF, budget=50)
        self.assertFalse(ctx["report"]["ok"])
        self.assertGreater(ctx["report"]["overflow"], 0)


class TestSerialisation(unittest.TestCase):
    def test_a_result_round_trips_through_json(self):
        result = G.generate("dual_loop", BRIEF)
        text = json.dumps(result.to_dict(), sort_keys=True)
        self.assertEqual(json.loads(text)["digest"], result.digest)


if __name__ == "__main__":
    unittest.main()
