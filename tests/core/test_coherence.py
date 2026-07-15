"""Coherence tests — ONE loop, ONE gate on every write path, ONE feedback channel.

Every test here FAILS on the pre-collapse repository and PASSES after it. They
are the proof that the behaviour changed, and they are the reason the change is
allowed to change behaviour on three surfaces.

The findings under test (audit/book_hitchhiker_agentic.md sec.4):

  #1  the feedback gate was a property of ``agent.planner.Planner``, so any
      planner that was not that class -- the A2A surface's ``_PlatePlanner``,
      for one -- fed the model ungated HEURISTIC diagnostics;
  #2  four loops, and ``core/pipeline.build`` (the shipping path) used the
      weakest one;
  #6  ``Diagnostic.soundness`` was stripped by ``to_dict()``, so the tier died
      at every JSON boundary;
  #10 the only harness with a write gate was reachable through the ACP editor.
"""

from __future__ import annotations

import unittest

from harnesscad.agents.agent.planner import EMIT_OPS_TOOL, Planner
from harnesscad.agents.agent.runner import run
from harnesscad.agents.llm.structured import ParsedOps
from harnesscad.core.cisp.ops import AddRectangle, Extrude, NewSketch
from harnesscad.core.harness import AgentHarness
from harnesscad.core.loop import HarnessSession
from harnesscad.eval.reliability.executor import SessionToolExecutor
from harnesscad.eval.verifiers.soundness import HEURISTIC, PROVEN, TIERS
from harnesscad.eval.verifiers.verify import Diagnostic, Severity
from harnesscad.io.backends.stub import StubBackend


def _plate_ops():
    return [NewSketch(), AddRectangle(sketch="sk1", x=0, y=0, w=20, h=10),
            Extrude(sketch="sk1", distance=5.0)]


class _RecordingPlanner:
    """A planner that is NOT ``agent.planner.Planner``.

    This is the shape the A2A surface ships (``_PlatePlanner``). It records the
    diagnostics the harness hands it, so we can assert on what the model would
    have been told.
    """

    def __init__(self, plans):
        self._plans = list(plans)
        self.seen_diagnostics = []

    def plan_parsed(self, brief, state_summary=None, diagnostics=None):
        self.seen_diagnostics.append(list(diagnostics or []))
        if self._plans:
            return ParsedOps(list(self._plans.pop(0)))
        return ParsedOps([])


# ---------------------------------------------------------------------------
# #6 -- the soundness tier must survive serialization.
# ---------------------------------------------------------------------------
class TestSoundnessOnTheWire(unittest.TestCase):
    def test_to_dict_carries_the_resolved_tier(self):
        d = Diagnostic(Severity.ERROR, "over-constrained", "negative DOF")
        self.assertEqual(d.to_dict()["soundness"], PROVEN)

    def test_a_stamped_tier_wins_over_the_code_index(self):
        d = Diagnostic(Severity.ERROR, "preflight-THICKNESS_TOO_LARGE", "m",
                       soundness=PROVEN)
        self.assertEqual(d.to_dict()["soundness"], PROVEN)

    def test_an_unknown_code_fails_closed_to_heuristic(self):
        d = Diagnostic(Severity.ERROR, "some-future-rule", "m")
        self.assertEqual(d.to_dict()["soundness"], HEURISTIC)

    def test_v1_is_still_byte_identical_for_the_frozen_experiment(self):
        d = Diagnostic(Severity.ERROR, "infeasible-plan", "m", where="sk1")
        self.assertEqual(
            d.to_dict_v1(),
            {"severity": "error", "code": "infeasible-plan", "message": "m",
             "where": "sk1"})
        self.assertNotIn("soundness", d.to_dict_v1())

    def test_the_tier_survives_an_applyops_json_boundary(self):
        # ApplyOpsResult.to_dict is what MCP, A2A and the tracer serialise.
        from harnesscad.core.cisp.protocol import ApplyOpsResult
        r = ApplyOpsResult(False, 0, "d", diagnostics=[
            Diagnostic(Severity.ERROR, "infeasible-plan", "guess")])
        self.assertEqual(r.to_dict()["diagnostics"][0]["soundness"], HEURISTIC)


# ---------------------------------------------------------------------------
# #1 -- the feedback gate is architectural, not a property of one planner class.
# ---------------------------------------------------------------------------
class TestFeedbackGateAtTheHarnessBoundary(unittest.TestCase):
    def _harness_with(self, planner, **kw):
        session = HarnessSession(StubBackend())
        return AgentHarness(session, planner, max_iterations=2, **kw)

    def test_a_heuristic_diagnostic_never_reaches_a_non_planner_planner(self):
        # A HEURISTIC finding surfaced by a harness-level verifier must not be
        # handed to a planner that has no gate of its own. This is the A2A bug.
        class _HeuristicVerifier:
            name = "precheck"   # declared HEURISTIC in the soundness table

            def check(self, backend, opdag):
                from harnesscad.eval.verifiers.verify import VerifyReport
                return VerifyReport([Diagnostic(
                    Severity.ERROR, "infeasible-plan",
                    "hole diameter 30 >= plate thickness 8",
                    soundness=HEURISTIC)])

        planner = _RecordingPlanner([_plate_ops(), _plate_ops()])
        h = self._harness_with(planner, verifiers=[_HeuristicVerifier()])
        h.run("a washer")

        self.assertGreaterEqual(len(planner.seen_diagnostics), 2)
        fed_back = planner.seen_diagnostics[1]
        codes = [d.get("code") if isinstance(d, dict) else d.code for d in fed_back]
        self.assertNotIn("infeasible-plan", codes)

    def test_the_withheld_finding_is_still_reported_to_the_caller(self):
        # Narrowed to the model, not silenced. It must still be in the run.
        class _HeuristicVerifier:
            name = "precheck"

            def check(self, backend, opdag):
                from harnesscad.eval.verifiers.verify import VerifyReport
                return VerifyReport([Diagnostic(
                    Severity.ERROR, "infeasible-plan", "a guess",
                    soundness=HEURISTIC)])

        h = self._harness_with(_RecordingPlanner([_plate_ops()]),
                               verifiers=[_HeuristicVerifier()])
        run_ = h.run("a washer")
        codes = [d["code"] for d in run_.diagnostics]
        self.assertIn("infeasible-plan", codes)

    def test_the_policy_is_a_parameter_not_a_second_loop(self):
        # The pre-tiering behaviour (feed everything back) is reachable as a
        # PARAMETER on the one loop -- which is how the pressure comparison is
        # re-run, rather than by keeping a second loop around.
        class _HeuristicVerifier:
            name = "precheck"

            def check(self, backend, opdag):
                from harnesscad.eval.verifiers.verify import VerifyReport
                return VerifyReport([Diagnostic(
                    Severity.ERROR, "infeasible-plan", "a guess",
                    soundness=HEURISTIC)])

        planner = _RecordingPlanner([_plate_ops(), _plate_ops()])
        h = self._harness_with(planner, verifiers=[_HeuristicVerifier()],
                               feedback_tiers=TIERS)
        h.run("a washer")
        fed_back = planner.seen_diagnostics[1]
        codes = [d.get("code") if isinstance(d, dict) else d.code for d in fed_back]
        self.assertIn("infeasible-plan", codes)


# ---------------------------------------------------------------------------
# #10 -- the write gate is on by default, on every surface.
# ---------------------------------------------------------------------------
class TestWriteGateOnEveryPath(unittest.TestCase):
    def test_the_default_harness_dispatches_through_the_gated_executor(self):
        h = AgentHarness(HarnessSession(StubBackend()), _RecordingPlanner([]))
        self.assertIsInstance(h.executor, SessionToolExecutor)

    def test_a_guardrail_violating_op_never_touches_the_session(self):
        # Extrude(distance=0) is a guardrail violation. Under the old default
        # (`session.apply_ops`) it reached the backend; the hard gate now blocks
        # it BEFORE the model is mutated (block-and-correct).
        session = HarnessSession(StubBackend())
        bad = [NewSketch(), AddRectangle(sketch="sk1", x=0, y=0, w=20, h=10),
               Extrude(sketch="sk1", distance=0.0)]
        h = AgentHarness(session, _RecordingPlanner([bad]), max_iterations=1)
        run_ = h.run("a zero-height plate")

        self.assertFalse(run_.ok)
        self.assertFalse(session.summary()["solid_present"])
        self.assertEqual(run_.trajectory[0]["rejected"]["reason"],
                         "guardrail-blocked")

    def test_gated_false_restores_the_ungated_path_explicitly(self):
        h = AgentHarness(HarnessSession(StubBackend()), _RecordingPlanner([]),
                         gated=False)
        self.assertIsNone(h.executor)

    def test_a_per_op_tool_executor_is_rejected_not_mis_called(self):
        # The old _dispatch duck-typed: handed a bare ToolExecutor (whose
        # signature is execute(op, session)) it fell back to fn(self.session,
        # ops) and silently called execute(op=session, session=ops).
        from harnesscad.eval.reliability.executor import ToolExecutor
        h = AgentHarness(HarnessSession(StubBackend()),
                         _RecordingPlanner([_plate_ops()]),
                         executor=ToolExecutor(), max_iterations=1)
        with self.assertRaises(TypeError):
            h.run("a plate")


# ---------------------------------------------------------------------------
# #2 -- one loop. The shipping path and `runner.run` are views of AgentHarness.
# ---------------------------------------------------------------------------
class TestOneLoop(unittest.TestCase):
    def test_runner_run_drives_the_harness_and_yields_a_trajectory_run(self):
        session = HarnessSession(StubBackend())
        result = run(session, _RecordingPlanner([_plate_ops()]), "a plate")
        self.assertTrue(result.ok)
        self.assertTrue(session.summary()["solid_present"])

    def test_runner_run_now_has_loop_detection_as_a_parameter(self):
        # The minimal loop had none. The one loop takes it as a collaborator.
        # Loop detection fires on the AGENT re-emitting the same stuck plan
        # ACROSS iterations, not on a single plan that repeats an op signature
        # (a plate is four identical distance constraints -- legitimate content,
        # not oscillation). So the fixture is a plan that never applies -- an
        # extrude of a sketch that does not exist -- re-emitted turn after turn:
        # the agent is stuck, never converges, and the loop is detected.
        from harnesscad.eval.reliability.loopdetect import LoopDetector
        session = HarnessSession(StubBackend())
        over = [Extrude(sketch="nope", distance=5.0)]   # never applies; re-emitted
        result = run(session, _RecordingPlanner([over] * 5), "loop me",
                     max_iters=5, loop_detector=LoopDetector())
        self.assertFalse(result.ok)

    def test_pipeline_build_returns_the_harness_trajectory(self):
        # The shipping path produced no trajectory at all before the collapse.
        from harnesscad.core.pipeline import build

        class _LLM:
            def complete(self, messages, tools=None, response_schema=None, **o):
                from harnesscad.agents.llm.base import CompletionResult
                import json
                return CompletionResult(text=json.dumps([
                    {"op": "new_sketch", "plane": "XY"},
                    {"op": "add_rectangle", "sketch": "sk1", "x": 0, "y": 0,
                     "w": 20, "h": 10},
                    {"op": "extrude", "sketch": "sk1", "distance": 5.0},
                ]))

            def stream(self, *a, **k):
                raise NotImplementedError

        out = build("a 20x10x5 plate", llm=_LLM(), backend="stub")
        self.assertTrue(out["ok"])
        self.assertIn("trajectory", out)
        self.assertIn("run_id", out)
        self.assertEqual(out["stop_reason"], "converged")
        self.assertTrue(out["trajectory"][0]["op_signatures"])


# ---------------------------------------------------------------------------
# The tool surface: the model gets the catalogue we export to strangers.
# ---------------------------------------------------------------------------
class TestToolSurface(unittest.TestCase):
    def test_emit_ops_carries_the_five_component_description(self):
        text = EMIT_OPS_TOOL.description
        for component in ("When to use:", "When NOT to use:", "Side effects:",
                          "Output:"):
            self.assertIn(component, text)

    def test_every_op_schema_reaches_the_model_with_its_description(self):
        items = EMIT_OPS_TOOL.parameters["properties"]["ops"]["items"]["anyOf"]
        titles = {s["title"] for s in items}
        for tag in ("new_sketch", "extrude", "fillet", "shell", "boolean"):
            self.assertIn(tag, titles)
        shell = next(s for s in items if s["title"] == "shell")
        self.assertIn("When NOT to use:", shell["description"])
        self.assertEqual(shell["properties"]["op"]["const"], "shell")

    def test_the_planner_actually_sends_the_tools(self):
        # use_tool defaulted to False: the good descriptions were built and then
        # not sent.
        self.assertTrue(Planner.__init__.__defaults__[0])


if __name__ == "__main__":
    unittest.main()
