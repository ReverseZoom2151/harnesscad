"""Behaviour tests for the AgentHarness ReAct orchestrator (harness.py).

Everything here is deterministic and offline: a MockPlanner (good / oscillating /
bad-then-good / never-converge variants) drives a real HarnessSession over the
dependency-free StubBackend, with an InMemoryTracer capturing the event stream.
No network, no wall clock.
"""

import unittest

from harnesscad.core.cisp.ops import NewSketch, AddRectangle, Constrain, Extrude
from harnesscad.io.backends.stub import StubBackend
from harnesscad.core.loop import HarnessSession
from harnesscad.eval.reliability.loopdetect import LoopDetector
from harnesscad.agents.llm.structured import ParsedOps
from harnesscad.core.trace import InMemoryTracer
from harnesscad.core.contract import Contract
from harnesscad.core.harness import AgentHarness, HarnessRun, HARNESS_EVENT_KINDS


# --- fixtures --------------------------------------------------------------
def good_ops():
    """A verifying plan: rectangle sketch, fully constrained, extruded to a solid.

    Mirrors tests/test_loop.py's valid batch: rectangle adds 4 DOF, four distance
    constraints zero them out, extrude yields a solid. 7 ops, all verify.
    """
    return (
        [NewSketch(), AddRectangle(sketch="sk1")]
        + [Constrain(kind="distance", a="e1", value=10.0) for _ in range(4)]
        + [Extrude(sketch="sk1", distance=5.0)]
    )


def bad_ops():
    """A plan whose FIRST op is rejected by the backend (unknown sketch ref).

    Because it fails on op[0], nothing is applied and the session state is left
    untouched — so a later good plan applies cleanly (no duplicate prefix)."""
    return [Extrude(sketch="nope", distance=5.0)]


class MockPlanner:
    """Returns a scripted ParsedOps per call; the last script repeats.

    Each script entry is either a list of ops or a string (a parse-error message).
    Records every (brief, state_summary, diagnostics) it was called with so tests
    can assert diagnostics were fed back for repair.
    """

    def __init__(self, scripts):
        self.scripts = scripts
        self.calls = 0
        self.seen = []

    def plan_parsed(self, brief, state_summary=None, diagnostics=None):
        self.seen.append((brief, state_summary, diagnostics))
        idx = min(self.calls, len(self.scripts) - 1)
        self.calls += 1
        item = self.scripts[idx]
        if isinstance(item, str):
            return ParsedOps([], error=item)
        return ParsedOps(list(item))


class NeverConvergePlanner:
    """Emits a DISTINCT rejected op every call: never converges, never oscillates."""

    def __init__(self):
        self.calls = 0

    def plan_parsed(self, brief, state_summary=None, diagnostics=None):
        self.calls += 1
        return ParsedOps([Extrude(sketch=f"missing{self.calls}", distance=5.0)])


def _fresh_session(tracer=None):
    return HarnessSession(StubBackend(), tracer=tracer)


# --- tests -----------------------------------------------------------------
class TestConvergence(unittest.TestCase):
    def test_good_brief_converges_with_checkpoint(self):
        tracer = InMemoryTracer()
        session = _fresh_session()
        harness = AgentHarness(
            session, MockPlanner([good_ops()]), tracer=tracer)
        run = harness.run("a 10x10x5 block")

        self.assertIsInstance(run, HarnessRun)
        self.assertTrue(run.ok)
        self.assertEqual(run.stop_reason, "converged")
        self.assertEqual(run.iterations, 1)
        self.assertEqual(run.applied, 7)
        self.assertTrue(run.contract_ok)
        # Sane trajectory: one iteration, converged, dispatch ok.
        self.assertEqual(len(run.trajectory), 1)
        entry = run.trajectory[0]
        self.assertTrue(entry["converged"])
        self.assertTrue(entry["dispatch_ok"])
        self.assertEqual(entry["applied"], 7)
        # A harness-level checkpoint was emitted on success.
        self.assertTrue(any(e["kind"] == "checkpoint" for e in tracer.events))

    def test_solid_actually_built(self):
        session = _fresh_session()
        harness = AgentHarness(session, MockPlanner([good_ops()]))
        run = harness.run("block")
        self.assertTrue(run.ok)
        self.assertTrue(session.summary()["solid_present"])
        self.assertEqual(session.summary()["feature_count"], 1)


class TestLoopDetection(unittest.TestCase):
    def test_oscillating_planner_halts_with_loop(self):
        session = _fresh_session()
        harness = AgentHarness(
            session,
            MockPlanner([bad_ops()]),          # same rejected op forever
            loop_detector=LoopDetector(window=6, threshold=3),
        )
        run = harness.run("oscillate")

        self.assertFalse(run.ok)
        self.assertEqual(run.stop_reason, "loop")
        # Detector trips on the 3rd emission (threshold=3): 3 iterations recorded.
        self.assertEqual(run.iterations, 3)
        self.assertTrue(run.trajectory[-1]["looped"])
        # Nothing was ever successfully applied.
        self.assertEqual(run.applied, 0)


class TestRepair(unittest.TestCase):
    def test_bad_first_plan_is_repaired(self):
        session = _fresh_session()
        planner = MockPlanner([bad_ops(), good_ops()])
        harness = AgentHarness(session, planner)
        run = harness.run("repair me")

        self.assertTrue(run.ok)
        self.assertEqual(run.stop_reason, "converged")
        self.assertEqual(run.iterations, 2)
        # Iteration 0 failed to dispatch; iteration 1 converged.
        self.assertFalse(run.trajectory[0]["dispatch_ok"])
        self.assertFalse(run.trajectory[0]["converged"])
        self.assertTrue(run.trajectory[1]["converged"])
        # The failure diagnostics were fed back into the second plan call.
        second_call_diagnostics = planner.seen[1][2]
        self.assertTrue(second_call_diagnostics)

    def test_parse_error_is_repaired(self):
        session = _fresh_session()
        planner = MockPlanner(["model emitted garbage", good_ops()])
        harness = AgentHarness(session, planner)
        run = harness.run("bad json first")

        self.assertTrue(run.ok)
        self.assertEqual(run.iterations, 2)
        # First trajectory entry carries the parse-error diagnostic.
        diags = run.trajectory[0]["diagnostics"]
        self.assertTrue(any(d["code"] == "plan-parse-error" for d in diags))


class TestMaxIterations(unittest.TestCase):
    def test_max_iterations_is_honored(self):
        session = _fresh_session()
        harness = AgentHarness(
            session, NeverConvergePlanner(), max_iterations=4)
        run = harness.run("never")

        self.assertFalse(run.ok)
        self.assertEqual(run.stop_reason, "max_iterations")
        self.assertEqual(run.iterations, 4)
        self.assertEqual(run.applied, 0)


class TestContract(unittest.TestCase):
    def test_satisfiable_contract_sets_contract_ok(self):
        session = _fresh_session()
        harness = AgentHarness(session, MockPlanner([good_ops()]))
        contract = Contract(name="block", min_features=1)
        run = harness.run("block", contract=contract)

        self.assertTrue(run.ok)
        self.assertTrue(run.contract_ok)
        self.assertEqual(run.stop_reason, "converged")

    def test_impossible_contract_stops_unsatisfied(self):
        session = _fresh_session()
        # Build once, then propose nothing more (empty plans): the contract's
        # feature_count=99 can never be met, so the run exhausts its iterations.
        planner = MockPlanner([good_ops(), []])
        harness = AgentHarness(session, planner, max_iterations=4)
        contract = Contract(name="impossible", feature_count=99)
        run = harness.run("impossible", contract=contract)

        self.assertFalse(run.ok)
        self.assertFalse(run.contract_ok)
        self.assertEqual(run.stop_reason, "max_iterations")
        # The geometry was still built (the contract, not the build, failed).
        self.assertTrue(session.summary()["solid_present"])


class TestTracing(unittest.TestCase):
    def test_events_share_stable_run_id(self):
        tracer = InMemoryTracer()
        session = _fresh_session()
        harness = AgentHarness(
            session, MockPlanner([good_ops()]), tracer=tracer)
        run = harness.run("trace me")

        self.assertTrue(tracer.events)
        run_ids = {e["run_id"] for e in tracer.events}
        self.assertEqual(run_ids, {run.run_id})
        kinds = {e["kind"] for e in tracer.events}
        self.assertIn("harness_start", kinds)
        self.assertIn("harness_end", kinds)
        # Every emitted kind is a declared harness kind.
        self.assertTrue(kinds.issubset(set(HARNESS_EVENT_KINDS)))

    def test_run_id_is_deterministic_across_runs(self):
        r1 = AgentHarness(_fresh_session(), MockPlanner([good_ops()])).run("same")
        r2 = AgentHarness(_fresh_session(), MockPlanner([good_ops()])).run("same")
        self.assertEqual(r1.run_id, r2.run_id)
        # Deterministic outcome too.
        self.assertEqual(r1.digest, r2.digest)


if __name__ == "__main__":
    unittest.main()
