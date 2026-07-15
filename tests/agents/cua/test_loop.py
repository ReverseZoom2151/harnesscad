"""The CUA loop, driven against a FAKE environment (no GUI, no model, no Ollama).

The point of the fake is to prove the wiring in isolation: that AgentHarness's
ReAct spine drives a capability-declaring Environment through
:class:`EnvironmentExecutor` / :class:`EnvSession`, that ``session.digest()`` does
NOT raise (the reason a GUI could not ride the session spine unmodified), that
"MEASURE -> correct" repairs a wrong plan, and that the tier tally counts only
verified actions. The live GUI half lives in test_grade.py behind
HARNESSCAD_CUA_LIVE=1.
"""

import unittest
from typing import Any, Dict, List

from harnesscad.agents.cua.briefs import Target
from harnesscad.agents.cua.loop import (
    ActionTier, EnvSession, EnvironmentExecutor, GeometryGradeVerifier,
    TierCounts, build_cua_harness,
)
from harnesscad.core.cisp.ops import AddRectangle, Extrude, NewSketch
from harnesscad.core.environment import Capabilities, Observation, StepResult, coerce_ops
from harnesscad.eval.verifiers.verify import Diagnostic, Severity


class FakeGuiEnv:
    """A minimal Environment that behaves like the FreeCAD GUI one for the loop:
    it BUILDS a box from a (new_sketch, add_rectangle, extrude) run and refuses
    anything else, it measures the built solid, and it has NO content digest."""

    CAPABILITIES = Capabilities(
        name="fake-gui", content_digest=False, supported_ops=(
            "new_sketch", "add_rectangle", "extrude"),
        resolve_before_act=True)

    def __init__(self) -> None:
        self._built: List[dict] = []
        self._outcomes: List[dict] = []
        self._rect = None
        self._dist = None

    def capabilities(self) -> Capabilities:
        return self.CAPABILITIES

    def reset(self) -> Observation:
        self._built, self._outcomes, self._rect, self._dist = [], [], None, None
        return self.observe()

    def step(self, action) -> StepResult:
        ops = coerce_ops(action)
        for op in ops:
            tag = getattr(type(op), "OP", "")
            if not self.CAPABILITIES.supports(tag):
                return StepResult(ok=False, verified=False, observation=self.observe(),
                                  diagnostics=[Diagnostic(Severity.ERROR, "unsupported",
                                               "fake-gui cannot do %s" % tag)])
            if tag == "add_rectangle":
                self._rect = (op.w, op.h)
            elif tag == "extrude":
                self._dist = op.distance
            self._built.append(op.to_dict())
        if self._rect and self._dist:
            self._outcomes.append({"ok": True, "recipe": "box", "tier": "semantic_gui"})
        return StepResult(ok=True, verified=True, observation=self.observe(),
                          info={"executed_ops": len(ops)})

    def observe(self) -> Observation:
        return Observation(kind="hybrid", state={"ops_built": list(self._built)},
                           digest=None)

    def _measure(self) -> Dict[str, Any]:
        w, h = self._rect
        d = self._dist
        return {"volume": w * h * d, "surface_area": 2 * (w * h + w * d + h * d),
                "bbox": [w, h, d], "center_of_mass": [w / 2, h / 2, d / 2],
                "faces": 6, "edges": 12, "vertices": 8, "solids": 1,
                "is_valid": True, "is_closed": True}

    def measure(self, q: str = "measure") -> Dict[str, Any]:
        if not (self._rect and self._dist):
            return {"solid_present": False, "error": "no solid"}
        m = self._measure()
        if q == "validity":
            return {"is_valid": True, "solids": 1, "faces": 6, "edges": 12}
        return m

    def export(self, fmt: str) -> str:
        return "ISO-10303-21; fake STEP"

    def close(self) -> None:
        pass


class _ScriptedPlanner:
    """A planner stand-in: emits a fixed op stream, or a wrong-then-right pair to
    exercise repair. Never touches an LLM."""

    def __init__(self, plans: List[List]) -> None:
        self._plans = plans
        self._i = 0

    def plan_parsed(self, brief, state_summary=None, diagnostics=None):
        from harnesscad.agents.llm.structured import ParsedOps
        ops = self._plans[min(self._i, len(self._plans) - 1)]
        self._i += 1
        return ParsedOps(list(ops))


class TestEnvSessionIsHonest(unittest.TestCase):
    def test_digest_is_opaque_and_does_not_raise(self):
        env = FakeGuiEnv(); env.reset()
        sess = EnvSession(env)
        # THE reason a GUI could not ride the HarnessSession spine unmodified: the
        # harness calls digest() every iteration. Ours returns a token, never raises.
        d = sess.digest()
        self.assertTrue(d.startswith("envops-"))
        self.assertIsNone(sess.backend)
        self.assertIsNone(sess.opdag)
        self.assertIsNone(sess.checkpoint("x"))

    def test_digest_moves_when_ops_are_built(self):
        env = FakeGuiEnv(); env.reset()
        sess = EnvSession(env)
        before = sess.digest()
        env.step([NewSketch(plane="XY"), AddRectangle(w=30, h=20), Extrude(distance=10)])
        self.assertNotEqual(before, sess.digest())


class TestGradeVerifierDrivesRepair(unittest.TestCase):
    def test_target_miss_is_an_error_diagnostic(self):
        env = FakeGuiEnv(); env.reset()
        env.step([NewSketch(plane="XY"), AddRectangle(w=10, h=10), Extrude(distance=10)])
        v = GeometryGradeVerifier(env, Target(volume=6000.0, bbox=(30, 20, 10)))
        diags = v.check(None, None).diagnostics
        self.assertTrue(any(d.code == "target-miss" for d in diags))

    def test_target_hit_has_no_error(self):
        env = FakeGuiEnv(); env.reset()
        env.step([NewSketch(plane="XY"), AddRectangle(w=30, h=20), Extrude(distance=10)])
        v = GeometryGradeVerifier(env, Target(volume=6000.0, bbox=(30, 20, 10)))
        self.assertEqual(v.check(None, None).diagnostics, [])

    def test_no_solid_is_an_error_not_a_crash(self):
        env = FakeGuiEnv(); env.reset()
        v = GeometryGradeVerifier(env, Target(volume=1.0, bbox=(1, 1, 1)))
        diags = v.check(None, None).diagnostics
        self.assertTrue(any(d.code == "no-solid" for d in diags))


class TestFullLoopReusesAgentHarness(unittest.TestCase):
    """The whole point: AgentHarness (unmodified) drives the Environment."""

    def _run(self, plans):
        env = FakeGuiEnv(); env.reset()
        counts = TierCounts()
        harness, executor = build_cua_harness(
            env, _ScriptedPlanner(plans),
            target=Target(volume=6000.0, bbox=(30, 20, 10)),
            counts=counts, max_iterations=4)
        run = harness.run("a 30x20x10 block")
        return env, run, counts

    def test_correct_plan_converges_first_iteration(self):
        env, run, counts = self._run([
            [NewSketch(plane="XY"), AddRectangle(w=30, h=20), Extrude(distance=10)]])
        self.assertTrue(run.ok)
        self.assertEqual(run.stop_reason, "converged")
        # One tier-1 semantic-GUI action, nothing scripted, nothing in the viewport.
        self.assertEqual(counts.semantic_gui, 1)
        self.assertEqual(counts.script, 0)
        self.assertEqual(counts.viewport_pick, 0)

    def test_wrong_plan_is_repaired_via_measure_then_correct(self):
        # First plan builds the wrong volume; the grade verifier feeds it back and
        # the second plan fixes it. This is MEASURE -> correct on the harness spine.
        env, run, counts = self._run([
            [NewSketch(plane="XY"), AddRectangle(w=10, h=10), Extrude(distance=10)],
            [NewSketch(plane="XY"), AddRectangle(w=30, h=20), Extrude(distance=10)]])
        self.assertTrue(run.ok)
        self.assertGreaterEqual(run.iterations, 2)

    def test_the_digest_call_never_raised(self):
        # If EnvSession.digest() had raised (as the real state_digest does) the run
        # would have thrown; reaching here at all is the assertion.
        env, run, counts = self._run([
            [NewSketch(plane="XY"), AddRectangle(w=30, h=20), Extrude(distance=10)]])
        self.assertTrue(run.trajectory)
        self.assertTrue(all("digest" in e for e in run.trajectory))


class TestExecutorTierTally(unittest.TestCase):
    def test_refused_op_is_counted_and_not_a_tier_action(self):
        from harnesscad.core.cisp.ops import Fillet
        env = FakeGuiEnv(); env.reset()
        counts = TierCounts()
        ex = EnvironmentExecutor(env, counts)
        res = ex.apply_ops([Fillet(edges=("|Z",), radius=2.0)])
        self.assertFalse(res.ok)
        self.assertEqual(counts.refused, 1)
        self.assertEqual(counts.semantic_gui, 0)


if __name__ == "__main__":
    unittest.main()
