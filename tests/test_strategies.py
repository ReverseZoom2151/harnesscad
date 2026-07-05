"""Tests for the reliability strategies (Best-of-N + verifier; Reflexion).

Everything is deterministic and dependency-free: a ``MockPlanner`` stands in for
``agent.planner.Planner`` (same ``plan(brief, state_summary=None,
diagnostics=None)`` surface), a fresh ``StubBackend``-backed ``HarnessSession`` is
the ``session_factory``, and a real ``MemoryStore`` records Reflexion's insights.
No LLM, no network, no geometry kernel.
"""

import unittest
from typing import Any, Dict, List, Optional

from backends.stub import StubBackend
from cisp.ops import (
    Op, NewSketch, AddRectangle, Constrain, Extrude,
)
from loop import HarnessSession
from memory.store import MemoryStore

from reliability.strategies import (
    best_of_n,
    default_scorer,
    BestOfNResult,
    ReflexionLoop,
    ReflexionResult,
    heuristic_reflect,
)


# --- fixtures ---------------------------------------------------------------
def good_plate_ops() -> List[Op]:
    """A plan the StubBackend accepts + verifies: sketch -> rectangle -> extrude."""
    return [
        NewSketch(plane="XY"),
        AddRectangle(sketch="sk1", x=0, y=0, w=20, h=10),
        Extrude(sketch="sk1", distance=5.0),
    ]


def bad_ref_ops() -> List[Op]:
    """A plan the backend BLOCKS: extrude references a sketch that never existed."""
    return [Extrude(sketch="nope", distance=5.0)]


def over_constrained_ops() -> List[Op]:
    """A plan that APPLIES but FAILS verify: rectangle driven over-constrained.

    A rectangle contributes 4 DOF; three 'distance' constraints remove < 4, but a
    'coincident' (removes 2) plus three 'distance' (removes 3) drives dof to -1 ->
    the SketchConstraintCheck emits an ERROR, so apply_ops rolls back and returns
    ok=False with an 'over-constrained' diagnostic.
    """
    return [
        NewSketch(plane="XY"),
        AddRectangle(sketch="sk1", x=0, y=0, w=20, h=10),
        Constrain(kind="coincident", a="e1"),
        Constrain(kind="distance", a="e1", value=20.0),
        Constrain(kind="distance", a="e1", value=10.0),
        Constrain(kind="distance", a="e1", value=5.0),
    ]


def stub_session_factory() -> HarnessSession:
    return HarnessSession(StubBackend())


class QueuePlanner:
    """Pops a queued plan per ``plan()`` call. Matches the Planner surface.

    ``plans`` is a list of op-lists; the last one repeats once exhausted so extra
    Best-of-N draws are well-defined.
    """

    def __init__(self, plans: List[List[Op]]) -> None:
        self._plans = list(plans)
        self.calls: List[Dict[str, Any]] = []

    def plan(self, brief: str, state_summary: Optional[dict] = None,
             diagnostics: Optional[list] = None) -> List[Op]:
        self.calls.append({"brief": brief, "state_summary": state_summary,
                           "diagnostics": diagnostics})
        if len(self._plans) > 1:
            return self._plans.pop(0)
        return self._plans[0]


class InsightGatedPlanner:
    """Returns a good plan ONLY once a recalled insight appears in the brief.

    This makes Reflexion's convergence *causally* driven by the recall step: until
    the loop writes an insight to semantic memory and prepends it on the next
    attempt, the planner keeps returning the failing plan.
    """

    def __init__(self, marker: str, good: List[Op], bad: List[Op]) -> None:
        self.marker = marker
        self.good = good
        self.bad = bad
        self.calls: List[str] = []

    def plan(self, brief: str, state_summary: Optional[dict] = None,
             diagnostics: Optional[list] = None) -> List[Op]:
        self.calls.append(brief)
        return self.good if self.marker in brief else self.bad


# --- Best-of-N --------------------------------------------------------------
class TestBestOfN(unittest.TestCase):
    def test_picks_ok_over_blocked_candidate(self):
        # Candidate 0 is blocked (bad ref), candidate 1 builds cleanly.
        planner = QueuePlanner([bad_ref_ops(), good_plate_ops()])
        res = best_of_n(planner, stub_session_factory, "make a plate", n=2)
        self.assertIsInstance(res, BestOfNResult)
        self.assertTrue(res.ok)
        self.assertEqual(res.winner_index, 1)
        self.assertTrue(res.winner.result.ok)
        # Both candidates are reported.
        self.assertEqual(len(res.candidates), 2)
        self.assertFalse(res.candidates[0].ok)

    def test_prefers_fewer_diagnostics_then_more_applied(self):
        # Candidate 0 applies-but-fails-verify (over-constrained, carries an ERROR
        # diagnostic); candidate 1 is clean. Winner must be the clean one.
        planner = QueuePlanner([over_constrained_ops(), good_plate_ops()])
        res = best_of_n(planner, stub_session_factory, "make a plate", n=2)
        self.assertTrue(res.ok)
        self.assertEqual(res.winner_index, 1)

    def test_all_candidates_fail_still_returns_a_winner(self):
        planner = QueuePlanner([bad_ref_ops()])  # every draw is blocked
        res = best_of_n(planner, stub_session_factory, "make a plate", n=3)
        self.assertFalse(res.ok)
        self.assertIsNotNone(res.winner)
        self.assertEqual(len(res.candidates), 3)

    def test_seeded_briefs_differ_per_candidate(self):
        planner = QueuePlanner([good_plate_ops()])
        best_of_n(planner, stub_session_factory, "make a plate", n=3)
        briefs = [c["brief"] for c in planner.calls]
        self.assertEqual(len(briefs), 3)
        # Candidate 0 is the untouched brief; later draws carry a distinct seed.
        self.assertEqual(briefs[0], "make a plate")
        self.assertIn("seed=1", briefs[1])
        self.assertIn("seed=2", briefs[2])
        self.assertEqual(len(set(briefs)), 3)

    def test_deterministic(self):
        def run() -> int:
            planner = QueuePlanner([bad_ref_ops(), good_plate_ops()])
            return best_of_n(planner, stub_session_factory, "brief", n=2).winner_index
        self.assertEqual(run(), run())

    def test_ties_resolve_to_earliest_candidate(self):
        # Two identical good plans tie on score -> the lowest index wins (stable max).
        planner = QueuePlanner([good_plate_ops()])
        res = best_of_n(planner, stub_session_factory, "make a plate", n=2)
        self.assertTrue(res.ok)
        self.assertEqual(res.winner_index, 0)

    def test_custom_scorer_is_respected(self):
        # A scorer that inverts preference (favour NOT-ok) should pick a failing one.
        planner = QueuePlanner([bad_ref_ops(), good_plate_ops()])
        res = best_of_n(planner, stub_session_factory, "brief", n=2,
                        scorer=lambda r: (0 if r.ok else 1,))
        self.assertEqual(res.winner_index, 0)
        self.assertFalse(res.winner.result.ok)

    def test_planner_exception_becomes_failed_candidate(self):
        class Boom:
            def plan(self, brief, state_summary=None, diagnostics=None):
                if "seed=1" in brief:
                    return good_plate_ops()
                raise ValueError("bad model output")
        res = best_of_n(Boom(), stub_session_factory, "brief", n=2)
        self.assertTrue(res.ok)
        self.assertEqual(res.winner_index, 1)
        self.assertIsNotNone(res.candidates[0].error)

    def test_default_scorer_ordering(self):
        from cisp.protocol import ApplyOpsResult
        from verifiers.verify import Diagnostic, Severity
        ok_clean = ApplyOpsResult(True, 3, "d", [])
        ok_warn = ApplyOpsResult(True, 3, "d",
                                 [Diagnostic(Severity.WARNING, "w", "m")])
        fail = ApplyOpsResult(False, 1, "d",
                              [Diagnostic(Severity.ERROR, "e", "m")])
        self.assertGreater(default_scorer(ok_clean), default_scorer(ok_warn))
        self.assertGreater(default_scorer(ok_warn), default_scorer(fail))

    def test_rejects_n_below_one(self):
        planner = QueuePlanner([good_plate_ops()])
        with self.assertRaises(ValueError):
            best_of_n(planner, stub_session_factory, "brief", n=0)


# --- Reflexion --------------------------------------------------------------
class TestReflexion(unittest.TestCase):
    def test_converges_after_reflection_and_recall(self):
        # The planner only emits a good plan once the recalled insight is in the
        # brief, so convergence proves the write->recall path works end to end.
        good = good_plate_ops()
        marker = "PRIOR INSIGHTS"
        planner = InsightGatedPlanner(marker, good=good, bad=bad_ref_ops())
        mem = MemoryStore()
        loop = ReflexionLoop(planner, stub_session_factory, mem, max_attempts=3)

        res = loop.run("make a plate")
        self.assertIsInstance(res, ReflexionResult)
        self.assertTrue(res.converged)
        # Attempt 0 failed (no insight yet), attempt 1 recalled + converged.
        self.assertEqual(len(res.attempts), 2)
        self.assertFalse(res.attempts[0].ok)
        self.assertTrue(res.attempts[1].ok)
        # An insight was written to semantic memory on the first failure...
        stored = mem.get_semantic("reflexion:insights", [])
        self.assertTrue(stored)
        self.assertIsNotNone(res.attempts[0].insight)
        # ...and recalled into the second attempt's context.
        self.assertIn(res.attempts[0].insight, stored)
        self.assertIn(marker, res.attempts[1].brief)
        self.assertEqual(res.attempts[1].recalled, stored)

    def test_writes_actionable_insight_for_bad_ref(self):
        planner = InsightGatedPlanner("PRIOR INSIGHTS", good=good_plate_ops(),
                                      bad=bad_ref_ops())
        mem = MemoryStore()
        loop = ReflexionLoop(planner, stub_session_factory, mem, max_attempts=3)
        loop.run("make a plate")
        stored = mem.get_semantic("reflexion:insights", [])
        # The bad-ref diagnostic maps to the "create ... before referencing" lesson.
        self.assertTrue(any("before referencing" in s for s in stored))

    def test_does_not_converge_when_planner_never_fixes(self):
        # Planner ignores insights and always returns the blocked plan.
        planner = QueuePlanner([bad_ref_ops()])
        mem = MemoryStore()
        loop = ReflexionLoop(planner, stub_session_factory, mem, max_attempts=3)
        res = loop.run("make a plate")
        self.assertFalse(res.converged)
        self.assertEqual(len(res.attempts), 3)
        self.assertFalse(res.final_result.ok)
        # Insight was still learned + persisted (deduped to one entry).
        self.assertEqual(len(mem.get_semantic("reflexion:insights", [])), 1)

    def test_first_attempt_success_writes_no_insight(self):
        planner = QueuePlanner([good_plate_ops()])
        mem = MemoryStore()
        loop = ReflexionLoop(planner, stub_session_factory, mem, max_attempts=3)
        res = loop.run("make a plate")
        self.assertTrue(res.converged)
        self.assertEqual(len(res.attempts), 1)
        self.assertIsNone(res.attempts[0].insight)
        self.assertEqual(mem.get_semantic("reflexion:insights", []), [])

    def test_records_episodic_trajectory(self):
        planner = InsightGatedPlanner("PRIOR INSIGHTS", good=good_plate_ops(),
                                      bad=bad_ref_ops())
        mem = MemoryStore()
        loop = ReflexionLoop(planner, stub_session_factory, mem, max_attempts=3)
        loop.run("make a plate")
        # One episodic entry per attempt (fail then ok).
        outcomes = [e.outcome for e in mem.episodic]
        self.assertEqual(outcomes, ["failed", "ok"])

    def test_injected_reflect_critic_is_used(self):
        seen = {}

        def critic(diagnostics, brief):
            seen["called"] = True
            return "CUSTOM_INSIGHT_TOKEN"

        planner = InsightGatedPlanner("CUSTOM_INSIGHT_TOKEN",
                                      good=good_plate_ops(), bad=bad_ref_ops())
        mem = MemoryStore()
        loop = ReflexionLoop(planner, stub_session_factory, mem,
                             reflect=critic, max_attempts=3)
        res = loop.run("make a plate")
        self.assertTrue(seen.get("called"))
        self.assertTrue(res.converged)
        self.assertIn("CUSTOM_INSIGHT_TOKEN",
                      mem.get_semantic("reflexion:insights", []))

    def test_deterministic(self):
        def run() -> bool:
            planner = InsightGatedPlanner("PRIOR INSIGHTS", good=good_plate_ops(),
                                          bad=bad_ref_ops())
            loop = ReflexionLoop(planner, stub_session_factory, MemoryStore(),
                                 max_attempts=3)
            return loop.run("make a plate").converged
        self.assertEqual(run(), run())


class TestHeuristicReflect(unittest.TestCase):
    def test_maps_known_codes(self):
        from verifiers.verify import Diagnostic, Severity
        d = [Diagnostic(Severity.ERROR, "over-constrained", "sketch sk1 over-constrained")]
        self.assertIn("over-constrained", heuristic_reflect(d, "brief"))

    def test_coplanar_message_keyword_fallback(self):
        from verifiers.verify import Diagnostic, Severity
        d = [Diagnostic(Severity.ERROR, "boolean-failed", "faces are coplanar")]
        self.assertIn("offset", heuristic_reflect(d, "brief"))

    def test_ignores_non_error_severity(self):
        from verifiers.verify import Diagnostic, Severity
        d = [Diagnostic(Severity.WARNING, "under-constrained", "m")]
        # A warning-only report still returns a generic (never empty) insight.
        out = heuristic_reflect(d, "brief")
        self.assertTrue(out)
        self.assertNotIn("under-constrained", out)

    def test_unknown_code_generic_insight(self):
        from verifiers.verify import Diagnostic, Severity
        d = [Diagnostic(Severity.ERROR, "weird-code", "m")]
        out = heuristic_reflect(d, "brief")
        self.assertIn("weird-code", out)

    def test_accepts_dict_diagnostics(self):
        d = [{"severity": "error", "code": "bad-ref", "message": "unknown sketch"}]
        self.assertIn("before referencing", heuristic_reflect(d, "brief"))


if __name__ == "__main__":
    unittest.main()
