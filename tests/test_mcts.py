"""Tests for the MCTS search tier (strategies.mcts).

Deterministic and dependency-free, matching the style of ``test_strategies.py``:
a fresh ``StubBackend``-backed ``HarnessSession`` is the ``session_factory``, and
expansions are injected as fixed op-menus so a "good" and a "bad" continuation are
offered at the root. No LLM, no network, no geometry kernel.
"""

import unittest
from typing import Any, List, Optional

from backends.stub import StubBackend
from cisp.ops import Op, NewSketch, AddRectangle, Constrain, Extrude
from cisp.protocol import ApplyOpsResult
from loop import HarnessSession
from verify import Diagnostic, Severity, VerifyReport

from strategies.mcts import (
    mcts_search,
    MctsResult,
    MctsNode,
    default_reward,
    menu_expansion,
    planner_expansion,
)


# --- fixtures ---------------------------------------------------------------
def good_plate_ops() -> List[Op]:
    """A continuation the StubBackend accepts + verifies (ok=True)."""
    return [
        NewSketch(plane="XY"),
        AddRectangle(sketch="sk1", x=0, y=0, w=20, h=10),
        Extrude(sketch="sk1", distance=5.0),
    ]


def bad_ref_ops() -> List[Op]:
    """A continuation the backend BLOCKS: extrude references a missing sketch."""
    return [Extrude(sketch="nope", distance=5.0)]


def over_constrained_ops() -> List[Op]:
    """Applies but FAILS verify -> ok=False and carries an ERROR diagnostic."""
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


class NullPlanner:
    """Stands in for agent.planner.Planner; never called when expansion injected."""

    def plan(self, brief: str, state_summary: Optional[dict] = None,
             diagnostics: Optional[list] = None) -> List[Op]:  # pragma: no cover
        raise AssertionError("planner should not be called with injected expansion")


def _child_for(root: MctsNode, first_op_kind: type) -> Optional[MctsNode]:
    """Find the root child whose first appended op is of the given type."""
    for ch in root.children:
        rootlen = len(root.ops)
        if len(ch.ops) > rootlen and isinstance(ch.ops[rootlen], first_op_kind):
            return ch
    return None


# --- core search behaviour --------------------------------------------------
class TestMctsSearch(unittest.TestCase):
    def test_returns_higher_reward_sequence(self):
        # Root offers a good (verifies) and a bad (blocked) continuation.
        expansion = menu_expansion([good_plate_ops(), bad_ref_ops()], max_depth=3)
        res = mcts_search(NullPlanner(), stub_session_factory, "make a plate",
                          iterations=30, expansion=expansion, seed=0)
        self.assertIsInstance(res, MctsResult)
        self.assertTrue(res.ok)
        self.assertEqual([op.OP for op in res.best_ops],
                         [op.OP for op in good_plate_ops()])
        self.assertGreater(res.best_score, 0.7)  # ok-branch reward is >= 0.7

    def test_concentrates_visits_on_good_branch(self):
        expansion = menu_expansion([good_plate_ops(), bad_ref_ops()], max_depth=3)
        res = mcts_search(NullPlanner(), stub_session_factory, "brief",
                          iterations=40, expansion=expansion, seed=0)
        good = _child_for(res.root, NewSketch)   # good branch starts with a sketch
        bad = _child_for(res.root, Extrude)      # bad branch is a lone extrude
        self.assertIsNotNone(good)
        self.assertIsNotNone(bad)
        # UCB1 should pour the budget into the higher-value (good) branch.
        self.assertGreater(good.visits, bad.visits)

    def test_reports_iterations_and_tree_size(self):
        expansion = menu_expansion([good_plate_ops(), bad_ref_ops()], max_depth=1)
        res = mcts_search(NullPlanner(), stub_session_factory, "brief",
                          iterations=12, expansion=expansion, seed=1)
        self.assertEqual(res.iterations, 12)
        # root + 2 children (both menu items expanded once; max_depth=1 stops there)
        self.assertEqual(res.tree_size, 3)
        # Visit accounting is conserved: root sees every iteration.
        self.assertEqual(res.root.visits, 12)

    def test_deterministic_same_tree_twice(self):
        def run():
            expansion = menu_expansion([good_plate_ops(), bad_ref_ops()],
                                       max_depth=3)
            r = mcts_search(NullPlanner(), stub_session_factory, "brief",
                            iterations=25, expansion=expansion, seed=7)
            visits = tuple(sorted(ch.visits for ch in r.root.children))
            return ([op.OP for op in r.best_ops], round(r.best_score, 9),
                    r.tree_size, visits)
        self.assertEqual(run(), run())

    def test_reward_prefers_fewer_diagnostics_picks_clean_branch(self):
        # Both branches are ok-ish, but the over-constrained one FAILS verify and
        # carries an ERROR diagnostic; the clean plate has none. Default reward
        # (fewer diagnostics preferred) must select the clean branch.
        expansion = menu_expansion([over_constrained_ops(), good_plate_ops()],
                                   max_depth=6)
        res = mcts_search(NullPlanner(), stub_session_factory, "brief",
                          iterations=40, expansion=expansion, seed=0)
        self.assertTrue(res.ok)
        self.assertEqual([op.OP for op in res.best_ops],
                         [op.OP for op in good_plate_ops()])

    def test_custom_reward_is_respected(self):
        # A reward that INVERTS preference (favours not-ok) should pick the blocked
        # branch even though it fails verification.
        expansion = menu_expansion([good_plate_ops(), bad_ref_ops()], max_depth=3)

        def inverted(apply_result, verify_report) -> float:
            return 0.0 if apply_result.ok else 1.0

        res = mcts_search(NullPlanner(), stub_session_factory, "brief",
                          iterations=30, expansion=expansion, reward=inverted,
                          seed=0)
        self.assertFalse(res.ok)
        self.assertEqual([op.OP for op in res.best_ops],
                         [op.OP for op in bad_ref_ops()])

    def test_terminal_root_when_no_continuations(self):
        # An expansion that offers nothing -> the root (empty ops) is the result.
        empty_expansion = menu_expansion([], max_depth=3)
        res = mcts_search(NullPlanner(), stub_session_factory, "brief",
                          iterations=5, expansion=empty_expansion, seed=0)
        self.assertEqual(res.best_ops, [])
        self.assertEqual(res.tree_size, 1)
        self.assertIsNotNone(res.best_result)

    def test_zero_iterations_still_returns_root_result(self):
        expansion = menu_expansion([good_plate_ops()], max_depth=3)
        res = mcts_search(NullPlanner(), stub_session_factory, "brief",
                          iterations=0, expansion=expansion, seed=0)
        self.assertEqual(res.iterations, 0)
        self.assertEqual(res.tree_size, 1)
        self.assertIsNotNone(res.best_result)  # trivial: the empty root sequence

    def test_root_ops_seeds_the_search(self):
        # Start partway: root already holds a sketch+rectangle; the menu extrudes.
        prefix = [NewSketch(plane="XY"),
                  AddRectangle(sketch="sk1", x=0, y=0, w=20, h=10)]
        expansion = menu_expansion([[Extrude(sketch="sk1", distance=5.0)]])
        res = mcts_search(NullPlanner(), stub_session_factory, "brief",
                          iterations=6, expansion=expansion, seed=0,
                          root_ops=prefix)
        self.assertTrue(res.ok)
        self.assertEqual([op.OP for op in res.best_ops],
                         ["new_sketch", "add_rectangle", "extrude"])

    def test_rejects_negative_iterations(self):
        expansion = menu_expansion([good_plate_ops()])
        with self.assertRaises(ValueError):
            mcts_search(NullPlanner(), stub_session_factory, "brief",
                        iterations=-1, expansion=expansion)


# --- default reward ---------------------------------------------------------
class TestDefaultReward(unittest.TestCase):
    def test_ok_beats_not_ok(self):
        ok = ApplyOpsResult(True, 3, "d", [])
        fail = ApplyOpsResult(False, 1, "d",
                              [Diagnostic(Severity.ERROR, "e", "m")])
        self.assertGreater(default_reward(ok, VerifyReport(ok.diagnostics)),
                           default_reward(fail, VerifyReport(fail.diagnostics)))

    def test_fewer_diagnostics_preferred(self):
        clean = ApplyOpsResult(True, 3, "d", [])
        warn = ApplyOpsResult(True, 3, "d",
                              [Diagnostic(Severity.WARNING, "w", "m")])
        self.assertGreater(default_reward(clean, None),
                           default_reward(warn, None))

    def test_more_applied_preferred(self):
        few = ApplyOpsResult(True, 1, "d", [])
        many = ApplyOpsResult(True, 5, "d", [])
        self.assertGreater(default_reward(many, None), default_reward(few, None))

    def test_bounded_and_none_is_zero(self):
        best = ApplyOpsResult(True, 100, "d", [])
        self.assertLessEqual(default_reward(best, None), 1.0)
        self.assertEqual(default_reward(None, None), 0.0)


# --- default (planner-driven) expansion -------------------------------------
class TestPlannerExpansion(unittest.TestCase):
    def test_returns_tail_beyond_prefix(self):
        class WholePlanPlanner:
            def plan(self, brief, state_summary=None, diagnostics=None):
                return good_plate_ops()

        expand = planner_expansion(WholePlanPlanner(), k=1)
        rng = __import__("random").Random(0)
        # From the empty root, the continuation is the whole plan.
        conts = expand([], "brief", rng)
        self.assertEqual(len(conts), 1)
        self.assertEqual([op.OP for op in conts[0]],
                         [op.OP for op in good_plate_ops()])
        # From a 2-op prefix, only the tail (the extrude) is offered.
        conts = expand(good_plate_ops()[:2], "brief", rng)
        self.assertEqual([op.OP for op in conts[0]], ["extrude"])

    def test_dedupes_identical_tails(self):
        class SamePlanPlanner:
            def plan(self, brief, state_summary=None, diagnostics=None):
                return good_plate_ops()

        expand = planner_expansion(SamePlanPlanner(), k=3)
        rng = __import__("random").Random(0)
        conts = expand([], "brief", rng)
        self.assertEqual(len(conts), 1)  # 3 identical draws collapse to one

    def test_end_to_end_with_default_expansion(self):
        # No injected expansion: the search drives the planner itself.
        class PlatePlanner:
            def plan(self, brief, state_summary=None, diagnostics=None):
                return good_plate_ops()

        res = mcts_search(PlatePlanner(), stub_session_factory, "make a plate",
                          iterations=10, seed=0, max_depth=4)
        self.assertTrue(res.ok)
        self.assertEqual([op.OP for op in res.best_ops],
                         [op.OP for op in good_plate_ops()])


if __name__ == "__main__":
    unittest.main()
