"""Tests for Agent-S Behavior-Best-of-N deterministic trajectory selection."""

import unittest

from harnesscad.agents.cua.best_of_n_trajectory import (
    BehaviorBestOfN, Selection, TrajectoryCandidate, TrajectoryVerdict,
    behavior_score, oracle_score,
)


class _FakeDiff:
    def __init__(self, agree):
        self.agree = agree


class _FakeGrade:
    def __init__(self, solved, gui_valid, agree, gate_ok, target_ok):
        self.solved = solved
        self.gui_valid = gui_valid
        self.diff = _FakeDiff(agree)
        self.gate_ok = gate_ok
        self.target_ok = target_ok


def _solved(action_count=3, refusals=0):
    return TrajectoryVerdict(solved=True, gui_valid=True, differential_agree=True,
                             gate_ok=True, target_ok=True,
                             action_count=action_count, refusals=refusals)


class TestVerdictFromGrade(unittest.TestCase):
    def test_reads_gradelike_object(self):
        g = _FakeGrade(True, True, True, True, True)
        v = TrajectoryVerdict.from_grade(g, action_count=5)
        self.assertTrue(v.solved and v.differential_agree and v.gate_ok)
        self.assertEqual(v.action_count, 5)

    def test_unsolved_grade(self):
        g = _FakeGrade(False, True, False, True, False)
        v = TrajectoryVerdict.from_grade(g)
        self.assertFalse(v.solved)
        self.assertFalse(v.differential_agree)


class TestOracleScore(unittest.TestCase):
    def test_solved_dominates_unsolved(self):
        solved = TrajectoryCandidate("a", verdict=_solved())
        # An unsolved candidate that ticks every OTHER box still loses.
        partial = TrajectoryCandidate("b", verdict=TrajectoryVerdict(
            solved=False, gui_valid=True, differential_agree=True,
            gate_ok=True, target_ok=True))
        self.assertGreater(oracle_score(solved), oracle_score(partial))

    def test_no_verdict_scores_zero(self):
        self.assertEqual(oracle_score(TrajectoryCandidate("x")), 0.0)


class TestBehaviorScore(unittest.TestCase):
    def test_progress_and_terminate_beat_a_loop(self):
        good = TrajectoryCandidate("g", progressed=True, terminated=True)
        looper = TrajectoryCandidate("l", progressed=True, looped=True)
        self.assertGreater(behavior_score(good), behavior_score(looper))


class TestSelect(unittest.TestCase):
    def test_picks_the_solved_candidate(self):
        cands = [
            TrajectoryCandidate("a", verdict=TrajectoryVerdict(gui_valid=True)),
            TrajectoryCandidate("b", verdict=_solved()),
            TrajectoryCandidate("c"),
        ]
        sel = BehaviorBestOfN().select(cands)
        self.assertEqual(sel.best.id, "b")
        self.assertEqual(sel.ranked[0].id, "b")

    def test_tie_break_prefers_fewer_actions(self):
        # Two equally-solved trajectories -> the shorter one wins.
        cands = [
            TrajectoryCandidate("long", verdict=_solved(action_count=9)),
            TrajectoryCandidate("short", verdict=_solved(action_count=3)),
        ]
        sel = BehaviorBestOfN().select(cands)
        self.assertEqual(sel.best.id, "short")

    def test_tie_break_then_prefers_fewer_refusals(self):
        cands = [
            TrajectoryCandidate("x", verdict=_solved(action_count=3, refusals=2)),
            TrajectoryCandidate("y", verdict=_solved(action_count=3, refusals=0)),
        ]
        self.assertEqual(BehaviorBestOfN().select(cands).best.id, "y")

    def test_deterministic_regardless_of_input_order(self):
        a = TrajectoryCandidate("a", verdict=_solved(action_count=5))
        b = TrajectoryCandidate("b", verdict=_solved(action_count=3))
        c = TrajectoryCandidate("c", verdict=TrajectoryVerdict(gui_valid=True))
        s1 = BehaviorBestOfN().select([a, b, c])
        s2 = BehaviorBestOfN().select([c, b, a])
        self.assertEqual([x.id for x in s1.ranked], [x.id for x in s2.ranked])

    def test_empty_selection(self):
        sel = BehaviorBestOfN().select([])
        self.assertIsNone(sel.best)
        self.assertIn("no candidates", sel.rationale)

    def test_custom_scorer_used(self):
        cands = [TrajectoryCandidate("g", progressed=True, terminated=True),
                 TrajectoryCandidate("l", looped=True)]
        sel = BehaviorBestOfN(scorer=behavior_score).select(cands)
        self.assertEqual(sel.best.id, "g")


if __name__ == "__main__":
    unittest.main()
