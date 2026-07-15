"""`tool_reward` is now reachable from the loop. It used to have two importers:
a dispatch table and its own unit test."""

from __future__ import annotations

import unittest

from harnesscad.agents.agent import trace_reward as tr
from harnesscad.core.cisp.ops import parse_op
from harnesscad.core.loop import HarnessSession
from harnesscad.core.trace import InMemoryTracer
from harnesscad.io.backends.frep import FRepBackend


PERFECT = [
    {"index": 0, "op": "new_sketch", "reward": 1.0, "reason": "applied+verified"},
    {"index": 1, "op": "extrude", "reward": 1.0, "reason": "applied+verified"},
]
BROKEN = [
    {"index": 0, "op": "new_sketch", "reward": 1.0, "reason": "applied+verified"},
    {"index": 1, "op": "add_rectangle", "reward": 1.0, "reason": "applied+verified"},
    {"index": 2, "op": "fillet", "reward": 0.0, "reason": "verify-failed"},
]


class TestCreditAssignment(unittest.TestCase):
    def test_first_divergence_names_one_op(self):
        self.assertIsNone(tr.first_divergence(PERFECT))
        self.assertEqual(tr.first_divergence(BROKEN), 2)

    def test_step_accuracy(self):
        self.assertEqual(tr.step_accuracy(PERFECT), 1.0)
        self.assertAlmostEqual(tr.step_accuracy(BROKEN), 2 / 3)
        self.assertEqual(tr.step_accuracy([]), 0.0)

    def test_trajectory_is_completed_only_when_no_op_broke_it(self):
        self.assertTrue(tr.trajectory_from_steps(PERFECT).completed)
        self.assertFalse(tr.trajectory_from_steps(BROKEN).completed)


class TestAggregateReward(unittest.TestCase):
    def test_process_reward_separates_two_failures_that_the_outcome_conflates(self):
        # Outcome-only supervision gives BOTH of these the same score: 0.
        # The process reward does not: one plan got 2/3 of the way, the other 0.
        nearly = tr.reward_for_steps(BROKEN, orm_verdict=False)
        nothing = tr.reward_for_steps(
            [{"index": 0, "op": "extrude", "reward": 0.0, "reason": "backend-rejected"}],
            orm_verdict=False)
        self.assertEqual(nearly.outcome, nothing.outcome)   # both 0: ORM is blind
        self.assertGreater(nearly.step_mean, nothing.step_mean)
        self.assertGreater(nearly.total, nothing.total)

    def test_outcome_requires_a_completed_trajectory(self):
        r = tr.reward_for_steps(BROKEN, orm_verdict=True)
        self.assertEqual(r.outcome, 0.0)

    def test_weights_are_honoured(self):
        r = tr.reward_for_steps(PERFECT, orm_verdict=True, alpha=2.0, beta=3.0,
                                gamma=0.0)
        self.assertAlmostEqual(r.total, 2.0 * 1.0 + 3.0 * 1.0)


class TestWiredToTheLoop(unittest.TestCase):
    def test_a_real_session_scores_itself(self):
        s = HarnessSession(FRepBackend())
        s.apply_ops([parse_op(dict(o)) for o in (
            {"op": "new_sketch", "plane": "XY"},
            {"op": "add_rectangle", "sketch": "sk1", "x": 0, "y": 0, "w": 50, "h": 30},
            {"op": "extrude", "sketch": "sk1", "distance": 6},
        )])
        r = tr.reward_for_session(s, orm_verdict=True)
        self.assertEqual(r.step_mean, 1.0)
        self.assertEqual(r.outcome, 1.0)

    def test_a_trajectory_rebuilds_from_the_trace_alone(self):
        tracer = InMemoryTracer()
        s = HarnessSession(FRepBackend(), tracer=tracer)
        s.apply_ops([parse_op({"op": "new_sketch", "plane": "XY"})])
        traj = tr.trajectory_from_trace(tracer.events)
        self.assertEqual(len(traj), 1)
        self.assertTrue(traj.completed)


if __name__ == "__main__":
    unittest.main()
