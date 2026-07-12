"""Tests for agent.toolcad_reward."""

import unittest

from agent.toolcad_tool_schema import ToolCall, default_toolcad_library
from agent.toolcad_trajectory import ToolTrajectory, rollout
from agent.toolcad_reward import (
    aggregate_reward,
    format_reward,
    mean_step_reward,
    outcome_reward,
    score_tool_selection,
    step_execution_rewards,
)


def _good_text():
    return ("<think>a</think><tool_call>{}</tool_call>"
            "<tool_response>ok</tool_response>")


class FormatRewardTest(unittest.TestCase):
    def test_good(self):
        self.assertEqual(format_reward(_good_text()), 0.5)

    def test_bad(self):
        self.assertEqual(format_reward("<tool_call>x</tool_call>"), 0.0)


class StepRewardTest(unittest.TestCase):
    def setUp(self):
        self.lib = default_toolcad_library()

    def _traj(self):
        calls = [
            ("a", ToolCall("create_simple_sketch",
                           {"profile": "r", "sketch_name": "s1"})),
            ("b", ToolCall("extrude_face",
                           {"sketch_name": "s1", "distance": 5, "name": "p1"})),
            ("c", ToolCall("boolean_operation",
                           {"base_object_name": "missing",
                            "tool_object_name": "p1", "operation": "cut"})),
        ]
        return rollout(calls, self.lib)

    def test_per_step(self):
        rewards = step_execution_rewards(self._traj())
        self.assertEqual(rewards, (1.0, 1.0, 0.0))

    def test_mean(self):
        self.assertAlmostEqual(mean_step_reward(self._traj()), 2.0 / 3.0)

    def test_empty_mean(self):
        self.assertEqual(mean_step_reward(ToolTrajectory()), 0.0)


class OutcomeRewardTest(unittest.TestCase):
    def test_requires_completion(self):
        traj = ToolTrajectory(completed=False)
        self.assertEqual(outcome_reward(True, traj), 0.0)

    def test_completed_and_verdict(self):
        traj = ToolTrajectory(completed=True)
        self.assertEqual(outcome_reward(True, traj), 1.0)

    def test_verdict_false(self):
        traj = ToolTrajectory(completed=True)
        self.assertEqual(outcome_reward(False, traj), 0.0)


class AggregateRewardTest(unittest.TestCase):
    def setUp(self):
        self.lib = default_toolcad_library()
        calls = [
            ("a", ToolCall("create_simple_sketch",
                           {"profile": "r", "sketch_name": "s1"})),
            ("b", ToolCall("extrude_face",
                           {"sketch_name": "s1", "distance": 5, "name": "p1"})),
        ]
        self.traj = rollout(calls, self.lib, completed=True)

    def test_all_terms(self):
        r = aggregate_reward(self.traj, orm_verdict=True,
                             format_text=_good_text())
        self.assertEqual(r.outcome, 1.0)
        self.assertEqual(r.step_mean, 1.0)
        self.assertEqual(r.fmt, 0.5)
        self.assertEqual(r.total, 2.5)

    def test_weighted(self):
        r = aggregate_reward(self.traj, orm_verdict=True,
                             format_text=_good_text(),
                             alpha=2.0, beta=1.0, gamma=0.0)
        self.assertEqual(r.total, 2.0 * 1.0 + 1.0 * 1.0 + 0.0)

    def test_negative_weight_rejected(self):
        with self.assertRaises(ValueError):
            aggregate_reward(self.traj, orm_verdict=True,
                             format_text=_good_text(), alpha=-1.0)


class ToolSelectionScoreTest(unittest.TestCase):
    def setUp(self):
        self.gold = [
            ToolCall("set_coord_system", {"origin": [0, 0, 0]}),
            ToolCall("create_simple_sketch", {"profile": "rect"}),
            ToolCall("extrude_face", {"sketch_name": "s1", "distance": 5}),
        ]

    def test_perfect(self):
        pred = [
            ToolCall("set_coord_system", {"origin": (0, 0, 0)}),  # tuple vs list
            ToolCall("create_simple_sketch", {"profile": "rect"}),
            ToolCall("extrude_face", {"sketch_name": "s1", "distance": 5}),
        ]
        s = score_tool_selection(pred, self.gold)
        self.assertEqual(s.selection_accuracy, 1.0)
        self.assertEqual(s.argument_accuracy, 1.0)
        self.assertEqual(s.length_penalty, 1.0)
        self.assertEqual(s.reward, 1.0)

    def test_right_tool_wrong_arg(self):
        pred = [
            ToolCall("set_coord_system", {"origin": [0, 0, 0]}),
            ToolCall("create_simple_sketch", {"profile": "circle"}),
            ToolCall("extrude_face", {"sketch_name": "s1", "distance": 99}),
        ]
        s = score_tool_selection(pred, self.gold)
        self.assertEqual(s.selection_accuracy, 1.0)
        self.assertAlmostEqual(s.argument_accuracy, 1.0 / 3.0)

    def test_wrong_tool(self):
        pred = [
            ToolCall("boolean_operation", {}),
            ToolCall("create_simple_sketch", {"profile": "rect"}),
            ToolCall("extrude_face", {"sketch_name": "s1", "distance": 5}),
        ]
        s = score_tool_selection(pred, self.gold)
        self.assertAlmostEqual(s.selection_accuracy, 2.0 / 3.0)

    def test_over_generation_penalty(self):
        pred = [
            ToolCall("set_coord_system", {"origin": [0, 0, 0]}),
            ToolCall("create_simple_sketch", {"profile": "rect"}),
            ToolCall("extrude_face", {"sketch_name": "s1", "distance": 5}),
            ToolCall("extrude_face", {"sketch_name": "s2", "distance": 1}),
            ToolCall("extrude_face", {"sketch_name": "s3", "distance": 1}),
        ]
        s = score_tool_selection(pred, self.gold)
        self.assertEqual(s.n_pred, 5)
        self.assertEqual(s.length_penalty, 1.0 / 3.0)  # extra == 2
        self.assertAlmostEqual(s.reward, 1.0 * (1.0 / 3.0))

    def test_empty_gold(self):
        s = score_tool_selection([], [])
        self.assertEqual(s.selection_accuracy, 0.0)
        self.assertEqual(s.length_penalty, 1.0)


if __name__ == "__main__":
    unittest.main()
