"""Tests for agent.toolcad_metrics."""

import unittest

from harnesscad.agents.agent.tool_schema import ToolCall, default_toolcad_library
from harnesscad.agents.agent.tool_trajectory import ToolTrajectory, rollout
from harnesscad.agents.agent.tool_metrics import (
    count_redundant_calls,
    effective_progress,
    interaction_cost,
    success_rate,
    summarize,
    tools_per_task,
)


class MetricsBase(unittest.TestCase):
    def setUp(self):
        self.lib = default_toolcad_library()

    def _traj(self, completed=True):
        calls = [
            ("a", ToolCall("create_simple_sketch",
                           {"profile": "r", "sketch_name": "s1"})),
            ("b", ToolCall("extrude_face",
                           {"sketch_name": "s1", "distance": 5, "name": "p1"})),
            ("c", ToolCall("boolean_operation",
                           {"base_object_name": "missing",
                            "tool_object_name": "p1", "operation": "cut"})),
        ]
        return rollout(calls, self.lib, completed=completed)


class BasicMetricsTest(MetricsBase):
    def test_tools_per_task(self):
        self.assertEqual(tools_per_task(self._traj()), 3)

    def test_success_rate(self):
        self.assertAlmostEqual(success_rate(self._traj()), 2.0 / 3.0)

    def test_empty_success_rate(self):
        self.assertEqual(success_rate(ToolTrajectory()), 0.0)

    def test_effective_progress(self):
        # 2 producing successes (sketch, extrude) out of 3 calls.
        self.assertAlmostEqual(effective_progress(self._traj()), 2.0 / 3.0)


class RedundancyTest(MetricsBase):
    def test_duplicate_calls(self):
        calls = [
            ("a", ToolCall("set_coord_system", {"origin": [0, 0, 0]})),
            ("a2", ToolCall("set_coord_system", {"origin": [0, 0, 0]})),
        ]
        traj = rollout(calls, self.lib)
        self.assertEqual(count_redundant_calls(traj), 1)

    def test_no_redundancy(self):
        self.assertEqual(count_redundant_calls(self._traj()), 0)

    def test_boolean_self_op(self):
        calls = [
            ("mk", ToolCall("extrude_face",
                            {"sketch_name": "s1", "distance": 1, "name": "a"})),
            ("selfop", ToolCall("boolean_operation",
                                {"base_object_name": "a", "tool_object_name": "a",
                                 "operation": "fuse"})),
        ]
        traj = rollout(calls, self.lib)
        self.assertEqual(count_redundant_calls(traj), 1)

    def test_list_args_hashable(self):
        calls = [
            ("m", ToolCall("multiple_fuse", {"object_names": ["a", "b"]})),
            ("m2", ToolCall("multiple_fuse", {"object_names": ["a", "b"]})),
        ]
        traj = rollout(calls, self.lib)
        self.assertEqual(count_redundant_calls(traj), 1)


class SummaryTest(MetricsBase):
    def test_summary_bundle(self):
        m = summarize(self._traj())
        self.assertEqual(m.tools_per_task, 3)
        self.assertAlmostEqual(m.success_rate, 2.0 / 3.0)
        self.assertEqual(m.redundant_calls, 0)
        self.assertTrue(m.completed)

    def test_efficiency_incomplete_is_zero(self):
        m = summarize(self._traj(completed=False))
        self.assertEqual(m.efficiency, 0.0)

    def test_efficiency_completed(self):
        m = summarize(self._traj(completed=True))
        # no redundancy -> efficiency == success_rate
        self.assertAlmostEqual(m.efficiency, 2.0 / 3.0)

    def test_efficiency_penalized_by_redundancy(self):
        calls = [
            ("a", ToolCall("set_coord_system", {"origin": [0, 0, 0]})),
            ("a2", ToolCall("set_coord_system", {"origin": [0, 0, 0]})),
        ]
        traj = rollout(calls, self.lib, completed=True)
        m = summarize(traj)
        # both succeed -> success_rate 1.0; 1 redundant of 2 -> factor 0.5
        self.assertAlmostEqual(m.efficiency, 1.0 * 0.5)


class InteractionCostTest(MetricsBase):
    def test_averages(self):
        traj = self._traj()
        cost = interaction_cost(traj, [100, 200, 300], [10.0, 20.0, 30.0])
        self.assertEqual(cost.total_tokens, 600)
        self.assertAlmostEqual(cost.avg_tokens_per_call, 200.0)
        self.assertAlmostEqual(cost.avg_latency_ms, 20.0)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            interaction_cost(self._traj(), [1, 2], [1.0, 2.0])

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            interaction_cost(self._traj(), [1, 2, -3], [1.0, 2.0, 3.0])

    def test_empty_averages(self):
        cost = interaction_cost(ToolTrajectory(), [], [])
        self.assertEqual(cost.avg_tokens_per_call, 0.0)
        self.assertEqual(cost.avg_latency_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
