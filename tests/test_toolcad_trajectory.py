"""Tests for agent.toolcad_trajectory."""

import unittest

from harnesscad.agents.agent.toolcad_tool_schema import ToolCall, default_toolcad_library
from harnesscad.agents.agent.toolcad_trajectory import (
    ToolTrajectory,
    check_format_order,
    parse_react_trajectory,
    render_step,
    rollout,
)


class RolloutTest(unittest.TestCase):
    def setUp(self):
        self.lib = default_toolcad_library()

    def test_rollout_success_chain(self):
        calls = [
            ("sketch base", ToolCall("create_simple_sketch",
                                     {"profile": "rect", "sketch_name": "s1"})),
            ("extrude base", ToolCall("extrude_face",
                                      {"sketch_name": "s1", "distance": 5, "name": "p1"})),
        ]
        traj = rollout(calls, self.lib)
        self.assertEqual(len(traj), 2)
        self.assertEqual(traj.num_success, 2)
        self.assertEqual(traj.num_fail, 0)
        self.assertTrue(traj.completed)

    def test_rollout_records_failure(self):
        calls = [
            ("bad boolean", ToolCall("boolean_operation",
                                     {"base_object_name": "x", "tool_object_name": "y",
                                      "operation": "cut"})),
        ]
        traj = rollout(calls, self.lib)
        self.assertEqual(traj.num_fail, 1)
        self.assertFalse(traj.steps[0].succeeded)

    def test_tool_calls_property(self):
        calls = [("t", ToolCall("set_coord_system", {"origin": [0, 0, 0]}))]
        traj = rollout(calls, self.lib)
        self.assertEqual(traj.tool_calls[0].name, "set_coord_system")


class FormatOrderTest(unittest.TestCase):
    def test_valid_order(self):
        text = (
            "<think>plan</think>"
            "<tool_call>{}</tool_call>"
            "<tool_response>ok</tool_response>"
        )
        self.assertTrue(check_format_order(text))

    def test_valid_two_steps(self):
        step = ("<think>a</think><tool_call>b</tool_call>"
                "<tool_response>c</tool_response>")
        self.assertTrue(check_format_order(step + step))

    def test_wrong_order(self):
        text = (
            "<tool_call>{}</tool_call>"
            "<think>plan</think>"
            "<tool_response>ok</tool_response>"
        )
        self.assertFalse(check_format_order(text))

    def test_missing_tag(self):
        text = "<think>plan</think><tool_call>{}</tool_call>"
        self.assertFalse(check_format_order(text))

    def test_empty(self):
        self.assertFalse(check_format_order(""))


class ParseTest(unittest.TestCase):
    def test_parse_success_and_completion(self):
        text = (
            '<think>make a plate</think>'
            '<tool_call>{"name": "create_simple_sketch", '
            '"arguments": {"profile": "rect"}}</tool_call>'
            '<tool_response>success: created sketch</tool_response>'
            '<answer>COMPLETED</answer>'
        )
        traj = parse_react_trajectory(text)
        self.assertEqual(len(traj), 1)
        self.assertTrue(traj.completed)
        self.assertTrue(traj.steps[0].succeeded)
        self.assertEqual(traj.steps[0].call.name, "create_simple_sketch")
        self.assertEqual(traj.steps[0].call.arguments["profile"], "rect")

    def test_parse_failure_label(self):
        text = (
            '<think>t</think>'
            '<tool_call>{"name": "boolean_operation", "arguments": {}}</tool_call>'
            '<tool_response>Boolean operation failed. Error: no operand</tool_response>'
        )
        traj = parse_react_trajectory(text)
        self.assertFalse(traj.steps[0].succeeded)
        self.assertFalse(traj.completed)

    def test_mismatched_counts_raise(self):
        text = '<think>t</think><tool_call>{"name":"x"}</tool_call>'
        with self.assertRaises(ValueError):
            parse_react_trajectory(text)

    def test_bad_json_raises(self):
        text = ('<think>t</think><tool_call>not json</tool_call>'
                '<tool_response>ok</tool_response>')
        with self.assertRaises(ValueError):
            parse_react_trajectory(text)

    def test_body_without_name_raises(self):
        text = ('<think>t</think><tool_call>{"arguments": {}}</tool_call>'
                '<tool_response>ok</tool_response>')
        with self.assertRaises(ValueError):
            parse_react_trajectory(text)


class RenderRoundTripTest(unittest.TestCase):
    def test_render_then_reparse(self):
        lib = default_toolcad_library()
        calls = [("plan", ToolCall("create_simple_sketch",
                                   {"profile": "rect", "sketch_name": "s1"}))]
        traj = rollout(calls, lib)
        rendered = render_step(traj.steps[0]) + "<answer>COMPLETED</answer>"
        self.assertTrue(check_format_order(render_step(traj.steps[0])))
        reparsed = parse_react_trajectory(rendered)
        self.assertEqual(reparsed.steps[0].call.name, "create_simple_sketch")
        self.assertTrue(reparsed.completed)


if __name__ == "__main__":
    unittest.main()
