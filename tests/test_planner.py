"""End-to-end tests for the NL->CISP planner and the correction runner.

Uses `MockLLM` (canned CISP-op JSON) + a real StubBackend/HarnessSession — no
network, no API keys. The headline case: brief "make a 20x10x5 plate" -> mock
emits NewSketch/AddRectangle/constraints/Extrude -> the runner drives the loop
to ok=True.
"""

import json
import unittest

from harnesscad.core.cisp.ops import NewSketch, AddRectangle, Constrain, Extrude
from harnesscad.io.backends.stub import StubBackend
from harnesscad.core.loop import HarnessSession
from harnesscad.agents.llm.base import CompletionResult, ToolCall
from harnesscad.agents.agent.planner import Planner, PlanError, EMIT_OPS_TOOL
from harnesscad.agents.agent.runner import run
from harnesscad.agents.agent.system_prompt import SYSTEM_PROMPT, op_vocabulary, build_system_prompt

from tests.test_llm import MockLLM, plate_ops_json


def _under_constrained_plate_json() -> str:
    """A plate with only 2 of 4 needed constraints -> under-constrained (warning,
    still ok=True in the harness) but useful for correction tests when paired."""
    ops = (
        [{"op": "new_sketch", "plane": "XY"},
         {"op": "add_rectangle", "sketch": "sk1", "x": 0, "y": 0, "w": 20, "h": 10}]
        + [{"op": "constrain", "kind": "distance", "a": "e1", "value": 10.0} for _ in range(2)]
    )
    return json.dumps(ops)


def _over_constrained_plate_json() -> str:
    """Rectangle + 5 distance constraints -> over-constrained ERROR (rolled back)."""
    ops = (
        [{"op": "new_sketch", "plane": "XY"},
         {"op": "add_rectangle", "sketch": "sk1", "x": 0, "y": 0, "w": 20, "h": 10}]
        + [{"op": "constrain", "kind": "distance", "a": "e1", "value": 10.0} for _ in range(5)]
    )
    return json.dumps(ops)


class TestSystemPrompt(unittest.TestCase):
    def test_prompt_lists_every_op(self):
        vocab = op_vocabulary()
        for tag in ("new_sketch", "add_point", "add_line", "add_circle",
                    "add_rectangle", "constrain", "extrude", "fillet", "boolean"):
            self.assertIn(tag, vocab)

    def test_prompt_states_rules_and_contract(self):
        self.assertIn("mechanical CAD design agent", SYSTEM_PROMPT)
        self.assertIn("JSON array", SYSTEM_PROMPT)
        self.assertIn("constrain", SYSTEM_PROMPT.lower())
        # deterministic + idempotent
        self.assertEqual(SYSTEM_PROMPT, build_system_prompt())


class TestPlanner(unittest.TestCase):
    def test_plan_returns_validated_ops(self):
        planner = Planner(MockLLM([plate_ops_json()]))
        ops = planner.plan("make a 20x10x5 plate")
        self.assertEqual(len(ops), 7)
        self.assertIsInstance(ops[0], NewSketch)
        self.assertIsInstance(ops[1], AddRectangle)
        self.assertIsInstance(ops[-1], Extrude)

    def test_plan_builds_messages_with_brief_and_state(self):
        mock = MockLLM([plate_ops_json()])
        planner = Planner(mock)
        planner.plan("make a plate", state_summary={"feature_count": 0})
        # system + user
        sent = mock.calls[0]
        self.assertEqual(sent[0].role, "system")
        self.assertEqual(sent[0].content, SYSTEM_PROMPT)
        self.assertIn("make a plate", sent[1].content)
        self.assertIn("CURRENT MODEL STATE", sent[1].content)

    def test_plan_includes_diagnostics_in_prompt(self):
        mock = MockLLM([plate_ops_json()])
        planner = Planner(mock)
        diags = [{"severity": "error", "code": "over-constrained",
                  "message": "sketch sk1 is over-constrained", "where": "sk1"}]
        planner.plan("fix it", diagnostics=diags)
        user_msg = mock.calls[0][1].content
        self.assertIn("PRIOR ATTEMPT FAILED", user_msg)
        self.assertIn("over-constrained", user_msg)

    def test_plan_bad_output_raises_planerror(self):
        planner = Planner(MockLLM(["not valid json {"]))
        with self.assertRaises(PlanError):
            planner.plan("make a plate")

    def test_plan_parsed_does_not_raise(self):
        planner = Planner(MockLLM(["garbage"]))
        parsed = planner.plan_parsed("make a plate")
        self.assertFalse(parsed.ok)
        self.assertIsInstance(parsed.error, str)

    def test_planner_with_tool_call_response(self):
        tc = ToolCall("emit_ops", plate_ops_json())
        planner = Planner(MockLLM([CompletionResult(tool_calls=[tc])]), use_tool=True)
        ops = planner.plan("make a 20x10x5 plate")
        self.assertEqual(len(ops), 7)

    def test_tool_spec_shape(self):
        self.assertEqual(EMIT_OPS_TOOL.name, "emit_ops")


class TestRunnerEndToEnd(unittest.TestCase):
    def test_plate_brief_drives_loop_to_ok(self):
        session = HarnessSession(StubBackend())
        planner = Planner(MockLLM([plate_ops_json()]))
        result = run(session, planner, "make a 20x10x5 plate")
        self.assertTrue(result.ok)
        self.assertEqual(result.applied, 7)
        summary = session.summary()
        self.assertTrue(summary["solid_present"])
        self.assertEqual(summary["feature_count"], 1)

    def test_runner_recovers_from_bad_first_plan(self):
        # First plan over-constrains (ERROR -> rolled back, not ok); second plan
        # is correct. The runner must feed diagnostics back and converge.
        session = HarnessSession(StubBackend())
        planner = Planner(MockLLM([
            _over_constrained_plate_json(),
            plate_ops_json(),
        ]))
        result = run(session, planner, "make a 20x10x5 plate", max_iters=5)
        self.assertTrue(result.ok)
        self.assertEqual(result.applied, 7)
        self.assertTrue(session.summary()["solid_present"])

    def test_runner_feeds_diagnostics_to_second_plan(self):
        session = HarnessSession(StubBackend())
        mock = MockLLM([_over_constrained_plate_json(), plate_ops_json()])
        planner = Planner(mock)
        run(session, planner, "make a 20x10x5 plate", max_iters=5)
        # Second call must carry the failure diagnostics from the first apply.
        self.assertEqual(len(mock.calls), 2)
        second_user_msg = mock.calls[1][1].content
        self.assertIn("PRIOR ATTEMPT FAILED", second_user_msg)
        self.assertIn("over-constrained", second_user_msg)

    def test_runner_recovers_from_unparseable_first_plan(self):
        session = HarnessSession(StubBackend())
        planner = Planner(MockLLM(["not json", plate_ops_json()]))
        result = run(session, planner, "make a plate", max_iters=5)
        self.assertTrue(result.ok)
        self.assertEqual(result.applied, 7)

    def test_runner_gives_up_after_max_iters(self):
        # Always over-constrains -> never ok. Runner returns the last (failed) result.
        session = HarnessSession(StubBackend())
        planner = Planner(MockLLM([_over_constrained_plate_json()] * 10))
        result = run(session, planner, "make a plate", max_iters=3)
        self.assertFalse(result.ok)

    def test_runner_bad_reference_is_fed_back(self):
        # Extrude a non-existent sketch -> block-and-correct rejects; then fix.
        session = HarnessSession(StubBackend())
        bad = json.dumps([{"op": "extrude", "sketch": "nope", "distance": 5.0}])
        planner = Planner(MockLLM([bad, plate_ops_json()]))
        result = run(session, planner, "make a plate", max_iters=5)
        self.assertTrue(result.ok)


if __name__ == "__main__":
    unittest.main()
