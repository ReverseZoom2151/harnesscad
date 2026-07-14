"""The harness write path must run every op through the approval gate.

``AgentHarness`` dispatches through a ``SessionToolExecutor``; this pins the
approval half of that executor:

  * a tier-3 op planned by the model is REFUSED in a headless run and NEVER
    reaches the session (before the fix the executor auto-approved it silently);
  * the harness accepts an approver / an explicit policy, and the decision is
    recorded on the executor's audit log;
  * a tier-2 plan is unaffected (the gate is not a brake on ordinary modelling).
"""

import unittest
from dataclasses import dataclass
from typing import ClassVar

from harnesscad.agents.llm.structured import ParsedOps
from harnesscad.core.cisp.ops import Op
from harnesscad.core.harness import AgentHarness
from harnesscad.core.loop import HarnessSession
from harnesscad.io.backends.stub import StubBackend
from harnesscad.io.surfaces.ui.approval import ApprovalPolicy


@dataclass(frozen=True)
class DeleteOp(Op):
    """A tier-3 (destructive) op: 'delete' classifies as REQUIRE."""

    OP: ClassVar[str] = "delete"
    target: str = "body1"


class SpySession(HarnessSession):
    """A real session that records every apply_ops it is asked to run."""

    def __init__(self):
        super().__init__(StubBackend())
        self.applied_batches = []

    def apply_ops(self, ops):
        self.applied_batches.append(list(ops))
        return super().apply_ops(ops)


class DeletePlanner:
    """Plans one destructive op, then nothing."""

    def __init__(self):
        self.calls = 0

    def plan_parsed(self, brief, state_summary=None, diagnostics=None):
        self.calls += 1
        return ParsedOps([DeleteOp()])


class HarnessApprovalTest(unittest.TestCase):
    def test_tier3_op_is_refused_and_never_reaches_the_session(self):
        session = SpySession()
        harness = AgentHarness(session, DeletePlanner(), max_iterations=1)
        run = harness.run("delete the body")

        self.assertFalse(run.ok)
        self.assertEqual(session.applied_batches, [])  # never applied
        self.assertEqual(run.applied, 0)
        rejected = run.trajectory[0]["rejected"]
        self.assertIsNotNone(rejected)
        self.assertEqual(rejected["reason"], "approval-denied")
        record = harness.executor.executor.approval_audit[-1]
        self.assertEqual(record["decided_by"], "policy:headless-refuse")

    def test_an_approver_can_let_the_tier3_op_through(self):
        session = SpySession()
        asked = []
        harness = AgentHarness(session, DeletePlanner(), max_iterations=1,
                               approval=lambda op: asked.append(op) or True)
        harness.run("delete the body")
        self.assertEqual(len(asked), 1)
        self.assertEqual(len(session.applied_batches), 1)
        self.assertEqual(harness.executor.executor.approval_audit[-1]["decided_by"],
                         "human:approver")

    def test_explicit_policy_object_is_used(self):
        session = SpySession()
        policy = ApprovalPolicy.headless_auto_approve(
            "unattended batch: operator signed off", surface="harness")
        harness = AgentHarness(session, DeletePlanner(), max_iterations=1,
                               approval=policy)
        harness.run("delete the body")
        self.assertEqual(len(session.applied_batches), 1)
        self.assertEqual(policy.audit[-1].decided_by, "policy:headless-auto-approve")


if __name__ == "__main__":
    unittest.main()
