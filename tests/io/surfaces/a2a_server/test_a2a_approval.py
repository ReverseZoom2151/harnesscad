"""The A2A public surface must gate its one destructive action: the STEP export.

``A2AHandler._execute`` called ``backend.export("step")`` unconditionally -- a
tier-3, irreversible hand-off on a public surface, with no consent step and no
record. The handler now decides through an ``ApprovalPolicy``:

  * with a REFUSE policy the export NEVER runs and the task fails with an
    auditable refusal (this test FAILS before the fix: the artifact is produced);
  * the default policy still delivers the artifact, but only because it is an
    EXPLICIT ``headless_auto_approve`` whose reason is recorded on the artifact.
"""

import unittest

from harnesscad.agents.llm.structured import ParsedOps
from harnesscad.core.cisp.ops import AddRectangle, Constrain, Extrude, NewSketch
from harnesscad.core.harness import AgentHarness
from harnesscad.core.loop import HarnessSession
from harnesscad.io.backends.stub import StubBackend
from harnesscad.io.surfaces.a2a_server.handler import A2AHandler
from harnesscad.io.surfaces.ui.approval import ApprovalPolicy


def plate_ops():
    return (
        [NewSketch(), AddRectangle(sketch="sk1")]
        + [Constrain(kind="distance", a="e1", value=20.0) for _ in range(4)]
        + [Extrude(sketch="sk1", distance=5.0)]
    )


class MockPlanner:
    def plan_parsed(self, brief, state_summary=None, diagnostics=None):
        if state_summary and state_summary.get("solid_present"):
            return ParsedOps([])
        return ParsedOps(list(plate_ops()))


class SpyBackend(StubBackend):
    """Records every export so a test can prove the export never ran."""

    def __init__(self):
        super().__init__()
        self.exports = []

    def export(self, fmt):
        self.exports.append(fmt)
        return super().export(fmt)


class _Factory:
    """Builds a harness per task and keeps the backends for inspection."""

    def __init__(self):
        self.backends = []

    def __call__(self):
        backend = SpyBackend()
        self.backends.append(backend)
        return AgentHarness(HarnessSession(backend), MockPlanner())


def _send(text="a 20mm plate"):
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {"message": {
            "role": "user",
            "parts": [{"kind": "text", "text": text}],
            "messageId": "m1",
            "kind": "message",
        }},
    }


class A2AExportRefusedTest(unittest.TestCase):
    def test_refuse_policy_blocks_the_export_and_fails_the_task(self):
        factory = _Factory()
        handler = A2AHandler(factory, approval=ApprovalPolicy(surface="a2a"))
        task = handler.dispatch(_send())["result"]

        self.assertEqual(task["status"]["state"], "failed")
        self.assertEqual(task.get("artifacts", []), [])
        # THE POINT: the destructive call never happened.
        self.assertEqual(factory.backends[-1].exports, [])
        record = handler.approval.audit[-1]
        self.assertEqual(record.op_name, "export")
        self.assertFalse(record.approved)
        self.assertEqual(record.decided_by, "policy:headless-refuse")


class A2ADefaultPolicyIsExplicitTest(unittest.TestCase):
    def test_default_export_is_auto_approved_but_recorded_on_the_artifact(self):
        factory = _Factory()
        handler = A2AHandler(factory)  # default: explicit headless auto-approve
        task = handler.dispatch(_send())["result"]

        self.assertEqual(task["status"]["state"], "completed")
        self.assertEqual(factory.backends[-1].exports, ["step"])
        approval = task["artifacts"][0]["metadata"]["approval"]
        self.assertTrue(approval["approved"])
        self.assertEqual(approval["decided_by"], "policy:headless-auto-approve")
        self.assertIn("message/send", approval["reason"])

    def test_human_approver_is_consulted_for_the_export(self):
        asked = []
        handler = A2AHandler(_Factory(), approval=ApprovalPolicy(
            lambda name: asked.append(name) or True, principal="operator"))
        task = handler.dispatch(_send())["result"]
        self.assertEqual(task["status"]["state"], "completed")
        self.assertEqual(asked, ["export"])


if __name__ == "__main__":
    unittest.main()
