"""The MCP public surface must not execute a destructive tool without consent.

``tools/call`` carried ``destructiveHint`` annotations for ``export`` / ``reset``
/ ``delete`` and then ran them anyway: a public, unauthenticated surface with no
human gate. These tests pin the gate closed.

Before the fix, ``test_export_is_refused_in_a_headless_server`` and
``test_reset_is_refused_in_a_headless_server`` both FAIL (the call returns a
successful CallToolResult and the model state is wiped / the artifact written).
"""

import unittest

from harnesscad.core.cisp.ops import Extrude, NewSketch
from harnesscad.core.loop import HarnessSession
from harnesscad.io.backends.stub import StubBackend
from harnesscad.io.surfaces.mcp.server import MCPServer
from harnesscad.io.surfaces.ui.approval import ApprovalPolicy


def _req(msg_id, method, params=None):
    msg = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def _call(name, arguments=None):
    return _req(1, "tools/call", {"name": name, "arguments": arguments or {}})


class MCPHeadlessRefusalTest(unittest.TestCase):
    def setUp(self):
        self.session = HarnessSession(StubBackend())
        self.server = MCPServer(session=self.session)  # headless: no approver

    def test_export_is_refused_in_a_headless_server(self):
        resp = self.server.handle(_call("export", {"fmt": "step"}))
        result = resp["result"]
        self.assertTrue(result["isError"])
        record = result["structuredContent"]["approvalDenied"]
        self.assertEqual(record["op"], "export")
        self.assertEqual(record["tier"], "require")
        self.assertEqual(record["risk"], "high")
        self.assertEqual(record["decided_by"], "policy:headless-refuse")
        # Nothing was exported: the tool never ran.
        self.assertNotIn("content", result.get("structuredContent", {}))

    def test_reset_is_refused_and_the_model_survives(self):
        self.session.apply_ops([NewSketch()])
        before = self.session.digest()
        resp = self.server.handle(_call("reset"))
        self.assertTrue(resp["result"]["isError"])
        self.assertEqual(self.session.digest(), before)  # state NOT wiped

    def test_tier2_op_tool_still_executes_and_is_recorded(self):
        resp = self.server.handle(_call("new_sketch", {"plane": "XY"}))
        self.assertFalse(resp["result"].get("isError"))
        record = self.server.approval.audit[-1]
        self.assertEqual(record.op_name, "new_sketch")
        self.assertTrue(record.approved)
        self.assertEqual(record.decided_by, "policy:tier-2")

    def test_readonly_tool_needs_no_gate(self):
        resp = self.server.handle(_call("query", {"what": "summary"}))
        self.assertFalse(resp["result"].get("isError"))
        self.assertEqual(self.server.approval.audit[-1].decided_by, "policy:tier-1")


class MCPApprovedExportTest(unittest.TestCase):
    def test_human_approver_lets_the_export_through(self):
        session = HarnessSession(StubBackend())
        asked = []
        server = MCPServer(session=session, approval=ApprovalPolicy(
            lambda name: asked.append(name) or True,
            principal="operator", surface="mcp"))
        resp = server.handle(_call("export", {"fmt": "step"}))
        self.assertFalse(resp["result"].get("isError"))
        self.assertEqual(asked, ["export"])
        self.assertEqual(server.approval.audit[-1].decided_by, "human:operator")

    def test_explicit_headless_auto_approve_lets_the_export_through(self):
        session = HarnessSession(StubBackend())
        session.apply_ops([NewSketch()])
        server = MCPServer(session=session,
                           approval=ApprovalPolicy.headless_auto_approve(
                               "trusted embedder: exports are sandboxed",
                               surface="mcp"))
        resp = server.handle(_call("export", {"fmt": "step"}))
        self.assertFalse(resp["result"].get("isError"))
        self.assertEqual(server.approval.audit[-1].decided_by,
                         "policy:headless-auto-approve")

    def test_denying_human_blocks_the_export(self):
        session = HarnessSession(StubBackend())
        server = MCPServer(session=session,
                           approval=ApprovalPolicy(lambda name: False))
        resp = server.handle(_call("export", {"fmt": "step"}))
        self.assertTrue(resp["result"]["isError"])
        self.assertIn("approvalDenied", resp["result"]["structuredContent"])


class MCPUnknownToolStillErrorsTest(unittest.TestCase):
    def test_unknown_tool_is_still_invalid_params_not_a_denial(self):
        server = MCPServer(session=HarnessSession(StubBackend()))
        resp = server.handle(_call("nope"))
        self.assertIn("error", resp)

    def test_extrude_still_reaches_the_verifier(self):
        # A tier-2 op that the verifier rejects must still come back as a tool
        # error with diagnostics -- the gate must not swallow the self-correction
        # channel.
        server = MCPServer(session=HarnessSession(StubBackend()))
        resp = server.handle(_call("extrude", {"sketch": "nope", "distance": 5.0}))
        self.assertTrue(resp["result"]["isError"])
        self.assertIn("diagnostics", resp["result"]["structuredContent"])


if __name__ == "__main__":
    unittest.main()
