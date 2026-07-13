"""Tests for the Zed ACP agent adapter (surfaces.acp).

Drives ``ACPAgent`` in-process over a MOCK bidirectional transport — no real
stdio or network. The mock captures every outbound frame and auto-answers the
agent's outbound requests (``session/request_permission`` / ``fs/write_text_file``)
synchronously, so the whole permission round-trip runs deterministically on one
thread.

Covers:
  * initialize returns the ACP handshake.
  * session/new returns a sessionId and wires a live session.
  * session/prompt on a good brief (StubBackend + a MockPlanner plate plan)
    runs and emits agent_message_chunk + tool_call + tool_call_update + plan and
    a terminal stopReason=end_turn.
  * a Tier-3 op (export) triggers session/request_permission; an injected
    allow_once proceeds (tool_call_update completed) while reject_once fails it
    (tool_call_update failed).
  * the 3-tier mapping: measure -> no permission (read), modify -> edit,
    export -> permission (execute).
  * session/cancel maps to stopReason=cancelled.
"""

import unittest

from harnesscad.core.cisp.ops import AddRectangle, Extrude, NewSketch
from harnesscad.core.cisp.protocol import ApplyOpsResult
from harnesscad.agents.llm.structured import ParsedOps
from harnesscad.eval.reliability.executor import ToolExecutor

from harnesscad.io.surfaces.acp.agent import ACPAgent, BridgingExecutor, PromptCancelled
from harnesscad.io.surfaces.acp.bridge import ACPBridge, kind_for
from harnesscad.io.surfaces.acp.jsonrpc import Connection
from harnesscad.io.surfaces.ui.approval import ApprovalTier, tier_for


# --- test doubles ----------------------------------------------------------
class MockTransport:
    """A fake bidirectional channel: captures outbound frames and auto-answers
    the agent's outbound requests from a scripted queue."""

    def __init__(self, permission_script=None):
        self.sent = []
        self.connection = None  # set after Connection is built
        self.permission_script = list(permission_script or [])

    def send(self, msg):
        self.sent.append(msg)
        # Auto-respond to agent-initiated REQUESTS (they carry id + method).
        if "method" in msg and "id" in msg:
            self._auto_respond(msg)

    def _auto_respond(self, msg):
        method = msg["method"]
        if method == "session/request_permission":
            option = self.permission_script.pop(0) if self.permission_script \
                else "reject_once"
            result = {"outcome": {"outcome": "selected", "optionId": option}}
        else:  # e.g. fs/write_text_file
            result = {}
        self.connection.deliver(
            {"jsonrpc": "2.0", "id": msg["id"], "result": result})

    # --- capture helpers --------------------------------------------------
    def updates(self):
        """All session/update payloads, in order."""
        return [m["params"]["update"] for m in self.sent
                if m.get("method") == "session/update"]

    def update_kinds(self):
        return [u.get("sessionUpdate") for u in self.updates()]

    def requests(self, method):
        return [m for m in self.sent if m.get("method") == method and "id" in m]


class MockPlanner:
    """Returns a fixed op list once; accepts the harness's kwargs."""

    def __init__(self, ops):
        self.ops = list(ops)

    def plan_parsed(self, brief, state_summary=None, diagnostics=None):
        return ParsedOps(list(self.ops))


class CancelPlanner:
    """Flips the bridge cancel flag when planning, to exercise session/cancel."""

    def __init__(self, bridge, ops):
        self.bridge = bridge
        self.ops = list(ops)

    def plan_parsed(self, brief, state_summary=None, diagnostics=None):
        self.bridge.cancelled = True
        return ParsedOps(list(self.ops))


class MockSession:
    """A minimal session: apply_ops always succeeds (isolates the gate path)."""

    backend = None

    def apply_ops(self, ops):
        return ApplyOpsResult(True, len(ops), "digest", [])

    def digest(self):
        return "digest"


class ExportOp:
    """A synthetic Tier-3 op (name 'export' classifies as REQUIRE)."""

    OP = "export"

    def to_dict(self):
        return {"op": "export"}


class MeasureOp:
    """A synthetic Tier-1 op (name 'measure' classifies as AUTO)."""

    OP = "measure"

    def to_dict(self):
        return {"op": "measure"}


def _plate_plan():
    return [
        NewSketch(plane="XY"),
        AddRectangle(sketch="sk1", w=10.0, h=10.0),
        Extrude(sketch="sk1", distance=2.0),
    ]


def _make_agent(permission_script=None, planner=None, backend="stub"):
    transport = MockTransport(permission_script=permission_script)
    connection = Connection(transport.send)
    transport.connection = connection
    factory = (lambda be: planner) if planner is not None \
        else (lambda be: MockPlanner(_plate_plan()))
    agent = ACPAgent(connection, backend=backend, planner_factory=factory)
    return agent, transport, connection


# --- tests -----------------------------------------------------------------
class TestHandshake(unittest.TestCase):
    def test_initialize_returns_handshake(self):
        agent, _t, _c = _make_agent()
        result = agent.initialize({"clientCapabilities": {"fs": {}}})
        self.assertEqual(result["protocolVersion"], 1)
        self.assertEqual(result["agentInfo"]["name"], "harnesscad")
        self.assertFalse(result["agentCapabilities"]["loadSession"])
        self.assertIn("promptCapabilities", result["agentCapabilities"])

    def test_initialize_reads_fs_capability(self):
        agent, _t, _c = _make_agent()
        agent.initialize({"clientCapabilities": {"fs": {"writeTextFile": True}}})
        self.assertTrue(agent.client_can_write_fs)


class TestSessionNew(unittest.TestCase):
    def test_session_new_returns_id_and_wires_session(self):
        agent, _t, _c = _make_agent()
        agent.initialize({})
        result = agent.session_new({"cwd": "/work"})
        sid = result["sessionId"]
        self.assertTrue(sid.startswith("sess-"))
        self.assertIn(sid, agent._sessions)
        entry = agent._sessions[sid]
        self.assertIsNotNone(entry["harness"])
        self.assertIsInstance(entry["bridge"], ACPBridge)


class TestPromptGoodBrief(unittest.TestCase):
    def test_prompt_runs_and_emits_updates(self):
        agent, transport, _c = _make_agent()
        agent.initialize({})
        sid = agent.session_new({"cwd": "/work"})["sessionId"]
        result = agent.session_prompt({
            "sessionId": sid,
            "prompt": [{"type": "text", "text": "a 10mm square plate"}],
        })
        kinds = transport.update_kinds()
        self.assertIn("agent_message_chunk", kinds)
        self.assertIn("tool_call", kinds)
        self.assertIn("tool_call_update", kinds)
        self.assertIn("plan", kinds)
        # Tier-2 (NOTIFY) ops require no permission round-trip.
        self.assertEqual(transport.requests("session/request_permission"), [])
        self.assertEqual(result["stopReason"], "end_turn")

    def test_tool_call_updates_are_completed(self):
        agent, transport, _c = _make_agent()
        agent.initialize({})
        sid = agent.session_new({"cwd": "/work"})["sessionId"]
        agent.session_prompt({"sessionId": sid,
                              "prompt": [{"type": "text", "text": "plate"}]})
        completes = [u for u in transport.updates()
                     if u.get("sessionUpdate") == "tool_call_update"]
        self.assertTrue(completes)
        self.assertTrue(all(u["status"] == "completed" for u in completes))

    def test_plan_entries_shape(self):
        agent, transport, _c = _make_agent()
        agent.initialize({})
        sid = agent.session_new({"cwd": "/work"})["sessionId"]
        agent.session_prompt({"sessionId": sid,
                              "prompt": [{"type": "text", "text": "plate"}]})
        plans = [u for u in transport.updates() if u.get("sessionUpdate") == "plan"]
        self.assertTrue(plans)
        for entry in plans[-1]["entries"]:
            self.assertIn("content", entry)
            self.assertIn("priority", entry)
            self.assertIn("status", entry)

    def test_step_delivered_inline_without_fs(self):
        agent, transport, _c = _make_agent()
        agent.initialize({"clientCapabilities": {"fs": {}}})  # no writeTextFile
        sid = agent.session_new({"cwd": "/work"})["sessionId"]
        agent.session_prompt({"sessionId": sid,
                              "prompt": [{"type": "text", "text": "plate"}]})
        resources = [u for u in transport.updates()
                     if u.get("content", {}).get("type") == "resource"]
        self.assertTrue(resources)
        self.assertEqual(resources[0]["content"]["resource"]["mimeType"],
                         "application/step")

    def test_step_delivered_via_fs_when_supported(self):
        agent, transport, _c = _make_agent()
        agent.initialize({"clientCapabilities": {"fs": {"writeTextFile": True}}})
        sid = agent.session_new({"cwd": "/work"})["sessionId"]
        agent.session_prompt({"sessionId": sid,
                              "prompt": [{"type": "text", "text": "plate"}]})
        writes = transport.requests("fs/write_text_file")
        self.assertEqual(len(writes), 1)
        self.assertTrue(writes[0]["params"]["path"].endswith("/out.step"))


class TestTierThreePermission(unittest.TestCase):
    def _run_export(self, option):
        transport = MockTransport(permission_script=[option])
        connection = Connection(transport.send)
        transport.connection = connection
        bridge = ACPBridge(connection, "sess-1", "/work",
                           client_can_write_fs=False)
        tx = ToolExecutor(approval=lambda op: bridge.request_permission(op))
        executor = BridgingExecutor(MockSession(), bridge, tx)
        result = executor.apply_ops([ExportOp()])
        return transport, result

    def test_allow_once_proceeds(self):
        transport, result = self._run_export("allow_once")
        self.assertEqual(len(transport.requests("session/request_permission")), 1)
        self.assertTrue(result.ok)
        updates = transport.update_kinds()
        self.assertIn("tool_call", updates)
        completes = [u for u in transport.updates()
                     if u.get("sessionUpdate") == "tool_call_update"]
        self.assertTrue(completes)
        self.assertEqual(completes[-1]["status"], "completed")

    def test_reject_once_fails(self):
        transport, result = self._run_export("reject_once")
        self.assertEqual(len(transport.requests("session/request_permission")), 1)
        self.assertFalse(result.ok)
        failed = [u for u in transport.updates()
                  if u.get("sessionUpdate") == "tool_call_update"]
        self.assertTrue(failed)
        self.assertEqual(failed[-1]["status"], "failed")

    def test_permission_request_shape(self):
        transport, _result = self._run_export("allow_once")
        req = transport.requests("session/request_permission")[0]
        params = req["params"]
        self.assertEqual(params["title"], "export")
        self.assertIn("risk", params["description"])
        option_ids = {o["optionId"] for o in params["options"]}
        self.assertEqual(option_ids,
                         {"allow_once", "allow_always", "reject_once", "reject_always"})


class TestThreeTierMapping(unittest.TestCase):
    def test_measure_no_permission(self):
        transport = MockTransport()
        connection = Connection(transport.send)
        transport.connection = connection
        bridge = ACPBridge(connection, "sess-1", "/work")
        tx = ToolExecutor(approval=lambda op: bridge.request_permission(op))
        executor = BridgingExecutor(MockSession(), bridge, tx)
        executor.apply_ops([MeasureOp()])
        self.assertEqual(transport.requests("session/request_permission"), [])

    def test_tier_and_kind_mapping(self):
        self.assertIs(tier_for(MeasureOp()), ApprovalTier.AUTO)
        self.assertIs(tier_for(NewSketch()), ApprovalTier.NOTIFY)
        self.assertIs(tier_for(ExportOp()), ApprovalTier.REQUIRE)
        self.assertEqual(kind_for(MeasureOp()), "read")
        self.assertEqual(kind_for(NewSketch()), "edit")
        self.assertEqual(kind_for(ExportOp()), "execute")


class TestCancel(unittest.TestCase):
    def test_cancel_flag_yields_cancelled(self):
        agent, _t, _c = _make_agent()
        agent.initialize({})
        sid = agent.session_new({"cwd": "/work"})["sessionId"]
        entry = agent._sessions[sid]
        # Inject a planner that flips the cancel flag mid-run.
        entry["harness"].planner = CancelPlanner(entry["bridge"], _plate_plan())
        result = agent.session_prompt({
            "sessionId": sid, "prompt": [{"type": "text", "text": "plate"}]})
        self.assertEqual(result["stopReason"], "cancelled")

    def test_session_cancel_sets_flag(self):
        agent, _t, _c = _make_agent()
        agent.initialize({})
        sid = agent.session_new({"cwd": "/work"})["sessionId"]
        agent.session_cancel({"sessionId": sid})
        self.assertTrue(agent._sessions[sid]["bridge"].cancelled)


if __name__ == "__main__":
    unittest.main()
