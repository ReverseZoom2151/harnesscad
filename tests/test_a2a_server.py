"""Tests for the HarnessCAD A2A protocol server (surfaces/a2a_server).

Deterministic and offline: a MockPlanner emitting a constrained plate drives a
real AgentHarness over the dependency-free StubBackend. Most assertions run the
JSON-RPC dispatcher in-process (no socket); one end-to-end test binds an
ephemeral port and speaks HTTP to prove the transport is wired correctly.
"""

import json
import threading
import unittest
import urllib.request

from harnesscad.core.cisp.ops import AddRectangle, Constrain, Extrude, NewSketch
from harnesscad.io.backends.stub import StubBackend
from harnesscad.core.harness import AgentHarness
from harnesscad.agents.llm.structured import ParsedOps
from harnesscad.core.loop import HarnessSession
from harnesscad.io.surfaces.a2a_server import wire
from harnesscad.io.surfaces.a2a_server.app import make_server
from harnesscad.io.surfaces.a2a_server.card import build_agent_card
from harnesscad.io.surfaces.a2a_server.handler import A2AHandler


# --- fixtures --------------------------------------------------------------
def plate_ops():
    """A verifying plan: rectangle sketch, fully constrained, extruded to a solid."""
    return (
        [NewSketch(), AddRectangle(sketch="sk1")]
        + [Constrain(kind="distance", a="e1", value=20.0) for _ in range(4)]
        + [Extrude(sketch="sk1", distance=5.0)]
    )


class MockPlanner:
    """Emits the plate plan once, then empty plans (so the harness converges)."""

    def plan_parsed(self, brief, state_summary=None, diagnostics=None):
        if state_summary and state_summary.get("solid_present"):
            return ParsedOps([])
        return ParsedOps(list(plate_ops()))


def harness_factory():
    return AgentHarness(HarnessSession(StubBackend()), MockPlanner())


def make_handler():
    return A2AHandler(harness_factory)


def send_request(text="a 20mm plate", **msg_extra):
    message = {
        "role": "user",
        "parts": [{"kind": "text", "text": text}],
        "messageId": "m1",
        "kind": "message",
    }
    message.update(msg_extra)
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {"message": message},
    }


# --- agent card ------------------------------------------------------------
class TestAgentCard(unittest.TestCase):
    def test_card_shape(self):
        card = build_agent_card(url="http://example.test:9100/").to_dict()
        self.assertEqual(card["name"], "HarnessCAD")
        self.assertEqual(card["url"], "http://example.test:9100/")
        self.assertEqual(card["protocolVersion"], "0.3.0")
        self.assertEqual(card["preferredTransport"], "JSONRPC")
        self.assertTrue(card["capabilities"]["streaming"])
        self.assertFalse(card["capabilities"]["pushNotifications"])
        skills = card["skills"]
        self.assertEqual(len(skills), 1)
        skill = skills[0]
        self.assertEqual(skill["id"], "text-to-cad")
        self.assertIn("cad", skill["tags"])
        self.assertIn("model/step", skill["outputModes"])


# --- message/send ----------------------------------------------------------
class TestMessageSend(unittest.TestCase):
    def test_send_completes_with_step_artifact(self):
        handler = make_handler()
        resp = handler.dispatch(send_request())
        self.assertEqual(resp["jsonrpc"], "2.0")
        self.assertNotIn("error", resp)
        task = resp["result"]
        self.assertEqual(task["kind"], "task")
        self.assertEqual(task["status"]["state"], "completed")
        self.assertEqual(len(task["artifacts"]), 1)
        artifact = task["artifacts"][0]
        self.assertEqual(artifact["name"], "out.step")
        parts = artifact["parts"]
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0]["kind"], "file")
        file_obj = parts[0]["file"]
        self.assertEqual(file_obj["mimeType"], "model/step")
        self.assertIn("bytes", file_obj)
        self.assertTrue(file_obj["bytes"])  # non-empty base64 payload

    def test_send_registers_task_retrievable_via_get(self):
        handler = make_handler()
        resp = handler.dispatch(send_request())
        task_id = resp["result"]["id"]
        got = handler.dispatch(
            {"jsonrpc": "2.0", "id": 2, "method": "tasks/get", "params": {"id": task_id}}
        )
        self.assertEqual(got["result"]["id"], task_id)
        self.assertEqual(got["result"]["status"]["state"], "completed")


# --- tasks/get -------------------------------------------------------------
class TestTasksGet(unittest.TestCase):
    def test_unknown_task_returns_task_not_found(self):
        handler = make_handler()
        resp = handler.dispatch(
            {"jsonrpc": "2.0", "id": 3, "method": "tasks/get", "params": {"id": "nope"}}
        )
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], wire.ERROR_TASK_NOT_FOUND)


# --- tasks/cancel ----------------------------------------------------------
class TestTasksCancel(unittest.TestCase):
    def test_cancel_submitted_task_succeeds(self):
        handler = make_handler()
        task = handler.store.create()
        task.submit()  # SUBMITTED is cancelable
        resp = handler.dispatch(
            {"jsonrpc": "2.0", "id": 4, "method": "tasks/cancel",
             "params": {"id": task.taskId}}
        )
        self.assertNotIn("error", resp)
        self.assertEqual(resp["result"]["status"]["state"], "canceled")

    def test_cancel_completed_task_is_not_cancelable(self):
        handler = make_handler()
        send = handler.dispatch(send_request())
        task_id = send["result"]["id"]  # already completed
        resp = handler.dispatch(
            {"jsonrpc": "2.0", "id": 5, "method": "tasks/cancel",
             "params": {"id": task_id}}
        )
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], wire.ERROR_TASK_NOT_CANCELABLE)

    def test_cancel_unknown_task_returns_task_not_found(self):
        handler = make_handler()
        resp = handler.dispatch(
            {"jsonrpc": "2.0", "id": 6, "method": "tasks/cancel", "params": {"id": "x"}}
        )
        self.assertEqual(resp["error"]["code"], wire.ERROR_TASK_NOT_FOUND)


# --- error codes -----------------------------------------------------------
class TestErrors(unittest.TestCase):
    def test_unknown_method(self):
        handler = make_handler()
        resp = handler.dispatch({"jsonrpc": "2.0", "id": 7, "method": "bogus/method"})
        self.assertEqual(resp["error"]["code"], wire.ERROR_METHOD_NOT_FOUND)

    def test_push_notification_not_supported(self):
        handler = make_handler()
        resp = handler.dispatch(
            {"jsonrpc": "2.0", "id": 8,
             "method": "tasks/pushNotificationConfig/set", "params": {}}
        )
        self.assertEqual(
            resp["error"]["code"], wire.ERROR_PUSH_NOTIFICATION_NOT_SUPPORTED
        )


# --- message/stream --------------------------------------------------------
class TestMessageStream(unittest.TestCase):
    def test_stream_yields_status_and_artifact_frames(self):
        handler = make_handler()
        request = {
            "jsonrpc": "2.0", "id": 9, "method": "message/stream",
            "params": {"message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "a plate"}],
                "kind": "message",
            }},
        }
        frames = list(handler.stream(request))
        self.assertTrue(frames)
        results = [json.loads(f[len("data: "):].strip())["result"] for f in frames]
        kinds = [r["kind"] for r in results]
        self.assertIn("status-update", kinds)
        self.assertIn("artifact-update", kinds)
        # Terminal frame is a final status-update reaching 'completed'.
        last = results[-1]
        self.assertEqual(last["kind"], "status-update")
        self.assertTrue(last["final"])
        self.assertEqual(last["status"]["state"], "completed")


# --- end-to-end over a real socket -----------------------------------------
class TestHttpEndToEnd(unittest.TestCase):
    def setUp(self):
        self.server = make_server(harness_factory, host="127.0.0.1", port=0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def _url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def test_agent_card_over_http(self):
        with urllib.request.urlopen(self._url("/.well-known/agent-card.json")) as r:
            card = json.loads(r.read().decode())
        self.assertEqual(card["name"], "HarnessCAD")
        self.assertEqual(card["skills"][0]["id"], "text-to-cad")
        self.assertEqual(card["protocolVersion"], "0.3.0")

    def test_message_send_over_http(self):
        payload = json.dumps(send_request()).encode()
        req = urllib.request.Request(
            self._url("/"), data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as r:
            resp = json.loads(r.read().decode())
        self.assertNotIn("error", resp)
        task = resp["result"]
        self.assertEqual(task["status"]["state"], "completed")
        self.assertEqual(task["artifacts"][0]["parts"][0]["file"]["mimeType"], "model/step")


if __name__ == "__main__":
    unittest.main()
