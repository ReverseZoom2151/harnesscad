"""Conformance tests for the real Google A2A protocol JSON-RPC binding.

These assert the SPEC wire shapes from a2a-protocol.org (as opposed to the
internal-convenience shapes exercised in test_a2a.py):

  - TaskState wire values, including the hyphenated "input-required" and the
    added rejected/auth-required/unknown states, plus their legal transitions.
  - The A2A ``Task`` object shape emitted by ``Task.to_a2a`` -- top-level ``id``,
    nested ``status``, ``kind:"task"``.
  - The spec ``file`` Part (FilePart) round-trip and the first-class ``Artifact``.
  - AgentCard emitting url/protocolVersion/preferredTransport, and AgentSkill.
  - A2AMessage emitting ``kind:"message"``.
  - The hyphenated wire event kinds via ``to_wire_event``.
"""

import json
import unittest

from a2a.messages import (
    A2AMessage,
    AgentCard,
    AgentSkill,
    Artifact,
    Part,
    PART_FILE,
    ROLE_AGENT,
)
from a2a.task import (
    EVENT_ARTIFACT_UPDATE,
    EVENT_STATUS_UPDATE,
    IllegalTransition,
    Task,
    TaskState,
    monotonic_counter,
    to_wire_event,
)


def _roundtrip(obj_dict):
    return json.loads(json.dumps(obj_dict))


class TestTaskStateWireValues(unittest.TestCase):
    def test_input_required_is_hyphenated(self):
        self.assertEqual(TaskState.INPUT_REQUIRED.value, "input-required")

    def test_added_states_present_with_wire_values(self):
        self.assertEqual(TaskState.REJECTED.value, "rejected")
        self.assertEqual(TaskState.AUTH_REQUIRED.value, "auth-required")
        self.assertEqual(TaskState.UNKNOWN.value, "unknown")

    def test_rejected_is_terminal(self):
        from a2a.task import TERMINAL_STATES

        self.assertIn(TaskState.REJECTED, TERMINAL_STATES)


class TestNewTransitions(unittest.TestCase):
    def test_submitted_to_rejected_legal(self):
        t = Task()
        t.submit().reject()
        self.assertEqual(t.state, TaskState.REJECTED)

    def test_working_to_auth_required_legal(self):
        t = Task()
        t.submit().start().require_auth()
        self.assertEqual(t.state, TaskState.AUTH_REQUIRED)

    def test_auth_required_recovers_and_cancels(self):
        t = Task()
        t.submit().start().require_auth().start()
        self.assertEqual(t.state, TaskState.WORKING)
        t2 = Task()
        t2.submit().start().require_auth().cancel()
        self.assertEqual(t2.state, TaskState.CANCELED)

    def test_auth_required_to_completed_illegal(self):
        t = Task()
        t.submit().start().require_auth()
        with self.assertRaises(IllegalTransition):
            t.complete()


class TestTaskSpecSerialization(unittest.TestCase):
    def test_to_a2a_shape(self):
        t = Task(taskId="task-1", contextId="ctx-1", clock=monotonic_counter())
        prompt = A2AMessage(role=ROLE_AGENT, parts=(Part.from_text("working on it"),))
        t.submit().start(message=prompt)
        d = _roundtrip(t.to_a2a())
        # top-level id (NOT taskId) and kind discriminator
        self.assertEqual(d["id"], "task-1")
        self.assertNotIn("taskId", d)
        self.assertEqual(d["kind"], "task")
        self.assertEqual(d["contextId"], "ctx-1")
        # nested status holds CURRENT state
        self.assertEqual(d["status"]["state"], "working")
        self.assertIn("timestamp", d["status"])
        self.assertEqual(d["status"]["message"]["parts"][0]["text"], "working on it")
        # history is the Message[] conversation
        self.assertEqual(d["history"][0]["kind"], "message")
        self.assertIsInstance(d["artifacts"], list)

    def test_to_a2a_unsubmitted_state_unknown(self):
        t = Task(taskId="t0", contextId="c0")
        d = t.to_a2a()
        self.assertEqual(d["status"]["state"], "unknown")


class TestFilePart(unittest.TestCase):
    def test_from_file_bytes_roundtrip(self):
        p = Part.from_file(name="part.step", mime_type="model/step", bytes_b64="QUJD")
        self.assertEqual(p.kind, PART_FILE)
        d = _roundtrip(p.to_dict())
        self.assertEqual(d["kind"], "file")
        self.assertEqual(d["file"]["name"], "part.step")
        self.assertEqual(d["file"]["mimeType"], "model/step")
        self.assertEqual(d["file"]["bytes"], "QUJD")
        self.assertNotIn("uri", d["file"])
        self.assertEqual(Part.from_dict(d), p)

    def test_from_file_uri_roundtrip(self):
        p = Part.from_file(name="mesh.stl", uri="mem://mesh/1")
        d = _roundtrip(p.to_dict())
        self.assertEqual(d["file"]["uri"], "mem://mesh/1")
        self.assertNotIn("bytes", d["file"])
        self.assertEqual(Part.from_dict(d), p)


class TestArtifact(unittest.TestCase):
    def test_roundtrip(self):
        art = Artifact(
            artifactId="a-1",
            name="bracket",
            description="the finished bracket",
            parts=(Part.from_text("summary"), Part.from_file(uri="mem://b/1")),
            metadata={"volume": 12.0},
        )
        d = _roundtrip(art.to_dict())
        self.assertEqual(d["artifactId"], "a-1")
        self.assertEqual(d["name"], "bracket")
        self.assertEqual(d["description"], "the finished bracket")
        self.assertEqual(len(d["parts"]), 2)
        self.assertEqual(d["parts"][1]["kind"], "file")
        self.assertEqual(d["metadata"]["volume"], 12.0)
        self.assertEqual(Artifact.from_dict(d), art)

    def test_minimal_roundtrip(self):
        art = Artifact(artifactId="a-2", parts=(Part.from_text("x"),))
        d = _roundtrip(art.to_dict())
        self.assertNotIn("name", d)
        self.assertNotIn("metadata", d)
        self.assertEqual(Artifact.from_dict(d), art)


class TestAgentCardConformance(unittest.TestCase):
    def test_emits_spec_fields(self):
        card = AgentCard(
            name="Modeler",
            description="Emits CAD ops.",
            url="https://example/a2a",
            capabilities={"streaming": True, "stateTransitionHistory": True},
            skills=(AgentSkill(id="extrude", name="Extrude", tags=("cad",)).to_dict(),),
            defaultInputModes=("text/plain",),
            defaultOutputModes=("model/step",),
        )
        d = _roundtrip(card.to_dict())
        self.assertEqual(d["url"], "https://example/a2a")
        self.assertTrue(d["protocolVersion"])
        self.assertEqual(d["preferredTransport"], "JSONRPC")
        self.assertEqual(d["defaultInputModes"], ["text/plain"])
        self.assertEqual(d["capabilities"]["stateTransitionHistory"], True)
        self.assertEqual(AgentCard.from_dict(d), card)

    def test_endpoints_deprecated_alias_populates_url(self):
        card = AgentCard(name="Verifier", endpoints={"a2a": "https://v/a2a"})
        self.assertEqual(card.url, "https://v/a2a")
        d = _roundtrip(card.to_dict())
        self.assertEqual(d["url"], "https://v/a2a")
        self.assertEqual(AgentCard.from_dict(d), card)

    def test_agent_skill_shape(self):
        skill = AgentSkill(
            id="fillet",
            name="Fillet",
            description="round an edge",
            tags=("cad", "edge"),
            examples=("fillet all edges 2mm",),
            inputModes=("text/plain",),
            outputModes=("model/step",),
        )
        d = _roundtrip(skill.to_dict())
        self.assertEqual(d["id"], "fillet")
        self.assertEqual(d["name"], "Fillet")
        self.assertEqual(d["tags"], ["cad", "edge"])
        self.assertEqual(d["examples"], ["fillet all edges 2mm"])
        self.assertEqual(d["inputModes"], ["text/plain"])
        self.assertEqual(AgentSkill.from_dict(d), skill)


class TestA2AMessageKind(unittest.TestCase):
    def test_emits_kind_message(self):
        msg = A2AMessage(role=ROLE_AGENT, parts=(Part.from_text("hi"),))
        d = _roundtrip(msg.to_dict())
        self.assertEqual(d["kind"], "message")

    def test_reference_task_ids_and_extensions_roundtrip(self):
        msg = A2AMessage(
            role=ROLE_AGENT,
            parts=(Part.from_text("hi"),),
            referenceTaskIds=("t1", "t2"),
            extensions=("https://ext/1",),
        )
        d = _roundtrip(msg.to_dict())
        self.assertEqual(d["referenceTaskIds"], ["t1", "t2"])
        self.assertEqual(d["extensions"], ["https://ext/1"])
        self.assertEqual(A2AMessage.from_dict(d), msg)


class TestWireEventKinds(unittest.TestCase):
    def test_status_and_artifact_event_kinds_hyphenated(self):
        events = []
        t = Task(taskId="t1", contextId="c1", clock=monotonic_counter())
        t.subscribe(events.append)
        t.submit().start()
        t.add_artifact(Part.from_file(uri="mem://a/1"))
        wire = [to_wire_event(e) for e in events]
        self.assertEqual(wire[0]["kind"], "status-update")
        self.assertEqual(wire[-1]["kind"], "artifact-update")
        # internal event dicts are unchanged (mapping happens only at the seam)
        self.assertEqual(events[0]["kind"], EVENT_STATUS_UPDATE)
        self.assertEqual(events[-1]["kind"], EVENT_ARTIFACT_UPDATE)


if __name__ == "__main__":
    unittest.main()
