"""Tests for the A2A internal message vocabulary and task lifecycle.

Covers: AgentCard + A2AMessage round-trip JSON; the three Part variants; the
Task guarded state machine (legal path + illegal transition raises); contextId
grouping in the TaskStore; SSE-style subscriber callbacks firing on status and
artifact changes; and TaskStore retrieval by id and by context.
"""

import json
import unittest

from a2a.messages import (
    AgentCard,
    A2AMessage,
    Part,
    PART_TEXT,
    PART_DATA,
    PART_ARTIFACT,
    ROLE_AGENT,
    ROLE_USER,
    agent_message,
    user_message,
)
from a2a.task import (
    EVENT_ARTIFACT_UPDATE,
    EVENT_STATUS_UPDATE,
    IllegalTransition,
    Task,
    TaskState,
    TaskStore,
    monotonic_counter,
)


def _roundtrip(obj_dict):
    """JSON-encode then decode a dict, proving it is fully JSON-serialisable."""
    return json.loads(json.dumps(obj_dict))


class TestParts(unittest.TestCase):
    def test_text_variant(self):
        p = Part.from_text("hello")
        self.assertEqual(p.kind, PART_TEXT)
        self.assertEqual(p.text, "hello")
        self.assertEqual(Part.from_dict(_roundtrip(p.to_dict())), p)

    def test_data_variant(self):
        p = Part.from_data({"op": "extrude", "distance": 5.0})
        self.assertEqual(p.kind, PART_DATA)
        self.assertEqual(p.data["op"], "extrude")
        self.assertEqual(Part.from_dict(_roundtrip(p.to_dict())), p)

    def test_artifact_variant(self):
        p = Part.from_artifact({"name": "part.step", "mimeType": "model/step", "uri": "mem://1"})
        self.assertEqual(p.kind, PART_ARTIFACT)
        self.assertEqual(p.artifact["name"], "part.step")
        self.assertEqual(Part.from_dict(_roundtrip(p.to_dict())), p)

    def test_frozen(self):
        p = Part.from_text("x")
        with self.assertRaises(Exception):
            p.text = "y"  # type: ignore[misc]


class TestAgentCard(unittest.TestCase):
    def test_roundtrip_json(self):
        card = AgentCard(
            name="Modeler",
            description="Emits CAD ops from a plan.",
            capabilities={"streaming": True, "pushNotifications": True},
            skills=({"id": "extrude", "description": "extrude a sketch"},),
            endpoints={"a2a": "https://example/a2a"},
            version="0.1.0",
        )
        restored = AgentCard.from_dict(_roundtrip(card.to_dict()))
        self.assertEqual(restored, card)
        self.assertTrue(restored.capabilities["streaming"])
        self.assertEqual(restored.skills[0]["id"], "extrude")

    def test_defaults(self):
        card = AgentCard(name="Verifier")
        d = _roundtrip(card.to_dict())
        self.assertEqual(d["capabilities"], {})
        self.assertEqual(d["skills"], [])
        self.assertEqual(AgentCard.from_dict(d), card)


class TestA2AMessage(unittest.TestCase):
    def test_roundtrip_json_mixed_parts(self):
        msg = A2AMessage(
            role=ROLE_AGENT,
            parts=(
                Part.from_text("done"),
                Part.from_data({"volume": 12.0}),
                Part.from_artifact({"name": "m.step"}),
            ),
            contextId="ctx-1",
            taskId="task-1",
            messageId="m-1",
            metadata={"tokens": 42},
        )
        restored = A2AMessage.from_dict(_roundtrip(msg.to_dict()))
        self.assertEqual(restored, msg)
        self.assertEqual(restored.contextId, "ctx-1")
        self.assertEqual(restored.taskId, "task-1")
        self.assertEqual(restored.metadata["tokens"], 42)
        self.assertEqual(len(restored.parts), 3)

    def test_role_helpers_and_text(self):
        u = user_message(Part.from_text("make a bracket"), contextId="c")
        a = agent_message(Part.from_text("ok "), Part.from_text("bracket"))
        self.assertEqual(u.role, ROLE_USER)
        self.assertEqual(a.role, ROLE_AGENT)
        self.assertEqual(a.text(), "ok bracket")


class TestTaskLifecycle(unittest.TestCase):
    def test_legal_path_submitted_working_completed(self):
        t = Task(taskId="t1", contextId="c1", clock=monotonic_counter())
        self.assertIsNone(t.state)
        t.submit()
        self.assertEqual(t.state, TaskState.SUBMITTED)
        t.start()
        self.assertEqual(t.state, TaskState.WORKING)
        t.complete()
        self.assertEqual(t.state, TaskState.COMPLETED)
        self.assertTrue(t.is_terminal)
        self.assertEqual(
            [h.state for h in t.history],
            [TaskState.SUBMITTED, TaskState.WORKING, TaskState.COMPLETED],
        )
        # history timestamps are monotonic
        ts = [h.ts for h in t.history]
        self.assertEqual(ts, sorted(ts))

    def test_input_required_cycle(self):
        t = Task()
        t.submit().start().require_input().start().complete()
        self.assertEqual(t.state, TaskState.COMPLETED)

    def test_illegal_transition_raises(self):
        t = Task()
        t.submit()
        # submitted -> completed is not a legal edge
        with self.assertRaises(IllegalTransition):
            t.complete()
        # state is unchanged after the rejected transition
        self.assertEqual(t.state, TaskState.SUBMITTED)

    def test_illegal_from_terminal(self):
        t = Task()
        t.submit().start().complete()
        with self.assertRaises(IllegalTransition):
            t.start()

    def test_double_submit_illegal(self):
        t = Task()
        t.submit()
        with self.assertRaises(IllegalTransition):
            t.submit()


class TestSubscribers(unittest.TestCase):
    def test_status_callbacks_fire(self):
        events = []
        t = Task(taskId="t1", contextId="c1", clock=monotonic_counter())
        t.subscribe(events.append)
        t.submit().start().complete()
        kinds = [e["kind"] for e in events]
        self.assertEqual(kinds, [EVENT_STATUS_UPDATE] * 3)
        states = [e["data"]["state"] for e in events]
        self.assertEqual(states, ["submitted", "working", "completed"])
        # terminal event flagged final; others not
        self.assertFalse(events[0]["data"]["final"])
        self.assertTrue(events[-1]["data"]["final"])
        # events carry correlation ids
        self.assertTrue(all(e["taskId"] == "t1" and e["contextId"] == "c1" for e in events))

    def test_artifact_callback_fires(self):
        events = []
        t = Task()
        t.subscribe(events.append)
        t.submit().start()
        art = Part.from_artifact({"name": "out.step"})
        t.add_artifact(art)
        self.assertEqual(events[-1]["kind"], EVENT_ARTIFACT_UPDATE)
        # data.artifact is the serialised Part (kind="artifact", payload nested)
        self.assertEqual(events[-1]["data"]["artifact"]["artifact"]["name"], "out.step")
        self.assertEqual(t.artifacts, [art])

    def test_unsubscribe(self):
        events = []
        t = Task()
        off = t.subscribe(events.append)
        t.submit()
        off()
        t.start()
        self.assertEqual(len(events), 1)

    def test_message_carried_in_event(self):
        events = []
        t = Task()
        t.subscribe(events.append)
        prompt = A2AMessage(role=ROLE_AGENT, parts=(Part.from_text("need a fillet radius"),))
        t.submit().start().require_input(message=prompt)
        self.assertEqual(events[-1]["data"]["state"], "input_required")
        self.assertEqual(events[-1]["data"]["message"]["parts"][0]["text"], "need a fillet radius")


class TestTaskStore(unittest.TestCase):
    def test_retrieval_by_id(self):
        store = TaskStore(clock=monotonic_counter())
        t = store.create(contextId="c1")
        self.assertIs(store.get(t.taskId), t)
        self.assertIn(t.taskId, store)
        self.assertIsNone(store.get("missing"))

    def test_context_groups_tasks(self):
        store = TaskStore()
        a = store.create(contextId="ctx-A")
        b = store.create(contextId="ctx-A")
        c = store.create(contextId="ctx-B")
        grouped = store.by_context("ctx-A")
        self.assertEqual([t.taskId for t in grouped], [a.taskId, b.taskId])
        self.assertEqual([t.taskId for t in store.by_context("ctx-B")], [c.taskId])
        self.assertEqual(len(store), 3)
        self.assertCountEqual(store.contexts(), ["ctx-A", "ctx-B"])

    def test_put_external_task(self):
        store = TaskStore()
        t = Task(taskId="ext", contextId="c9")
        store.put(t)
        self.assertIs(store.get("ext"), t)
        self.assertEqual(store.by_context("c9"), [t])

    def test_put_is_idempotent_in_context_index(self):
        store = TaskStore()
        t = store.create(contextId="c1")
        store.put(t)  # re-register same task
        self.assertEqual(store.by_context("c1"), [t])


if __name__ == "__main__":
    unittest.main()
