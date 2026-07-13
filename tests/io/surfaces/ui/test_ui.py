"""Tests for the ui package — the typed SSE event contract and 3-tier approval.

Covers (per docs/blueprint.md sec.14):
  - every EventType round-trips through to_sse()/parse_sse
  - EventStream yields well-formed SSE frames (and a terminal done)
  - tier_for maps representative ops (export=REQUIRE, extrude=NOTIFY,
    measure/render=AUTO) and MCP annotations auto-assign tiers
  - ApprovalGate auto-proceeds Tier-1, notifies Tier-2, emits approval_required
    for Tier-3, with a risk indicator + dry-run preview
  - DryRunPreview summarises an op without mutating a supplied state snapshot
  - batching collapses related Tier-3 approvals into one event
"""

import json
import unittest

from harnesscad.core.cisp.ops import (
    NewSketch, AddCircle, AddRectangle, Constrain, Extrude, Fillet, Boolean,
)
from harnesscad.io.surfaces.ui.events import (
    EVENT_TYPES, EventStream, EventType, UIEvent, parse_sse, parse_stream,
)
from harnesscad.io.surfaces.ui.approval import (
    ApprovalGate, ApprovalTier, DryRunPreview, RiskLevel,
    tier_for, tier_from_annotations, risk_for,
)


# --- events ----------------------------------------------------------------
class TestEventWireFormat(unittest.TestCase):
    def test_all_event_types_covered(self):
        self.assertEqual(
            set(EVENT_TYPES),
            {"status", "thinking", "token", "tool_call", "tool_result",
             "approval_required", "action_rejected", "done"},
        )

    def test_to_sse_shape(self):
        ev = UIEvent.status("regenerating")
        wire = ev.to_sse()
        self.assertTrue(wire.startswith("event: status\n"))
        self.assertIn("data: ", wire)
        self.assertTrue(wire.endswith("\n\n"))

    def test_every_event_type_round_trips(self):
        samples = [
            UIEvent.status("working"),
            UIEvent.thinking("decompose the brief"),
            UIEvent.token("ex"),
            UIEvent.tool_call("extrude", {"sketch": "sk1", "distance": 5.0}, "c1"),
            UIEvent.tool_result("extrude", {"digest": "abc"}, "c1", ok=True),
            UIEvent.approval_required("export", "high", {"summary": "write file"}),
            UIEvent.action_rejected("boolean", "verify-failed", [{"code": "nm"}]),
            UIEvent.done(ok=True),
        ]
        # one sample per event type
        self.assertEqual({s.type for s in samples}, set(EventType))
        for ev in samples:
            restored = parse_sse(ev.to_sse())
            self.assertEqual(restored.type, ev.type)
            self.assertEqual(restored.data, ev.data)

    def test_parse_accepts_type_string(self):
        ev = UIEvent("done", {"ok": True})
        self.assertIs(ev.type, EventType.DONE)

    def test_parse_rejects_missing_event_field(self):
        with self.assertRaises(ValueError):
            parse_sse("data: {}\n\n")

    def test_data_is_single_line_json(self):
        ev = UIEvent.tool_call("boolean", {"kind": "cut", "target": "a"})
        wire = ev.to_sse()
        data_line = [l for l in wire.splitlines() if l.startswith("data:")]
        self.assertEqual(len(data_line), 1)
        payload = json.loads(data_line[0][len("data:"):].strip())
        self.assertEqual(payload["name"], "boolean")


class TestEventStream(unittest.TestCase):
    def test_stream_yields_wellformed_frames(self):
        stream = EventStream([
            UIEvent.status("start"),
            UIEvent.tool_call("extrude", {"sketch": "sk1"}),
            UIEvent.done(),
        ])
        frames = list(stream)
        for f in frames:
            self.assertTrue(f.startswith("event: "))
            self.assertTrue(f.endswith("\n\n"))
        # round-trips back to the same events
        parsed = parse_stream("".join(frames))
        self.assertEqual([p.type for p in parsed],
                         [EventType.STATUS, EventType.TOOL_CALL, EventType.DONE])

    def test_stream_appends_terminal_done(self):
        stream = EventStream([UIEvent.status("only")])
        parsed = parse_stream(stream.to_sse())
        self.assertIs(parsed[-1].type, EventType.DONE)

    def test_stream_does_not_double_done(self):
        stream = EventStream([UIEvent.status("x"), UIEvent.done()])
        parsed = parse_stream(stream.to_sse())
        self.assertEqual(sum(1 for p in parsed if p.type is EventType.DONE), 1)


# --- tier classification ---------------------------------------------------
class TestTierFor(unittest.TestCase):
    def test_export_is_require(self):
        self.assertIs(tier_for("export"), ApprovalTier.REQUIRE)
        self.assertIs(tier_for("export_step"), ApprovalTier.REQUIRE)
        self.assertIs(tier_for("delete"), ApprovalTier.REQUIRE)

    def test_modify_ops_are_notify(self):
        for op in (Extrude(sketch="sk1", distance=5.0),
                   Fillet(edges=(1,), radius=1.0),
                   Boolean(kind="cut", target="a", tool="b"),
                   NewSketch(), AddCircle(sketch="sk1"), Constrain(kind="distance")):
            self.assertIs(tier_for(op), ApprovalTier.NOTIFY, op)

    def test_read_measure_render_are_auto(self):
        for name in ("measure", "render", "query", "read_bbox", "measure_mass"):
            self.assertIs(tier_for(name), ApprovalTier.AUTO, name)

    def test_annotations_auto_assign(self):
        self.assertIs(tier_from_annotations({"destructive": True}),
                      ApprovalTier.REQUIRE)
        self.assertIs(tier_from_annotations({"read_only": True}),
                      ApprovalTier.AUTO)
        self.assertIs(tier_from_annotations({}), ApprovalTier.NOTIFY)

    def test_risk_tracks_tier(self):
        self.assertIs(risk_for("export"), RiskLevel.HIGH)
        self.assertIs(risk_for(Extrude()), RiskLevel.MEDIUM)
        self.assertIs(risk_for("measure"), RiskLevel.LOW)


# --- dry-run preview -------------------------------------------------------
class TestDryRunPreview(unittest.TestCase):
    def test_summarises_without_mutating_snapshot(self):
        snapshot = {"solids_intent": 2, "features": ["sketch"]}
        original = json.loads(json.dumps(snapshot))  # deep copy for comparison
        preview = DryRunPreview.for_op(Extrude(sketch="sk1", distance=5.0), snapshot)
        # snapshot untouched
        self.assertEqual(snapshot, original)
        # before mirrors input, after projects intent
        self.assertEqual(preview.before.get("solids_intent"), 2)
        self.assertEqual(preview.after.get("solids_intent"), 3)
        self.assertTrue(preview.mutates)
        self.assertIn("extrude", preview.summary)
        self.assertTrue(preview.changes)

    def test_readonly_op_marked_non_mutating(self):
        preview = DryRunPreview.for_op("measure")
        self.assertFalse(preview.mutates)

    def test_preview_serialises(self):
        d = DryRunPreview.for_op(Fillet(edges=(1, 2), radius=2.0)).to_dict()
        self.assertEqual(d["op"], "fillet")
        self.assertIn("summary", d)
        self.assertIn("before", d)
        self.assertIn("after", d)


# --- approval gate ---------------------------------------------------------
class TestApprovalGate(unittest.TestCase):
    def setUp(self):
        self.emitted = []
        self.gate = ApprovalGate(emit=self.emitted.append)

    def test_tier1_auto_proceeds_silently(self):
        d = self.gate.evaluate("measure")
        self.assertTrue(d.auto_proceed)
        self.assertFalse(d.requires_approval)
        self.assertIsNone(d.event)
        self.assertEqual(self.emitted, [])

    def test_tier2_proceeds_with_notification(self):
        d = self.gate.evaluate(Extrude(sketch="sk1", distance=5.0))
        self.assertTrue(d.auto_proceed)
        self.assertFalse(d.requires_approval)
        # a status notification was emitted, but it is not a blocking approval
        self.assertEqual(len(self.emitted), 1)
        self.assertIs(self.emitted[0].type, EventType.STATUS)

    def test_tier3_requires_approval_and_emits_event(self):
        d = self.gate.evaluate("export")
        self.assertFalse(d.auto_proceed)
        self.assertTrue(d.requires_approval)
        self.assertEqual(len(self.emitted), 1)
        ev = self.emitted[0]
        self.assertIs(ev.type, EventType.APPROVAL_REQUIRED)
        # carries risk indicator + dry-run preview
        self.assertEqual(ev.data["risk"], "high")
        self.assertIn("preview", ev.data)
        self.assertIn("summary", ev.data["preview"])

    def test_may_proceed_helper(self):
        self.assertTrue(self.gate.may_proceed("render"))
        self.assertFalse(self.gate.may_proceed("delete"))

    def test_state_provider_feeds_preview(self):
        gate = ApprovalGate(state_provider=lambda: {"solids_intent": 5})
        d = gate.evaluate(Extrude(sketch="sk1", distance=2.0))
        self.assertEqual(d.preview.before.get("solids_intent"), 5)
        self.assertEqual(d.preview.after.get("solids_intent"), 6)


class TestBatching(unittest.TestCase):
    def test_batch_groups_tier3_into_one_event(self):
        emitted = []
        gate = ApprovalGate(emit=emitted.append)
        decisions = gate.batch_evaluate([
            "export_step", "export_stl", "delete",
        ])
        self.assertEqual(len(decisions), 3)
        self.assertTrue(all(d.requires_approval for d in decisions))
        # exactly one approval_required event for the whole group
        approvals = [e for e in emitted if e.type is EventType.APPROVAL_REQUIRED]
        self.assertEqual(len(approvals), 1)
        batch = approvals[0].data["batch"]
        self.assertEqual(len(batch), 3)
        # all decisions share the one batch event
        self.assertTrue(all(d.event is approvals[0] for d in decisions))

    def test_batch_mixes_tiers_correctly(self):
        emitted = []
        gate = ApprovalGate(emit=emitted.append)
        decisions = gate.batch_evaluate([
            "measure",                              # AUTO
            Extrude(sketch="sk1", distance=5.0),    # NOTIFY
            "export",                               # REQUIRE
        ])
        tiers = [d.tier for d in decisions]
        self.assertEqual(
            tiers, [ApprovalTier.AUTO, ApprovalTier.NOTIFY, ApprovalTier.REQUIRE])
        approvals = [e for e in emitted if e.type is EventType.APPROVAL_REQUIRED]
        self.assertEqual(len(approvals), 1)
        self.assertEqual(len(approvals[0].data["batch"]), 1)


if __name__ == "__main__":
    unittest.main()
