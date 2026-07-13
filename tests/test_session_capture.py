import json
import unittest

from harnesscad.core.cisp.ops import Extrude, NewSketch
from harnesscad.data.dataengine.session_capture import Consent, ModelingSessionCapture


class SessionCaptureTests(unittest.TestCase):
    def make_capture(self, redactor=None):
        return ModelingSessionCapture(
            "session-42",
            Consent(True, subject="participant-7"),
            provenance={"app": "HarnessCAD", "source": "desktop", "version": "test"},
            redactor=redactor,
        )

    def test_aligns_ui_and_tool_events_to_decisions(self):
        capture = self.make_capture()
        capture.record_event(1, "ui", "prompt", {"text": "make a shaft"})
        proposal = capture.propose(2, [NewSketch(), Extrude("sk1", 10)])
        capture.record_event(2.5, "tool", "preview_rendered", {"view": "iso"},
                             proposal_id=proposal.proposal_id)
        capture.decide(proposal.proposal_id, 3, True)

        record = capture.export()
        decision = record["op_decisions"][0]
        self.assertEqual("accepted", decision["status"])
        self.assertEqual(decision["proposed_ops"], decision["accepted_ops"])
        linked = [e for e in record["events"] if e["proposal_id"] == proposal.proposal_id]
        self.assertEqual(
            ["ops_proposed", "preview_rendered", "ops_accepted"],
            [event["kind"] for event in linked],
        )

    def test_rejection_preserves_reason_and_no_accepted_ops(self):
        capture = self.make_capture()
        proposal = capture.propose(4, [NewSketch()])
        capture.decide(proposal.proposal_id, 5, False, reason="wrong plane")
        decision = capture.export()["op_decisions"][0]
        self.assertEqual("rejected", decision["status"])
        self.assertEqual("wrong plane", decision["reason"])
        self.assertEqual([], decision["accepted_ops"])

    def test_export_is_deterministic_and_orders_explicit_timestamps(self):
        def build():
            capture = self.make_capture()
            capture.record_event(5, "ui", "later", {"b": 2, "a": 1})
            capture.record_event(1, "system", "earlier", {})
            proposal = capture.propose(2, [NewSketch()])
            capture.decide(proposal.proposal_id, 3, True)
            return capture.to_json()

        self.assertEqual(build(), build())
        record = json.loads(build())
        self.assertEqual(
            ["earlier", "ops_proposed", "ops_accepted", "later"],
            [event["kind"] for event in record["events"]],
        )

    def test_consent_is_enforced(self):
        capture = ModelingSessionCapture("private", Consent(False))
        capture.record_event(1, "ui", "prompt", {"text": "secret"})
        with self.assertRaises(PermissionError):
            capture.export()
        evaluation_only = ModelingSessionCapture("eval", Consent(True, scope="evaluation"))
        with self.assertRaises(PermissionError):
            evaluation_only.export()

    def test_redaction_hook_sees_and_redacts_whole_record(self):
        def redact(record):
            record["provenance"].pop("user_email", None)
            for event in record["events"]:
                if "text" in event["payload"]:
                    event["payload"]["text"] = "[REDACTED]"
            return record

        capture = ModelingSessionCapture(
            "redacted", Consent(True), provenance={"user_email": "a@example.com"},
            redactor=redact,
        )
        capture.record_event(1, "ui", "prompt", {"text": "client secret"})
        record = capture.export()
        self.assertNotIn("user_email", record["provenance"])
        self.assertEqual("[REDACTED]", record["events"][0]["payload"]["text"])

    def test_accepted_ops_can_capture_human_adjustment(self):
        capture = self.make_capture()
        proposal = capture.propose(1, [Extrude("sk1", 10)])
        capture.decide(
            proposal.proposal_id, 2, True,
            accepted_ops=[Extrude("sk1", 8)],
            reason="reduced length",
        )
        decision = capture.export()["op_decisions"][0]
        self.assertEqual(8, decision["accepted_ops"][0]["distance"])
        self.assertEqual(10, decision["proposed_ops"][0]["distance"])


if __name__ == "__main__":
    unittest.main()
