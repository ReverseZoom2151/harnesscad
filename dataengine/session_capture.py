"""Consent-aware capture of modeling events and CISP operation decisions.

The recorder aligns timestamped UI/tool events with proposed, accepted, and
rejected operation batches.  It emits deterministic, training-ready records;
screen/video capture and model inference are intentionally outside its scope.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence


Redactor = Callable[[Any], Any]


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda p: str(p[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass(frozen=True)
class Consent:
    granted: bool
    scope: str = "training"
    subject: Optional[str] = None
    policy_version: str = "1"

    def to_dict(self) -> dict:
        return {
            "granted": self.granted,
            "scope": self.scope,
            "subject": self.subject,
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True)
class CaptureEvent:
    timestamp: float
    sequence: int
    channel: str
    kind: str
    payload: Any
    proposal_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "sequence": self.sequence,
            "channel": self.channel,
            "kind": self.kind,
            "payload": _plain(self.payload),
            "proposal_id": self.proposal_id,
        }


@dataclass
class OpDecision:
    proposal_id: str
    timestamp: float
    proposed_ops: list[dict]
    status: str = "proposed"
    decision_timestamp: Optional[float] = None
    reason: Optional[str] = None
    accepted_ops: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "timestamp": self.timestamp,
            "proposed_ops": _plain(self.proposed_ops),
            "status": self.status,
            "decision_timestamp": self.decision_timestamp,
            "reason": self.reason,
            "accepted_ops": _plain(self.accepted_ops),
        }


class ModelingSessionCapture:
    """In-memory capture with explicit timestamps and deterministic export."""

    def __init__(
        self,
        session_id: str,
        consent: Consent,
        *,
        provenance: Optional[Mapping[str, Any]] = None,
        redactor: Optional[Redactor] = None,
    ) -> None:
        if not session_id:
            raise ValueError("session_id is required")
        self.session_id = session_id
        self.consent = consent
        self.provenance = _plain(provenance or {})
        self.redactor = redactor or (lambda value: value)
        self.events: list[CaptureEvent] = []
        self.decisions: dict[str, OpDecision] = {}
        self._sequence = 0
        self._proposal_sequence = 0

    def record_event(
        self,
        timestamp: float,
        channel: str,
        kind: str,
        payload: Any,
        *,
        proposal_id: Optional[str] = None,
    ) -> CaptureEvent:
        if channel not in {"ui", "tool", "system"}:
            raise ValueError(f"unsupported event channel: {channel}")
        self._sequence += 1
        event = CaptureEvent(
            float(timestamp), self._sequence, channel, str(kind), _plain(payload), proposal_id
        )
        self.events.append(event)
        return event

    def propose(
        self, timestamp: float, ops: Sequence[Any], *, context: Optional[Mapping[str, Any]] = None
    ) -> OpDecision:
        serialized = [_plain(op) for op in ops]
        if not serialized:
            raise ValueError("proposal must contain at least one op")
        self._proposal_sequence += 1
        seed = json.dumps(serialized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        token = hashlib.sha256(
            f"{self.session_id}|{self._proposal_sequence}|{seed}".encode("utf-8")
        ).hexdigest()[:16]
        proposal_id = f"proposal-{token}"
        decision = OpDecision(proposal_id, float(timestamp), serialized)
        self.decisions[proposal_id] = decision
        self.record_event(
            timestamp, "tool", "ops_proposed",
            {"ops": serialized, "context": _plain(context or {})},
            proposal_id=proposal_id,
        )
        return decision

    def decide(
        self,
        proposal_id: str,
        timestamp: float,
        accepted: bool,
        *,
        reason: Optional[str] = None,
        accepted_ops: Optional[Sequence[Any]] = None,
    ) -> OpDecision:
        decision = self.decisions[proposal_id]
        if decision.status != "proposed":
            raise ValueError(f"proposal {proposal_id} is already {decision.status}")
        decision.status = "accepted" if accepted else "rejected"
        decision.decision_timestamp = float(timestamp)
        decision.reason = reason
        decision.accepted_ops = (
            [_plain(op) for op in accepted_ops]
            if accepted_ops is not None
            else (list(decision.proposed_ops) if accepted else [])
        )
        self.record_event(
            timestamp, "ui", f"ops_{decision.status}",
            {"reason": reason, "ops": decision.accepted_ops},
            proposal_id=proposal_id,
        )
        return decision

    def export(self) -> dict:
        """Return a deterministic record, refusing use without training consent."""
        if not self.consent.granted or self.consent.scope not in {"training", "training+evaluation"}:
            raise PermissionError("session is not consented for training export")
        events = sorted(self.events, key=lambda event: (event.timestamp, event.sequence))
        decisions = sorted(self.decisions.values(), key=lambda item: (item.timestamp, item.proposal_id))
        body = {
            "schema_version": 1,
            "session_id": self.session_id,
            "consent": self.consent.to_dict(),
            "provenance": self.provenance,
            "events": [event.to_dict() for event in events],
            "op_decisions": [decision.to_dict() for decision in decisions],
        }
        redacted = self.redactor(_plain(body))
        if not isinstance(redacted, Mapping):
            raise TypeError("redactor must return a mapping for the exported record")
        return _plain(redacted)

    def to_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(
            self.export(), sort_keys=True, ensure_ascii=False,
            separators=None if indent else (",", ":"), indent=indent,
        )
