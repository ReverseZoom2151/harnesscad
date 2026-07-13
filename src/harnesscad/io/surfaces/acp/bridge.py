"""Bridge — translate HarnessCAD UIEvents / trace into ACP ``session/update``.

The ACP wire vocabulary is different from HarnessCAD's internal one. This module
is the sole translation seam, mapping the eight ``surfaces.ui.events.UIEvent``
kinds and the harness op stream onto ACP ``session/update`` notifications, and
running the two outbound round-trips (``session/request_permission``,
``fs/write_text_file``) the agent initiates mid-turn.

UIEvent -> session/update mapping (ACP protocol v1):

  status            -> agent_message_chunk   (text)
  thinking          -> agent_thought_chunk   (text)
  token             -> agent_message_chunk   (text)
  tool_call         -> tool_call             {toolCallId, title, kind, status:in_progress}
  tool_result       -> tool_call_update      {toolCallId, status: completed|failed}
  approval_required -> (an outbound session/request_permission round-trip)
  action_rejected   -> tool_call_update      {toolCallId, status: failed}
  plan (op list)    -> plan                  {entries:[{content, priority, status}]}

``kind`` is derived from the three-tier approval model (``surfaces.ui.approval``):
read/measure (AUTO) -> "read"; modify ops (NOTIFY) -> "edit"; export/delete
(REQUIRE) -> "execute".

STEP delivery: on a successful run the STEP text is delivered either by an
outbound ``fs/write_text_file`` (when the client advertised ``fs.writeTextFile``)
or, failing that, inlined as an ACP resource ``ContentBlock`` on an
``agent_message_chunk``.

Determinism: tool-call ids are content-derived (a hash of the op's canonical
JSON plus its ordinal), never a uuid or wall clock.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from harnesscad.core.cisp.ops import canonical_json
from harnesscad.io.surfaces.ui.approval import (
    ApprovalTier, DryRunPreview, op_name, risk_for, tier_for,
)
from harnesscad.io.surfaces.ui.events import EventType, UIEvent


# --- ACP enum-ish constants (kept as literals per the ACP schema) ----------
KIND_READ = "read"
KIND_EDIT = "edit"
KIND_EXECUTE = "execute"

STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

PLAN_PENDING = "pending"
PLAN_IN_PROGRESS = "in_progress"
PLAN_COMPLETED = "completed"

# Permission option ids the agent offers for a Tier-3 op.
PERMISSION_OPTIONS = (
    {"optionId": "allow_once", "name": "Allow once", "kind": "allow_once"},
    {"optionId": "allow_always", "name": "Allow always", "kind": "allow_always"},
    {"optionId": "reject_once", "name": "Reject once", "kind": "reject_once"},
    {"optionId": "reject_always", "name": "Reject always", "kind": "reject_always"},
)
_ALLOW_OPTIONS = frozenset({"allow_once", "allow_always"})

# tier -> ACP tool-call kind, and tier -> ACP plan-entry priority.
_TIER_KIND = {
    ApprovalTier.AUTO: KIND_READ,
    ApprovalTier.NOTIFY: KIND_EDIT,
    ApprovalTier.REQUIRE: KIND_EXECUTE,
}
_TIER_PRIORITY = {
    ApprovalTier.AUTO: "low",
    ApprovalTier.NOTIFY: "medium",
    ApprovalTier.REQUIRE: "high",
}


def kind_for(op: Any) -> str:
    """ACP tool-call ``kind`` for an op, via the three-tier approval model."""
    return _TIER_KIND[tier_for(op)]


def tool_call_id(op: Any, index: int) -> str:
    """Deterministic, content-derived tool-call id (no uuid, no clock)."""
    try:
        blob = canonical_json(op)
    except Exception:
        blob = op_name(op)
    h = hashlib.sha256(f"{index}|{blob}".encode("utf-8")).hexdigest()
    return f"tc-{h[:12]}"


class ACPBridge:
    """Owns the per-session translation of events into ``session/update`` and the
    two outbound round-trips. One bridge is created per ACP session.
    """

    def __init__(self, connection, session_id: str, cwd: str,
                 client_can_write_fs: bool = False) -> None:
        self.connection = connection
        self.session_id = session_id
        self.cwd = cwd
        self.client_can_write_fs = client_can_write_fs
        # Cancellation flag the loop consults (set by session/cancel).
        self.cancelled = False
        # The tool-call id currently being processed — consulted by
        # request_permission so the outbound permission request references the
        # same call the client already saw as a ``tool_call`` update.
        self._active_call_id: str = ""
        # Plan state (index-aligned with the current op batch).
        self._plan_entries: List[Dict[str, Any]] = []
        self._plan_call_ids: List[str] = []

    # --- low-level send ---------------------------------------------------
    def _update(self, update: Dict[str, Any]) -> None:
        """Send one ``session/update`` notification carrying ``update``."""
        self.connection.notify("session/update", {
            "sessionId": self.session_id,
            "update": update,
        })

    # --- UIEvent translation ---------------------------------------------
    def translate(self, event: UIEvent) -> Optional[Dict[str, Any]]:
        """Pure UIEvent -> session/update ``update`` dict (or None if not a
        one-way notification, e.g. approval_required / done)."""
        t = event.type
        d = event.data
        if t is EventType.STATUS:
            return {"sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": d.get("message", "")}}
        if t is EventType.THINKING:
            return {"sessionUpdate": "agent_thought_chunk",
                    "content": {"type": "text", "text": d.get("text", "")}}
        if t is EventType.TOKEN:
            return {"sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": d.get("text", "")}}
        if t is EventType.TOOL_CALL:
            return {"sessionUpdate": "tool_call",
                    "toolCallId": d.get("call_id", ""),
                    "title": d.get("name", ""),
                    "kind": d.get("kind", KIND_EDIT),
                    "status": STATUS_IN_PROGRESS}
        if t is EventType.TOOL_RESULT:
            ok = d.get("ok", True)
            return {"sessionUpdate": "tool_call_update",
                    "toolCallId": d.get("call_id", ""),
                    "status": STATUS_COMPLETED if ok else STATUS_FAILED}
        if t is EventType.ACTION_REJECTED:
            return {"sessionUpdate": "tool_call_update",
                    "toolCallId": d.get("call_id", ""),
                    "status": STATUS_FAILED}
        # approval_required and done are not one-way session/updates.
        return None

    def emit(self, event: UIEvent) -> None:
        """Translate and send a UIEvent, if it maps to a session/update."""
        update = self.translate(event)
        if update is not None:
            self._update(update)

    # --- convenience emitters used by the executor bridge -----------------
    def emit_tool_call(self, op: Any, call_id: str) -> None:
        self.emit(UIEvent.tool_call(
            name=op_name(op), args={}, call_id=call_id, kind=kind_for(op)))

    def emit_tool_result(self, op: Any, call_id: str, ok: bool) -> None:
        self.emit(UIEvent.tool_result(
            name=op_name(op), result={}, call_id=call_id, ok=ok))

    def emit_action_rejected(self, op: Any, call_id: str, reason: str,
                             diagnostics: Optional[List] = None) -> None:
        self.emit(UIEvent.action_rejected(
            name=op_name(op), reason=reason,
            diagnostics=diagnostics or [], call_id=call_id))

    # --- plan -------------------------------------------------------------
    def start_plan(self, ops: List[Any]) -> None:
        """Build and send the full plan entry list (all pending) for ``ops``.

        Per the ACP spec the COMPLETE entry list is sent on every change; this
        seeds it. ``plan_call_id`` / ``plan_mark`` then keep it in sync.
        """
        self._plan_entries = []
        self._plan_call_ids = []
        for i, op in enumerate(ops):
            preview = DryRunPreview.for_op(op)
            self._plan_entries.append({
                "content": preview.summary,
                "priority": _TIER_PRIORITY[tier_for(op)],
                "status": PLAN_PENDING,
            })
            self._plan_call_ids.append(tool_call_id(op, i))
        self._send_plan()

    def plan_call_id(self, index: int) -> str:
        return self._plan_call_ids[index]

    def plan_mark(self, index: int, status: str) -> None:
        if 0 <= index < len(self._plan_entries):
            self._plan_entries[index]["status"] = status
            self._send_plan()

    def _send_plan(self) -> None:
        self._update({"sessionUpdate": "plan",
                      "entries": [dict(e) for e in self._plan_entries]})

    # --- outbound: permission round-trip ----------------------------------
    def request_permission(self, op: Any) -> bool:
        """BLOCKING ``session/request_permission`` round-trip for a Tier-3 op.

        title = op name; description = the dry-run preview summary + risk;
        options = allow_once / allow_always / reject_once / reject_always.
        Returns True iff the client selected an ``allow_*`` option.
        """
        name = op_name(op)
        preview = DryRunPreview.for_op(op)
        risk = risk_for(op)
        params = {
            "sessionId": self.session_id,
            "toolCall": {
                "toolCallId": self._active_call_id,
                "title": name,
                "kind": kind_for(op),
                "status": STATUS_IN_PROGRESS,
            },
            "title": name,
            "description": f"{preview.summary} [risk: {risk}]",
            "options": [dict(o) for o in PERMISSION_OPTIONS],
        }
        result = self.connection.request("session/request_permission", params)
        return self._permission_granted(result)

    @staticmethod
    def _permission_granted(result: Any) -> bool:
        """Decode a ``RequestPermissionResponse`` outcome into allow/deny."""
        if not isinstance(result, dict):
            return False
        outcome = result.get("outcome")
        # ACP nests the verdict under ``outcome``; tolerate a flat shape too.
        if isinstance(outcome, dict):
            if outcome.get("outcome") == "cancelled":
                return False
            option = outcome.get("optionId")
        else:
            option = result.get("optionId")
        return option in _ALLOW_OPTIONS

    # --- STEP delivery ----------------------------------------------------
    def deliver_step(self, content: str, filename: str = "out.step") -> Dict[str, Any]:
        """Deliver the exported STEP text to the client.

        Uses an outbound ``fs/write_text_file`` when the client advertised
        ``fs.writeTextFile``; otherwise inlines the STEP as an ACP resource
        ContentBlock on an ``agent_message_chunk``. Returns a small record of
        what was done (for logging / tests).
        """
        path = self.cwd.rstrip("/\\") + "/" + filename
        if self.client_can_write_fs:
            self.connection.request("fs/write_text_file", {
                "sessionId": self.session_id,
                "path": path,
                "content": content,
            })
            return {"delivery": "fs", "path": path}
        self._update({
            "sessionUpdate": "agent_message_chunk",
            "content": {
                "type": "resource",
                "resource": {
                    "uri": "file://" + path,
                    "mimeType": "application/step",
                    "text": content,
                },
            },
        })
        return {"delivery": "inline", "path": path}
