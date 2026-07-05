"""Pure wire translators between HarnessCAD internals and the A2A JSON-RPC wire.

Everything here is a pure function (no I/O, no state) so it is trivially unit-
testable and reusable by both the in-process ``handler`` dispatch and the HTTP
``app`` transport:

  - ``a2a_message_from_params`` — a ``MessageSendParams`` dict -> ``A2AMessage``.
  - ``event_to_wire`` — an internal ``Task`` event dict (underscore ``kind``,
    nested ``data``) -> the spec ``TaskStatusUpdateEvent`` /
    ``TaskArtifactUpdateEvent`` shape (hyphenated ``kind``; ``final`` on status;
    ``append``/``lastChunk`` on artifact).
  - JSON-RPC 2.0 helpers ``rpc_result`` / ``rpc_error`` and the SSE ``sse_frame``
    that wraps a JSON-RPC *result* as ``data: <json>\\n\\n``.

The internal event shapes mirror ``a2a.task.Task._emit`` exactly; if that emitter
changes, this is the single seam to update.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from a2a.messages import A2AMessage
from a2a.task import EVENT_ARTIFACT_UPDATE, EVENT_STATUS_UPDATE

# --- A2A JSON-RPC error codes (a2a-protocol.org) ---------------------------
ERROR_TASK_NOT_FOUND = -32001
ERROR_TASK_NOT_CANCELABLE = -32002
ERROR_PUSH_NOTIFICATION_NOT_SUPPORTED = -32003
ERROR_UNSUPPORTED_OPERATION = -32004
# --- standard JSON-RPC 2.0 codes -------------------------------------------
ERROR_PARSE = -32700
ERROR_INVALID_REQUEST = -32600
ERROR_METHOD_NOT_FOUND = -32601
ERROR_INVALID_PARAMS = -32602
ERROR_INTERNAL = -32603


def a2a_message_from_params(params: Dict[str, Any]) -> A2AMessage:
    """Extract the ``A2AMessage`` from a ``message/send``/``message/stream`` params.

    Accepts the spec ``MessageSendParams`` shape ``{"message": {...}, ...}`` and,
    leniently, a bare message object (one carrying a ``role``).
    """
    if not isinstance(params, dict):
        raise ValueError("params must be a JSON object")
    msg = params.get("message")
    if msg is None and "role" in params:
        msg = params
    if not isinstance(msg, dict):
        raise ValueError("missing required 'message' object in params")
    return A2AMessage.from_dict(msg)


# --- event translation -----------------------------------------------------
def status_update_wire(event: Dict[str, Any]) -> Dict[str, Any]:
    """Internal ``status_update`` event -> A2A ``TaskStatusUpdateEvent``."""
    data = event.get("data") or {}
    status: Dict[str, Any] = {"state": data.get("state")}
    if data.get("message") is not None:
        status["message"] = data["message"]
    if event.get("ts") is not None:
        status["timestamp"] = event["ts"]
    return {
        "kind": "status-update",
        "taskId": event.get("taskId"),
        "contextId": event.get("contextId"),
        "status": status,
        "final": bool(data.get("final")),
    }


def artifact_update_wire(event: Dict[str, Any]) -> Dict[str, Any]:
    """Internal ``artifact_update`` event -> A2A ``TaskArtifactUpdateEvent``."""
    data = event.get("data") or {}
    return {
        "kind": "artifact-update",
        "taskId": event.get("taskId"),
        "contextId": event.get("contextId"),
        "artifact": data.get("artifact"),
        "append": False,
        "lastChunk": True,
    }


def event_to_wire(event: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch an internal ``Task`` event dict to its A2A wire event shape."""
    kind = event.get("kind")
    if kind == EVENT_STATUS_UPDATE:
        return status_update_wire(event)
    if kind == EVENT_ARTIFACT_UPDATE:
        return artifact_update_wire(event)
    # Defensive: pass through unknown events unchanged.
    return dict(event)


# --- JSON-RPC 2.0 envelope helpers -----------------------------------------
def rpc_result(rpc_id: Any, result: Any) -> Dict[str, Any]:
    """A JSON-RPC 2.0 success response object."""
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


def rpc_error(
    rpc_id: Any, code: int, message: str, data: Optional[Any] = None
) -> Dict[str, Any]:
    """A JSON-RPC 2.0 error response object."""
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": rpc_id, "error": err}


def sse_frame(obj: Dict[str, Any]) -> str:
    """Wrap a JSON-RPC response object as one SSE frame: ``data: <json>\\n\\n``."""
    return "data: " + json.dumps(obj, separators=(",", ":")) + "\n\n"
