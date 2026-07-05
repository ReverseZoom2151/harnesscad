"""Bidirectional JSON-RPC 2.0 framing over stdio for the ACP agent.

This is the one genuinely new transport piece versus ``surfaces/server.py``'s
one-directional ``serve_stdio``: an ACP agent must not only *answer* requests
from the client, it must *initiate* requests mid-turn (``session/request_permission``,
``fs/write_text_file``) and BLOCK awaiting the client's reply before it can
continue applying ops.

Wire format (ACP): JSON-RPC 2.0, newline-delimited — exactly one JSON message
per line, no embedded newlines. ``stderr`` is left free for logs.

Design (stdlib-only, threading + queue):

  * ``Connection`` is transport-agnostic: it is handed a single ``send``
    callable (``dict -> None``) that writes one JSON message to the peer, and it
    is fed inbound messages via ``deliver``. This lets a real stdio reader thread
    OR an in-process mock drive exactly the same object.
  * Outbound *requests* the agent initiates get a monotonically increasing id and
    a pending-id future (a ``queue.Queue`` of size 1). ``request`` writes the
    frame then blocks on that queue until ``deliver`` routes the matching response
    back — the blocking round-trip the permission gate needs.
  * Outbound *notifications* (``notify``) are fire-and-forget.
  * Inbound responses (a message with an ``id`` but no ``method``) are matched to
    their pending future by ``deliver``; inbound requests/notifications (a message
    with a ``method``) are returned from ``deliver`` for the caller to dispatch.

Determinism: request ids are a simple 1,2,3,... counter (no wall clock, no uuid).
"""

from __future__ import annotations

import itertools
import json
import queue
import threading
from typing import Any, Callable, Dict, Optional


JSONRPC_VERSION = "2.0"

# JSON-RPC standard error codes used by the agent side.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INTERNAL_ERROR = -32603


class RPCError(Exception):
    """Raised inside ``request`` when the peer answers with an ``error`` object."""

    def __init__(self, error: Dict[str, Any]) -> None:
        self.code = error.get("code")
        self.message = error.get("message", "")
        self.data = error.get("data")
        super().__init__(f"[{self.code}] {self.message}")


def is_response(msg: Dict[str, Any]) -> bool:
    """True iff ``msg`` is a response to a request we sent (has id, no method)."""
    return isinstance(msg, dict) and "method" not in msg and "id" in msg


def is_request(msg: Dict[str, Any]) -> bool:
    """True iff ``msg`` is an inbound request (has method AND id)."""
    return isinstance(msg, dict) and "method" in msg and msg.get("id") is not None


def is_notification(msg: Dict[str, Any]) -> bool:
    """True iff ``msg`` is an inbound notification (method, no id)."""
    return isinstance(msg, dict) and "method" in msg and msg.get("id") is None


class Connection:
    """A bidirectional JSON-RPC 2.0 endpoint over one ``send`` sink.

    Thread-safe: ``request`` may be called from a worker thread while a reader
    thread calls ``deliver`` with the matching response.
    """

    def __init__(self, send: Callable[[Dict[str, Any]], None]) -> None:
        self._send = send
        self._pending: Dict[Any, "queue.Queue"] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    # --- outbound ---------------------------------------------------------
    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Send a one-way notification (no response expected)."""
        self._send({
            "jsonrpc": JSONRPC_VERSION,
            "method": method,
            "params": params if params is not None else {},
        })

    def request(self, method: str, params: Optional[Dict[str, Any]] = None,
                timeout: Optional[float] = None) -> Any:
        """Send a request and BLOCK until the peer's response arrives.

        Returns the response ``result``; raises ``RPCError`` on an error reply.
        The pending future is registered BEFORE the frame is written so a
        synchronous in-process peer (a test transport) may fulfil it re-entrantly
        during ``_send`` without a race.
        """
        with self._lock:
            rid = next(self._ids)
            box: "queue.Queue" = queue.Queue(maxsize=1)
            self._pending[rid] = box
        try:
            self._send({
                "jsonrpc": JSONRPC_VERSION,
                "id": rid,
                "method": method,
                "params": params if params is not None else {},
            })
            msg = box.get(timeout=timeout)
        finally:
            with self._lock:
                self._pending.pop(rid, None)
        err = msg.get("error")
        if err is not None:
            raise RPCError(err)
        return msg.get("result")

    def respond(self, rid: Any, result: Any = None,
                error: Optional[Dict[str, Any]] = None) -> None:
        """Answer an inbound request identified by ``rid``."""
        msg: Dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "id": rid}
        if error is not None:
            msg["error"] = error
        else:
            msg["result"] = result
        self._send(msg)

    # --- inbound ----------------------------------------------------------
    def deliver(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Route one inbound message.

        A response is matched to its pending future and ``None`` is returned. A
        request/notification is returned to the caller to dispatch.
        """
        if is_response(msg):
            with self._lock:
                box = self._pending.get(msg.get("id"))
            if box is not None:
                box.put(msg)
            return None
        return msg


def encode(msg: Dict[str, Any]) -> str:
    """Serialise one message to a single newline-free JSON line."""
    return json.dumps(msg, separators=(",", ":"), sort_keys=True)


def decode(line: str) -> Dict[str, Any]:
    """Parse one JSON line into a message dict."""
    return json.loads(line)
