"""HarnessCAD as a Google A2A (Agent-to-Agent) protocol server.

Exposes the text-to-CAD harness over the A2A JSON-RPC 2.0 HTTP binding
(protocol v0.3.0): an ``AgentCard`` at ``/.well-known/agent-card.json`` plus the
``message/send``, ``message/stream``, ``tasks/get``, ``tasks/cancel`` and
``tasks/resubscribe`` methods. Stdlib-only (``http.server``); the geometry work
runs through the existing ``AgentHarness``.

Layout:
  - ``card``    — the HarnessCAD ``AgentCard`` (one skill: ``text-to-cad``).
  - ``wire``    — pure translators (params->message, event->A2A, JSON-RPC/SSE).
  - ``handler`` — the JSON-RPC dispatcher backed by an ``a2a.TaskStore``.
  - ``app``     — the ``http.server`` transport (``make_server`` / ``serve``).
  - ``__main__``— the CLI (``python -m surfaces.a2a_server``).
"""

from __future__ import annotations

from harnesscad.io.surfaces.a2a_server.app import make_server, serve
from harnesscad.io.surfaces.a2a_server.auth import (
    AuthError,
    Authenticator,
    Principal,
)
from harnesscad.io.surfaces.a2a_server.card import build_agent_card
from harnesscad.io.surfaces.a2a_server.handler import A2AHandler

__all__ = [
    "A2AHandler",
    "AuthError",
    "Authenticator",
    "Principal",
    "build_agent_card",
    "make_server",
    "serve",
]
