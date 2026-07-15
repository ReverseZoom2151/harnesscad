"""The HTTP transport: a stdlib ``http.server`` skin over ``A2AHandler``.

Two routes, matching the A2A JSON-RPC-over-HTTP binding:

  - ``GET /.well-known/agent-card.json`` -> the ``AgentCard`` JSON.
  - ``POST /`` -> a JSON-RPC 2.0 call. For ``message/stream`` and
    ``tasks/resubscribe`` the response is ``text/event-stream`` and frames are
    flushed as the task progresses; every other method gets a single JSON body.

Concurrency is ``ThreadingHTTPServer`` (one thread per connection) so a streaming
request never blocks other callers. ``make_server`` returns an unstarted server
(handy for tests that bind port 0 and drive it in a thread); ``serve`` builds one
and runs it forever.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional

from harnesscad.agents.a2a.messages import AgentCard
from harnesscad.io.surfaces.a2a_server import wire
from harnesscad.io.surfaces.a2a_server.auth import AuthError, Authenticator
from harnesscad.io.surfaces.a2a_server.card import build_agent_card
from harnesscad.io.surfaces.a2a_server.handler import A2AHandler

AGENT_CARD_PATH = "/.well-known/agent-card.json"
_STREAMING_METHODS = ("message/stream", "tasks/resubscribe")


def make_request_handler(
    handler: A2AHandler,
    card: AgentCard,
    authenticator: Optional[Authenticator] = None,
):
    """Build a ``BaseHTTPRequestHandler`` subclass bound to a handler + card.

    ``authenticator`` gates every JSON-RPC POST (see ``_authenticate``); when
    omitted a no-op (auth-disabled) ``Authenticator`` is used so local dev keeps
    working. The Agent Card GET stays public — it is the discovery document that
    tells a client how to authenticate.
    """
    auth = authenticator if authenticator is not None else Authenticator()

    class _A2ARequestHandler(BaseHTTPRequestHandler):
        server_version = "HarnessCAD-A2A/0.3.0"

        # -- routing -------------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            if self.path.split("?", 1)[0] == AGENT_CARD_PATH:
                self._send_json(200, card.to_dict())
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib naming
            # Authenticate EVERY RPC before parsing/dispatching anything.
            if not self._authenticate():
                return
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            try:
                request = json.loads(body) if body else None
            except json.JSONDecodeError as exc:
                self._send_json(
                    200, wire.rpc_error(None, wire.ERROR_PARSE, f"parse error: {exc}")
                )
                return
            if not isinstance(request, dict):
                self._send_json(
                    200,
                    wire.rpc_error(
                        None, wire.ERROR_INVALID_REQUEST, "request must be an object"
                    ),
                )
                return
            if request.get("method") in _STREAMING_METHODS:
                self._stream(request)
            else:
                self._send_json(200, handler.dispatch(request))

        # -- authentication ------------------------------------------------
        def _authenticate(self) -> bool:
            """Verify the request; on failure send 401/403 and return False."""
            try:
                auth.authenticate(self.headers)
                return True
            except AuthError as exc:
                self._send_auth_error(exc)
                return False

        def _send_auth_error(self, exc: AuthError) -> None:
            payload = json.dumps({"error": exc.message}).encode("utf-8")
            self.send_response(exc.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            if exc.www_authenticate:
                self.send_header("WWW-Authenticate", exc.www_authenticate)
            self.end_headers()
            self.wfile.write(payload)

        # -- responses -----------------------------------------------------
        def _stream(self, request: dict) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for frame in handler.stream(request):
                self.wfile.write(frame.encode("utf-8"))
                self.wfile.flush()

        def _send_json(self, status: int, obj: Any) -> None:
            payload = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: Any) -> None:  # silence per-request logging
            pass

    return _A2ARequestHandler


def make_server(
    harness_factory: Callable[[], Any],
    host: str = "127.0.0.1",
    port: int = 9100,
    card: Optional[AgentCard] = None,
    authenticator: Optional[Authenticator] = None,
) -> ThreadingHTTPServer:
    """Build (but do not start) a ThreadingHTTPServer serving the A2A endpoint.

    ``authenticator`` defaults to ``Authenticator.from_env()`` so production can
    require auth (``HARNESSCAD_A2A_AUTH=1`` + a secret) while local dev, with the
    env unset, stays open. The resolved authenticator also shapes the Agent
    Card's ``security`` block.
    """
    auth = authenticator if authenticator is not None else Authenticator.from_env()
    handler = A2AHandler(harness_factory)
    resolved_card = card if card is not None else build_agent_card(
        url=f"http://{host}:{port}/", authenticator=auth
    )
    request_handler = make_request_handler(handler, resolved_card, auth)
    return ThreadingHTTPServer((host, port), request_handler)


def serve(
    harness_factory: Callable[[], Any],
    host: str = "127.0.0.1",
    port: int = 9100,
    card: Optional[AgentCard] = None,
    authenticator: Optional[Authenticator] = None,
) -> None:
    """Build the server and run it forever (blocks)."""
    httpd = make_server(harness_factory, host, port, card, authenticator)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass
    finally:
        httpd.server_close()
