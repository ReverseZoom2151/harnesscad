"""McpClient -- a spec-compliant Model Context Protocol *client* (stdlib only).

The rest of ``surfaces/mcp`` SERVES the CAD environment as an MCP endpoint
(:class:`surfaces.mcp.server.MCPServer`). This module is the mirror image: it
lets the harness CONSUME an external MCP server, so HarnessCAD can drive a
third-party CAD MCP tool (or any other server) the same way an LLM host would.

It reuses the repo's JSON-RPC codec (:mod:`surfaces.mcp.jsonrpc`) and the
server's negotiated ``PROTOCOL_VERSION`` / ``SUPPORTED_PROTOCOLS`` constants, so
the two sides cannot drift on the wire format or the accepted revisions.

Lifecycle (per the MCP spec ``basic/lifecycle``):

  1. client sends ``initialize`` (protocolVersion, capabilities, clientInfo);
  2. server returns its capabilities + serverInfo + a protocolVersion;
  3. if that version is not one the client supports, the client DISCONNECTS
     (:class:`McpProtocolError`) rather than speaking a dialect it cannot;
  4. client sends the ``notifications/initialized`` notification;
  5. normal operation: ``tools/list``, ``tools/call``, ``resources/list``,
     ``resources/read``, ``prompts/list``, ``prompts/get``, ``ping``.

Two transports, selectable by construction:

  - **stdio**  -- spawn the server as a subprocess (:mod:`subprocess`) and speak
    newline-delimited JSON-RPC over its stdin/stdout (:meth:`McpClient.stdio`);
  - **Streamable HTTP** -- POST JSON-RPC to a URL (stdlib :mod:`urllib`), parsing
    either an ``application/json`` reply or a ``text/event-stream`` (SSE) body,
    with bearer-token / custom-header / API-key auth (:meth:`McpClient.http`).

Request/response correlation is by JSON-RPC ``id``; out-of-order responses are
buffered so a notification handler can itself issue a request (a
``notifications/tools/list_changed`` triggers a tools refresh, re-entrantly).
Every failure is a typed :class:`McpError`: a dead transport is an
:class:`McpConnectionError`, a version clash is an :class:`McpProtocolError`, an
un-advertised capability is an :class:`McpCapabilityError`, and a JSON-RPC error
frame from the server is an :class:`McpRpcError`. The client never fabricates a
result for a capability the server did not offer.

``main(argv)`` exposes a ``--selfcheck`` that runs the full handshake +
``tools/list`` + ``tools/call`` round-trip against an IN-PROCESS fake server (no
subprocess, no network), proving the client end to end.
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence

from harnesscad.io.surfaces.mcp import jsonrpc
from harnesscad.io.surfaces.mcp.server import PROTOCOL_VERSION, SUPPORTED_PROTOCOLS

CLIENT_NAME = "harnesscad-client"
CLIENT_VERSION = "0.2.1"

# What the harness advertises AS A CLIENT. It consumes tools/resources/prompts
# but offers no roots/sampling/elicitation back to the server, so the object is
# intentionally empty -- an honest, minimal capability set.
DEFAULT_CLIENT_CAPABILITIES: Dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# Typed errors -- a failure is never a fabricated result
# --------------------------------------------------------------------------- #
class McpError(Exception):
    """Base class for every MCP client failure."""


class McpConnectionError(McpError):
    """The transport is unreachable, closed, or died mid-exchange."""


class McpProtocolError(McpError):
    """Handshake failure -- e.g. a protocol version the client cannot speak."""


class McpCapabilityError(McpError):
    """The server did not advertise the capability a call requires."""


class McpRpcError(McpError):
    """The server returned a JSON-RPC error frame for a request."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__("JSON-RPC error %d: %s" % (code, message))
        self.code = code
        self.rpc_message = message
        self.data = data


# --------------------------------------------------------------------------- #
# JSON-RPC request/notification builders (responses are built by jsonrpc.*)
# --------------------------------------------------------------------------- #
def _request(id: Any, method: str, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    msg: Dict[str, Any] = {"jsonrpc": jsonrpc.JSONRPC_VERSION, "id": id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def _notification(method: str, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    msg: Dict[str, Any] = {"jsonrpc": jsonrpc.JSONRPC_VERSION, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


# --------------------------------------------------------------------------- #
# Transports. A transport is a bidirectional JSON-RPC message channel:
#   send_message(obj)          -- push one JSON-RPC object toward the server
#   next_message() -> obj|None -- pull the next inbound object, or None at EOF
#   close()                    -- release the channel
# --------------------------------------------------------------------------- #
class _Transport:
    """Abstract JSON-RPC message channel."""

    def send_message(self, message: Dict[str, Any]) -> None:
        raise NotImplementedError

    def next_message(self) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class StdioTransport(_Transport):
    """Newline-delimited JSON-RPC over a spawned subprocess' stdin/stdout.

    Mirrors :func:`surfaces.mcp.stdio.serve_stdio` from the other side: one JSON
    object per line, stdout is the pure MCP channel, and the server's stderr is
    kept off the protocol stream.
    """

    def __init__(self, command: Sequence[str], *, env: Optional[Mapping[str, str]] = None,
                 cwd: Optional[str] = None,
                 stderr: Optional[int] = subprocess.DEVNULL) -> None:
        if not command:
            raise McpConnectionError("stdio transport needs a non-empty command")
        try:
            self._proc = subprocess.Popen(
                list(command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                env=dict(env) if env is not None else None,
                cwd=cwd,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise McpConnectionError("failed to spawn server %r: %s" % (list(command), exc))

    def send_message(self, message: Dict[str, Any]) -> None:
        proc = self._proc
        if proc.stdin is None or proc.poll() is not None:
            raise McpConnectionError("server process is not accepting input")
        try:
            proc.stdin.write(json.dumps(message) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise McpConnectionError("write to server failed: %s" % exc)

    def next_message(self) -> Optional[Dict[str, Any]]:
        proc = self._proc
        if proc.stdout is None:
            return None
        while True:
            line = proc.stdout.readline()
            if line == "":
                return None  # EOF -- the server closed its output stream
            line = line.strip()
            if not line:
                continue
            try:
                return jsonrpc.parse(line)
            except json.JSONDecodeError as exc:
                raise McpConnectionError("server emitted non-JSON line: %s" % exc)

    def close(self) -> None:
        proc = self._proc
        # Spec stdio shutdown: close stdin, wait, then escalate to terminate/kill.
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        if proc.poll() is None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        for stream in (proc.stdout,):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass


class HttpTransport(_Transport):
    """Streamable-HTTP JSON-RPC: POST a message, buffer the reply message(s).

    Each ``send_message`` POSTs the JSON-RPC object. A request's reply body is
    either a single ``application/json`` object or a ``text/event-stream`` (SSE)
    sequence of objects; every object is buffered and handed out by
    :meth:`next_message`. A notification POST yields ``202 Accepted`` with no
    body, so nothing is buffered and the caller does not wait.

    Auth: ``bearer_token`` sets ``Authorization: Bearer ...``; ``headers`` adds
    arbitrary headers (e.g. an ``X-API-Key``). After the handshake, the
    negotiated ``MCP-Protocol-Version`` and any server-issued ``Mcp-Session-Id``
    are echoed on every request, per the transport spec.
    """

    def __init__(self, url: str, *, bearer_token: Optional[str] = None,
                 headers: Optional[Mapping[str, str]] = None,
                 timeout: float = 30.0) -> None:
        self._url = url
        self._timeout = timeout
        self._queue: Deque[Dict[str, Any]] = collections.deque()
        self._base_headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if bearer_token:
            self._base_headers["Authorization"] = "Bearer " + bearer_token
        if headers:
            for key, value in headers.items():
                self._base_headers[str(key)] = str(value)
        self.protocol_version: Optional[str] = None
        self.session_id: Optional[str] = None

    def _headers(self) -> Dict[str, str]:
        out = dict(self._base_headers)
        if self.protocol_version:
            out["MCP-Protocol-Version"] = self.protocol_version
        if self.session_id:
            out["Mcp-Session-Id"] = self.session_id
        return out

    def send_message(self, message: Dict[str, Any]) -> None:
        body = json.dumps(message).encode("utf-8")
        req = urllib.request.Request(
            self._url, data=body, headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                session = resp.headers.get("Mcp-Session-Id")
                if session:
                    self.session_id = session
                content_type = (resp.headers.get("Content-Type") or "").lower()
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")
            except OSError:
                detail = ""
            raise McpConnectionError(
                "HTTP %s from server: %s" % (exc.code, detail or exc.reason))
        except (urllib.error.URLError, OSError) as exc:
            raise McpConnectionError("server unreachable at %s: %s" % (self._url, exc))
        if not raw:
            return  # 202 Accepted (notification) -- nothing to buffer
        if "text/event-stream" in content_type:
            for obj in self._parse_sse(raw.decode("utf-8", "replace")):
                self._queue.append(obj)
        else:
            self._buffer_json(raw.decode("utf-8", "replace"))

    def _buffer_json(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        try:
            parsed = jsonrpc.parse(text)
        except json.JSONDecodeError as exc:
            raise McpConnectionError("server sent non-JSON body: %s" % exc)
        if isinstance(parsed, list):
            for obj in parsed:
                if isinstance(obj, dict):
                    self._queue.append(obj)
        elif isinstance(parsed, dict):
            self._queue.append(parsed)

    @staticmethod
    def _parse_sse(text: str) -> List[Dict[str, Any]]:
        """Extract JSON-RPC objects from the ``data:`` fields of an SSE stream."""
        out: List[Dict[str, Any]] = []
        for block in text.replace("\r\n", "\n").split("\n\n"):
            data_lines = [ln[5:].lstrip() for ln in block.split("\n")
                          if ln.startswith("data:")]
            if not data_lines:
                continue
            payload = "\n".join(data_lines).strip()
            if not payload:
                continue
            try:
                obj = jsonrpc.parse(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
        return out

    def next_message(self) -> Optional[Dict[str, Any]]:
        if self._queue:
            return self._queue.popleft()
        return None

    def close(self) -> None:
        self._queue.clear()


# --------------------------------------------------------------------------- #
# The client
# --------------------------------------------------------------------------- #
class McpClient:
    """A connected MCP client over a :class:`_Transport`.

    Construct directly with a transport, or use the :meth:`stdio` / :meth:`http`
    / :meth:`in_process` factories. :meth:`connect` runs the handshake; the
    ``tools/*``, ``resources/*`` and ``prompts/*`` helpers are usable after it.
    Also a context manager: ``with McpClient.stdio(cmd) as c: ...``.
    """

    def __init__(self, transport: _Transport, *, name: str = CLIENT_NAME,
                 version: str = CLIENT_VERSION,
                 capabilities: Optional[Dict[str, Any]] = None) -> None:
        self._transport = transport
        self._name = name
        self._version = version
        self._client_capabilities = (
            dict(capabilities) if capabilities is not None
            else dict(DEFAULT_CLIENT_CAPABILITIES))
        self._id = 0
        self._orphans: Dict[Any, Dict[str, Any]] = {}
        self._initialized = False
        self.protocol_version: Optional[str] = None
        self.server_capabilities: Dict[str, Any] = {}
        self.server_info: Dict[str, Any] = {}
        self.instructions: Optional[str] = None
        # Caches, invalidated (and eagerly refreshed) on *_list_changed.
        self._tools: Optional[List[Dict[str, Any]]] = None
        self._resources: Optional[List[Dict[str, Any]]] = None
        self._prompts: Optional[List[Dict[str, Any]]] = None

    # --- factories --------------------------------------------------------
    @classmethod
    def stdio(cls, command: Sequence[str], *, env: Optional[Mapping[str, str]] = None,
              cwd: Optional[str] = None, **kwargs: Any) -> "McpClient":
        """A client that spawns ``command`` and speaks JSON-RPC over its stdio."""
        return cls(StdioTransport(command, env=env, cwd=cwd), **kwargs)

    @classmethod
    def http(cls, url: str, *, bearer_token: Optional[str] = None,
             headers: Optional[Mapping[str, str]] = None,
             timeout: float = 30.0, **kwargs: Any) -> "McpClient":
        """A client that POSTs JSON-RPC to ``url`` (Streamable HTTP)."""
        return cls(HttpTransport(url, bearer_token=bearer_token, headers=headers,
                                 timeout=timeout), **kwargs)

    @classmethod
    def in_process(cls, handler: "_InProcessServer", **kwargs: Any) -> "McpClient":
        """A client wired to an in-process handler (no subprocess, no network)."""
        transport = _InProcessTransport(handler)
        return cls(transport, **kwargs)

    # --- lifecycle --------------------------------------------------------
    def connect(self) -> Dict[str, Any]:
        """Run ``initialize`` + version check + ``notifications/initialized``.

        Returns the server's ``initialize`` result. Raises
        :class:`McpProtocolError` (and closes the transport) if the server picks
        a protocol version this client does not support.
        """
        result = self._send_request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": self._client_capabilities,
            "clientInfo": {"name": self._name, "version": self._version},
        })
        if not isinstance(result, dict):
            self.close()
            raise McpProtocolError("initialize returned a non-object result")
        version = result.get("protocolVersion")
        if version not in SUPPORTED_PROTOCOLS:
            self.close()
            raise McpProtocolError(
                "server protocol %r is not supported (client speaks %s)"
                % (version, ", ".join(SUPPORTED_PROTOCOLS)))
        self.protocol_version = version
        caps = result.get("capabilities")
        self.server_capabilities = caps if isinstance(caps, dict) else {}
        info = result.get("serverInfo")
        self.server_info = info if isinstance(info, dict) else {}
        instr = result.get("instructions")
        self.instructions = instr if isinstance(instr, str) else None
        # Tell an HTTP transport the negotiated version so it can header it.
        if isinstance(self._transport, HttpTransport):
            self._transport.protocol_version = version
        self._initialized = True
        self._send_notification("notifications/initialized")
        return result

    def close(self) -> None:
        """Release the transport (subprocess shutdown / HTTP teardown)."""
        try:
            self._transport.close()
        except McpError:
            raise
        except Exception:  # noqa: BLE001 -- close must not raise on cleanup
            pass
        finally:
            self._initialized = False

    def __enter__(self) -> "McpClient":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # --- operations -------------------------------------------------------
    def ping(self) -> Dict[str, Any]:
        """A liveness round-trip (``ping`` -> ``{}``)."""
        return self._send_request("ping", None)

    def list_tools(self, *, force: bool = False) -> List[Dict[str, Any]]:
        """The server's tools. Cached; ``force`` re-fetches from the server."""
        self._require_capability("tools")
        if self._tools is None or force:
            result = self._send_request("tools/list", None)
            self._tools = self._items(result, "tools")
        return list(self._tools)

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None
                  ) -> Dict[str, Any]:
        """Invoke a tool. Returns the ``CallToolResult`` (``isError`` may be set)."""
        self._require_capability("tools")
        params: Dict[str, Any] = {"name": name, "arguments": arguments or {}}
        return self._send_request("tools/call", params)

    def list_resources(self, *, force: bool = False) -> List[Dict[str, Any]]:
        """The server's resources. Cached; ``force`` re-fetches."""
        self._require_capability("resources")
        if self._resources is None or force:
            result = self._send_request("resources/list", None)
            self._resources = self._items(result, "resources")
        return list(self._resources)

    def read_resource(self, uri: str) -> List[Dict[str, Any]]:
        """Materialise a resource -> its ``contents`` list."""
        self._require_capability("resources")
        result = self._send_request("resources/read", {"uri": uri})
        return self._items(result, "contents")

    def list_prompts(self, *, force: bool = False) -> List[Dict[str, Any]]:
        """The server's prompts. Cached; ``force`` re-fetches."""
        self._require_capability("prompts")
        if self._prompts is None or force:
            result = self._send_request("prompts/list", None)
            self._prompts = self._items(result, "prompts")
        return list(self._prompts)

    def get_prompt(self, name: str, arguments: Optional[Dict[str, Any]] = None
                   ) -> Dict[str, Any]:
        """Render a prompt template -> ``{messages: [...], description?}``."""
        self._require_capability("prompts")
        params: Dict[str, Any] = {"name": name}
        if arguments:
            params["arguments"] = arguments
        return self._send_request("prompts/get", params)

    # --- internals --------------------------------------------------------
    def _require_capability(self, name: str) -> None:
        if not self._initialized:
            raise McpConnectionError("client is not connected; call connect() first")
        if self.server_capabilities.get(name) is None:
            raise McpCapabilityError(
                "server does not advertise the '%s' capability" % name)

    @staticmethod
    def _items(result: Any, key: str) -> List[Dict[str, Any]]:
        if not isinstance(result, dict):
            return []
        value = result.get(key)
        return list(value) if isinstance(value, list) else []

    def _send_request(self, method: str, params: Optional[Dict[str, Any]]) -> Any:
        self._id += 1
        rid = self._id
        self._transport.send_message(_request(rid, method, params))
        return self._await(rid)

    def _send_notification(self, method: str,
                           params: Optional[Dict[str, Any]] = None) -> None:
        self._transport.send_message(_notification(method, params))

    def _await(self, rid: Any) -> Any:
        """Pump inbound messages until the response for ``rid`` arrives.

        Notifications are dispatched in-line; responses to *other* ids are
        buffered so a re-entrant request (from a notification handler) still
        finds its own response, whatever order they arrive in.
        """
        while True:
            if rid in self._orphans:
                return self._unwrap(self._orphans.pop(rid))
            msg = self._transport.next_message()
            if msg is None:
                raise McpConnectionError(
                    "connection closed while awaiting response to id %r" % rid)
            if not isinstance(msg, dict):
                continue
            if "method" in msg:
                if "id" in msg:
                    # A server-initiated request. We advertise no client
                    # capabilities, so decline it rather than hang the server.
                    self._transport.send_message(jsonrpc.error(
                        msg.get("id"), jsonrpc.METHOD_NOT_FOUND,
                        "client does not implement '%s'" % msg.get("method")))
                else:
                    self._handle_notification(msg)
                continue
            mid = msg.get("id")
            if mid == rid:
                return self._unwrap(msg)
            self._orphans[mid] = msg

    @staticmethod
    def _unwrap(response: Dict[str, Any]) -> Any:
        if "error" in response:
            err = response.get("error") or {}
            raise McpRpcError(
                int(err.get("code", jsonrpc.INTERNAL_ERROR)),
                str(err.get("message", "unknown error")),
                err.get("data"))
        return response.get("result")

    def _handle_notification(self, msg: Dict[str, Any]) -> None:
        method = msg.get("method")
        # A list-changed notice invalidates the cache and eagerly re-fetches, so
        # the client's view tracks the server. The re-entrant request is safe:
        # _await buffers any out-of-order response by id.
        if method == "notifications/tools/list_changed":
            self._tools = None
            if self.server_capabilities.get("tools") is not None:
                self.list_tools(force=True)
        elif method == "notifications/resources/list_changed":
            self._resources = None
            if self.server_capabilities.get("resources") is not None:
                self.list_resources(force=True)
        elif method == "notifications/prompts/list_changed":
            self._prompts = None
            if self.server_capabilities.get("prompts") is not None:
                self.list_prompts(force=True)
        # Every other notification (progress, logging, ...) is acknowledged by
        # being read; there is nothing this client must do with it.


# --------------------------------------------------------------------------- #
# In-process transport + a tiny fake server, for --selfcheck (no OS resources)
# --------------------------------------------------------------------------- #
class _InProcessServer:
    """Duck-typed in-process server: ``handle(msg) -> dict | None``.

    It may push server-initiated notifications by calling ``attach``'d
    transport's :meth:`_InProcessTransport.push`.
    """

    def attach(self, transport: "_InProcessTransport") -> None:
        raise NotImplementedError

    def handle(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class _InProcessTransport(_Transport):
    """Synchronous channel: a sent message is handled immediately in-process."""

    def __init__(self, server: _InProcessServer) -> None:
        self._server = server
        self._queue: Deque[Dict[str, Any]] = collections.deque()
        self._closed = False
        server.attach(self)

    def push(self, message: Dict[str, Any]) -> None:
        """Enqueue a server-initiated message (e.g. a notification)."""
        self._queue.append(message)

    def send_message(self, message: Dict[str, Any]) -> None:
        if self._closed:
            raise McpConnectionError("in-process transport is closed")
        response = self._server.handle(message)
        if response is not None:
            self._queue.append(response)

    def next_message(self) -> Optional[Dict[str, Any]]:
        if self._queue:
            return self._queue.popleft()
        return None

    def close(self) -> None:
        self._closed = True
        self._queue.clear()


class _FakeMcpServer(_InProcessServer):
    """A minimal spec-shaped server used only by ``--selfcheck``.

    It negotiates the handshake, lists a single ``echo`` tool, executes it, and
    on the first ``tools/call`` mutates its own catalogue AND pushes a
    ``notifications/tools/list_changed`` -- so a correct client refreshes and
    then sees two tools, proving notification-driven refresh.
    """

    def __init__(self) -> None:
        self._transport: Optional[_InProcessTransport] = None
        self._tools: List[Dict[str, Any]] = [{
            "name": "echo",
            "description": "Echo the given text back to the caller.",
            "inputSchema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        }]

    def attach(self, transport: _InProcessTransport) -> None:
        self._transport = transport

    def handle(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = msg.get("method")
        if jsonrpc.is_notification(msg):
            return None
        msg_id = msg.get("id")
        params = msg.get("params") if isinstance(msg.get("params"), dict) else {}

        if method == "initialize":
            return jsonrpc.result(msg_id, {
                "protocolVersion": PROTOCOL_VERSION,
                # Advertise ONLY tools, so --selfcheck can prove the client
                # refuses an un-advertised capability with a typed error.
                "capabilities": {"tools": {"listChanged": True}},
                "serverInfo": {"name": "fake-mcp", "version": "0.0.0"},
                "instructions": "In-process fake server for McpClient --selfcheck.",
            })
        if method == "ping":
            return jsonrpc.result(msg_id, {})
        if method == "tools/list":
            return jsonrpc.result(msg_id, {"tools": list(self._tools)})
        if method == "tools/call":
            return self._call(msg_id, params)
        return jsonrpc.error(msg_id, jsonrpc.METHOD_NOT_FOUND,
                             "fake server: unknown method %r" % method)

    def _call(self, msg_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        if name != "echo":
            return jsonrpc.error(msg_id, jsonrpc.INVALID_PARAMS,
                                 "fake server: unknown tool %r" % name)
        # Grow the catalogue and announce it BEFORE returning the call result, so
        # the client sees the notification ahead of its own response.
        if not any(t["name"] == "echo_twice" for t in self._tools):
            self._tools.append({
                "name": "echo_twice",
                "description": "Echo the given text back twice.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            })
            if self._transport is not None:
                self._transport.push(_notification(
                    "notifications/tools/list_changed", None))
        text = str(arguments.get("text", ""))
        return jsonrpc.result(msg_id, {
            "content": [{"type": "text", "text": text}],
            "isError": False,
        })


# --------------------------------------------------------------------------- #
# --selfcheck: exercise the client end to end against the fake server
# --------------------------------------------------------------------------- #
def _selfcheck(out: Any = None) -> int:
    out = out if out is not None else sys.stdout

    def say(line: str) -> None:
        out.write(line + "\n")

    client = McpClient.in_process(_FakeMcpServer())
    try:
        init = client.connect()
        assert client._initialized, "client did not mark itself initialized"
        assert init.get("serverInfo", {}).get("name") == "fake-mcp", \
            "unexpected serverInfo"
        assert client.protocol_version in SUPPORTED_PROTOCOLS, \
            "protocol not negotiated"
        say("handshake OK -- server=%s proto=%s" % (
            client.server_info.get("name"), client.protocol_version))

        assert client.ping() == {}, "ping did not return an empty result"
        say("ping OK")

        tools = client.list_tools()
        names = [t.get("name") for t in tools]
        assert names == ["echo"], "expected one 'echo' tool, got %r" % names
        say("tools/list OK -- %r" % names)

        result = client.call_tool("echo", {"text": "hello harnesscad"})
        assert result.get("isError") is False, "echo reported isError"
        content = result.get("content", [])
        assert content and content[0].get("text") == "hello harnesscad", \
            "echo round-trip mismatch: %r" % content
        say("tools/call OK -- echo returned %r" % content[0].get("text"))

        # The call pushed a tools/list_changed notification; the client should
        # have refreshed its cache re-entrantly, so it now sees the new tool
        # WITHOUT us forcing a re-fetch.
        refreshed = [t.get("name") for t in client.list_tools()]
        assert refreshed == ["echo", "echo_twice"], \
            "list_changed refresh did not update cache: %r" % refreshed
        say("notifications/tools/list_changed OK -- refreshed to %r" % refreshed)

        # Absent-capability degradation is a typed error, never a fabrication.
        try:
            client.read_resource("state://model")
        except McpCapabilityError:
            say("capability guard OK -- resources absent -> McpCapabilityError")
        else:
            raise AssertionError("read_resource should have raised McpCapabilityError")
    except AssertionError as exc:
        say("SELFCHECK FAILED: %s" % exc)
        return 1
    except McpError as exc:
        say("SELFCHECK FAILED (McpError): %s" % exc)
        return 1
    finally:
        client.close()

    say("selfcheck: all checks passed")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.io.surfaces.mcp.client",
        description="HarnessCAD Model Context Protocol CLIENT (consume external "
                    "MCP servers over stdio or Streamable HTTP)")
    sub = parser.add_argument_group("actions")
    sub.add_argument("--selfcheck", action="store_true",
                     help="exercise the client against an in-process fake server")
    parser.add_argument("--stdio", nargs=argparse.REMAINDER, default=None,
                        metavar="CMD",
                        help="connect over stdio to CMD [ARGS...] and list its tools")
    parser.add_argument("--http", default=None, metavar="URL",
                        help="connect over Streamable HTTP to URL and list its tools")
    parser.add_argument("--bearer", default=None, metavar="TOKEN",
                        help="bearer token for --http (Authorization: Bearer ...)")
    parser.add_argument("--header", action="append", default=[], metavar="K:V",
                        help="extra header for --http (repeatable), e.g. X-API-Key:abc")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.selfcheck:
        return _selfcheck()

    if args.stdio:
        client = McpClient.stdio(args.stdio)
        return _describe(client)

    if args.http:
        headers: Dict[str, str] = {}
        for item in args.header:
            key, sep, value = item.partition(":")
            if not sep:
                print("bad --header %r; expected K:V" % item, file=sys.stderr)
                return 2
            headers[key.strip()] = value.strip()
        client = McpClient.http(args.http, bearer_token=args.bearer, headers=headers)
        return _describe(client)

    parser.print_help()
    return 0


def _describe(client: McpClient) -> int:
    """Connect, print serverInfo + tool/resource/prompt names, disconnect."""
    try:
        client.connect()
        report: Dict[str, Any] = {
            "serverInfo": client.server_info,
            "protocolVersion": client.protocol_version,
            "capabilities": sorted(client.server_capabilities),
        }
        if client.server_capabilities.get("tools") is not None:
            report["tools"] = [t.get("name") for t in client.list_tools()]
        if client.server_capabilities.get("resources") is not None:
            report["resources"] = [r.get("uri") for r in client.list_resources()]
        if client.server_capabilities.get("prompts") is not None:
            report["prompts"] = [p.get("name") for p in client.list_prompts()]
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except McpError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
