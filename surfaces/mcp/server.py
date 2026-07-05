"""MCPServer — a spec-compliant Model Context Protocol server over the CAD env.

This is the real MCP surface (protocol ``2025-11-25``) built on the existing
:class:`surfaces.mcp.tools.ToolCatalog` (tools = CISP ops + aux, resources =
model-state observations, prompts = op templates) and a live
:class:`loop.HarnessSession`.

``handle(msg) -> dict | None`` is a *pure*, in-process JSON-RPC dispatcher (it
returns ``None`` for notifications, which get no response); the newline-delimited
stdio transport that wraps it lives in :mod:`surfaces.mcp.stdio`.

Dispatch map (JSON-RPC method -> MCP semantics):

  - ``initialize``        handshake + capabilities + protocol-version negotiation
  - ``notifications/*``   acknowledged silently (return ``None``)
  - ``ping``              -> ``{}``
  - ``tools/list``        cleaned tool objects (spec keys + relocated ``_meta``)
  - ``tools/call``        run a tool -> ``CallToolResult`` (op rejection => isError)
  - ``resources/list``    model-state observation descriptors
  - ``resources/read``    materialise a resource (unknown uri => -32002)
  - ``prompts/list``      op templates (internal ``template`` key stripped)
  - ``prompts/get``       render a template with arguments -> messages

Stdlib only; deterministic; no network.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional

from loop import HarnessSession
from surfaces.mcp import jsonrpc
from surfaces.mcp.jsonrpc import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    RESOURCE_NOT_FOUND,
)
from surfaces.mcp.tools import (
    MCPError,
    ToolCatalog,
    ToolExecutionError,
    ToolValidationError,
    UnknownToolError,
)
from surfaces.server import _make_backend

# The MCP protocol revisions this server can speak. The first is preferred; the
# others are accepted for backwards compatibility during negotiation.
PROTOCOL_VERSION = "2025-11-25"
SUPPORTED_PROTOCOLS = ("2025-11-25", "2025-06-18", "2025-03-26")
SERVER_NAME = "harnesscad"
SERVER_VERSION = "0.2.1"

SERVER_INSTRUCTIONS = (
    "A transactional CAD modelling environment exposed as MCP tools. Each tool "
    "is a typed CISP operation (sketch / constraint / feature) or a read-only "
    "query; every mutating call is verified and block-and-corrected, and a "
    "rejected op returns isError with diagnostics you can use to self-correct. "
    "Read resources for model state and use prompts for common part templates."
)

# Keys that survive into a spec ``tools/list`` tool object; everything else on a
# ToolDefinition.to_mcp() dict is relocated under ``_meta`` (namespaced).
_SPEC_TOOL_KEYS = frozenset(
    {"name", "title", "description", "inputSchema", "outputSchema", "annotations"}
)
_META_PREFIX = "com.harnesscad/"


def _negotiate_protocol(client_version: Any) -> str:
    """Echo the client's protocol version if we support it, else our preferred."""
    if client_version in SUPPORTED_PROTOCOLS:
        return client_version
    return PROTOCOL_VERSION


def _clean_tool(tool_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only spec tool keys; relocate the rest under a namespaced ``_meta``."""
    cleaned: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}
    for key, value in tool_obj.items():
        if key in _SPEC_TOOL_KEYS:
            if value is not None:
                cleaned[key] = value
        else:
            meta[_META_PREFIX + key] = value
    if meta:
        cleaned["_meta"] = meta
    return cleaned


def _resolve_template(value: Any, arguments: Dict[str, Any]) -> Any:
    """Substitute ``{arg}`` placeholders in a prompt template value.

    A string that is *exactly* a single ``{name}`` placeholder is replaced by the
    argument's typed value (so numbers stay numbers); any other string has each
    ``{name}`` occurrence textually substituted. Lists/dicts recurse.
    """
    if isinstance(value, str):
        stripped = value.strip()
        if (stripped.startswith("{") and stripped.endswith("}")
                and stripped[1:-1] in arguments):
            return arguments[stripped[1:-1]]
        out = value
        for name, val in arguments.items():
            out = out.replace("{" + name + "}", str(val))
        return out
    if isinstance(value, list):
        return [_resolve_template(v, arguments) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_template(v, arguments) for k, v in value.items()}
    return value


class MCPServer:
    """A session-scoped MCP endpoint over a ToolCatalog + a HarnessSession."""

    def __init__(self, backend: str = "stub", *, session: Optional[HarnessSession] = None,
                 catalog: Optional[ToolCatalog] = None) -> None:
        self.catalog = catalog if catalog is not None else ToolCatalog()
        if session is not None:
            self.session = session
            self.backend = session.backend
            self.backend_name = type(session.backend).__name__
            self.backend_note = None
        else:
            self.backend, self.backend_name, self.backend_note = _make_backend(backend)
            self.session = HarnessSession(self.backend)

    # --- dispatch ---------------------------------------------------------
    def handle(self, msg: Any) -> Optional[Dict[str, Any]]:
        """Pure JSON-RPC dispatch. Returns ``None`` for notifications."""
        if not isinstance(msg, dict):
            return jsonrpc.error(None, jsonrpc.INVALID_REQUEST,
                                 "request must be a JSON object")
        method = msg.get("method")
        msg_id = msg.get("id")
        # Notifications (no id) and any notifications/* method get no response.
        if jsonrpc.is_notification(msg):
            return None
        if isinstance(method, str) and method.startswith("notifications/"):
            return None

        params = msg.get("params")
        if not isinstance(params, dict):
            params = {}

        try:
            if method == "initialize":
                return jsonrpc.result(msg_id, self._initialize(params))
            if method == "ping":
                return jsonrpc.result(msg_id, {})
            if method == "tools/list":
                return jsonrpc.result(msg_id, self._tools_list())
            if method == "tools/call":
                return self._tools_call(msg_id, params)
            if method == "resources/list":
                return jsonrpc.result(msg_id, {"resources": self.catalog.resources()})
            if method == "resources/read":
                return self._resources_read(msg_id, params)
            if method == "prompts/list":
                return jsonrpc.result(msg_id, self._prompts_list())
            if method == "prompts/get":
                return self._prompts_get(msg_id, params)
            return jsonrpc.error(msg_id, METHOD_NOT_FOUND,
                                 f"unknown method '{method}'")
        except MCPError as exc:
            return jsonrpc.error(msg_id, jsonrpc.code_for(exc), exc.message,
                                 exc.data or None)

    # --- initialize -------------------------------------------------------
    def _initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        protocol = _negotiate_protocol(params.get("protocolVersion"))
        return {
            "protocolVersion": protocol,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"listChanged": False},
                "prompts": {"listChanged": False},
            },
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": SERVER_INSTRUCTIONS,
        }

    # --- tools ------------------------------------------------------------
    def _tools_list(self) -> Dict[str, Any]:
        return {"tools": [_clean_tool(t) for t in self.catalog.to_mcp()]}

    def _tools_call(self, msg_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str):
            return jsonrpc.error(msg_id, INVALID_PARAMS,
                                 "tools/call: 'name' must be a string")
        if not isinstance(arguments, dict):
            return jsonrpc.error(msg_id, INVALID_PARAMS,
                                 "tools/call: 'arguments' must be an object")
        try:
            result = self.catalog.call(name, arguments, session=self.session)
        except (UnknownToolError, ToolValidationError) as exc:
            return jsonrpc.error(msg_id, INVALID_PARAMS, exc.message, exc.data or None)
        except ToolExecutionError as exc:
            # A rejected op is NOT a JSON-RPC error: it is a tool result with
            # isError:true carrying the diagnostics as the self-correction channel.
            structured = {
                "diagnostics": exc.data.get("diagnostics", []),
                "rejected": exc.data.get("rejected"),
                "reward": exc.data.get("reward"),
            }
            call_result = {
                "content": [{"type": "text", "text": exc.message}],
                "structuredContent": structured,
                "isError": True,
            }
            return jsonrpc.result(msg_id, call_result)
        return jsonrpc.result(msg_id, result.to_call_result())

    # --- resources --------------------------------------------------------
    def _resources_read(self, msg_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        uri = params.get("uri")
        if not isinstance(uri, str):
            return jsonrpc.error(msg_id, INVALID_PARAMS,
                                 "resources/read: 'uri' must be a string")
        try:
            obj = self.catalog.read_resource(uri, self.session)
        except UnknownToolError as exc:
            return jsonrpc.error(msg_id, RESOURCE_NOT_FOUND, exc.message,
                                 exc.data or None)
        contents = [{
            "uri": uri,
            "mimeType": "application/json",
            "text": json.dumps(obj),
        }]
        return jsonrpc.result(msg_id, {"contents": contents})

    # --- prompts ----------------------------------------------------------
    def _prompts_list(self) -> Dict[str, Any]:
        prompts = []
        for entry in self.catalog.prompts():
            prompts.append({k: v for k, v in entry.items() if k != "template"})
        return {"prompts": prompts}

    def _prompts_get(self, msg_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        entry = None
        for candidate in self.catalog.prompts():
            if candidate.get("name") == name:
                entry = candidate
                break
        if entry is None:
            return jsonrpc.error(msg_id, INVALID_PARAMS, f"unknown prompt '{name}'")

        template = copy.deepcopy(entry.get("template", []))
        resolved_ops = _resolve_template(template, arguments)
        human = (
            f"Build '{entry['name']}': {entry.get('description', '')}\n"
            "Apply the following CISP ops in order (call the matching tools):\n"
            + json.dumps(resolved_ops, indent=2)
        )
        message = {
            "role": "user",
            "content": {"type": "text", "text": human},
        }
        out: Dict[str, Any] = {"messages": [message]}
        if entry.get("description"):
            out["description"] = entry["description"]
        return jsonrpc.result(msg_id, out)
