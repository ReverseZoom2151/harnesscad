"""JSON-RPC 2.0 helpers for the MCP server (stdlib only).

MCP speaks JSON-RPC 2.0 over a newline-delimited stdio transport. This module is
the thin, dependency-free codec + code table the server dispatches through:

  - :func:`parse` / :func:`is_notification` classify an incoming message;
  - :func:`result` / :func:`error` build spec-shaped responses;
  - the integer ``*_ERROR`` constants are the JSON-RPC (and MCP) error codes;
  - :func:`code_for` maps the tool-layer :class:`MCPError` subclasses onto them.

Everything is plain dicts + ``json``; no framing is done here (that lives in
:mod:`surfaces.mcp.stdio`).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from surfaces.mcp.tools import (
    MCPError,
    ToolExecutionError,
    ToolValidationError,
    UnknownToolError,
)

# --- JSON-RPC 2.0 error codes (and the MCP RESOURCE_NOT_FOUND extension) ----
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
RESOURCE_NOT_FOUND = -32002

JSONRPC_VERSION = "2.0"


def parse(text: str) -> Any:
    """Parse a JSON-RPC message from ``text``; raises ``json.JSONDecodeError``."""
    return json.loads(text)


def is_notification(msg: Any) -> bool:
    """A JSON-RPC notification is a request object carrying no ``id``."""
    return isinstance(msg, dict) and "id" not in msg


def result(id: Any, obj: Any) -> Dict[str, Any]:
    """A successful JSON-RPC response envelope."""
    return {"jsonrpc": JSONRPC_VERSION, "id": id, "result": obj}


def error(id: Any, code: int, message: str,
          data: Optional[Any] = None) -> Dict[str, Any]:
    """A JSON-RPC error response envelope."""
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": JSONRPC_VERSION, "id": id, "error": err}


def code_for(exc: MCPError) -> int:
    """Map a tool-layer :class:`MCPError` onto a JSON-RPC error code.

    Bad tool name / bad arguments are *invalid params* (-32602); an op the
    kernel/verifier rejects is an internal execution failure (the server usually
    surfaces this as an ``isError`` tool result instead, so this is a fallback).
    """
    if isinstance(exc, (UnknownToolError, ToolValidationError)):
        return INVALID_PARAMS
    if isinstance(exc, ToolExecutionError):
        return INTERNAL_ERROR
    return INTERNAL_ERROR
