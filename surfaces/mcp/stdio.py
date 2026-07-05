"""Stdio transport for the MCP server — newline-delimited JSON over stdin/stdout.

Modelled on :meth:`surfaces.server.CISPServer.serve_stdio`: read one JSON-RPC
message per line, dispatch it through :meth:`MCPServer.handle`, and write each
non-``None`` response as ``json + "\\n"`` to stdout, flushing every line.

**Stdio purity** is enforced: only MCP JSON-RPC frames are ever written to
stdout (notifications and blank lines produce no output; a parse failure yields a
JSON-RPC ``-32700`` frame with a null id). Anything diagnostic must go to stderr,
never stdout, so the stream stays a clean MCP channel.
"""

from __future__ import annotations

import json
import sys
from typing import Optional, TextIO

from surfaces.mcp import jsonrpc
from surfaces.mcp.server import MCPServer


def serve_stdio(server: MCPServer, stdin: Optional[TextIO] = None,
                stdout: Optional[TextIO] = None) -> None:
    """Run the read-line / dispatch / write-line loop until stdin is exhausted."""
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = jsonrpc.parse(line)
        except json.JSONDecodeError as exc:
            response = jsonrpc.error(None, jsonrpc.PARSE_ERROR, f"parse error: {exc}")
        else:
            response = server.handle(msg)
        if response is None:
            continue
        stdout.write(json.dumps(response) + "\n")
        stdout.flush()
