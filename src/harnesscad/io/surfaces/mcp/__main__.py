"""``python -m surfaces.mcp`` — launch the MCP server over stdio.

Reuses the backend-selection pattern from :mod:`surfaces.server` (stub by
default; cadquery when available, falling back to stub with a note on stderr).
"""

from __future__ import annotations

import sys
from typing import List, Optional

from harnesscad.io.surfaces.mcp.server import MCPServer
from harnesscad.io.surfaces.mcp.stdio import serve_stdio


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m surfaces.mcp",
        description="HarnessCAD Model Context Protocol server (stdio transport)")
    parser.add_argument("--backend", default="stub", choices=["stub", "cadquery"])
    args = parser.parse_args(argv)

    server = MCPServer(backend=args.backend)
    # Backend notes go to stderr only — stdout is a pure MCP JSON channel.
    if server.backend_note:
        print(server.backend_note, file=sys.stderr)
    serve_stdio(server)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
