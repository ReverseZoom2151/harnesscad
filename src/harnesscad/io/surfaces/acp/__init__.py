"""surfaces.acp — the Zed Agent Client Protocol (ACP) agent adapter.

Lets any ACP client (e.g. Zed) drive HarnessCAD text-to-CAD in-editor over
newline-delimited JSON-RPC 2.0 on stdio (ACP protocol v1).

Layers (stdlib-only, deterministic):

  * ``jsonrpc`` — bidirectional JSON-RPC 2.0 framing with pending-id futures so
    the agent can initiate ``session/request_permission`` / ``fs/write_text_file``
    mid-turn and BLOCK on the client's reply.
  * ``bridge``  — translate HarnessCAD ``UIEvent``s / trace + the op stream into
    ACP ``session/update`` notifications, and run the two outbound round-trips.
  * ``agent``   — ``ACPAgent`` implementing the agent method set (initialize,
    session/new, session/prompt, session/cancel) over a ``HarnessSession`` +
    ``AgentHarness`` with a ToolExecutor-gated approval path.
  * ``__main__`` — ``python -m surfaces.acp``, the process a client spawns.
"""

from __future__ import annotations

from harnesscad.io.surfaces.acp.agent import ACPAgent, BridgingExecutor, BridgeTracer, PromptCancelled
from harnesscad.io.surfaces.acp.bridge import ACPBridge
from harnesscad.io.surfaces.acp.jsonrpc import Connection, RPCError

__all__ = [
    "ACPAgent",
    "ACPBridge",
    "BridgingExecutor",
    "BridgeTracer",
    "PromptCancelled",
    "Connection",
    "RPCError",
]
