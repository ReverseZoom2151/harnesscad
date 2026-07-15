"""MCP-style tool-server surface for the harnesscad CAD environment.

This package realises docs/blueprint.md sec.5 ("expose the environment as an
MCP server") and sec.9 ("typed op schema, 5-component tool descriptions, tools
raise typed errors) *without* pulling in the MCP SDK. The MCP concepts are
modelled as plain dataclasses / JSON so a real FastMCP transport drops in later:

  - **tools = action space**  -> :class:`mcp.tools.ToolCatalog` (one tool per
    CISP op + measure / query / verify / export / reset / render).
  - **resources = observations** -> :meth:`ToolCatalog.resources` (feature tree,
    validity, measurements).
  - **prompts = op templates** -> :meth:`ToolCatalog.prompts`.
  - **tool-result carries a reward field** -> :class:`mcp.tools.ToolResult`.
  - **a reset tool** and destructive/read-only **annotations** ->
    :mod:`mcp.annotations` (auto-derived approval tiers, sec.14).
  - **the Gym interface** (reset/step/state/render/close) ->
    :class:`mcp.gym.CADGymEnv`, wrapping a HarnessSession + backend, hybrid
    observation (B-rep summary JSON + a render hook), verifier-derived reward,
    and never leaking a ground-truth answer into the observation.

Nothing here imports an MCP library; ``ToolCatalog.to_mcp()`` emits the JSON
tool schema a FastMCP server would register. Stdlib only.
"""

from __future__ import annotations

from harnesscad.io.surfaces.mcp.annotations import (
    Annotations,
    TIER_AUTO,
    TIER_NOTIFY,
    TIER_REQUIRE,
    annotate,
    approval_tier,
)
from harnesscad.io.surfaces.mcp.tools import (
    MCPError,
    ParamSpec,
    ToolCatalog,
    ToolDefinition,
    ToolDescription,
    ToolExecutionError,
    ToolResult,
    ToolValidationError,
    UnknownToolError,
    reward_from_apply,
    reward_from_verify,
)
from harnesscad.io.surfaces.mcp.gym import CADGymEnv
from harnesscad.io.surfaces.mcp.client import (
    McpCapabilityError,
    McpClient,
    McpConnectionError,
    McpError,
    McpProtocolError,
    McpRpcError,
)

__all__ = [
    "Annotations",
    "TIER_AUTO",
    "TIER_NOTIFY",
    "TIER_REQUIRE",
    "annotate",
    "approval_tier",
    "MCPError",
    "ParamSpec",
    "ToolCatalog",
    "ToolDefinition",
    "ToolDescription",
    "ToolExecutionError",
    "ToolResult",
    "ToolValidationError",
    "UnknownToolError",
    "reward_from_apply",
    "reward_from_verify",
    "CADGymEnv",
    "McpClient",
    "McpError",
    "McpConnectionError",
    "McpProtocolError",
    "McpCapabilityError",
    "McpRpcError",
]
