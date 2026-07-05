"""Tool annotations + approval tiers (HARNESS_BLUEPRINT.md sec.5 & sec.14).

Every tool carries MCP-style behavioural hints (``readOnlyHint`` /
``destructiveHint`` / ``idempotentHint``) that are *auto-assigned* from the tool
name, so the three-tier approval policy from sec.14 can be derived mechanically:

  - **Tier 1 (auto)**    read-only tools: ``render`` / ``measure`` / ``query`` /
    ``verify`` / ``run_check``.
  - **Tier 2 (notify)**  modifying ops (sketch/extrude/fillet/boolean/...): the
    default tier.
  - **Tier 3 (require)**  destructive / irreversible tools: ``export`` /
    ``delete`` / ``reset``.

These are plain dataclasses / dicts; a FastMCP server would pass ``to_dict()``
straight through as the tool's ``annotations`` block.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- approval tiers (sec.14) ----------------------------------------------
TIER_AUTO = 1     # read / measure / render -> execute without asking
TIER_NOTIFY = 2   # modify -> execute but notify
TIER_REQUIRE = 3  # export / delete / irreversible -> require explicit approval

_TIER_NAMES = {TIER_AUTO: "auto", TIER_NOTIFY: "notify", TIER_REQUIRE: "require"}

# Tools whose primary effect is destructive / irreversible (external artifact or
# state wipe). ``delete`` is listed ahead of a future delete op.
DESTRUCTIVE_TOOLS = frozenset({"export", "delete", "reset"})

# Tools that only observe the model and never mutate it.
READ_ONLY_TOOLS = frozenset({"render", "measure", "query", "verify", "run_check"})


@dataclass(frozen=True)
class Annotations:
    """MCP-style behavioural hints for a tool. All hints default false."""

    readOnlyHint: bool = False
    destructiveHint: bool = False
    idempotentHint: bool = False
    openWorldHint: bool = False

    @property
    def tier(self) -> int:
        """Approval tier auto-derived from the hints (sec.14)."""
        if self.destructiveHint:
            return TIER_REQUIRE
        if self.readOnlyHint:
            return TIER_AUTO
        return TIER_NOTIFY

    @property
    def tier_name(self) -> str:
        return _TIER_NAMES[self.tier]

    def to_dict(self) -> dict:
        return {
            "readOnlyHint": self.readOnlyHint,
            "destructiveHint": self.destructiveHint,
            "idempotentHint": self.idempotentHint,
            "openWorldHint": self.openWorldHint,
            "tier": self.tier,
            "tierName": self.tier_name,
        }


def annotate(tool_name: str) -> Annotations:
    """Auto-assign annotations for ``tool_name``.

    export/delete/reset -> destructive; render/measure/query/verify -> read-only;
    everything else (the modifying ops) -> the default (notify) tier.
    """
    name = tool_name.lower()
    if name in DESTRUCTIVE_TOOLS:
        # reset is idempotent (resetting twice == resetting once); export is not.
        return Annotations(destructiveHint=True, idempotentHint=(name == "reset"))
    if name in READ_ONLY_TOOLS:
        return Annotations(readOnlyHint=True, idempotentHint=True)
    return Annotations()


def approval_tier(tool_name: str) -> int:
    """Convenience: the sec.14 approval tier for ``tool_name``."""
    return annotate(tool_name).tier
