"""ui — the UI event contract for harnesscad (docs/blueprint.md sec.14).

Two framework-free layers, stdlib-only, no web server:

  - ``ui.events``  — the typed SSE event protocol. An ``EventType`` enum
    (status/thinking/token/tool_call/tool_result/approval_required/
    action_rejected/done) and a ``UIEvent`` that serialises to the SSE wire
    format (``event: <type>\\ndata: <json>\\n\\n``) via ``to_sse()`` and parses
    back via ``parse_sse``. ``EventStream`` yields SSE strings from a sequence
    of events so it can feed any transport.

  - ``ui.approval`` — the three-tier approval model. An ``ApprovalTier`` enum
    (AUTO/NOTIFY/REQUIRE), ``tier_for(op)`` mapping CISP ops to tiers, a
    ``RiskLevel`` indicator, a ``DryRunPreview`` describing predicted geometry
    changes without mutating state, and an ``ApprovalGate`` that decides whether
    an op may auto-proceed or needs human approval (emitting an
    ``approval_required`` UIEvent for Tier-3) with batching support.

The two layers meet at the ``approval_required`` event: the gate builds a
``UIEvent`` of that type carrying the risk indicator and the dry-run preview.
"""

from __future__ import annotations

from harnesscad.io.surfaces.ui.events import (
    EVENT_TYPES,
    EventStream,
    EventType,
    UIEvent,
    parse_sse,
    parse_stream,
)
from harnesscad.io.surfaces.ui.approval import (
    ApprovalDecision,
    ApprovalGate,
    ApprovalTier,
    DryRunPreview,
    RiskLevel,
    tier_for,
    tier_from_annotations,
    risk_for,
)

__all__ = [
    # events
    "EventType",
    "UIEvent",
    "EventStream",
    "EVENT_TYPES",
    "parse_sse",
    "parse_stream",
    # approval
    "ApprovalTier",
    "RiskLevel",
    "DryRunPreview",
    "ApprovalGate",
    "ApprovalDecision",
    "tier_for",
    "tier_from_annotations",
    "risk_for",
]
