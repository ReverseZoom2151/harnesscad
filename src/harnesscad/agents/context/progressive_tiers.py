"""Token-budgeted progressive-tier reading over a ranked memory set.

The progressive memory model stores every memory node at
three tiers of increasing fidelity and cost -- tier 0 ``summary`` (index),
tier 1 ``detailed_summary`` (a lazy 3-8 sentence expansion), tier 2 ``raw``
(the full original text) -- and its ``read_memory(node_id, depth)`` API lets an
agent open "the shallowest tier that answers the question and only pay the
token cost for depth it actually needs" (README: *3-tier progressive
retrieval*). The orchestrator separately renders context windows *passive-first,
then recent-active fills the remaining slots* (``get_context_window``).

Neither behaviour is a fixed policy in CoMeT's own code -- depth is chosen by
hand and the window budget is a node *count*, not a token budget. This module
turns both into one deterministic planner: given ranked candidates, each with
the token cost of its content at every tier, and a hard token budget, it emits
a **read plan** -- the exact tier to open each node at -- that

  1. always includes pinned (passive/both) nodes first, claiming budget ahead
     of everyone else;
  2. admits the remaining nodes at tier 0 in priority order until the next
     summary no longer fits (the rest are dropped);
  3. spends whatever budget is left *deepening* nodes one tier at a time,
     highest priority first, with risk-flagged nodes escalated ahead of the
     rest because a high-risk node's summary is likely insufficient (the README
     ``risk_level=high`` warning path).

Everything is a pure function of the inputs. Ties break by ``node_id`` so the
plan never depends on iteration order, and no wall clock or randomness is used.
The cost of reading a node at tier ``t`` is ``costs[t]`` (tiers are alternative
reads, not cumulative) -- the caller supplies real token counts or uses
``estimate_tokens`` for a stdlib character heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "SUMMARY",
    "DETAIL",
    "RAW",
    "TierProfile",
    "PlannedRead",
    "ReadPlan",
    "estimate_tokens",
    "make_profile",
    "plan_reads",
]

SUMMARY = 0
DETAIL = 1
RAW = 2


def estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    """Deterministic token estimate: ``ceil(len(text) / chars_per_token)``.

    A stdlib stand-in for a real tokenizer -- monotonic in length, never zero
    for non-empty text, so a node always costs something to read.
    """
    if chars_per_token < 1:
        raise ValueError("chars_per_token must be >= 1")
    n = len(text)
    if n == 0:
        return 0
    return (n + chars_per_token - 1) // chars_per_token


@dataclass(frozen=True)
class TierProfile:
    """One candidate node and the token cost of reading it at each tier.

    ``costs`` is a 3-tuple ``(summary, detail, raw)``; each entry is the cost of
    opening the node *at that tier* (not cumulative). ``priority`` orders
    admission and deepening -- **lower is more important** (e.g. a retrieval
    rank). ``pinned`` nodes (passive/both recall mode) are always admitted.
    ``risk`` marks nodes whose summary is likely insufficient, so they are
    deepened toward raw before lower-risk peers.
    """

    node_id: str
    costs: Tuple[int, int, int]
    priority: int = 0
    pinned: bool = False
    risk: bool = False

    def __post_init__(self) -> None:
        if len(self.costs) != 3:
            raise ValueError("costs must be a 3-tuple (summary, detail, raw)")
        if any(c < 0 for c in self.costs):
            raise ValueError("costs must be non-negative")


def make_profile(
    node_id: str,
    summary: str,
    detailed_summary: str = "",
    raw: str = "",
    *,
    priority: int = 0,
    pinned: bool = False,
    risk: bool = False,
    chars_per_token: int = 4,
) -> TierProfile:
    """Build a :class:`TierProfile` by estimating token costs from text.

    A missing ``detailed_summary`` falls back to the summary's cost, and a
    missing ``raw`` falls back to the detail cost -- so an absent deeper tier is
    never *cheaper* than a shallower one, which keeps deepening monotonic.
    """
    c0 = estimate_tokens(summary, chars_per_token)
    c1 = estimate_tokens(detailed_summary, chars_per_token) if detailed_summary else c0
    c2 = estimate_tokens(raw, chars_per_token) if raw else c1
    return TierProfile(
        node_id=node_id,
        costs=(c0, max(c1, c0), max(c2, c1, c0)),
        priority=priority,
        pinned=pinned,
        risk=risk,
    )


@dataclass(frozen=True)
class PlannedRead:
    """The tier chosen for one admitted node, and what it costs there."""

    node_id: str
    tier: int
    tokens: int


@dataclass
class ReadPlan:
    reads: List[PlannedRead] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)
    total_tokens: int = 0
    budget: int = 0

    @property
    def remaining(self) -> int:
        return self.budget - self.total_tokens

    def tier_of(self, node_id: str) -> Optional[int]:
        for r in self.reads:
            if r.node_id == node_id:
                return r.tier
        return None

    def to_dict(self) -> dict:
        return {
            "reads": [
                {"node_id": r.node_id, "tier": r.tier, "tokens": r.tokens}
                for r in self.reads
            ],
            "dropped": list(self.dropped),
            "total_tokens": self.total_tokens,
            "budget": self.budget,
            "remaining": self.remaining,
        }


def _admission_key(p: TierProfile) -> Tuple[int, str]:
    # Priority first (lower = more important), then node_id for a stable order.
    return (p.priority, p.node_id)


def _escalation_key(p: TierProfile) -> Tuple[int, int, str]:
    # Risk-flagged nodes escalate first (they want raw), then by priority,
    # then node_id. All-deterministic.
    return (0 if p.risk else 1, p.priority, p.node_id)


def plan_reads(
    profiles: Sequence[TierProfile],
    budget: int,
    *,
    max_tier: int = RAW,
) -> ReadPlan:
    """Assign each node the deepest tier that fits a shared token budget.

    Two-phase greedy, fully deterministic:

    **Admit.** Pinned nodes are admitted at tier 0 first (in priority order),
    claiming budget unconditionally -- if even the pinned summaries overflow the
    budget the lowest-priority pinned nodes are dropped so the plan never
    exceeds the budget. Remaining nodes are then admitted at tier 0 in priority
    order until the next summary does not fit; everyone after that is dropped.

    **Deepen.** With the leftover budget, repeatedly take the escalation-ranked
    admitted node that can still move one tier deeper *and* whose next-tier cost
    delta fits, and deepen it by one tier. Risk-flagged nodes are deepened
    before others. Stops when no admitted node can deepen within budget.

    Returns a :class:`ReadPlan`; ``reads`` is sorted by ``(priority, node_id)``.
    """
    if budget < 0:
        raise ValueError("budget must be >= 0")
    if not 0 <= max_tier <= RAW:
        raise ValueError("max_tier must be in [0, 2]")

    by_id = {p.node_id: p for p in profiles}
    if len(by_id) != len(profiles):
        raise ValueError("duplicate node_id in profiles")

    ordered = sorted(profiles, key=_admission_key)
    pinned = [p for p in ordered if p.pinned]
    rest = [p for p in ordered if not p.pinned]

    tier: Dict[str, int] = {}
    spent = 0

    # Phase 1a: pinned nodes claim budget first, lowest priority dropped on
    # overflow (drop from the tail = least important pinned).
    admitted_pinned: List[TierProfile] = []
    for p in pinned:
        if spent + p.costs[SUMMARY] <= budget:
            tier[p.node_id] = SUMMARY
            spent += p.costs[SUMMARY]
            admitted_pinned.append(p)
    dropped_pinned = [p.node_id for p in pinned if p.node_id not in tier]

    # Phase 1b: remaining nodes at tier 0, priority order, until one won't fit.
    dropped_rest: List[str] = []
    stop = False
    for p in rest:
        if not stop and spent + p.costs[SUMMARY] <= budget:
            tier[p.node_id] = SUMMARY
            spent += p.costs[SUMMARY]
        else:
            stop = True
            dropped_rest.append(p.node_id)

    # Phase 2: deepen admitted nodes one tier at a time.
    admitted = [by_id[nid] for nid in tier]
    esc_order = sorted(admitted, key=_escalation_key)
    progress = True
    while progress:
        progress = False
        for p in esc_order:
            t = tier[p.node_id]
            if t >= max_tier:
                continue
            delta = p.costs[t + 1] - p.costs[t]
            if delta <= 0 or spent + delta <= budget:
                tier[p.node_id] = t + 1
                spent += max(0, delta)
                progress = True

    reads = [
        PlannedRead(node_id=nid, tier=t, tokens=by_id[nid].costs[t])
        for nid, t in tier.items()
    ]
    reads.sort(key=lambda r: (by_id[r.node_id].priority, r.node_id))

    return ReadPlan(
        reads=reads,
        dropped=sorted(dropped_pinned + dropped_rest),
        total_tokens=spent,
        budget=budget,
    )
