"""Terse meta-tag rendering for a memory context map.

Mined from CoMeT's orchestrator (``get_session_context`` / ``_to_row`` /
``_pick_highest`` / ``_render_with_origin_merge``). CoMeT tags every memory node
along four independent axes -- ORIGIN (where the content came from), ACT (what
kind of action the turn was), KIND (semantic flags), IMPORTANCE -- but a node
often carries *several* tags on one axis (bundle supersede, drifted action
flags). Rendering all of them turns the context map into noise, so the
orchestrator collapses each axis to its single highest-priority tag and prints
a compact ``(O:USER A:EDIT F:FEEDBACK I:H)`` block, then merges runs of
consecutive same-origin rows into one bundle line.

Two deterministic rules make the output terse without losing signal:

  * **One tag per axis, by priority.** Higher-priority tags win; ACT is dropped
    on user-authored rows (a user turn is self-describing); IMPORTANCE only
    surfaces the extremes H/L (MED is the default and pure noise).
  * **Consecutive-origin bundling.** A run of >= 2 rows sharing a non-USER
    origin is chunked into groups of at most ``max_merge`` and rendered as a
    single line with a merged id and semicolon-joined summaries.

Both are pure functions of the tag/row inputs -- transferable to any CAD memory
map that tags nodes by provenance (drawing / STEP import / solver run / user
edit) and needs a bounded, readable rendering. Priority maps mirror CoMeT's
defaults but are overridable so callers can bring their own tag namespace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

__all__ = [
    "ORIGIN_PRIORITY",
    "ACT_PRIORITY",
    "KIND_PRIORITY",
    "Row",
    "pick_highest",
    "short_tag_block",
    "render_rows",
]

# CoMeT's defaults (orchestrator._ORIGIN_PRIORITY etc.), kept overridable.
ORIGIN_PRIORITY: Dict[str, int] = {
    "ORIGIN:USER": 100,
    "ORIGIN:SUBAGENT_RESULT": 80,
    "ORIGIN:SESSION_HANDOFF": 75,
    "ORIGIN:PROJECT_GOAL": 70,
    "ORIGIN:TOOL_BUNDLE": 60,
    "ORIGIN:META_BUNDLE": 60,
    "ORIGIN:FILE_EDIT": 55,
    "ORIGIN:FILE_WRITE": 55,
    "ORIGIN:CODE": 55,
    "ORIGIN:TERMINAL_EXEC": 52,
    "ORIGIN:CROSS_SESSION_MESSAGE": 45,
    "ORIGIN:EXTERNAL": 30,
}

ACT_PRIORITY: Dict[str, int] = {
    "FLAG:ACT_FAIL": 100,
    "FLAG:ACT_EDIT": 80,
    "FLAG:ACT_EXECUTE": 70,
    "FLAG:ACT_DIAGNOSE": 60,
    "FLAG:ACT_FETCH": 40,
    "FLAG:ACT_PLAN": 30,
    "FLAG:ACT_DECIDE": 20,
    "FLAG:ACT_NONE": 0,
}

KIND_PRIORITY: Dict[str, int] = {
    "FLAG:SKILL": 100,
    "FLAG:WORKFLOW": 95,
    "FLAG:USER_REJECT": 80,
    "FLAG:USER_FEEDBACK": 70,
    "FLAG:PASSIVE": 10,
}


def pick_highest(tags: Sequence[str], priority: Mapping[str, int]) -> str:
    """Return the highest-priority tag from ``tags`` per ``priority``.

    Unknown tags score 0 and lose to any priority-mapped tag. Returns ``''``
    when no tag has positive priority. Ties break by tag string, so the choice
    is a pure function of the (unordered) tag set.
    """
    best_tag = ""
    best_score = 0
    for t in sorted(tags):
        score = priority.get(t, 0)
        if score > best_score:
            best_score = score
            best_tag = t
    return best_tag


def short_tag_block(
    tags: Sequence[str],
    *,
    origin_priority: Mapping[str, int] = ORIGIN_PRIORITY,
    act_priority: Mapping[str, int] = ACT_PRIORITY,
    kind_priority: Mapping[str, int] = KIND_PRIORITY,
) -> str:
    """Render the one-per-axis ``(O:.. A:.. F:.. I:..)`` block, or ``''``.

    ACT is suppressed on ``ORIGIN:USER`` rows; IMPORTANCE only shows H/L (MED
    suppressed). Empty when no axis contributes.
    """
    origin_tag = pick_highest(tags, origin_priority)
    act_tag = pick_highest(tags, act_priority)
    kind_tag = pick_highest(tags, kind_priority)

    importance = None
    for t in sorted(tags):
        if t.startswith("IMPORTANCE:"):
            importance = t[len("IMPORTANCE:"):].upper()
            break

    parts: List[str] = []
    if origin_tag:
        parts.append(f"O:{origin_tag[len('ORIGIN:'):]}")
    if act_tag and origin_tag != "ORIGIN:USER":
        parts.append(f"A:{act_tag[len('FLAG:ACT_'):]}")
    if kind_tag:
        parts.append(f"F:{kind_tag[len('FLAG:'):]}")
    if importance in ("HIGH", "LOW"):
        parts.append(f"I:{importance[0]}")

    return f"({' '.join(parts)})" if parts else ""


@dataclass(frozen=True)
class Row:
    """One memory-map row awaiting rendering."""

    node_id: str
    summary: str = ""
    trigger: str = ""
    tags: Sequence[str] = field(default_factory=tuple)
    recall_mode: str = "active"


def _origin_of(row: Row) -> str:
    return pick_highest(row.tags, ORIGIN_PRIORITY)


def _render_one(row: Row, origin_priority, act_priority, kind_priority) -> str:
    block = short_tag_block(
        row.tags,
        origin_priority=origin_priority,
        act_priority=act_priority,
        kind_priority=kind_priority,
    )
    tag_str = f"{block} " if block else ""
    prefix = "(passive) " if row.recall_mode in ("passive", "both") else ""
    return f"[{row.node_id}] {tag_str}{prefix}{row.summary} | {row.trigger}"


def render_rows(
    rows: Sequence[Row],
    *,
    max_merge: int = 3,
    origin_priority: Mapping[str, int] = ORIGIN_PRIORITY,
    act_priority: Mapping[str, int] = ACT_PRIORITY,
    kind_priority: Mapping[str, int] = KIND_PRIORITY,
) -> List[str]:
    """Render rows, bundling consecutive same-non-USER-origin runs.

    A run of >= 2 adjacent rows with the same origin (other than
    ``ORIGIN:USER``) is chunked into groups of at most ``max_merge`` and
    rendered as one line: a merged id (``prefix_h1+h2+h3`` from the id suffixes)
    and the group's summaries joined by ``'; '``, carrying the first row's tag
    block and trigger. Rows are consumed in the given order, so ordering is the
    caller's responsibility (chronological, in CoMeT); the transformation itself
    is deterministic.
    """
    if max_merge < 1:
        raise ValueError("max_merge must be >= 1")

    def render_one(r: Row) -> str:
        return _render_one(r, origin_priority, act_priority, kind_priority)

    out: List[str] = []
    i = 0
    n = len(rows)
    while i < n:
        row = rows[i]
        origin = _origin_of(row)
        if origin and origin != "ORIGIN:USER":
            j = i + 1
            while j < n and _origin_of(rows[j]) == origin:
                j += 1
            group = rows[i:j]
            if len(group) >= 2:
                for cs in range(0, len(group), max_merge):
                    chunk = group[cs:cs + max_merge]
                    if len(chunk) == 1:
                        out.append(render_one(chunk[0]))
                    else:
                        merged_nids = "+".join(r.node_id.split("_")[-1] for r in chunk)
                        first_prefix = "_".join(chunk[0].node_id.split("_")[:-1])
                        merged_id = (
                            f"{first_prefix}_{merged_nids}" if first_prefix else merged_nids
                        )
                        block = short_tag_block(
                            chunk[0].tags,
                            origin_priority=origin_priority,
                            act_priority=act_priority,
                            kind_priority=kind_priority,
                        )
                        tag_str = f"{block} " if block else ""
                        prefix = "(passive) " if chunk[0].recall_mode in ("passive", "both") else ""
                        merged_summaries = "; ".join(r.summary for r in chunk if r.summary)
                        out.append(
                            f"[{merged_id}] {tag_str}{prefix}{merged_summaries} | {chunk[0].trigger}"
                        )
                i = j
                continue
        out.append(render_one(row))
        i += 1
    return out
