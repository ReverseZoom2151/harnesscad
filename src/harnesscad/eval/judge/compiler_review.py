"""Compiler-as-a-review: a deterministic structural grader for sketch-extrude
op sequences.

A CAD compiler classifies
each failure into one of four modes -- ``format`` / ``geometry`` / ``extrusion``
/ ``boolean`` -- and renders a feedback message that can be re-injected into the
generator's prompt (the CRM "generate -> review -> refine" loop).

Every category except the ones that genuinely need a B-rep kernel can be
decided by structural inspection of the op list alone. Concretely we validate:

* **format** -- the sequence is a list of typed ops, terminated by ``end``,
  with only known opcodes and required fields present.
* **geometry** -- each ``sketch`` op has at least one loop, every loop has >= 3
  vertices (or is a circle), and no loop is degenerate (zero-area / collinear).
* **extrusion** -- every ``extrude`` has a non-zero depth and a preceding
  sketch to extrude.
* **boolean** -- every boolean references a known kind and has an existing base
  solid to combine against (a ``cut`` / ``intersect`` with no base is invalid).

The op schema is:

    [
        {"type": "sketch",  "loops": [{"points": [[x, y], ...]}, ...]},
        {"type": "extrude", "depth": 1.0, "boolean": "union"},
        ...
        {"type": "end"},
    ]

This is a *checkable-property* grader (not a learned judge): the same op list
always yields the same verdict, so it is safe as a reward signal or a gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "ReviewResult",
    "VALID_OPCODES",
    "BOOLEAN_KINDS",
    "review_sequence",
    "feedback_message",
]

#: Opcodes the structural reviewer understands.
VALID_OPCODES = ("sketch", "extrude", "boolean", "end")

#: Boolean kinds a solid combination may declare.
BOOLEAN_KINDS = ("union", "cut", "intersect")

#: The four failure categories, ordered by the stage at which they surface.
CATEGORIES = ("format", "geometry", "extrusion", "boolean")


@dataclass(frozen=True)
class ReviewResult:
    """Verdict of a structural compiler review.

    ``ok`` is True iff the sequence passes every structural check. On failure
    ``category`` is one of :data:`CATEGORIES`, ``op_index`` points at the
    offending op (``None`` for whole-sequence faults), and ``reason`` is a
    terse human/LLM-readable explanation.
    """

    ok: bool
    category: Optional[str] = None
    op_index: Optional[int] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "category": self.category,
            "op_index": self.op_index,
            "reason": self.reason,
        }


def _fail(category: str, reason: str, op_index: Optional[int] = None) -> ReviewResult:
    return ReviewResult(ok=False, category=category, op_index=op_index, reason=reason)


def _loop_is_degenerate(points: Sequence[Sequence[float]]) -> bool:
    """True iff a polygon loop encloses (numerically) zero area.

    Uses the shoelace formula. A loop whose vertices are all collinear or
    coincident has zero signed area and cannot bound a face.
    """
    n = len(points)
    if n < 3:
        return True
    area2 = 0.0
    for i in range(n):
        x1, y1 = points[i][0], points[i][1]
        x2, y2 = points[(i + 1) % n][0], points[(i + 1) % n][1]
        area2 += x1 * y2 - x2 * y1
    return abs(area2) < 1e-9


def _review_sketch(op: Dict[str, Any], i: int) -> Optional[ReviewResult]:
    loops = op.get("loops")
    if not loops:
        return _fail("geometry", f"sketch #{i} has no loop", i)
    for li, loop in enumerate(loops):
        # A circle loop is specified by centre+radius rather than a polyline.
        if "radius" in loop or loop.get("type") == "circle":
            r = float(loop.get("radius", 0.0))
            if r <= 0.0:
                return _fail("geometry", f"sketch #{i} loop #{li} has non-positive radius", i)
            continue
        pts = loop.get("points") or []
        if len(pts) < 3:
            return _fail("geometry", f"sketch #{i} loop #{li} has fewer than 3 vertices", i)
        if any(len(p) < 2 for p in pts):
            return _fail("geometry", f"sketch #{i} loop #{li} has a malformed point", i)
        if _loop_is_degenerate(pts):
            return _fail("geometry", f"sketch #{i} loop #{li} is degenerate (zero area)", i)
    return None


def review_sequence(operations: Sequence[Dict[str, Any]]) -> ReviewResult:
    """Structurally review a sketch-extrude op sequence. Deterministic.

    Returns a passing :class:`ReviewResult` iff the sequence is well-formed and
    every op is structurally buildable; otherwise the first failure, classified
    into :data:`CATEGORIES`.
    """
    if not isinstance(operations, (list, tuple)):
        return _fail("format", "sequence is not a list of ops")
    if not operations:
        return _fail("format", "empty op sequence")

    # --- format: typed ops, known opcodes, terminated by <end> ---
    for i, op in enumerate(operations):
        if not isinstance(op, dict):
            return _fail("format", f"op #{i} is not a dict", i)
        if "type" not in op:
            return _fail("format", f"op #{i} has no 'type' field", i)
        if op["type"] not in VALID_OPCODES:
            return _fail("format", f"op #{i} unknown opcode {op['type']!r}", i)
    if operations[-1].get("type") != "end":
        return _fail("format", "missing <end> terminator")

    # --- semantic pass: geometry / extrusion / boolean ---
    have_pending_sketch = False
    solid_count = 0
    for i, op in enumerate(operations[:-1]):
        kind = op["type"]
        if kind == "sketch":
            bad = _review_sketch(op, i)
            if bad is not None:
                return bad
            have_pending_sketch = True
        elif kind == "extrude":
            if not have_pending_sketch:
                return _fail("extrusion", f"extrude #{i} has no preceding sketch", i)
            depth = float(op.get("depth", 0.0))
            if abs(depth) < 1e-9:
                return _fail("extrusion", f"extrude #{i} has zero depth", i)
            boolean = op.get("boolean", "union")
            if boolean not in BOOLEAN_KINDS:
                return _fail("boolean", f"extrude #{i} unknown boolean {boolean!r}", i)
            if boolean in ("cut", "intersect") and solid_count == 0:
                return _fail("boolean", f"extrude #{i} {boolean} has no base solid", i)
            have_pending_sketch = False
            solid_count += 1
        elif kind == "boolean":
            bkind = op.get("kind", "union")
            if bkind not in BOOLEAN_KINDS:
                return _fail("boolean", f"boolean #{i} unknown kind {bkind!r}", i)
            if solid_count == 0:
                return _fail("boolean", f"boolean #{i} has no base solid", i)

    if solid_count == 0:
        return _fail("geometry", "empty solid (no extrusion produced)")
    return ReviewResult(ok=True)


#: Feedback templates, keyed by category, ready to append to a re-prompt.
_FEEDBACK: Dict[str, str] = {
    "format": (
        "The previous CAD sequence is malformed: {reason}. "
        "Re-emit a complete sketch-and-extrude sequence terminated by an <end> op."
    ),
    "geometry": (
        "The sketch in operation #{op} is geometrically invalid: {reason}. "
        "Re-design the sketch so every loop is closed and non-degenerate."
    ),
    "extrusion": (
        "Extrusion #{op} is invalid: {reason}. "
        "Ensure the extrusion has a non-zero depth and a valid preceding sketch."
    ),
    "boolean": (
        "Boolean operation #{op} is invalid: {reason}. "
        "Ensure a base solid exists before a cut/intersect and the kind is one of "
        "union/cut/intersect."
    ),
}


def feedback_message(result: ReviewResult) -> str:
    """Render a re-promptable feedback string from a failing review.

    Empty string when ``result.ok``. The message is the deterministic diagnostic
    the CRM loop injects verbatim into the next generation prompt.
    """
    if result.ok:
        return ""
    tpl = _FEEDBACK.get(result.category or "format", "Compiler error: {reason}.")
    op = result.op_index if result.op_index is not None else "?"
    return tpl.format(op=op, reason=result.reason or "unknown")
