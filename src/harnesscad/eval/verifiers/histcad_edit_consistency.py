"""HistCAD parametric-edit-consistency check.

HistCAD's central claim is that *explicit* geometric constraints make edits
propagate correctly: when a parameter changes, related geometry updates while
constraints are preserved; without constraints the same edit breaks semantic
relations (tangency, equality, concentricity) even if loops stay closed (paper
Sec. III-D3, Fig. 4).

This module makes that claim testable deterministically. It provides:

  * :func:`constraint_residual` — a numeric residual for each of the ten
    constraint types given concrete primitive geometry (0 == satisfied);
  * :func:`check_constraints` — evaluate all constraints of a sketch, returning
    satisfied / violated with residuals;
  * :func:`edit_consistency` — apply a parameter edit, then report whether the
    edited sketch still satisfies its constraints (constraint-preserving edit)
    versus the same edit ignoring constraints (which produces non-zero
    residuals on dependent primitives).

Works on ``reconstruction.histcad_sequence`` primitives (Line / Circle / Arc)
and Constraint records. Stdlib-only, deterministic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad.domain.reconstruction.sequences.histcad_sequence import Line, Circle, Arc, Constraint

_TOL = 1e-6


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _line_dir(l: Line) -> Tuple[float, float]:
    return (l.x2 - l.x1, l.y2 - l.y1)


def _norm(v: Tuple[float, float]) -> float:
    return math.hypot(v[0], v[1])


def _prim_endpoints(p) -> List[Tuple[float, float]]:
    ep = p.endpoints()
    return [tuple(e) for e in ep]


def _center(p):
    if isinstance(p, Circle):
        return (p.cx, p.cy)
    if isinstance(p, Arc):
        # circumcentre of the three points
        ax, ay = p.xs, p.ys
        bx, by = p.xm, p.ym
        cx, cy = p.xe, p.ye
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < _TOL:
            return ((ax + cx) / 2.0, (ay + cy) / 2.0)
        ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay)
              + (cx * cx + cy * cy) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx)
              + (cx * cx + cy * cy) * (bx - ax)) / d
        return (ux, uy)
    return None


def _radius(p):
    if isinstance(p, Circle):
        return abs(p.r)
    if isinstance(p, Arc):
        c = _center(p)
        return math.hypot(p.xs - c[0], p.ys - c[1])
    return None


# ---------------------------------------------------------------------------
# Per-constraint residuals
# ---------------------------------------------------------------------------
def constraint_residual(ctype: str, prims: Sequence) -> float:
    """Return a non-negative residual for one constraint (0 == satisfied)."""
    if ctype == "horizontal":
        l = prims[0]
        return abs(l.y2 - l.y1)
    if ctype == "vertical":
        l = prims[0]
        return abs(l.x2 - l.x1)
    if ctype == "parallel":
        d1, d2 = _line_dir(prims[0]), _line_dir(prims[1])
        return abs(d1[0] * d2[1] - d1[1] * d2[0])  # cross product
    if ctype in ("perpendicular", "normal"):
        d1, d2 = _line_dir(prims[0]), _line_dir(prims[1])
        return abs(d1[0] * d2[0] + d1[1] * d2[1])  # dot product
    if ctype == "equal":
        # equal length (lines) or equal radius (circles/arcs)
        a, b = prims[0], prims[1]
        ra, rb = _radius(a), _radius(b)
        if ra is not None and rb is not None:
            return abs(ra - rb)
        la = _norm(_line_dir(a)) if isinstance(a, Line) else 0.0
        lb = _norm(_line_dir(b)) if isinstance(b, Line) else 0.0
        return abs(la - lb)
    if ctype == "concentric":
        ca, cb = _center(prims[0]), _center(prims[1])
        return math.hypot(ca[0] - cb[0], ca[1] - cb[1])
    if ctype == "coincident":
        # nearest endpoint pair distance between two primitives
        ea = _prim_endpoints(prims[0])
        eb = _prim_endpoints(prims[1])
        if not ea or not eb:
            return 0.0
        return min(math.hypot(pa[0] - pb[0], pa[1] - pb[1])
                   for pa in ea for pb in eb)
    if ctype == "tangent":
        # circle/arc tangent to a line: |dist(center,line) - radius|
        circ = next((p for p in prims if _radius(p) is not None), None)
        line = next((p for p in prims if isinstance(p, Line)), None)
        if circ is None:
            # circle-circle tangency: dist(centers) == r1 + r2 or |r1 - r2|
            ca, cb = _center(prims[0]), _center(prims[1])
            ra, rb = _radius(prims[0]), _radius(prims[1])
            dist = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
            return min(abs(dist - (ra + rb)), abs(dist - abs(ra - rb)))
        if line is None:
            return 0.0
        c = _center(circ)
        r = _radius(circ)
        dx, dy = _line_dir(line)
        denom = math.hypot(dx, dy)
        if denom < _TOL:
            return 0.0
        # distance from center to infinite line through the segment
        num = abs(dy * (c[0] - line.x1) - dx * (c[1] - line.y1))
        return abs(num / denom - r)
    if ctype == "fix":
        return 0.0  # a fix constraint is always trivially satisfied in-place
    raise ValueError(f"unknown constraint type: {ctype!r}")


# ---------------------------------------------------------------------------
# Whole-sketch evaluation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConstraintCheck:
    ctype: str
    refs: Tuple[int, ...]
    residual: float
    satisfied: bool


@dataclass(frozen=True)
class ConsistencyReport:
    all_satisfied: bool
    max_residual: float
    checks: Tuple[ConstraintCheck, ...]

    @property
    def violated(self) -> Tuple[ConstraintCheck, ...]:
        return tuple(c for c in self.checks if not c.satisfied)


def check_constraints(primitives: Sequence,
                      constraints: Sequence,
                      tol: float = _TOL) -> ConsistencyReport:
    """Evaluate every constraint against concrete geometry."""
    checks: List[ConstraintCheck] = []
    max_res = 0.0
    for c in constraints:
        ctype = getattr(c, "ctype", None) or c[0]
        refs = tuple(getattr(c, "refs", None) if getattr(c, "refs", None) is not None else c[1])
        prims = [primitives[r] for r in refs]
        res = constraint_residual(ctype, prims)
        max_res = max(max_res, res)
        checks.append(ConstraintCheck(ctype, refs, res, res <= tol))
    return ConsistencyReport(all(c.satisfied for c in checks), max_res, tuple(checks))


# ---------------------------------------------------------------------------
# Edit propagation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EditResult:
    before: ConsistencyReport
    after: ConsistencyReport
    consistent: bool  # True iff constraints preserved after the edit


def edit_consistency(primitives: Sequence,
                     constraints: Sequence,
                     edited: Sequence,
                     tol: float = _TOL) -> EditResult:
    """Compare constraint satisfaction before vs after an edit.

    ``edited`` is the full primitive list after applying a parametric edit
    (optionally with constraint-driven propagation already applied by the
    caller). ``consistent`` is True iff the edited geometry still satisfies all
    constraints — i.e. the edit was constraint-preserving. An edit that ignores
    constraints leaves dependent primitives stale and produces a non-zero
    residual, so ``consistent`` is False.
    """
    before = check_constraints(primitives, constraints, tol)
    after = check_constraints(edited, constraints, tol)
    return EditResult(before, after, after.all_satisfied)


def propagate_equal_radius(primitives: Sequence,
                           constraints: Sequence,
                           source_index: int) -> List:
    """Propagate a source circle's radius to all ``equal``/``concentric`` peers.

    A minimal, deterministic associative-edit example: after editing the
    ``source_index`` circle, every circle joined to it by an ``equal``
    constraint is updated to the same radius. Returns a new primitive list.
    Demonstrates constraint-preserving propagation (paper Fig. 4).
    """
    prims = list(primitives)
    src = prims[source_index]
    if not isinstance(src, Circle):
        return prims
    target_r = abs(src.r)
    # build adjacency over 'equal' constraints
    peers = set()
    for c in constraints:
        ctype = getattr(c, "ctype", None) or c[0]
        refs = tuple(getattr(c, "refs", None) if getattr(c, "refs", None) is not None else c[1])
        if ctype == "equal" and source_index in refs:
            for r in refs:
                if r != source_index:
                    peers.add(r)
    for r in peers:
        p = prims[r]
        if isinstance(p, Circle):
            prims[r] = replace(p, r=target_r)
    return prims
