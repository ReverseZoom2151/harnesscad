"""Orthographic three-view consistency and projection-geometry validity checks.

CReFT-CAD's TriView2CAD data engine enforces two constraint classes when
synthesising a sample, and its "View Matching" post-tuning task asks whether two
orthographic views correspond to the same 3D object. Both reduce to deterministic
geometry checks that need no learned model:

  * **Inter-View consistency** (the classical third-angle correspondence rule):
    measurements annotated in one view must exactly match their counterparts in
    the others. Front and Top share the X (width) extent; Front and Side share
    the Z (height) extent; Top and Side share the Y (depth) extent. Three
    orthographic views can only come from one consistent solid when these shared
    extents agree.

  * **Intra-View validity**: within one projection every component must form a
    gap- and overlap-free contour (topological closure / physical validity), and
    paired dimensions must obey domain engineering constraints (e.g. "Cross-Bridge
    Pier Spacing" < "Cap Beam Cross-Bridge Dimension").

This module implements both as pure predicates over the :mod:`creft_projection`
view/rect types plus a small dimension-constraint evaluator. It is the
deterministic core of the paper's View-Matching and validity signals; the VLM
that *reads* views from raster images is the learned, out-of-scope part.

Pure stdlib, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from drawings.creft_projection import FRONT, SIDE, TOP, Rect, View

# Which 3D axis each (view, 2D-axis) pair measures.
_AXIS = {
    (FRONT, "h"): "x", (FRONT, "v"): "z",
    (TOP, "h"): "x", (TOP, "v"): "y",
    (SIDE, "h"): "y", (SIDE, "v"): "z",
}

# The three shared-extent correspondences between view pairs.
# (view_a, axis_a, view_b, axis_b, shared_3d_axis)
CORRESPONDENCES: Tuple[Tuple[str, str, str, str, str], ...] = (
    (FRONT, "h", TOP, "h", "x"),   # width
    (FRONT, "v", SIDE, "v", "z"),  # height
    (TOP, "v", SIDE, "h", "y"),    # depth
)


def _extent(view: View, axis: str) -> float:
    return view.horizontal_extent() if axis == "h" else view.vertical_extent()


@dataclass(frozen=True)
class ConsistencyResult:
    consistent: bool
    mismatches: Tuple[Dict[str, object], ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, object]:
        return {"consistent": self.consistent,
                "mismatches": [dict(m) for m in self.mismatches]}


def check_view_consistency(views: Dict[str, View],
                           tol: float = 1e-6) -> ConsistencyResult:
    """Check the three shared-extent correspondences between front/top/side.

    Returns a :class:`ConsistencyResult`; ``mismatches`` records each violated
    correspondence with the two measured extents and the shared axis. A view set
    missing any of the three views is reported as inconsistent.
    """
    mismatches: List[Dict[str, object]] = []
    for va, axa, vb, axb, shared in CORRESPONDENCES:
        if va not in views or vb not in views:
            mismatches.append({"axis": shared, "reason": "missing_view",
                               "views": (va, vb)})
            continue
        ea = _extent(views[va], axa)
        eb = _extent(views[vb], axb)
        if abs(ea - eb) > tol:
            mismatches.append({"axis": shared, "views": (va, vb),
                               "extent_a": ea, "extent_b": eb,
                               "delta": abs(ea - eb)})
    return ConsistencyResult(consistent=not mismatches,
                             mismatches=tuple(mismatches))


def views_match(views_a: Dict[str, View], views_b: Dict[str, View],
                tol: float = 1e-6) -> bool:
    """View-Matching task: do two view sets describe the same-sized 3D object?

    Compares the overall (width, height, depth) implied by each set. Each set must
    itself be self-consistent and the three shared extents must agree between sets.
    """
    if not check_view_consistency(views_a, tol).consistent:
        return False
    if not check_view_consistency(views_b, tol).consistent:
        return False
    da = implied_dimensions(views_a)
    db = implied_dimensions(views_b)
    return all(abs(da[k] - db[k]) <= tol for k in ("width", "height", "depth"))


def implied_dimensions(views: Dict[str, View]) -> Dict[str, float]:
    """Overall (width=X, height=Z, depth=Y) implied by a view set.

    Reads each extent from whichever view carries it, preferring the front view
    for width/height and the top view for depth.
    """
    width = height = depth = 0.0
    if FRONT in views:
        width = views[FRONT].horizontal_extent()
        height = views[FRONT].vertical_extent()
    elif TOP in views:
        width = views[TOP].horizontal_extent()
    if TOP in views:
        depth = views[TOP].vertical_extent()
        if not width:
            width = views[TOP].horizontal_extent()
    if not height and SIDE in views:
        height = views[SIDE].vertical_extent()
    if not depth and SIDE in views:
        depth = views[SIDE].horizontal_extent()
    return {"width": width, "height": height, "depth": depth}


# --------------------------------------------------------------------------- #
# Intra-view validity: gap- and overlap-free contour closure
# --------------------------------------------------------------------------- #
def _overlap_area(a: Rect, b: Rect) -> float:
    ou = max(0.0, min(a.umax, b.umax) - max(a.u, b.u))
    ov = max(0.0, min(a.vmax, b.vmax) - max(a.v, b.v))
    return ou * ov


def contour_has_overlap(view: View, tol: float = 1e-9) -> bool:
    """True if any two rectangles overlap with positive area (interior overlap)."""
    rects = list(view.rects)
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _overlap_area(rects[i], rects[j]) > tol:
                return True
    return False


def contour_is_connected(view: View, tol: float = 1e-9) -> bool:
    """True if the view's rectangles form one edge-connected (gap-free) contour.

    Two rectangles are connected when their bounding rectangles touch or overlap
    (shared edge or interior). A single component means no gaps between parts.
    """
    rects = list(view.rects)
    n = len(rects)
    if n <= 1:
        return True
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = rects[i], rects[j]
            gap_u = max(a.u, b.u) - min(a.umax, b.umax)
            gap_v = max(a.v, b.v) - min(a.vmax, b.vmax)
            if gap_u <= tol and gap_v <= tol:
                adj[i].append(j)
                adj[j].append(i)
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for nb in adj[node]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == n


def contour_is_valid(view: View, tol: float = 1e-9) -> bool:
    """A view is valid when it is gap-free (connected) AND overlap-free."""
    return contour_is_connected(view, tol) and not contour_has_overlap(view, tol)


# --------------------------------------------------------------------------- #
# Paired-dimension engineering constraints
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DimensionConstraint:
    """A binary ordering constraint ``left <op> right`` between two named dims."""

    left: str
    op: str   # one of "<", "<=", ">", ">=", "=="
    right: str

    def evaluate(self, params: Dict[str, float], tol: float = 1e-9) -> bool:
        a = float(params[self.left])
        b = float(params[self.right])
        if self.op == "<":
            return a < b - tol
        if self.op == "<=":
            return a <= b + tol
        if self.op == ">":
            return a > b + tol
        if self.op == ">=":
            return a >= b - tol
        if self.op == "==":
            return abs(a - b) <= tol
        raise ValueError("unknown op %r" % (self.op,))


def check_dimension_constraints(params: Dict[str, float],
                                constraints: Sequence[DimensionConstraint],
                                tol: float = 1e-9) -> List[DimensionConstraint]:
    """Return the constraints violated by ``params`` (empty -> all satisfied).

    Skips constraints referencing a parameter absent from ``params`` (they cannot
    be evaluated) rather than raising, so partial parameter sets are safe.
    """
    violated: List[DimensionConstraint] = []
    for c in constraints:
        if c.left not in params or c.right not in params:
            continue
        if not c.evaluate(params, tol):
            violated.append(c)
    return violated
