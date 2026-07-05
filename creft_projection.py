"""Forward orthographic projection of an axis-aligned solid to three views.

CReFT-CAD (NeurIPS 2025) frames CAD understanding as *orthographic projection
reasoning*: a 3D model is presented as three standard views (front, top, side)
and the model must recover its parameters. Its TriView2CAD data-engine renders a
3D model to three orthographic views before annotating dimensions. The rendering
itself is deterministic geometry — this module implements that forward step for a
solid modelled as a union of axis-aligned boxes (the modular, box-composed
"prefabricated pier" primitives the paper targets).

Conventions (Z-up, right-handed), matching the third-angle views a CAD sheet uses::

    Front view  — look along -Y onto the X-Z plane; horizontal = X, vertical = Z
    Top view    — look along -Z onto the X-Y plane; horizontal = X, vertical = Y
    Side view   — look along -X onto the Y-Z plane; horizontal = Y, vertical = Z

Each box projects to an axis-aligned rectangle in each view. A *view* is the set
of projected rectangles together with the view's overall 2D bounding extent. The
correspondence between shared axes across views (front/top share X, front/side
share Z, top/side share Y) is exactly the classical third-angle rule and is what
:mod:`creft_view_consistency` checks.

Pure stdlib, deterministic (no wall clock, no randomness).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

# View names.
FRONT = "front"
TOP = "top"
SIDE = "side"
VIEW_NAMES: Tuple[str, ...] = (FRONT, TOP, SIDE)


@dataclass(frozen=True)
class Box:
    """An axis-aligned box: origin corner (x, y, z) and positive sizes."""

    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float

    def __post_init__(self) -> None:
        for name in ("dx", "dy", "dz"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError("box %s must be positive, got %r"
                                 % (name, getattr(self, name)))

    @property
    def xmax(self) -> float:
        return self.x + self.dx

    @property
    def ymax(self) -> float:
        return self.y + self.dy

    @property
    def zmax(self) -> float:
        return self.z + self.dz


@dataclass(frozen=True)
class Rect:
    """An axis-aligned 2D rectangle: (u, v) low corner + positive (du, dv)."""

    u: float
    v: float
    du: float
    dv: float

    @property
    def umax(self) -> float:
        return self.u + self.du

    @property
    def vmax(self) -> float:
        return self.v + self.dv

    def area(self) -> float:
        return self.du * self.dv


@dataclass(frozen=True)
class View:
    """A projected orthographic view.

    ``horizontal`` / ``vertical`` name the 3D axis mapped to each 2D axis (used by
    the inter-view consistency checker). ``rects`` are the projected primitives.
    """

    name: str
    horizontal: str
    vertical: str
    rects: Tuple[Rect, ...] = field(default_factory=tuple)

    def bbox(self) -> Rect:
        """Overall bounding rectangle of the view (empty -> zero rect)."""
        if not self.rects:
            return Rect(0.0, 0.0, 0.0, 0.0)
        umin = min(r.u for r in self.rects)
        vmin = min(r.v for r in self.rects)
        umax = max(r.umax for r in self.rects)
        vmax = max(r.vmax for r in self.rects)
        return Rect(umin, vmin, umax - umin, vmax - vmin)

    def horizontal_extent(self) -> float:
        return self.bbox().du

    def vertical_extent(self) -> float:
        return self.bbox().dv


def _project_box(box: Box, view: str) -> Rect:
    if view == FRONT:
        return Rect(box.x, box.z, box.dx, box.dz)
    if view == TOP:
        return Rect(box.x, box.y, box.dx, box.dy)
    if view == SIDE:
        return Rect(box.y, box.z, box.dy, box.dz)
    raise ValueError("unknown view %r" % (view,))


def project_view(boxes: Sequence[Box], view: str) -> View:
    """Project ``boxes`` onto a single named view (front/top/side)."""
    axes = {FRONT: ("x", "z"), TOP: ("x", "y"), SIDE: ("y", "z")}
    if view not in axes:
        raise ValueError("unknown view %r" % (view,))
    h, v = axes[view]
    rects = tuple(_project_box(b, view) for b in boxes)
    return View(name=view, horizontal=h, vertical=v, rects=rects)


def project_three_views(boxes: Sequence[Box]) -> Dict[str, View]:
    """Project a box solid to all three orthographic views."""
    boxes = list(boxes)
    return {name: project_view(boxes, name) for name in VIEW_NAMES}


def model_bbox(boxes: Sequence[Box]) -> Tuple[float, float, float]:
    """Overall (dx, dy, dz) extent of the 3D model (empty -> zeros)."""
    boxes = list(boxes)
    if not boxes:
        return (0.0, 0.0, 0.0)
    xmin = min(b.x for b in boxes)
    ymin = min(b.y for b in boxes)
    zmin = min(b.z for b in boxes)
    xmax = max(b.xmax for b in boxes)
    ymax = max(b.ymax for b in boxes)
    zmax = max(b.zmax for b in boxes)
    return (xmax - xmin, ymax - ymin, zmax - zmin)


def _segments_from_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge a set of 1D closed intervals into disjoint covered spans."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged: List[Tuple[float, float]] = [ordered[0]]
    for lo, hi in ordered[1:]:
        last_lo, last_hi = merged[-1]
        if lo <= last_hi + 1e-12:
            merged[-1] = (last_lo, max(last_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def silhouette_area(view: View) -> float:
    """Area covered by the union of a view's rectangles (overlaps counted once).

    Deterministic sweep-line over the distinct vertical edges: at each x-slab the
    covered vertical length is the merged union of the active rectangles' spans.
    """
    rects = list(view.rects)
    if not rects:
        return 0.0
    xs = sorted({r.u for r in rects} | {r.umax for r in rects})
    total = 0.0
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        width = x1 - x0
        if width <= 0:
            continue
        mid = 0.5 * (x0 + x1)
        spans = [(r.v, r.vmax) for r in rects if r.u <= mid <= r.umax]
        covered = sum(hi - lo for lo, hi in _segments_from_intervals(spans))
        total += width * covered
    return total
