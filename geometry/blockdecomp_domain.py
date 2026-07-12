"""Rectilinear planar domain representation for RL block decomposition.

From *Reinforcement Learning for Block Decomposition of CAD Models*
(DiPrete, Garimella, Garcia Cardona & Ray, LANL, AAAI-2022). The paper trains a
Soft-Actor-Critic agent to decompose a **planar CAD model with straight,
axis-aligned edges** into well-shaped rectangular **blocks** (quads) that can be
mapped to a canonical meshable cube (Sec. "Our Approach", "Methodology"). The
*learned* policy (actor/critic/value networks, SplineCNN GNN, replay buffer,
entropy maximisation) is external and NOT modelled here; the underlying
**geometry is fully deterministic** and is what this module provides.

A domain is an axis-aligned rectilinear region. The paper generates its 49 test
shapes "by combining 2 to 10 rectangles" (Sec. "Data Sets"), so this module
represents a shape on the **coordinate mesh** induced by the union of all vertex
x- and y-lines: the shape is the set of grid *cells* whose interior is inside the
region. This representation is exact for rectilinear shapes, is closed under the
paper's axis-aligned cut operation, and makes connectivity / area / corner
queries robust and deterministic.

Provided here:

  * ``Shape`` -- (xs, ys, inside-cell-set) with area, bounding box, width/height,
    aspect ratio (longest bbox side / shortest, >= 1), area-weighted centroid,
    4-connectivity test, cell membership;
  * boundary tracing into ordered corner loops (outer CCW, holes CW) via
    directed-edge cancellation, and corner classification into *convex* (interior
    90 deg) / *reentrant* (270 deg) -- the model vertices the agent cuts from;
  * ``is_rectangle`` / ``is_quad`` -- the terminal "all quadrilateral blocks"
    test (a filled rectangle: connected, single loop, bbox area == area);
  * ``classify_angle`` -- interior-angle type name (acute/right/obtuse/
    reentrant/straight) used by the local-observation state features;
  * constructors ``from_rectangles`` (the paper's shape generator) and
    ``from_polygon`` (an axis-aligned corner loop).

Pure stdlib; deterministic (no wall clock, no randomness).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

Vec2 = Tuple[float, float]
Cell = Tuple[int, int]

_EPS = 1e-9


def classify_angle(v1: Vec2, v2: Vec2) -> str:
    """Classify the interior corner formed by edge vectors ``v1``, ``v2``.

    ``v1`` points from the corner to the previous neighbour, ``v2`` to the next
    neighbour (both emanate from the corner). Returns one of ``"acute"``,
    ``"right"``, ``"obtuse"``, ``"reentrant"`` or ``"straight"`` -- the corner
    types listed in the paper's local observation (Sec. "Local Observation").
    """
    ax, ay = v1
    bx, by = v2
    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)
    if na < _EPS or nb < _EPS:
        return "straight"
    dot = (ax * bx + ay * by) / (na * nb)
    dot = max(-1.0, min(1.0, dot))
    ang = math.degrees(math.acos(dot))  # interior angle between the two edges
    if abs(ang - 180.0) < 1e-6:
        return "straight"
    if abs(ang - 90.0) < 1e-6:
        # distinguish convex 90 from reentrant 270 via the turn direction below
        return "right"
    if ang < 90.0:
        return "acute"
    return "obtuse"


@dataclass(frozen=True)
class Corner:
    """A model vertex on the boundary of a :class:`Shape`."""

    pos: Vec2
    prev: Vec2  # previous corner along the loop
    nxt: Vec2  # next corner along the loop
    interior_angle: float  # degrees (90 convex, 270 reentrant for rectilinear)
    corner_type: str  # "convex" or "reentrant"
    loop: int


@dataclass(frozen=True)
class Shape:
    """A rectilinear region as inside cells over a shared coordinate mesh."""

    xs: Tuple[float, ...]
    ys: Tuple[float, ...]
    cells: FrozenSet[Cell]

    # ---- construction ---------------------------------------------------
    @staticmethod
    def from_rectangles(rects: Sequence[Tuple[float, float, float, float]]) -> "Shape":
        """Build the union of axis-aligned rectangles ``(x0, y0, x1, y1)``."""
        if not rects:
            raise ValueError("need at least one rectangle")
        norm = []
        xset, yset = set(), set()
        for x0, y0, x1, y1 in rects:
            lx, hx = (x0, x1) if x0 <= x1 else (x1, x0)
            ly, hy = (y0, y1) if y0 <= y1 else (y1, y0)
            if hx - lx < _EPS or hy - ly < _EPS:
                continue
            norm.append((lx, ly, hx, hy))
            xset.update((lx, hx))
            yset.update((ly, hy))
        if not norm:
            raise ValueError("all rectangles are degenerate")
        xs = tuple(sorted(xset))
        ys = tuple(sorted(yset))
        cells = set()
        for i in range(len(xs) - 1):
            cx = 0.5 * (xs[i] + xs[i + 1])
            for j in range(len(ys) - 1):
                cy = 0.5 * (ys[j] + ys[j + 1])
                for lx, ly, hx, hy in norm:
                    if lx < cx < hx and ly < cy < hy:
                        cells.add((i, j))
                        break
        return Shape(xs, ys, frozenset(cells))

    @staticmethod
    def from_polygon(vertices: Sequence[Vec2]) -> "Shape":
        """Build a shape from an axis-aligned closed corner loop ``vertices``."""
        if len(vertices) < 4:
            raise ValueError("need at least 4 vertices")
        xs = tuple(sorted({v[0] for v in vertices}))
        ys = tuple(sorted({v[1] for v in vertices}))
        cells = set()
        for i in range(len(xs) - 1):
            cx = 0.5 * (xs[i] + xs[i + 1])
            for j in range(len(ys) - 1):
                cy = 0.5 * (ys[j] + ys[j + 1])
                if _point_in_polygon(cx, cy, vertices):
                    cells.add((i, j))
        return Shape(xs, ys, frozenset(cells))

    def with_cells(self, cells: FrozenSet[Cell]) -> "Shape":
        """A sub-shape on the same mesh restricted to ``cells``."""
        return Shape(self.xs, self.ys, frozenset(cells))

    # ---- basic measures -------------------------------------------------
    @property
    def is_empty(self) -> bool:
        return not self.cells

    @property
    def num_cells(self) -> int:
        return len(self.cells)

    def cell_inside(self, i: int, j: int) -> bool:
        return (i, j) in self.cells

    def area(self) -> float:
        total = 0.0
        for (i, j) in self.cells:
            total += (self.xs[i + 1] - self.xs[i]) * (self.ys[j + 1] - self.ys[j])
        return total

    def bbox(self) -> Tuple[float, float, float, float]:
        if not self.cells:
            raise ValueError("empty shape has no bbox")
        imin = min(i for i, _ in self.cells)
        imax = max(i for i, _ in self.cells)
        jmin = min(j for _, j in self.cells)
        jmax = max(j for _, j in self.cells)
        return (self.xs[imin], self.ys[jmin], self.xs[imax + 1], self.ys[jmax + 1])

    def width(self) -> float:
        x0, _, x1, _ = self.bbox()
        return x1 - x0

    def height(self) -> float:
        _, y0, _, y1 = self.bbox()
        return y1 - y0

    def aspect_ratio(self) -> float:
        """Longest bounding-box side / shortest (>= 1), the paper's R_i."""
        w, h = self.width(), self.height()
        lo, hi = (h, w) if w >= h else (w, h)
        if lo < _EPS:
            return float("inf")
        return hi / lo

    def centroid(self) -> Vec2:
        """Area-weighted centroid of the region."""
        ax = ay = tot = 0.0
        for (i, j) in self.cells:
            w = self.xs[i + 1] - self.xs[i]
            h = self.ys[j + 1] - self.ys[j]
            a = w * h
            ax += a * 0.5 * (self.xs[i] + self.xs[i + 1])
            ay += a * 0.5 * (self.ys[j] + self.ys[j + 1])
            tot += a
        if tot < _EPS:
            raise ValueError("empty shape has no centroid")
        return (ax / tot, ay / tot)

    # ---- connectivity ---------------------------------------------------
    def connected_components(self) -> List[FrozenSet[Cell]]:
        remaining = set(self.cells)
        comps: List[FrozenSet[Cell]] = []
        while remaining:
            seed = next(iter(remaining))
            stack = [seed]
            remaining.discard(seed)
            comp = {seed}
            while stack:
                i, j = stack.pop()
                for ni, nj in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    if (ni, nj) in remaining:
                        remaining.discard((ni, nj))
                        comp.add((ni, nj))
                        stack.append((ni, nj))
            comps.append(frozenset(comp))
        return comps

    def is_connected(self) -> bool:
        return len(self.connected_components()) == 1 if self.cells else False

    # ---- boundary and corners ------------------------------------------
    def boundary_loops(self) -> List[List[Vec2]]:
        """Ordered corner loops (outer CCW, holes CW) of the region boundary."""
        # Directed CCW edges around each inside cell; interior shared edges
        # cancel with their reverse, leaving oriented boundary edges.
        edges: Dict[Vec2, Vec2] = {}
        raw = set()
        for (i, j) in self.cells:
            bl = (self.xs[i], self.ys[j])
            br = (self.xs[i + 1], self.ys[j])
            tr = (self.xs[i + 1], self.ys[j + 1])
            tl = (self.xs[i], self.ys[j + 1])
            for a, b in ((bl, br), (br, tr), (tr, tl), (tl, bl)):
                if (b, a) in raw:
                    raw.discard((b, a))
                else:
                    raw.add((a, b))
        adj: Dict[Vec2, List[Vec2]] = {}
        for a, b in raw:
            adj.setdefault(a, []).append(b)
        for k in adj:
            adj[k].sort()
        loops: List[List[Vec2]] = []
        used = set()
        for start in sorted(adj.keys()):
            for target in adj[start]:
                if (start, target) in used:
                    continue
                loop = [start]
                cur, nxt = start, target
                while True:
                    used.add((cur, nxt))
                    loop.append(nxt)
                    cur = nxt
                    if cur == start:
                        break
                    options = [t for t in adj.get(cur, []) if (cur, t) not in used]
                    if not options:
                        break
                    nxt = _pick_next(loop[-2], cur, options)
                loops.append(_merge_collinear(loop[:-1]))
        return [lp for lp in loops if len(lp) >= 4]

    def corners(self) -> List[Corner]:
        result: List[Corner] = []
        for li, loop in enumerate(self.boundary_loops()):
            n = len(loop)
            orient = _signed_area(loop)  # >0 CCW (outer), <0 CW (hole)
            for k in range(n):
                p = loop[k]
                pv = loop[(k - 1) % n]
                nv = loop[(k + 1) % n]
                d_in = (p[0] - pv[0], p[1] - pv[1])
                d_out = (nv[0] - p[0], nv[1] - p[1])
                cross = d_in[0] * d_out[1] - d_in[1] * d_out[0]
                # For a CCW loop, a convex corner turns left (cross > 0).
                convex = (cross > 0) if orient > 0 else (cross < 0)
                result.append(
                    Corner(
                        pos=p,
                        prev=pv,
                        nxt=nv,
                        interior_angle=90.0 if convex else 270.0,
                        corner_type="convex" if convex else "reentrant",
                        loop=li,
                    )
                )
        return result

    # ---- terminal quad test --------------------------------------------
    def is_rectangle(self) -> bool:
        """True iff the region is a single filled axis-aligned rectangle."""
        if not self.cells or not self.is_connected():
            return False
        x0, y0, x1, y1 = self.bbox()
        return abs(self.area() - (x1 - x0) * (y1 - y0)) < _EPS

    # A "quad block" in this paper is exactly a rectangle.
    is_quad = is_rectangle

    def num_corners(self) -> int:
        return len(self.corners())


def _point_in_polygon(x: float, y: float, poly: Sequence[Vec2]) -> bool:
    inside = False
    n = len(poly)
    for k in range(n):
        x0, y0 = poly[k]
        x1, y1 = poly[(k + 1) % n]
        if (y0 > y) != (y1 > y):
            xint = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xint:
                inside = not inside
    return inside


def _pick_next(prev: Vec2, cur: Vec2, options: List[Vec2]) -> Vec2:
    """At a pinch vertex prefer going straight, else the first option."""
    d_in = (cur[0] - prev[0], cur[1] - prev[1])
    for o in options:
        d_out = (o[0] - cur[0], o[1] - cur[1])
        if d_in[0] * d_out[1] - d_in[1] * d_out[0] == 0 and (
            d_in[0] * d_out[0] + d_in[1] * d_out[1] > 0
        ):
            return o
    return options[0]


def _merge_collinear(loop: List[Vec2]) -> List[Vec2]:
    n = len(loop)
    if n < 3:
        return loop
    out: List[Vec2] = []
    for k in range(n):
        p = loop[k]
        a = loop[(k - 1) % n]
        b = loop[(k + 1) % n]
        d1 = (p[0] - a[0], p[1] - a[1])
        d2 = (b[0] - p[0], b[1] - p[1])
        if d1[0] * d2[1] - d1[1] * d2[0] == 0:
            continue  # collinear: p is not a corner
        out.append(p)
    return out


def _signed_area(loop: Sequence[Vec2]) -> float:
    s = 0.0
    n = len(loop)
    for k in range(n):
        x0, y0 = loop[k]
        x1, y1 = loop[(k + 1) % n]
        s += x0 * y1 - x1 * y0
    return 0.5 * s
