"""HistCAD execution pipeline: loop/face reconstruction + replay validity.

HistCAD stores sketch primitives as an unordered set and, at execution time,
reconstructs loops and faces from geometric connectivity via rule-based
algorithms before translating each command into kernel calls (paper Sec.
III-C). This module implements the deterministic parts of that pipeline:

  * :func:`reconstruct_loops` — chain lines/arcs into closed loops by matching
    quantised endpoints (circles are self-closing loops), the connectivity
    inference that replaces an explicit face-loop hierarchy;
  * :func:`hierarchical_loops` — Algorithm 1: sort loops by area (descending)
    and classify each as an outer contour or an inner hole by containment,
    yielding a loop dictionary; plus a 2D oriented bounding box per sketch;
  * :func:`replay_validity` — walk a :class:`~reconstruction.histcad_sequence`
    ModelingSequence and report whether every feature is executable (closed
    loops exist, extrusion length non-zero, first boolean is ``create``, etc.).

Pure geometry, stdlib-only, deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

_QUANT = 1_000_000


def _qp(x: float, y: float) -> Tuple[int, int]:
    return (int(round(x * _QUANT)), int(round(y * _QUANT)))


# ---------------------------------------------------------------------------
# Loop reconstruction from unordered primitives
# ---------------------------------------------------------------------------
@dataclass
class LoopInfo:
    prim_indices: Tuple[int, ...]
    vertices: Tuple[Tuple[float, float], ...]
    closed: bool
    area: float
    is_circle: bool = False


def _poly_area(pts: Sequence[Tuple[float, float]]) -> float:
    """Signed polygon area via the shoelace formula; returns absolute area."""
    n = len(pts)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def reconstruct_loops(primitives: Sequence) -> List[LoopInfo]:
    """Group unordered primitives into closed loops by endpoint connectivity.

    Circles form their own closed loop. Lines/arcs are chained: primitives
    that share a quantised endpoint belong to the same connected component; a
    component whose every vertex has even degree (>= 2) is reported as closed.
    Deterministic: components and their traversal follow primitive index order.
    """
    loops: List[LoopInfo] = []

    # circles are standalone loops
    edge_prims: List[Tuple[int, Tuple[int, int], Tuple[int, int]]] = []
    for idx, p in enumerate(primitives):
        kind = getattr(p, "kind", None)
        if kind == "circle":
            r = abs(getattr(p, "r"))
            area = 3.141592653589793 * r * r
            loops.append(LoopInfo((idx,), ((p.cx, p.cy),), True, area, True))
        else:
            eps = p.endpoints()
            if len(eps) != 2:
                continue
            a = _qp(*eps[0])
            b = _qp(*eps[1])
            edge_prims.append((idx, a, b))

    # union-find over quantised vertices
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def find(v):
        parent.setdefault(v, v)
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    def union(u, v):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv

    for _, a, b in edge_prims:
        find(a)
        find(b)
        union(a, b)

    # group edges by component root
    comps: Dict[Tuple[int, int], List[int]] = {}
    for i, (idx, a, b) in enumerate(edge_prims):
        comps.setdefault(find(a), []).append(i)

    for root in sorted(comps.keys()):
        members = comps[root]
        degree: Dict[Tuple[int, int], int] = {}
        prim_idx: List[int] = []
        verts: List[Tuple[float, float]] = []
        seen_v = set()
        for m in members:
            idx, a, b = edge_prims[m]
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
            prim_idx.append(idx)
            p = primitives[idx]
            eps = p.endpoints()
            for e in eps:
                qv = _qp(*e)
                if qv not in seen_v:
                    seen_v.add(qv)
                    verts.append(e)
        closed = len(degree) >= 3 and all(d % 2 == 0 and d >= 2
                                          for d in degree.values())
        area = _poly_area(verts) if closed else 0.0
        loops.append(LoopInfo(tuple(sorted(prim_idx)), tuple(verts),
                              closed, area, False))
    return loops


# ---------------------------------------------------------------------------
# Algorithm 1: hierarchical loop extraction + OBB
# ---------------------------------------------------------------------------
@dataclass
class LoopNode:
    loop_id: int
    is_outer: bool
    holes: Tuple[int, ...]


def _bbox(pts: Sequence[Tuple[float, float]]):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_inside(inner, outer) -> bool:
    return (inner[0] >= outer[0] and inner[1] >= outer[1]
            and inner[2] <= outer[2] and inner[3] <= outer[3])


def hierarchical_loops(loops: Sequence[LoopInfo]):
    """Algorithm 1: classify loops into outer contours and inner holes.

    Returns ``(loop_dict, obb)`` where ``loop_dict`` maps each outer loop id to
    a :class:`LoopNode` (with its contained holes) and ``obb`` is the axis-
    aligned bounding box ``(minx, miny, maxx, maxy)`` over all closed loops.
    Loops are processed in descending area order (the paper's sort).
    """
    closed = [(i, lp) for i, lp in enumerate(loops) if lp.closed]
    order = sorted(closed, key=lambda t: (-t[1].area, t[0]))

    boxes: Dict[int, Tuple[float, float, float, float]] = {}
    for i, lp in closed:
        boxes[i] = _bbox(lp.vertices)

    loop_dict: Dict[int, LoopNode] = {}
    holes_of: Dict[int, List[int]] = {}
    for i, lp in order:
        is_outer = True
        for outer_id in list(loop_dict.keys()):
            if _bbox_inside(boxes[i], boxes[outer_id]):
                holes_of[outer_id].append(i)
                is_outer = False
                break
        if is_outer:
            loop_dict[i] = LoopNode(i, True, ())
            holes_of[i] = []
    for oid, node in loop_dict.items():
        loop_dict[oid] = LoopNode(oid, True, tuple(sorted(holes_of[oid])))

    all_pts: List[Tuple[float, float]] = []
    for _, lp in closed:
        all_pts.extend(lp.vertices)
    obb = _bbox(all_pts) if all_pts else (0.0, 0.0, 0.0, 0.0)
    return loop_dict, obb


# ---------------------------------------------------------------------------
# Replay validity
# ---------------------------------------------------------------------------
@dataclass
class FeatureReport:
    index: int
    valid: bool
    n_closed_loops: int
    errors: Tuple[str, ...]


@dataclass
class ReplayReport:
    valid: bool
    features: Tuple[FeatureReport, ...]

    @property
    def n_invalid(self) -> int:
        return sum(1 for f in self.features if not f.valid)


def replay_validity(seq) -> ReplayReport:
    """Walk a ModelingSequence and check each feature is executable.

    A feature is valid iff: at least one closed loop can be reconstructed from
    its sketch primitives, its extrusion length is non-zero, and its boolean op
    is legal for its position (the first feature must be ``create``; later
    features must NOT be ``create``).
    """
    reports: List[FeatureReport] = []
    for i, feat in enumerate(seq.features):
        errors: List[str] = []
        loops = reconstruct_loops(feat.sketch.primitives)
        n_closed = sum(1 for lp in loops if lp.closed)
        if n_closed == 0:
            errors.append("no-closed-loop")
        if feat.extrusion.length == 0:
            errors.append("zero-length-extrusion")
        if i == 0 and feat.boolean != "create":
            errors.append("first-feature-not-create")
        if i > 0 and feat.boolean == "create":
            errors.append("redundant-create")
        reports.append(FeatureReport(i, not errors, n_closed, tuple(errors)))
    overall = all(f.valid for f in reports) and bool(reports)
    return ReplayReport(overall, tuple(reports))
