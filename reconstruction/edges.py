"""Tolerance-aware normalization of extracted 2D edges."""

from __future__ import annotations

import math

from .model import Edge2D


def projection_feature(edge: Edge2D, tolerance: float) -> str:
    if edge.kind != "line":
        return "A"
    dx, dy = edge.end[0] - edge.start[0], edge.end[1] - edge.start[1]
    if abs(dy) <= tolerance:
        return "H"
    if abs(dx) <= tolerance:
        return "V"
    return "I"


def _q(value: float, tolerance: float) -> int:
    return round(value / tolerance)


def _key(edge: Edge2D, tolerance: float):
    points = tuple((_q(x, tolerance), _q(y, tolerance)) for x, y in edge.points)
    if edge.kind == "line":
        points = tuple(sorted(points))
    return edge.view, edge.kind, points


def _collinear(a: Edge2D, b: Edge2D, tolerance: float) -> bool:
    ax, ay = a.start
    bx, by = a.end
    cross = lambda p: (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax)
    length = max(math.dist(a.start, a.end), tolerance)
    return abs(cross(b.start)) / length <= tolerance and abs(cross(b.end)) / length <= tolerance


def _shared(a: Edge2D, b: Edge2D, tolerance: float) -> bool:
    return any(math.dist(x, y) <= tolerance
               for x in (a.start, a.end) for y in (b.start, b.end))


def normalize_edges(edges, tolerance: float) -> tuple[Edge2D, ...]:
    """Dedupe and repeatedly merge vertex-sharing collinear line segments."""
    unique: dict[object, Edge2D] = {}
    for edge in edges:
        if edge.kind == "line" and math.dist(edge.start, edge.end) <= tolerance:
            continue
        unique.setdefault(_key(edge, tolerance), edge)
    work = sorted(unique.values(), key=lambda e: (_key(e, tolerance), e.source_id))
    changed = True
    while changed:
        changed = False
        for i, left in enumerate(work):
            if left.kind != "line":
                continue
            for j in range(i + 1, len(work)):
                right = work[j]
                if right.kind != "line" or left.view != right.view:
                    continue
                if _shared(left, right, tolerance) and _collinear(left, right, tolerance):
                    points = (left.start, left.end, right.start, right.end)
                    pair = max(((a, b) for a in points for b in points),
                               key=lambda p: (math.dist(*p), p))
                    merged = Edge2D(left.view, "line", pair,
                                    left.hidden and right.hidden,
                                    "+".join(filter(None, (left.source_id, right.source_id))))
                    work = work[:i] + [merged] + work[i + 1:j] + work[j + 1:]
                    work.sort(key=lambda e: (_key(e, tolerance), e.source_id))
                    changed = True
                    break
            if changed:
                break
    return tuple(work)
