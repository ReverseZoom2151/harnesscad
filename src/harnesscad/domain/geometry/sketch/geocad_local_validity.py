"""Local-part validity for GeoCAD (Zhang et al. 2025, Sec. 4.1 "Validity").

GeoCAD's Prediction-Validity (PV) metric requires that a generated local part be a
well-formed loop (paper Sec. 3.1 + Sec. 4.1):

    "filtering out duplicates and discarding invalid ones (i.e., those that are not
     closed loops or involve intersecting line segments)"
    "Predicted local parts must form closed loops and must not contain intersecting
     line segments."

This module implements the deterministic geometric predicate behind that filter for
polyline loops: (a) the loop is *closed* (its ordered vertices form a cycle) and
(b) the loop is *simple* (no two non-adjacent edges cross, and adjacent edges only
meet at their shared endpoint). Together these are the closed-form component of PV;
the "renders to a valid 3D shape" part depends on the CAD kernel and is out of scope.

Pure computational geometry -- deterministic, integer-friendly, no learned model.
"""

from __future__ import annotations

from dataclasses import dataclass

Point = tuple[float, float]


def _orient(a: Point, b: Point, c: Point) -> float:
    """Signed area sign of triangle abc (>0 ccw, <0 cw, 0 collinear)."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, p: Point) -> bool:
    """Whether collinear point ``p`` lies on segment ab."""
    return (
        min(a[0], b[0]) - 1e-12 <= p[0] <= max(a[0], b[0]) + 1e-12
        and min(a[1], b[1]) - 1e-12 <= p[1] <= max(a[1], b[1]) + 1e-12
    )


def segments_properly_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    """Whether segments ab and cd cross (proper crossing or collinear overlap)."""
    d1 = _orient(c, d, a)
    d2 = _orient(c, d, b)
    d3 = _orient(a, b, c)
    d4 = _orient(a, b, d)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and \
            d1 != 0 and d2 != 0 and d3 != 0 and d4 != 0:
        return True
    # Collinear overlap counts as an intersection of line segments.
    if d1 == 0 and _on_segment(c, d, a):
        return True
    if d2 == 0 and _on_segment(c, d, b):
        return True
    if d3 == 0 and _on_segment(a, b, c):
        return True
    if d4 == 0 and _on_segment(a, b, d):
        return True
    return False


@dataclass(frozen=True)
class ValidityReport:
    """Why a loop is / isn't valid."""

    valid: bool
    closed: bool
    simple: bool
    reason: str = ""


def is_closed(vertices: list[Point], tol: float = 1e-9) -> bool:
    """A vertex list is closed if it has >= 3 distinct vertices forming a cycle.

    The cycle is implicit: edge i connects v_i to v_{i+1 mod n}. A trailing vertex
    equal to the first (explicit closure) is tolerated and dropped conceptually.
    """
    v = list(vertices)
    if len(v) >= 2 and abs(v[0][0] - v[-1][0]) <= tol and abs(v[0][1] - v[-1][1]) <= tol:
        v = v[:-1]
    if len(v) < 3:
        return False
    # No zero-length edge.
    n = len(v)
    for i in range(n):
        a, b = v[i], v[(i + 1) % n]
        if abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol:
            return False
    return True


def _normalise(vertices: list[Point], tol: float = 1e-9) -> list[Point]:
    v = list(vertices)
    if len(v) >= 2 and abs(v[0][0] - v[-1][0]) <= tol and abs(v[0][1] - v[-1][1]) <= tol:
        v = v[:-1]
    return v


def is_simple_polygon(vertices: list[Point]) -> bool:
    """Whether the closed polygon has no self-intersections."""
    v = _normalise(vertices)
    n = len(v)
    if n < 3:
        return False
    for i in range(n):
        a, b = v[i], v[(i + 1) % n]
        for j in range(i + 1, n):
            c, d = v[j], v[(j + 1) % n]
            # Skip edges sharing a vertex (adjacent edges).
            if j == i:
                continue
            if (j + 1) % n == i or (i + 1) % n == j:
                continue
            if segments_properly_intersect(a, b, c, d):
                return False
    return True


def check_loop(vertices: list[Point]) -> ValidityReport:
    """Full deterministic validity check: closed AND non-self-intersecting."""
    closed = is_closed(vertices)
    if not closed:
        return ValidityReport(False, False, False, "loop is not closed")
    simple = is_simple_polygon(vertices)
    if not simple:
        return ValidityReport(False, True, False, "loop has intersecting segments")
    return ValidityReport(True, True, True, "")


def prediction_validity_rate(loops: list[list[Point]]) -> float:
    """Fraction of predicted polyline loops that pass :func:`check_loop` (PV, Sec. 4.1)."""
    if not loops:
        return 0.0
    good = sum(1 for lp in loops if check_loop(lp).valid)
    return good / len(loops)
