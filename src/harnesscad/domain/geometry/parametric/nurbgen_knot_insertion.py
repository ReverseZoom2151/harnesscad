"""Knot insertion and refinement for NURBS curves (NURBGen).

Usama, Khan, Stricker & Afzal, *NURBGen: High-Fidelity Text-to-CAD Generation
through LLM-Driven NURBS Modeling* (AAAI 2026).

Knot insertion is the fundamental NURBS refinement operation: it adds a new knot
(and one new control point) *without changing the curve's shape*.  NURBGen's
pipeline relies on it implicitly -- OCCT's ``BRepBuilderAPI_NurbsConvert`` and
``Geom_BSplineSurface`` re-parameterise faces, and knot refinement is how a
NURBS is subdivided for tessellation, split at a parameter, or raised to a
target continuity/multiplicity before B-rep export.

This module implements Boehm's single-knot-insertion algorithm (NURBS Book A5.1)
on the *homogeneous* control points ``(w_i x_i, w_i y_i, ..., w_i)`` so it works
correctly for rational (weighted) curves, then projects back.  Repeated
insertion gives knot refinement and Bezier decomposition of a span.

Curve tuple convention matches :mod:`geometry.nurbgen_curve`:
``(control_points, weights, degree, knots)``.  Pure-Python stdlib, deterministic.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from harnesscad.domain.numeric.nurbs_basis import find_span, knot_multiplicities

Point = Tuple[float, ...]


def _to_homogeneous(control_points: Sequence[Point],
                    weights: Sequence[float]) -> List[List[float]]:
    hom = []
    for pt, w in zip(control_points, weights):
        hom.append([w * c for c in pt] + [w])
    return hom


def _from_homogeneous(hom: Sequence[Sequence[float]]
                      ) -> Tuple[List[Point], List[float]]:
    pts: List[Point] = []
    ws: List[float] = []
    for h in hom:
        w = h[-1]
        if abs(w) < 1e-14:
            raise ValueError("zero homogeneous weight")
        pts.append(tuple(h[c] / w for c in range(len(h) - 1)))
        ws.append(w)
    return pts, ws


def knot_span_multiplicity(knots: Sequence[float], u: float,
                           tol: float = 1e-12) -> int:
    """Return how many times ``u`` already appears in the knot vector."""
    return sum(1 for k in knots if abs(k - u) <= tol)


def insert_knot(control_points: Sequence[Point], weights: Sequence[float],
                degree: int, knots: Sequence[float], u: float, times: int = 1
                ) -> Tuple[List[Point], List[float], int, List[float]]:
    """Insert knot ``u`` ``times`` times (Boehm, NURBS Book A5.1).

    Returns a new ``(control_points, weights, degree, knots)`` describing the
    *same* curve with ``times`` extra control points.  The existing
    multiplicity plus ``times`` must not exceed ``degree`` (that would break the
    curve's continuity/definition).
    """
    if times < 1:
        raise ValueError("times must be >= 1")
    p = degree
    n = len(control_points) - 1
    if len(weights) != n + 1:
        raise ValueError("weights length must match control points")
    if len(knots) != n + p + 2:
        raise ValueError("knot vector has wrong length")
    s = knot_span_multiplicity(knots, u)
    if s + times > p:
        raise ValueError(
            "resulting multiplicity %d would exceed degree %d" % (s + times, p))
    if not (knots[p] - 1e-12 <= u <= knots[n + 1] + 1e-12):
        raise ValueError("u outside the curve's parameter domain")

    k = find_span(n, p, u, knots)
    # If u equals an existing knot, find_span may return the span start; ensure
    # k is the index of the last knot <= u.
    while k + 1 < len(knots) and knots[k + 1] <= u + 1e-12:
        k += 1

    Pw = _to_homogeneous(control_points, weights)
    dim = len(Pw[0])

    UP = list(knots)
    UQ = UP[:k + 1] + [u] * times + UP[k + 1:]

    Qw: List[List[float]] = [None] * (n + times + 1)
    for i in range(k - p + 1):
        Qw[i] = list(Pw[i])
    for i in range(k - s, n + 1):
        Qw[i + times] = list(Pw[i])

    # Temporary array of affected control points.
    Rw = [list(Pw[k - p + i]) for i in range(p - s + 1)]

    L = 0
    for j in range(1, times + 1):
        L = k - p + j
        for i in range(p - j - s + 1):
            # NURBS Book A5.1 alpha formula:
            alpha_den = UP[k + i + 1] - UP[L + i]
            alpha = 0.0 if abs(alpha_den) < 1e-14 else (u - UP[L + i]) / alpha_den
            Rw[i] = [alpha * Rw[i + 1][d] + (1.0 - alpha) * Rw[i][d]
                     for d in range(dim)]
        Qw[L] = list(Rw[0])
        Qw[k + times - j - s] = list(Rw[p - j - s])
    # Load the remaining altered control points from the auxiliary array Rw
    # (NURBS Book A5.1 final loop).
    for i in range(L + 1, k - s):
        Qw[i] = list(Rw[i - L])

    new_pts, new_w = _from_homogeneous(Qw)
    return new_pts, new_w, p, UQ


def refine_knots(control_points: Sequence[Point], weights: Sequence[float],
                 degree: int, knots: Sequence[float], new_knots: Sequence[float]
                 ) -> Tuple[List[Point], List[float], int, List[float]]:
    """Insert each value in ``new_knots`` (knot refinement) one at a time.

    Convenience wrapper: repeatedly calls :func:`insert_knot`.  The resulting
    curve is geometrically identical, with a denser control net.
    """
    cp, w, p, U = list(control_points), list(weights), degree, list(knots)
    for u in new_knots:
        cp, w, p, U = insert_knot(cp, w, p, U, u, 1)
    return cp, w, p, U


def decompose_span_to_bezier(control_points: Sequence[Point],
                             weights: Sequence[float], degree: int,
                             knots: Sequence[float], u: float
                             ) -> Tuple[List[Point], List[float], int,
                                        List[float]]:
    """Raise the interior knot ``u`` to full multiplicity ``degree``.

    After this the curve has a ``C^0`` join at ``u`` -- the standard first step
    of Bezier decomposition / curve splitting used before B-rep export.
    """
    s = knot_span_multiplicity(knots, u)
    need = degree - s
    if need <= 0:
        return list(control_points), list(weights), degree, list(knots)
    return insert_knot(control_points, weights, degree, knots, u, need)


def distinct_interior_knots(knots: Sequence[float], degree: int
                            ) -> List[float]:
    """Distinct interior knot values (strictly inside the clamped ends)."""
    pairs = knot_multiplicities(knots)
    if len(pairs) <= 2:
        return []
    return [v for v, _ in pairs[1:-1]]
