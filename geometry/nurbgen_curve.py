"""Rational B-spline (NURBS) curve evaluation and tessellation (NURBGen).

Usama, Khan, Stricker & Afzal, *NURBGen: High-Fidelity Text-to-CAD Generation
through LLM-Driven NURBS Modeling* (AAAI 2026), Eq. 1-2.

NURBGen emits each curve/face as control points, a degree, a knot vector and
rational weights.  This module implements the deterministic evaluation of a
NURBS *curve* (paper Eq. 1):

    C(u) = sum_i N_{i,p}(u) w_i P_i / sum_i N_{i,p}(u) w_i,   u in [u_p, u_{n+1}]

built on the Cox-de Boor basis functions from :mod:`numeric.nurbs_basis`.  The
existing ``geometry.dreamcad_rational_bezier`` covers only the Bernstein/Bezier
case (a single span, no knot vector); this adds the full non-uniform rational
B-spline curve, its analytic derivative/tangent, and NURBS -> polyline
tessellation.

A NURBS curve here is ``(control_points, weights, degree, knots)`` where
``control_points[i]`` are ``d``-tuples, ``weights[i] > 0``, and ``knots`` is a
non-decreasing vector of length ``len(control_points) + degree + 1``.

Pure-Python stdlib, deterministic.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from numeric.nurbs_basis import (
    basis_derivatives,
    basis_functions,
    find_span,
)

Point = Tuple[float, ...]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _check(control_points: Sequence[Point], weights: Sequence[float],
           degree: int, knots: Sequence[float]) -> Tuple[int, int]:
    npts = len(control_points)
    if npts == 0:
        raise ValueError("need at least one control point")
    if degree < 1:
        raise ValueError("degree must be >= 1")
    if npts < degree + 1:
        raise ValueError("need at least degree + 1 control points")
    if len(weights) != npts:
        raise ValueError("weights length must match control points")
    if any(w <= 0.0 for w in weights):
        raise ValueError("weights must be strictly positive")
    n = npts - 1
    if len(knots) != n + degree + 2:
        raise ValueError(
            "knot vector length %d != n + p + 2 = %d"
            % (len(knots), n + degree + 2))
    dim = len(control_points[0])
    for pt in control_points:
        if len(pt) != dim:
            raise ValueError("all control points must share a dimension")
    return n, dim


# ---------------------------------------------------------------------------
# Point evaluation (paper Eq. 1)
# ---------------------------------------------------------------------------

def curve_point(control_points: Sequence[Point], weights: Sequence[float],
                degree: int, knots: Sequence[float], u: float) -> Point:
    """Evaluate the NURBS curve ``C(u)`` (paper Eq. 1).

    Uses the projective (homogeneous) form: accumulate ``w_i N_i`` weighted
    control coordinates and the scalar ``sum w_i N_i``, then divide.
    """
    n, dim = _check(control_points, weights, degree, knots)
    span = find_span(n, degree, u, knots)
    N = basis_functions(span, u, degree, knots)
    numer = [0.0] * dim
    denom = 0.0
    for k in range(degree + 1):
        i = span - degree + k
        wN = weights[i] * N[k]
        denom += wN
        pt = control_points[i]
        for c in range(dim):
            numer[c] += wN * pt[c]
    if abs(denom) < 1e-14:
        raise ValueError("degenerate weight sum at u=%r" % u)
    return tuple(numer[c] / denom for c in range(dim))


# ---------------------------------------------------------------------------
# Derivatives via homogeneous coordinates (A4.2 style)
# ---------------------------------------------------------------------------

def curve_derivatives(control_points: Sequence[Point],
                      weights: Sequence[float], degree: int,
                      knots: Sequence[float], u: float,
                      order: int = 1) -> List[Point]:
    """Return ``[C(u), C'(u), ..., C^(order)(u)]`` of the NURBS curve.

    Computes derivatives of the homogeneous curve ``A(u) = sum w_i N_i P_i`` and
    the weight function ``w(u) = sum w_i N_i``, then applies the rational
    quotient rule (Leibniz):  ``C^(k) = (A^(k) - sum_{j=1..k} C(k,j) w^(j)
    C^(k-j)) / w``.
    """
    if order < 0:
        raise ValueError("order must be >= 0")
    n, dim = _check(control_points, weights, degree, knots)
    du = min(order, degree)
    span = find_span(n, degree, u, knots)
    ders = basis_derivatives(span, u, degree, knots, du)

    # Homogeneous derivatives: Aders[k] is a dim-vector, wders[k] a scalar.
    Aders = [[0.0] * dim for _ in range(order + 1)]
    wders = [0.0] * (order + 1)
    for k in range(du + 1):
        for local in range(degree + 1):
            i = span - degree + local
            wN = weights[i] * ders[k][local]
            wders[k] += wN
            pt = control_points[i]
            for c in range(dim):
                Aders[k][c] += wN * pt[c]

    if abs(wders[0]) < 1e-14:
        raise ValueError("degenerate weight sum at u=%r" % u)

    result: List[Point] = []
    ck = [[0.0] * dim for _ in range(order + 1)]
    for k in range(order + 1):
        v = list(Aders[k])
        for j in range(1, k + 1):
            binom = math.comb(k, j)
            for c in range(dim):
                v[c] -= binom * wders[j] * ck[k - j][c]
        for c in range(dim):
            ck[k][c] = v[c] / wders[0]
        result.append(tuple(ck[k]))
    return result


def curve_tangent(control_points: Sequence[Point], weights: Sequence[float],
                  degree: int, knots: Sequence[float], u: float) -> Point:
    """Unit tangent vector of the NURBS curve at ``u``.

    Raises if the first derivative vanishes (a cusp / degenerate parameter).
    """
    d = curve_derivatives(control_points, weights, degree, knots, u, 1)[1]
    length = math.sqrt(sum(c * c for c in d))
    if length < 1e-14:
        raise ValueError("zero-length tangent at u=%r" % u)
    return tuple(c / length for c in d)


# ---------------------------------------------------------------------------
# Tessellation: NURBS curve -> polyline
# ---------------------------------------------------------------------------

def tessellate_curve(control_points: Sequence[Point],
                     weights: Sequence[float], degree: int,
                     knots: Sequence[float], samples: int = 32
                     ) -> List[Point]:
    """Sample the NURBS curve into a polyline of ``samples + 1`` points.

    Parameters sweep uniformly over the valid domain ``[knots[p],
    knots[n+1]]``.  This is the deterministic NURBS -> polyline step NURBGen's
    B-rep viewer would use for display/point-cloud sampling.
    """
    n, _ = _check(control_points, weights, degree, knots)
    if samples < 1:
        raise ValueError("samples must be >= 1")
    u0 = knots[degree]
    u1 = knots[n + 1]
    pts: List[Point] = []
    for k in range(samples + 1):
        u = u0 + (u1 - u0) * (k / samples)
        pts.append(curve_point(control_points, weights, degree, knots, u))
    return pts


def polyline_length(points: Sequence[Point]) -> float:
    """Total Euclidean length of a polyline (sum of segment lengths)."""
    total = 0.0
    for a, b in zip(points, points[1:]):
        total += math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    return total


# ---------------------------------------------------------------------------
# A ready-made analytic case: the NURBS quarter/full circle
# ---------------------------------------------------------------------------

def nurbs_circle_quadrant(radius: float = 1.0) -> Tuple[
        List[Point], List[float], int, List[float]]:
    """Return a rational quadratic NURBS representing a 90-degree circular arc.

    Control points ``(r,0), (r,r), (0,r)`` with weights ``1, 1/sqrt2, 1`` on the
    clamped knot vector ``[0,0,0,1,1,1]`` trace an exact quarter circle -- the
    textbook demonstration that NURBS (unlike polynomial Beziers) represent
    conics exactly.  Used as an analytic ground truth in tests.
    """
    r = float(radius)
    cps: List[Point] = [(r, 0.0), (r, r), (0.0, r)]
    w = [1.0, math.sqrt(2.0) / 2.0, 1.0]
    knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    return cps, w, 2, knots
