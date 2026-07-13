"""Deterministic least-squares fitting of PPA sketch primitives from 2D points
(Wang et al., "Parametric Primitive Analysis of CAD Sketches with Vision
Transformer", IEEE T-II 2024).

The paper's primitive network *regresses* line / circle / arc / point parameters
from a rasterised sketch. The learned ViT regressor is out of scope, but the
classical geometric-fitting counterpart it approximates -- "given a cluster of 2D
sample points, recover the best-fit parametric primitive" -- is fully deterministic
and is exactly the kind of building block the paper's Related-Work calls "the
long-standing challenge of fitting parameterized primitives" (Sec. II-A). This
module provides those closed-form fits, returning :class:`reconstruction.ppa_primitive.Primitive`
objects in the paper's Table-I parameterisation:

  * :func:`fit_point`  -- centroid of the points.
  * :func:`fit_line`   -- total-least-squares (orthogonal-regression) line via the
    2x2 covariance eigenvector; endpoints are the extreme projections onto that line.
  * :func:`fit_circle` -- Kasa algebraic circle fit (solve the linear normal
    equations for centre + radius).
  * :func:`fit_arc`    -- Kasa circle for the support, then start / mid / end points
    resampled on the circle at the angular span of the data (Table-I 3-point arc).

Each fit also returns an RMS residual so callers can pick the best-matching primitive
type. Pure stdlib -- a small symmetric-eigenvalue helper and a 3x3 Gaussian solve.
"""

from __future__ import annotations

import math

from harnesscad.domain.reconstruction import ppa_primitive as pp

Point = tuple[float, float]


def _centroid(points):
    n = len(points)
    return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n)


def fit_point(points) -> tuple[pp.Primitive, float]:
    """Fit a point primitive: the centroid. Residual = RMS distance to centroid."""
    if not points:
        raise ValueError("need at least one point")
    cx, cy = _centroid(points)
    rms = math.sqrt(sum((x - cx) ** 2 + (y - cy) ** 2 for x, y in points) / len(points))
    return pp.point((cx, cy)), rms


def fit_line(points) -> tuple[pp.Primitive, float]:
    """Total-least-squares line fit; endpoints are extreme projections onto the line.

    Uses the eigenvector of the 2x2 covariance matrix for the largest eigenvalue as
    the line direction (orthogonal regression, robust to vertical lines). Residual is
    the RMS orthogonal distance.
    """
    if len(points) < 2:
        raise ValueError("need at least two points to fit a line")
    cx, cy = _centroid(points)
    sxx = sum((x - cx) ** 2 for x, y in points)
    syy = sum((y - cy) ** 2 for x, y in points)
    sxy = sum((x - cx) * (y - cy) for x, y in points)
    # Largest-eigenvalue eigenvector of [[sxx, sxy], [sxy, syy]].
    dx, dy = _dominant_eigenvector(sxx, sxy, syy)
    # Project points onto the direction; endpoints = min/max projection.
    ts = [(x - cx) * dx + (y - cy) * dy for x, y in points]
    tmin, tmax = min(ts), max(ts)
    p1 = (cx + tmin * dx, cy + tmin * dy)
    p2 = (cx + tmax * dx, cy + tmax * dy)
    # Orthogonal residual: distance to the infinite line through centroid.
    nx, ny = -dy, dx
    rms = math.sqrt(sum(((x - cx) * nx + (y - cy) * ny) ** 2 for x, y in points)
                    / len(points))
    return pp.line(p1, p2), rms


def _dominant_eigenvector(a: float, b: float, c: float) -> tuple[float, float]:
    """Unit eigenvector for the larger eigenvalue of ``[[a, b], [b, c]]``."""
    # Eigenvalues of a symmetric 2x2: (a+c)/2 +/- sqrt(((a-c)/2)^2 + b^2).
    tr = a + c
    disc = math.sqrt(((a - c) / 2.0) ** 2 + b * b)
    lam = tr / 2.0 + disc  # larger eigenvalue
    # Eigenvector: solve (a - lam) x + b y = 0.
    if abs(b) > 1e-12:
        vx, vy = lam - c, b
    elif a >= c:
        vx, vy = 1.0, 0.0
    else:
        vx, vy = 0.0, 1.0
    norm = math.hypot(vx, vy)
    if norm < 1e-15:
        return 1.0, 0.0
    return vx / norm, vy / norm


def _solve3(mat, rhs):
    """Solve a 3x3 linear system by Gaussian elimination with partial pivoting."""
    a = [list(row) + [r] for row, r in zip(mat, rhs)]
    n = 3
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-15:
            raise ValueError("singular system")
        a[col], a[piv] = a[piv], a[col]
        for r in range(n):
            if r == col:
                continue
            f = a[r][col] / a[col][col]
            for k in range(col, n + 1):
                a[r][k] -= f * a[col][k]
    return [a[r][n] / a[r][r] for r in range(n)]


def _kasa_circle(points):
    """Kasa algebraic circle fit: returns ``(cx, cy, r)``.

    Minimises sum ( (x-cx)^2 + (y-cy)^2 - r^2 )^2 by solving the linear system in
    ``(A, B, C)`` where the circle is ``x^2 + y^2 + A x + B y + C = 0``.
    """
    n = len(points)
    sx = sy = sxx = syy = sxy = sxz = syz = sz = 0.0
    for x, y in points:
        z = x * x + y * y
        sx += x
        sy += y
        sxx += x * x
        syy += y * y
        sxy += x * y
        sxz += x * z
        syz += y * z
        sz += z
    mat = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, float(n)]]
    rhs = [-sxz, -syz, -sz]
    A, B, C = _solve3(mat, rhs)
    cx, cy = -A / 2.0, -B / 2.0
    r2 = cx * cx + cy * cy - C
    r = math.sqrt(r2) if r2 > 0 else 0.0
    return cx, cy, r


def fit_circle(points) -> tuple[pp.Primitive, float]:
    """Kasa algebraic circle fit. Residual = RMS of ``|dist(p, centre) - r|``."""
    if len(points) < 3:
        raise ValueError("need at least three points to fit a circle")
    cx, cy, r = _kasa_circle(points)
    rms = math.sqrt(sum((math.hypot(x - cx, y - cy) - r) ** 2 for x, y in points)
                    / len(points))
    return pp.circle((cx, cy), r), rms


def fit_arc(points) -> tuple[pp.Primitive, float]:
    """Fit an arc: Kasa circle for support, then 3-point (start, mid, end) on-circle.

    ``start`` and ``end`` are the points at the extremes of the data's angular span
    about the fitted centre; ``mid`` is the on-circle point at the span midpoint (in
    the direction that keeps the samples inside the arc). Residual is the circle RMS.
    """
    if len(points) < 3:
        raise ValueError("need at least three points to fit an arc")
    cx, cy, r = _kasa_circle(points)
    angs = [math.atan2(y - cy, x - cx) for x, y in points]
    # Determine the minimal arc covering all sample angles: find the largest angular
    # gap between consecutive sorted angles; the arc spans the complement of that gap.
    order = sorted(angs)
    best_gap = -1.0
    gap_at = 0
    for i in range(len(order)):
        nxt = order[(i + 1) % len(order)] + (2 * math.pi if i + 1 == len(order) else 0)
        gap = nxt - order[i]
        if gap > best_gap:
            best_gap = gap
            gap_at = i
    start_ang = order[(gap_at + 1) % len(order)]
    span = 2 * math.pi - best_gap  # arc sweep (ccw from start)
    mid_ang = start_ang + span / 2.0
    end_ang = start_ang + span

    def on_circle(a):
        return (cx + r * math.cos(a), cy + r * math.sin(a))

    prim = pp.arc(on_circle(start_ang), on_circle(mid_ang), on_circle(end_ang))
    rms = math.sqrt(sum((math.hypot(x - cx, y - cy) - r) ** 2 for x, y in points)
                    / len(points))
    return prim, rms


def fit_best(points) -> tuple[pp.Primitive, float]:
    """Fit every applicable type and return the primitive with the smallest RMS.

    Point (1+ pts), line (2+), circle/arc (3+). Ties broken by fixed type order
    line < circle < arc < point for determinism.
    """
    candidates: list[tuple[float, int, pp.Primitive]] = []
    order = {pp.LINE: 0, pp.CIRCLE: 1, pp.ARC: 2, pp.POINT: 3}
    n = len(points)
    if n >= 2:
        prim, res = fit_line(points)
        candidates.append((res, order[prim.ptype], prim))
    if n >= 3:
        for fn in (fit_circle, fit_arc):
            try:
                prim, res = fn(points)
            except ValueError:
                continue  # collinear support -> circle/arc undefined; skip
            candidates.append((res, order[prim.ptype], prim))
    prim, res = fit_point(points)
    candidates.append((res, order[prim.ptype], prim))
    candidates.sort(key=lambda c: (c[0], c[1]))
    best = candidates[0]
    return best[2], best[0]
