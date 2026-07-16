"""NURBS curve, surface and closed-loop offsetting with an honesty contract.

Ported from kerf-main ``geom/offset.py`` (kerf-cad-core, GK-30/31/32), with the
supporting routines it imports from kerf's ``geom/nurbs.py`` (``make_circle_
nurbs``, ``make_arc_nurbs``, ``make_line_nurbs``) and ``geom/curve_toolkit.py``
(``_chord_params``, ``_pt_knots_from_params``, ``fit_curve``, ``interp_curve``)
folded in, since the harness has no least-squares B-spline fitter of its own.

What this module provides (kerf's public surface):

  * :func:`offset_curve`   -- planar curve offset by a signed distance ``d``
    along the right-side normal ``cross(plane_normal, tangent)``.  Analytic
    exact cases (straight line, kerf's 9-point rational circle, and circular
    arcs) are detected and offset exactly with ``actual_max_deviation = 0.0``
    and no refit; general curves are sampled, offset pointwise and refit by
    least squares within ``tol``.
  * :func:`offset_surface` -- offset along the analytic unit normal.  Planes
    and the standard rational revolution sphere are exact; general surfaces
    are grid-sampled, offset and refit with a two-stage (rows-then-columns)
    tensor-product least-squares fit.
  * :func:`offset_loop`    -- offset a closed planar loop of curves preserving
    connectivity: convex corners get an exact rational arc fillet of radius
    ``|d|``; concave corners are extended / trimmed to the intersection of the
    adjacent offset segments.

Honesty contract (kerf's): every approximating result reports the measured
``actual_max_deviation`` against the dense offset samples in its result dict
``{"ok": bool, "curve"/"surface": ..., "actual_max_deviation": float,
"reason": str, "analytic": str}``; analytic exact paths report ``0.0`` and
name the case taken in the ``"analytic"`` flag (``"line"``, ``"circle"``,
``"arc"``, ``"plane"``, ``"sphere"``, or ``""`` for the general refit path).
:func:`offset_loop` returns ``{"ok", "curves", "perimeter", "reason"}``.

Adaptations from kerf (documented, everything else is a faithful port):

  * numpy removed; pure stdlib, deterministic.
  * Curves are the harness plain-data 4-tuple ``(control_points, weights,
    degree, knots)`` and surfaces the 6-tuple ``(poles, weights, deg_u, deg_v,
    knots_u, knots_v)``; evaluation reuses
    :mod:`harnesscad.domain.geometry.parametric.nurbs_curve`
    (``curve_point`` / ``curve_derivatives``) and
    :mod:`harnesscad.domain.geometry.parametric.nurbs_surface`
    (``surface_point`` / ``surface_normal``) instead of kerf's ``de_boor`` /
    ``surface_evaluate``.  2-D control points are lifted to z=0; results are
    always 3-D.
  * Tangents along the general offset path use the analytic NURBS derivative
    (kerf used ``np.gradient`` over the samples), with a finite-difference
    fallback at cusps.
  * A straight-line analytic case (translate the control net, exact for any
    degree with a collinear net) and a circular-arc analytic case (circum-
    centre detection verified against dense samples at 1e-9) are added next
    to kerf's structural 9-point-circle detector, so line/circle/arc offsets
    are all exact with no refit.  The analytic circle/arc branch keeps kerf's
    convention that ``d > 0`` always grows the radius (``r_new = r + d``)
    regardless of parameterisation orientation.
  * Two kerf loop-fillet defects are corrected so corner arcs are tangent and
    gap-free: the fillet centre is placed at ``mid + sqrt(d^2 - h^2) *
    sag_dir`` (kerf used the sagitta ``d - sqrt(d^2 - h^2)``, putting the
    centre off the true corner), and the arc sweep is the short-way signed
    angle in (-pi, pi] (kerf forced the sweep sign from ``d``, which routed
    arcs the long way around).  Concave corners between straight segments are
    trimmed exactly to the intersection (kerf appended an extension line and
    left a gap); non-line segments keep kerf's extend-with-connector scheme
    but the connector to the next segment start is added so the loop closes.
  * kerf's surface refit fitted rows with Piegl-Tiller knots but re-evaluated
    them on mismatched uniform knots; this port uses one consistent
    uniform-parameter tensor-product least-squares fit (same knot vectors for
    fitting and the returned surface), keeping the reported deviation honest.
  * kerf's ``offset_curve_3d`` (geodesic offset on a surface) is not ported:
    it depends on kerf's closest-point surface inversion, which the harness
    does not have yet.

Relation to ``parametric/path_offset.py``: that module offsets 2-D
*polylines* with mitered corners (SolidPython port); this module offsets
*NURBS* curves and surfaces, handles exact conics analytically, inserts true
rational arc fillets rather than discretised ones, and reports a measured
deviation for every approximate result.

Pure stdlib, deterministic.
"""

from __future__ import annotations

import argparse
import math
from typing import List, NamedTuple, Optional, Sequence, Tuple

from harnesscad.domain.geometry.parametric.nurbs_curve import (
    curve_derivatives,
    curve_point,
)
from harnesscad.domain.geometry.parametric.nurbs_surface import (
    surface_normal,
    surface_point,
)
from harnesscad.domain.numeric.nurbs_basis import all_basis, uniform_clamped_knots

Vec3 = Tuple[float, float, float]

__all__ = [
    "CurveData",
    "SurfaceData",
    "make_line_curve",
    "make_circle_curve",
    "make_arc_curve",
    "fit_curve_points",
    "interp_curve_points",
    "offset_curve",
    "offset_surface",
    "offset_loop",
    "main",
]


class CurveData(NamedTuple):
    """Plain-data NURBS curve: the harness 4-tuple convention."""

    control_points: Tuple[Vec3, ...]
    weights: Tuple[float, ...]
    degree: int
    knots: Tuple[float, ...]


class SurfaceData(NamedTuple):
    """Plain-data NURBS surface: the harness 6-tuple convention."""

    poles: Tuple[Tuple[Vec3, ...], ...]
    weights: Tuple[Tuple[float, ...], ...]
    deg_u: int
    deg_v: int
    knots_u: Tuple[float, ...]
    knots_v: Tuple[float, ...]


# ---------------------------------------------------------------------------
# Small vector helpers (pure stdlib)
# ---------------------------------------------------------------------------

def _v3(p: Sequence[float]) -> Vec3:
    if len(p) == 3:
        return (float(p[0]), float(p[1]), float(p[2]))
    if len(p) == 2:
        return (float(p[0]), float(p[1]), 0.0)
    raise ValueError("points must be 2-D or 3-D, got dimension %d" % len(p))


def _sub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: Sequence[float], s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(_dot(a, a))


def _unit(a: Sequence[float]) -> Vec3:
    n = _norm(a)
    if n < 1e-300:
        raise ValueError("zero-length vector cannot be normalised")
    return (a[0] / n, a[1] / n, a[2] / n)


def _validate_distance(d: float) -> float:
    d = float(d)
    if math.isnan(d) or math.isinf(d):
        raise ValueError("offset distance must be finite, got %r" % d)
    return d


def _normalise_plane_normal(plane_normal: Optional[Sequence[float]]) -> Vec3:
    """kerf's plane-normal handling: default +z, pad short vectors, unit."""
    if plane_normal is None:
        return (0.0, 0.0, 1.0)
    vals = [float(c) for c in plane_normal][:3]
    while len(vals) < 3:
        vals.append(0.0)
    n = _norm(vals)
    if n < 1e-14:
        return (0.0, 0.0, 1.0)
    return (vals[0] / n, vals[1] / n, vals[2] / n)


# ---------------------------------------------------------------------------
# Curve / surface plain-data handling
# ---------------------------------------------------------------------------

def _as_curve(curve) -> CurveData:
    """Normalise a plain-data curve tuple to a 3-D ``CurveData``."""
    cps, w, degree, knots = curve
    cps3 = tuple(_v3(p) for p in cps)
    return CurveData(cps3, tuple(float(x) for x in w), int(degree),
                     tuple(float(k) for k in knots))


def _as_surface(surface) -> SurfaceData:
    poles, w, deg_u, deg_v, ku, kv = surface
    poles3 = tuple(tuple(_v3(p) for p in row) for row in poles)
    w3 = tuple(tuple(float(x) for x in row) for row in w)
    return SurfaceData(poles3, w3, int(deg_u), int(deg_v),
                       tuple(float(k) for k in ku), tuple(float(k) for k in kv))


def _curve_param_range(c: CurveData) -> Tuple[float, float]:
    n = len(c.control_points) - 1
    return float(c.knots[c.degree]), float(c.knots[n + 1])


def _surface_param_range(s: SurfaceData) -> Tuple[float, float, float, float]:
    n = len(s.poles) - 1
    m = len(s.poles[0]) - 1
    return (float(s.knots_u[s.deg_u]), float(s.knots_u[n + 1]),
            float(s.knots_v[s.deg_v]), float(s.knots_v[m + 1]))


def _eval_curve(c: CurveData, t: float) -> Vec3:
    return _v3(curve_point(c.control_points, c.weights, c.degree, c.knots, t))


def _sample_curve_pts(c: CurveData, num: int) -> Tuple[List[float], List[Vec3]]:
    """(params, points) at ``num`` uniformly spaced parameter values."""
    t0, t1 = _curve_param_range(c)
    num = max(3, int(num))
    ts = [t0 + (t1 - t0) * (i / (num - 1)) for i in range(num)]
    return ts, [_eval_curve(c, t) for t in ts]


def _curve_tangent_at(c: CurveData, t: float) -> Optional[Vec3]:
    """Analytic unit tangent, finite-difference fallback, None if degenerate."""
    try:
        d1 = curve_derivatives(c.control_points, c.weights, c.degree,
                               c.knots, t, 1)[1]
        n = _norm(_v3(d1))
        if n > 1e-14:
            return _unit(_v3(d1))
    except ValueError:
        pass
    t0, t1 = _curve_param_range(c)
    eps = (t1 - t0) * 1e-4
    a = _eval_curve(c, max(t - eps, t0))
    b = _eval_curve(c, min(t + eps, t1))
    v = _sub(b, a)
    n = _norm(v)
    return _unit(v) if n > 1e-14 else None


# ---------------------------------------------------------------------------
# Exact constructors (ported from kerf geom/nurbs.py)
# ---------------------------------------------------------------------------

def make_line_curve(p1: Sequence[float], p2: Sequence[float]) -> CurveData:
    """Straight segment as a degree-1 NURBS (kerf ``make_line_nurbs``)."""
    return CurveData((_v3(p1), _v3(p2)), (1.0, 1.0), 1, (0.0, 0.0, 1.0, 1.0))


def make_circle_curve(center: Sequence[float], radius: float,
                      x_axis: Optional[Sequence[float]] = None,
                      y_axis: Optional[Sequence[float]] = None) -> CurveData:
    """Exact full circle: the standard rational quadratic 9-point NURBS.

    Four quadratic rational Bezier segments (Piegl & Tiller sec. 7.5), ported
    from kerf ``make_circle_nurbs``: on-circle quadrant points and square-
    corner shoulder points, weights ``[1, s, 1, s, 1, s, 1, s, 1]`` with
    ``s = sqrt(2)/2`` and knots ``[0,0,0, 1/4,1/4, 1/2,1/2, 3/4,3/4, 1,1,1]``.
    """
    ctr = _v3(center)
    X = _unit(_v3(x_axis)) if x_axis is not None else (1.0, 0.0, 0.0)
    Y = _unit(_v3(y_axis)) if y_axis is not None else (0.0, 1.0, 0.0)
    r = float(radius)
    s = math.sqrt(2.0) / 2.0
    offs = [(r, 0.0), (r, r), (0.0, r), (-r, r), (-r, 0.0),
            (-r, -r), (0.0, -r), (r, -r), (r, 0.0)]
    cps = tuple(_add(ctr, _add(_scale(X, a), _scale(Y, b))) for a, b in offs)
    weights = (1.0, s, 1.0, s, 1.0, s, 1.0, s, 1.0)
    knots = (0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0)
    return CurveData(cps, weights, 2, knots)


def make_arc_curve(center: Sequence[float], radius: float,
                   start_angle: float, end_angle: float,
                   x_axis: Optional[Sequence[float]] = None,
                   y_axis: Optional[Sequence[float]] = None) -> CurveData:
    """Exact rational quadratic circular arc (kerf ``make_arc_nurbs``).

    Piegl & Tiller sec. 7.3 (A7.1): the sweep is split into
    ``ceil(|dtheta| / 90deg)`` exact rational quadratic Bezier segments whose
    shoulder points are the end-tangent intersections; shoulder weight is
    ``cos(|segment sweep| / 2)``.  Negative sweeps are supported.
    """
    ctr = _v3(center)
    X = _unit(_v3(x_axis)) if x_axis is not None else (1.0, 0.0, 0.0)
    Y = _unit(_v3(y_axis)) if y_axis is not None else (0.0, 1.0, 0.0)
    r = float(radius)
    theta = float(end_angle) - float(start_angle)
    if abs(theta) < 1e-14:
        raise ValueError("arc sweep must be non-zero")

    n_seg = max(1, int(math.ceil(abs(theta) / (math.pi / 2.0) - 1e-12)))
    dtheta = theta / n_seg
    w_mid = math.cos(abs(dtheta) / 2.0)

    def point(ang: float) -> Vec3:
        return _add(ctr, _add(_scale(X, r * math.cos(ang)),
                              _scale(Y, r * math.sin(ang))))

    def tangent(ang: float) -> Vec3:
        return _add(_scale(X, -math.sin(ang)), _scale(Y, math.cos(ang)))

    cps: List[Vec3] = [point(float(start_angle))]
    weights: List[float] = [1.0]
    a0 = float(start_angle)
    for _ in range(n_seg):
        a1 = a0 + dtheta
        p0, p2 = point(a0), point(a1)
        t0, t2 = tangent(a0), tangent(a1)
        # Solve p0 + alpha t0 = p2 - beta t2 in the local (X, Y) frame.
        m00, m01 = _dot(t0, X), -_dot(t2, X)
        m10, m11 = _dot(t0, Y), -_dot(t2, Y)
        r0, r1 = _dot(_sub(p2, p0), X), _dot(_sub(p2, p0), Y)
        det = m00 * m11 - m01 * m10
        alpha = (r0 * m11 - r1 * m01) / det if abs(det) > 1e-14 else 0.0
        cps.append(_add(p0, _scale(t0, alpha)))
        weights.append(w_mid)
        cps.append(p2)
        weights.append(1.0)
        a0 = a1

    knots: List[float] = [0.0, 0.0, 0.0]
    for k in range(1, n_seg):
        knots += [k / n_seg, k / n_seg]
    knots += [1.0, 1.0, 1.0]
    return CurveData(tuple(cps), tuple(weights), 2, tuple(knots))


# ---------------------------------------------------------------------------
# Least-squares B-spline fitting (ported from kerf geom/curve_toolkit.py)
# ---------------------------------------------------------------------------

def _chord_params(points: Sequence[Vec3]) -> List[float]:
    """Chord-length parameter sequence in [0, 1] (kerf ``_chord_params``)."""
    n = len(points)
    if n == 1:
        return [0.0]
    norms = [_norm(_sub(b, a)) for a, b in zip(points[:-1], points[1:])]
    total = sum(norms)
    if total < 1e-14:
        return [i / (n - 1) for i in range(n)]
    ts = [0.0]
    acc = 0.0
    for x in norms:
        acc += x
        ts.append(acc / total)
    ts[-1] = 1.0
    return ts


def _pt_knots_from_params(ts: Sequence[float], num_ctrl: int,
                          degree: int) -> List[float]:
    """Piegl-Tiller knot placement for least-squares fitting (P&T eq. 9.68)."""
    m = len(ts) - 1
    n = num_ctrl - 1
    p = degree
    knots = [0.0] * (n + p + 2)
    for i in range(len(knots) - (p + 1), len(knots)):
        knots[i] = 1.0
    num_interior = n - p
    if num_interior <= 0:
        return knots
    d = (m + 1) / (n - p + 1)
    for j in range(1, num_interior + 1):
        idx = int(j * d)
        alpha = j * d - idx
        idx = max(1, min(idx, m))
        knots[p + j] = (1.0 - alpha) * ts[idx - 1] + alpha * ts[idx]
    for k in range(p + 1, len(knots) - p - 1):
        knots[k] = max(knots[k], knots[k - 1])
    return knots


def _solve_linear(M: List[List[float]], B: List[List[float]]
                  ) -> List[List[float]]:
    """Solve M x = B (multiple right-hand sides) by Gaussian elimination."""
    n = len(M)
    a = [list(row) for row in M]
    b = [list(row) for row in B]
    scale = max((abs(x) for row in a for x in row), default=0.0)
    tol = 1e-14 * max(1.0, scale)
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) <= tol:
            raise ValueError("singular linear system in least-squares fit")
        if piv != col:
            a[col], a[piv] = a[piv], a[col]
            b[col], b[piv] = b[piv], b[col]
        inv = 1.0 / a[col][col]
        for r in range(col + 1, n):
            f = a[r][col] * inv
            if f == 0.0:
                continue
            for c in range(col, n):
                a[r][c] -= f * a[col][c]
            for c in range(len(b[r])):
                b[r][c] -= f * b[col][c]
    for col in range(n - 1, -1, -1):
        inv = 1.0 / a[col][col]
        for c in range(len(b[col])):
            acc = b[col][c]
            for k in range(col + 1, n):
                acc -= a[col][k] * b[k][c]
            b[col][c] = acc * inv
    return b


def _lstsq(A: List[List[float]], pts: Sequence[Vec3]) -> List[Vec3]:
    """Least-squares solve of A @ P = pts via the normal equations."""
    rows = len(A)
    cols = len(A[0])
    N = [[0.0] * cols for _ in range(cols)]
    R = [[0.0, 0.0, 0.0] for _ in range(cols)]
    for r in range(rows):
        Ar = A[r]
        p = pts[r]
        for i in range(cols):
            ai = Ar[i]
            if ai == 0.0:
                continue
            Ni = N[i]
            for j in range(cols):
                Ni[j] += ai * Ar[j]
            Ri = R[i]
            Ri[0] += ai * p[0]
            Ri[1] += ai * p[1]
            Ri[2] += ai * p[2]
    sol = _solve_linear(N, R)
    return [(row[0], row[1], row[2]) for row in sol]


def _basis_matrix(ts: Sequence[float], degree: int, knots: Sequence[float],
                  num_ctrl: int) -> List[List[float]]:
    return [all_basis(num_ctrl - 1, degree, t, knots) for t in ts]


def interp_curve_points(points: Sequence[Vec3], degree: int = 3) -> CurveData:
    """Interpolate a B-spline curve through ``points`` (kerf ``interp_curve``).

    Chord-length parameterisation, Piegl & Tiller averaging knots (9.3.6),
    collocation solve.
    """
    pts = [_v3(p) for p in points]
    n = len(pts)
    if n < 2:
        raise ValueError("interp_curve_points requires at least 2 points")
    degree = min(degree, n - 1)
    ts = _chord_params(pts)
    knots = uniform_clamped_knots(n - 1, degree)
    for j in range(1, n - degree):
        knots[j + degree] = sum(ts[j: j + degree]) / degree
    A = _basis_matrix(ts, degree, knots, n)
    ctrl = _lstsq(A, pts)
    return CurveData(tuple(ctrl), tuple([1.0] * n), degree, tuple(knots))


def fit_curve_points(points: Sequence[Vec3], degree: int = 3,
                     tolerance: float = 1e-3, max_ctrl: int = 64) -> dict:
    """Least-squares B-spline fit to ``points`` (kerf ``fit_curve``).

    Piegl-Tiller knot placement; the control-point count grows from
    ``degree + 1`` until the max deviation at the data parameters is within
    ``tolerance`` or ``max_ctrl`` is reached.  Never raises; returns
    ``{"ok", "curve", "deviation", "num_ctrl", "reason"}``.
    """
    try:
        pts = [_v3(p) for p in points]
        n = len(pts)
        if n < 2:
            return {"ok": False, "curve": None, "deviation": float("inf"),
                    "num_ctrl": 0, "reason": "need at least 2 points"}
        span = max(_norm(_sub(p, pts[0])) for p in pts)
        if span < 1e-14:
            curve = make_line_curve(pts[0], pts[0])
            return {"ok": True, "curve": curve, "deviation": 0.0,
                    "num_ctrl": 2, "reason": "degenerate: all points identical"}
        degree = min(degree, n - 1)
        ts = _chord_params(pts)

        curve = None
        dev = float("inf")
        num_ctrl = degree + 1
        for num_ctrl in range(degree + 1, min(max_ctrl + 1, n + 1)):
            knots = _pt_knots_from_params(ts, num_ctrl, degree)
            A = _basis_matrix(ts, degree, knots, num_ctrl)
            try:
                ctrl = _lstsq(A, pts)
            except ValueError:
                continue
            curve = CurveData(tuple(ctrl), tuple([1.0] * num_ctrl),
                              degree, tuple(knots))
            dev = max(_norm(_sub(_eval_curve(curve, t), p))
                      for t, p in zip(ts, pts))
            if dev <= tolerance:
                return {"ok": True, "curve": curve, "deviation": dev,
                        "num_ctrl": num_ctrl, "reason": ""}
        return {"ok": False, "curve": curve, "deviation": dev,
                "num_ctrl": num_ctrl,
                "reason": "tolerance %g not achieved; best deviation %.4g"
                          % (tolerance, dev)}
    except Exception as exc:  # kerf: never raise from fit_curve
        return {"ok": False, "curve": None, "deviation": float("inf"),
                "num_ctrl": 0, "reason": str(exc)}


# ---------------------------------------------------------------------------
# Analytic shape detectors
# ---------------------------------------------------------------------------

def _collinear_control_points(c: CurveData) -> Optional[Vec3]:
    """If every control point lies on one line, return the unit direction.

    Translation of a collinear control net translates the (rational) curve
    exactly, so such curves admit an exact offset.
    """
    cps = c.control_points
    first, last = cps[0], cps[-1]
    axis = _sub(last, first)
    if _norm(axis) < 1e-14:
        return None
    u = _unit(axis)
    for p in cps:
        rel = _sub(p, first)
        if _norm(_cross(rel, u)) > 1e-9:
            return None
    return u


def _is_rational_circle(c: CurveData) -> Optional[Tuple[Vec3, float]]:
    """kerf's structural detector for the exact 9-point rational circle.

    degree 2, 9 control points, weights ``[1, s, 1, s, 1, s, 1, s, 1]``
    (s = sqrt(2)/2, +-1e-9), 4 quadrant points equidistant from their
    centroid.  Returns (centre, radius) or None.
    """
    if c.degree != 2 or len(c.control_points) != 9:
        return None
    s = math.sqrt(2.0) / 2.0
    expected = (1.0, s, 1.0, s, 1.0, s, 1.0, s, 1.0)
    if any(abs(w - e) > 1e-9 for w, e in zip(c.weights, expected)):
        return None
    q = [c.control_points[i] for i in (0, 2, 4, 6)]
    centre = _scale(_add(_add(q[0], q[1]), _add(q[2], q[3])), 0.25)
    radii = [_norm(_sub(p, centre)) for p in q]
    r0 = radii[0]
    if any(abs(r - r0) > 1e-9 * max(1.0, r0) + 1e-12 for r in radii):
        return None
    if r0 < 1e-14:
        return None
    return centre, float(r0)


def _detect_circular_arc(c: CurveData, nrm: Vec3
                         ) -> Optional[Tuple[Vec3, float, Vec3, Vec3, float]]:
    """Detect an exact planar circular arc by circumcentre + dense check.

    Returns ``(centre, radius, x_axis, y_axis, sweep)`` such that the curve
    coincides (within 1e-9 relative) with ``centre + r*(cos(a) X + sin(a) Y)``
    for ``a`` in ``[0, sweep]``, or None.  Full closed circles are left to the
    structural circle detector / general path, matching kerf.
    """
    K = 33
    _, pts = _sample_curve_pts(c, K)
    p_start, p_end = pts[0], pts[-1]
    scale = max(_norm(_sub(p, p_start)) for p in pts)
    if scale < 1e-14:
        return None
    if _norm(_sub(p_end, p_start)) < 1e-9 * scale:
        return None  # closed: not an arc
    # Circumcentre of (start, mid, end) in the plane through p_start.
    p_mid = pts[K // 2]
    try:
        e1 = _unit(_sub(p_mid, p_start))
    except ValueError:
        return None
    e2 = _cross(nrm, e1)
    if _norm(e2) < 1e-9:
        return None
    e2 = _unit(e2)

    def to2d(p: Vec3) -> Tuple[float, float]:
        rel = _sub(p, p_start)
        return _dot(rel, e1), _dot(rel, e2)

    ax, ay = 0.0, 0.0
    bx, by = to2d(p_mid)
    cx, cy = to2d(p_end)
    d2 = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d2) < 1e-12 * scale * scale:
        return None  # nearly straight
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay)
          + (cx * cx + cy * cy) * (ay - by)) / d2
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx)
          + (cx * cx + cy * cy) * (bx - ax)) / d2
    centre = _add(p_start, _add(_scale(e1, ux), _scale(e2, uy)))
    r = _norm(_sub(p_start, centre))
    if r < 1e-14:
        return None
    tol = 1e-9 * max(1.0, r)
    for p in pts:
        if abs(_norm(_sub(p, centre)) - r) > tol:
            return None
        if abs(_dot(_sub(p, p_start), nrm)) > tol:
            return None  # not planar w.r.t. the given normal
    x_ax = _unit(_sub(p_start, centre))
    y_ax = _unit(_cross(nrm, x_ax))
    # Unwrapped sweep along the dense samples.
    sweep = 0.0
    prev = 0.0
    for p in pts[1:]:
        rel = _sub(p, centre)
        ang = math.atan2(_dot(rel, y_ax), _dot(rel, x_ax))
        dab = ang - prev
        while dab > math.pi:
            dab -= 2.0 * math.pi
        while dab <= -math.pi:
            dab += 2.0 * math.pi
        sweep += dab
        prev = ang
    if abs(sweep) < 1e-10:
        return None
    return centre, float(r), x_ax, y_ax, float(sweep)


# ---------------------------------------------------------------------------
# offset_curve
# ---------------------------------------------------------------------------

def offset_curve(curve, d: float, *, tol: float = 1e-4,
                 plane_normal: Optional[Sequence[float]] = None,
                 num_samples: int = 200) -> dict:
    """Planar curve offset by signed distance ``d`` along the right normal.

    Sign convention (kerf's): the right-side normal of the tangent ``T`` in
    plane ``N`` is ``R = normalise(cross(N, T))``; ``d > 0`` moves along
    ``+R``.  Analytic circle/arc results follow kerf's radius convention
    ``r_new = r + d`` (positive ``d`` grows the radius).

    Returns ``{"ok", "curve", "actual_max_deviation", "reason", "analytic"}``.
    ``actual_max_deviation`` is 0.0 for exact analytic results, else the max
    distance between the refit curve and the dense offset samples.

    Raises ``ValueError`` on non-finite ``d`` or a degenerate (zero-length)
    curve.
    """
    c = _as_curve(curve)
    d = _validate_distance(d)
    nrm = _normalise_plane_normal(plane_normal)

    # Validate the input curve is non-degenerate (kerf's 3-point check).
    t0, t1 = _curve_param_range(c)
    p0 = _eval_curve(c, t0)
    p_mid = _eval_curve(c, (t0 + t1) * 0.5)
    p1 = _eval_curve(c, t1)
    if _norm(_sub(p_mid, p0)) < 1e-14 and _norm(_sub(p1, p0)) < 1e-14:
        raise ValueError("curve is degenerate (zero length)")

    # --- analytic shortcut: straight line (translate the control net) ---
    line_dir = _collinear_control_points(c)
    if line_dir is not None:
        right = _cross(nrm, line_dir)
        rn = _norm(right)
        if rn > 1e-14:
            move = _scale(right, d / rn)
            new_cps = tuple(_add(p, move) for p in c.control_points)
            return {"ok": True,
                    "curve": CurveData(new_cps, c.weights, c.degree, c.knots),
                    "actual_max_deviation": 0.0, "reason": "",
                    "analytic": "line"}

    # --- analytic shortcut: exact rational 9-point circle ---
    circle_info = _is_rational_circle(c)
    if circle_info is not None:
        centre, r = circle_info
        r_new = r + d
        if r_new <= 0.0:
            return {"ok": False, "curve": None, "actual_max_deviation": 0.0,
                    "reason": "offset distance %g collapses circle of radius %g"
                              % (d, r),
                    "analytic": "circle"}
        x_raw = _sub(c.control_points[0], centre)
        x_ax = _unit(x_raw) if _norm(x_raw) > 1e-14 else (1.0, 0.0, 0.0)
        y_raw = _cross(nrm, x_ax)
        y_ax = _unit(y_raw) if _norm(y_raw) > 1e-14 else (0.0, 1.0, 0.0)
        return {"ok": True,
                "curve": make_circle_curve(centre, r_new, x_axis=x_ax,
                                           y_axis=y_ax),
                "actual_max_deviation": 0.0, "reason": "",
                "analytic": "circle"}

    # --- analytic shortcut: circular arc ---
    arc_info = _detect_circular_arc(c, nrm)
    if arc_info is not None:
        centre, r, x_ax, y_ax, sweep = arc_info
        r_new = r + d
        if r_new <= 0.0:
            return {"ok": False, "curve": None, "actual_max_deviation": 0.0,
                    "reason": "offset distance %g collapses arc of radius %g"
                              % (d, r),
                    "analytic": "arc"}
        return {"ok": True,
                "curve": make_arc_curve(centre, r_new, 0.0, sweep,
                                        x_axis=x_ax, y_axis=y_ax),
                "actual_max_deviation": 0.0, "reason": "",
                "analytic": "arc"}

    # --- general NURBS path: sample, offset pointwise, refit ---
    ts, pts = _sample_curve_pts(c, num_samples)
    offset_pts: List[Vec3] = []
    for t, p in zip(ts, pts):
        tan = _curve_tangent_at(c, t)
        if tan is None:
            offset_pts.append(p)
            continue
        right = _cross(nrm, tan)
        rn = _norm(right)
        if rn < 1e-14:
            offset_pts.append(p)
        else:
            offset_pts.append(_add(p, _scale(right, d / rn)))

    fit_degree = min(3, c.degree)
    result = fit_curve_points(offset_pts, degree=fit_degree, tolerance=tol,
                              max_ctrl=max(16, num_samples // 4))
    if result["ok"] and result["curve"] is not None:
        approx = result["curve"]
        actual_dev = float(result["deviation"])
    else:
        # kerf fallback: interpolate through all offset points, then measure
        # the residual at the chord parameter values.
        approx = interp_curve_points(offset_pts, degree=fit_degree)
        ts_ip = _chord_params(offset_pts)
        a0, a1 = _curve_param_range(approx)
        actual_dev = max(
            _norm(_sub(_eval_curve(approx, a0 + t * (a1 - a0)), p))
            for t, p in zip(ts_ip, offset_pts))

    ok = actual_dev <= tol if not math.isinf(actual_dev) else False
    return {"ok": ok, "curve": approx, "actual_max_deviation": actual_dev,
            "reason": "" if ok else "tolerance %g not achieved; best dev %.4g"
                                    % (tol, actual_dev),
            "analytic": ""}


# ---------------------------------------------------------------------------
# offset_surface
# ---------------------------------------------------------------------------

def _is_planar_surface(s: SurfaceData) -> Optional[Tuple[Vec3, Vec3]]:
    """Detect a degree-(1,1) 2x2 planar patch: (point, unit normal) or None."""
    if s.deg_u != 1 or s.deg_v != 1:
        return None
    if len(s.poles) != 2 or len(s.poles[0]) != 2:
        return None
    p00, p01 = s.poles[0][0], s.poles[0][1]
    p10, p11 = s.poles[1][0], s.poles[1][1]
    n = _cross(_sub(p10, p00), _sub(p01, p00))
    mag = _norm(n)
    if mag < 1e-12:
        return None
    un = _scale(n, 1.0 / mag)
    if abs(_dot(_sub(p11, p00), un)) > 1e-9:
        return None
    return p00, un


def _is_sphere_surface(s: SurfaceData) -> Optional[Tuple[Vec3, float]]:
    """kerf's structural detector for the rational revolution sphere.

    degree (2,2); first and last v-columns collapsed to the two poles; the
    middle v-row's weight-1 control points form the equator.  Returns
    (centre, radius) or None.
    """
    if s.deg_u != 2 or s.deg_v != 2:
        return None
    nu = len(s.poles)
    nv = len(s.poles[0])
    if nu < 5 or nv < 5:
        return None
    col0 = [s.poles[i][0] for i in range(nu)]
    colN = [s.poles[i][nv - 1] for i in range(nu)]
    if any(_norm(_sub(p, col0[0])) > 1e-9 for p in col0):
        return None
    if any(_norm(_sub(p, colN[0])) > 1e-9 for p in colN):
        return None
    south, north = col0[0], colN[0]
    centre = _scale(_add(south, north), 0.5)
    r_axis = _norm(_sub(north, south)) * 0.5
    if r_axis < 1e-14:
        return None
    j_mid = nv // 2
    on_pts = [s.poles[i][j_mid] for i in range(nu)
              if abs(s.weights[i][j_mid] - 1.0) < 1e-9]
    if len(on_pts) < 3:
        return None
    dists = [_norm(_sub(p, centre)) for p in on_pts]
    r_eq = sum(dists) / len(dists)
    if r_eq < 1e-14 or abs(r_eq - r_axis) / r_axis > 1e-3:
        return None
    mean = r_eq
    var = sum((x - mean) ** 2 for x in dists) / len(dists)
    if math.sqrt(var) / r_eq > 1e-6:
        return None
    return centre, float(r_eq)


def offset_surface(surface, d: float, *, tol: float = 1e-4,
                   grid_samples: int = 20) -> dict:
    """Surface offset along the analytic unit normal by signed distance ``d``.

    Analytic shortcuts: spheres (concentric scale of the control net) and
    planes (translate the poles along the normal) are exact.  General
    surfaces are sampled on a ``grid_samples x grid_samples`` grid, offset
    along the analytic normal, and refit with a two-stage tensor-product
    least-squares fit; the measured max deviation at the grid samples is
    reported.

    Returns ``{"ok", "surface", "actual_max_deviation", "reason",
    "analytic"}``.
    """
    s = _as_surface(surface)
    d = _validate_distance(d)

    sphere_info = _is_sphere_surface(s)
    if sphere_info is not None:
        centre, r = sphere_info
        r_new = r + d
        if r_new <= 0.0:
            return {"ok": False, "surface": None, "actual_max_deviation": 0.0,
                    "reason": "offset distance %g collapses sphere of radius %g"
                              % (d, r),
                    "analytic": "sphere"}
        scale = r_new / r
        new_poles = tuple(
            tuple(_add(centre, _scale(_sub(p, centre), scale)) for p in row)
            for row in s.poles)
        return {"ok": True,
                "surface": SurfaceData(new_poles, s.weights, s.deg_u, s.deg_v,
                                       s.knots_u, s.knots_v),
                "actual_max_deviation": 0.0, "reason": "",
                "analytic": "sphere"}

    plane_info = _is_planar_surface(s)
    if plane_info is not None:
        _, un = plane_info
        move = _scale(un, d)
        new_poles = tuple(tuple(_add(p, move) for p in row) for row in s.poles)
        return {"ok": True,
                "surface": SurfaceData(new_poles, s.weights, s.deg_u, s.deg_v,
                                       s.knots_u, s.knots_v),
                "actual_max_deviation": 0.0, "reason": "",
                "analytic": "plane"}

    # --- general path: grid sample -> offset along normal -> refit ---
    u_min, u_max, v_min, v_max = _surface_param_range(s)
    g = max(4, int(grid_samples))
    us = [u_min + (u_max - u_min) * (i / (g - 1)) for i in range(g)]
    vs = [v_min + (v_max - v_min) * (j / (g - 1)) for j in range(g)]
    grid_pts: List[List[Vec3]] = []
    for u in us:
        row = []
        for v in vs:
            P = _v3(surface_point(s.poles, s.weights, s.deg_u, s.deg_v,
                                  s.knots_u, s.knots_v, u, v))
            N = _v3(surface_normal(s.poles, s.weights, s.deg_u, s.deg_v,
                                   s.knots_u, s.knots_v, u, v))
            row.append(_add(P, _scale(N, d)))
        grid_pts.append(row)

    deg_u = min(3, s.deg_u)
    deg_v = min(3, s.deg_v)
    n_u_ctrl = max(deg_u + 1, min(g, 16))
    n_v_ctrl = max(deg_v + 1, min(g, 16))

    # Consistent uniform-parameter tensor-product fit (see module docstring):
    # the same parameter values and knot vectors are used for fitting and for
    # the returned surface.
    ts = [i / (g - 1) for i in range(g)]
    knots_v_new = _pt_knots_from_params(ts, n_v_ctrl, deg_v)
    knots_u_new = _pt_knots_from_params(ts, n_u_ctrl, deg_u)
    Av = _basis_matrix(ts, deg_v, knots_v_new, n_v_ctrl)
    Au = _basis_matrix(ts, deg_u, knots_u_new, n_u_ctrl)

    try:
        row_ctrl = [_lstsq(Av, grid_pts[i]) for i in range(g)]
        net: List[List[Vec3]] = [[(0.0, 0.0, 0.0)] * n_v_ctrl
                                 for _ in range(n_u_ctrl)]
        for j in range(n_v_ctrl):
            col = [row_ctrl[i][j] for i in range(g)]
            ctrl_u = _lstsq(Au, col)
            for a in range(n_u_ctrl):
                net[a][j] = ctrl_u[a]
    except ValueError as exc:
        return {"ok": False, "surface": None,
                "actual_max_deviation": float("inf"),
                "reason": "surface refit failed: %s" % exc, "analytic": ""}

    new_surf = SurfaceData(tuple(tuple(row) for row in net),
                           tuple(tuple([1.0] * n_v_ctrl)
                                 for _ in range(n_u_ctrl)),
                           deg_u, deg_v,
                           tuple(knots_u_new), tuple(knots_v_new))

    actual_dev = 0.0
    for i in range(g):
        for j in range(g):
            approx_pt = _v3(surface_point(
                new_surf.poles, new_surf.weights, new_surf.deg_u,
                new_surf.deg_v, new_surf.knots_u, new_surf.knots_v,
                ts[i], ts[j]))
            dev = _norm(_sub(approx_pt, grid_pts[i][j]))
            if dev > actual_dev:
                actual_dev = dev

    ok = actual_dev <= tol
    return {"ok": ok, "surface": new_surf,
            "actual_max_deviation": actual_dev,
            "reason": "" if ok else "tolerance %g not achieved; best dev %.4g"
                                    % (tol, actual_dev),
            "analytic": ""}


# ---------------------------------------------------------------------------
# offset_loop
# ---------------------------------------------------------------------------

def _intersect_lines_in_plane(p0: Vec3, v0: Vec3, p1: Vec3, v1: Vec3,
                              nrm: Vec3) -> Optional[Vec3]:
    """Intersect lines p0 + s*v0 and p1 + t*v1 lying in the plane of ``nrm``."""
    try:
        e1 = _unit(v0)
    except ValueError:
        return None
    e2 = _cross(nrm, e1)
    if _norm(e2) < 1e-12:
        return None
    e2 = _unit(e2)
    a00, a01 = _dot(v0, e1), -_dot(v1, e1)
    a10, a11 = _dot(v0, e2), -_dot(v1, e2)
    b0 = _dot(_sub(p1, p0), e1)
    b1 = _dot(_sub(p1, p0), e2)
    det = a00 * a11 - a01 * a10
    if abs(det) < 1e-14:
        return None
    sfac = (b0 * a11 - b1 * a01) / det
    return _add(p0, _scale(v0, sfac))


def _fillet_arc(p_end: Vec3, p_start: Vec3, abs_d: float, d: float,
                nrm: Vec3) -> Optional[CurveData]:
    """Tangent arc fillet of radius ``abs_d`` joining two offset endpoints.

    Ported from kerf's convex-corner branch with the centre and sweep fixes
    described in the module docstring: the centre sits at distance
    ``sqrt(d^2 - h^2)`` from the chord midpoint (which for straight adjacent
    segments is exactly the original corner point), and the sweep is the
    short-way signed angle so the arc bulges outward.
    """
    chord = _sub(p_start, p_end)
    chord_len = _norm(chord)
    if chord_len < 1e-12:
        return None
    mid = _scale(_add(p_end, p_start), 0.5)
    perp_chord = _cross(nrm, _scale(chord, 1.0 / chord_len))
    if _norm(perp_chord) < 1e-14:
        return None
    sag_dir = _scale(_unit(perp_chord), -1.0 if d > 0 else 1.0)
    h = chord_len / 2.0
    if abs_d < h - 1e-12:
        return None  # radius smaller than half-chord: degenerate
    centre = _add(mid, _scale(sag_dir, math.sqrt(max(0.0, abs_d * abs_d
                                                     - h * h))))
    v_start = _sub(p_end, centre)
    if _norm(v_start) < 1e-14:
        return None
    x_ax = _unit(v_start)
    y_raw = _cross(nrm, x_ax)
    y_ax = _unit(y_raw) if _norm(y_raw) > 1e-14 else (0.0, 1.0, 0.0)
    v_end = _sub(p_start, centre)
    ang_end = math.atan2(_dot(v_end, y_ax), _dot(v_end, x_ax))
    if abs(ang_end) < 1e-10:
        return None
    try:
        return make_arc_curve(centre, abs_d, 0.0, ang_end,
                              x_axis=x_ax, y_axis=y_ax)
    except ValueError:
        return None


def offset_loop(curves: Sequence, d: float, *,
                plane_normal: Optional[Sequence[float]] = None,
                tol: float = 1e-4, num_samples: int = 100) -> dict:
    """Offset a closed planar loop of curves preserving connectivity.

    Each segment is offset via :func:`offset_curve`.  At each corner between
    adjacent offset segments:

      * convex corner (``d * turn_sign < 0``) -> exact rational arc fillet
        of radius ``|d|``;
      * concave corner -> extend / trim to the intersection of the adjacent
        offset directions: straight segments are trimmed exactly, curved
        segments get straight connector lines through the intersection.

    Returns ``{"ok", "curves", "perimeter", "reason"}``; ``perimeter`` is
    the chord-length sum over 50 samples per output segment (kerf's measure).
    """
    if not curves:
        raise ValueError("curves list is empty")
    d = _validate_distance(d)
    nrm = _normalise_plane_normal(plane_normal)

    segs: List[CurveData] = []
    for crv in curves:
        res = offset_curve(crv, d, tol=tol, plane_normal=nrm,
                           num_samples=num_samples)
        if not res["ok"] or res["curve"] is None:
            return {"ok": False, "curves": [], "perimeter": 0.0,
                    "reason": "segment offset failed: %s" % res["reason"]}
        segs.append(res["curve"])

    n_segs = len(segs)
    abs_d = abs(d)

    def _sample50(seg: CurveData) -> List[Vec3]:
        return _sample_curve_pts(seg, 50)[1]

    def _perimeter(out: Sequence[CurveData]) -> float:
        total = 0.0
        for seg in out:
            pts = _sample50(seg)
            for a, b in zip(pts[:-1], pts[1:]):
                total += _norm(_sub(b, a))
        return total

    if abs_d < 1e-14:
        return {"ok": True, "curves": segs, "perimeter": _perimeter(segs),
                "reason": ""}

    starts, ends, tan_starts, tan_ends = [], [], [], []
    for seg in segs:
        t0, t1 = _curve_param_range(seg)
        starts.append(_eval_curve(seg, t0))
        ends.append(_eval_curve(seg, t1))
        tan_starts.append(_curve_tangent_at(seg, t0) or (0.0, 0.0, 0.0))
        tan_ends.append(_curve_tangent_at(seg, t1) or (0.0, 0.0, 0.0))

    # Classify each corner i (between seg i and seg i+1).
    corner_convex: List[bool] = []
    corner_isect: List[Optional[Vec3]] = []
    for i in range(n_segs):
        j = (i + 1) % n_segs
        turn_sign = _dot(_cross(tan_ends[i], tan_starts[j]), nrm)
        convex = d * turn_sign < 0.0
        corner_convex.append(convex)
        if convex:
            corner_isect.append(None)
        else:
            corner_isect.append(_intersect_lines_in_plane(
                ends[i], tan_ends[i], starts[j], tan_starts[j], nrm))

    is_line = [_collinear_control_points(seg) is not None for seg in segs]

    result_curves: List[CurveData] = []
    for i in range(n_segs):
        seg = segs[i]
        prev = (i - 1) % n_segs
        prev_isect = None if corner_convex[prev] else corner_isect[prev]
        this_isect = None if corner_convex[i] else corner_isect[i]

        if is_line[i]:
            # Trim / extend a straight offset segment exactly.
            new_start = prev_isect if prev_isect is not None else starts[i]
            new_end = this_isect if this_isect is not None else ends[i]
            if _norm(_sub(new_end, new_start)) > 1e-14:
                result_curves.append(make_line_curve(new_start, new_end))
        else:
            if prev_isect is not None and \
                    _norm(_sub(prev_isect, starts[i])) > 1e-12:
                result_curves.append(make_line_curve(prev_isect, starts[i]))
            result_curves.append(seg)
            if this_isect is not None and \
                    _norm(_sub(ends[i], this_isect)) > 1e-12:
                result_curves.append(make_line_curve(ends[i], this_isect))

        if corner_convex[i]:
            j = (i + 1) % n_segs
            arc = _fillet_arc(ends[i], starts[j], abs_d, d, nrm)
            if arc is not None:
                result_curves.append(arc)
            elif _norm(_sub(starts[j], ends[i])) > 1e-12:
                # degenerate fillet: straight connector keeps the loop closed
                result_curves.append(make_line_curve(ends[i], starts[j]))

    if not result_curves:
        return {"ok": False, "curves": [], "perimeter": 0.0,
                "reason": "no output segments generated"}

    return {"ok": True, "curves": result_curves,
            "perimeter": _perimeter(result_curves), "reason": ""}


# ---------------------------------------------------------------------------
# Selfcheck
# ---------------------------------------------------------------------------

def _dist_point_to_segment(p: Vec3, a: Vec3, b: Vec3) -> float:
    ab = _sub(b, a)
    denom = _dot(ab, ab)
    t = 0.0 if denom < 1e-300 else max(0.0, min(1.0, _dot(_sub(p, a), ab)
                                                / denom))
    return _norm(_sub(p, _add(a, _scale(ab, t))))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.geometry.parametric.offset_nurbs",
        description="NURBS curve / surface / closed-loop offsetting with "
                    "analytic exact cases and measured-deviation refits "
                    "(ported from kerf-cad-core geom/offset.py).",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="offset a line, a circle, an arc, a square loop "
                             "(outward + inward), a freeform curve and two "
                             "surfaces on synthetic geometry and verify the "
                             "analytic exactness and honesty contract.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    # 1. Straight line: analytic branch, exact offset at distance |d|.
    line = make_line_curve((0.0, 0.0, 0.0), (3.0, 4.0, 0.0))
    res = offset_curve(line, 1.0)
    assert res["ok"] and res["analytic"] == "line"
    assert res["actual_max_deviation"] == 0.0
    a, b = (0.0, 0.0, 0.0), (3.0, 4.0, 0.0)
    for t in (0.0, 0.25, 0.5, 1.0):
        p = _eval_curve(res["curve"], t)
        assert abs(_dist_point_to_segment(p, a, b) - 1.0) < 1e-12
    print("[selfcheck] line offset: analytic, deviation=0, distance exact")

    # 2. Circle: analytic branch, radius r + d within 1e-9.
    centre = (1.0, 2.0, 0.0)
    circle = make_circle_curve(centre, 2.0)
    res = offset_curve(circle, 0.5)
    assert res["ok"] and res["analytic"] == "circle"
    assert res["actual_max_deviation"] == 0.0
    _, pts = _sample_curve_pts(res["curve"], 97)
    assert all(abs(_norm(_sub(p, centre)) - 2.5) < 1e-9 for p in pts)
    collapse = offset_curve(make_circle_curve((0.0, 0.0, 0.0), 1.0), -1.5)
    assert not collapse["ok"] and "collapses" in collapse["reason"]
    print("[selfcheck] circle offset: analytic, radius r+d within 1e-9, "
          "collapse rejected")

    # 3. Arc: analytic branch, radius r + d within 1e-9.
    arc = make_arc_curve((0.0, 0.0, 0.0), 1.0, 0.0, math.pi / 2.0)
    res = offset_curve(arc, 0.25)
    assert res["ok"] and res["analytic"] == "arc"
    assert res["actual_max_deviation"] == 0.0
    _, pts = _sample_curve_pts(res["curve"], 97)
    assert all(abs(_norm(p) - 1.25) < 1e-9 for p in pts)
    print("[selfcheck] arc offset: analytic, radius r+d within 1e-9")

    # 4. Square loop, outward: sides preserved, 4 arc fillets at the convex
    #    corners, perimeter = old + 2*pi*d.
    square = [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0),
              (1.0, 0.0, 0.0)]  # clockwise: cross(z, T) points outward
    loop = [make_line_curve(square[k], square[(k + 1) % 4]) for k in range(4)]
    d_out = 0.25
    res = offset_loop(loop, d_out)
    assert res["ok"]
    arcs = [seg for seg in res["curves"] if seg.degree == 2]
    lines = [seg for seg in res["curves"] if seg.degree == 1]
    assert len(arcs) == 4 and len(lines) == 4
    expected = 4.0 + 2.0 * math.pi * d_out
    assert abs(res["perimeter"] - expected) < 5e-4, res["perimeter"]
    # connectivity: consecutive segments share endpoints
    for k in range(len(res["curves"])):
        seg = res["curves"][k]
        nxt = res["curves"][(k + 1) % len(res["curves"])]
        e = _eval_curve(seg, _curve_param_range(seg)[1])
        s = _eval_curve(nxt, _curve_param_range(nxt)[0])
        assert _norm(_sub(s, e)) < 1e-9
    print("[selfcheck] square loop outward: 4 convex arc fillets, perimeter "
          "%.6f ~ old + 2*pi*d = %.6f" % (res["perimeter"], expected))

    # 5. Square loop, inward: concave corners trimmed; perimeter = old - 8*d.
    d_in = 0.25
    res = offset_loop(loop, -d_in)
    assert res["ok"]
    assert all(seg.degree == 1 for seg in res["curves"])
    assert len(res["curves"]) == 4
    assert abs(res["perimeter"] - (4.0 - 8.0 * d_in)) < 1e-9
    print("[selfcheck] square loop inward: concave corners trimmed, "
          "perimeter %.6f = old - 8*d" % res["perimeter"])

    # 6. Freeform cubic: general path; reported deviation below requested tol.
    cps = ((0.0, 0.0, 0.0), (1.0, 1.5, 0.0), (2.0, -1.0, 0.0),
           (3.0, 0.5, 0.0), (4.0, 0.0, 0.0))
    knots = tuple(uniform_clamped_knots(4, 3))
    freeform = CurveData(cps, (1.0,) * 5, 3, knots)
    tol = 1e-3
    res = offset_curve(freeform, 0.2, tol=tol)
    assert res["analytic"] == ""
    assert res["ok"] and res["actual_max_deviation"] <= tol
    # sanity: offset samples sit ~0.2 from the base curve
    _, base_pts = _sample_curve_pts(freeform, 400)
    _, off_pts = _sample_curve_pts(res["curve"], 21)
    for p in off_pts[1:-1]:
        dist = min(_dist_point_to_segment(p, a, b)
                   for a, b in zip(base_pts[:-1], base_pts[1:]))
        assert abs(dist - 0.2) < 2e-2
    print("[selfcheck] freeform offset: refit deviation %.3g <= tol %.3g"
          % (res["actual_max_deviation"], tol))

    # 7. Plane surface: analytic exact translation along the normal.
    plane = SurfaceData((((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                         ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0))),
                        ((1.0, 1.0), (1.0, 1.0)), 1, 1,
                        (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0))
    res = offset_surface(plane, 1.0)
    assert res["ok"] and res["analytic"] == "plane"
    assert res["actual_max_deviation"] == 0.0
    ns = res["surface"]
    p = surface_point(ns.poles, ns.weights, ns.deg_u, ns.deg_v,
                      ns.knots_u, ns.knots_v, 0.5, 0.5)
    assert abs(p[2] - 1.0) < 1e-12
    print("[selfcheck] plane surface offset: analytic, exact")

    # 8. Freeform bump surface: general refit path within tolerance.
    bump_poles = tuple(
        tuple((float(i), float(j),
               0.15 if (1 <= i <= 2 and 1 <= j <= 2) else 0.0)
              for j in range(4))
        for i in range(4))
    bump = SurfaceData(bump_poles, tuple((1.0,) * 4 for _ in range(4)),
                       2, 2, tuple(uniform_clamped_knots(3, 2)),
                       tuple(uniform_clamped_knots(3, 2)))
    stol = 5e-3
    res = offset_surface(bump, 0.05, tol=stol)
    assert res["analytic"] == ""
    assert res["ok"] and res["actual_max_deviation"] <= stol
    print("[selfcheck] bump surface offset: refit deviation %.3g <= tol %.3g"
          % (res["actual_max_deviation"], stol))

    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
