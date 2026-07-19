"""NURBS curve, surface and closed-loop offsetting with an honesty contract.

The module offsets planar NURBS curves, NURBS surfaces and closed planar
loops of curves.  Two qualitatively different result paths exist and the
result dictionaries always say which one was taken:

  * an ANALYTIC path, taken when the input is recognised as one of the
    shapes whose offset is again a shape of the same family.  These results
    are exact: the offset is produced by a closed-form reconstruction rather
    than by approximation, and ``actual_max_deviation`` is ``0.0``.  The
    recognised families are the straight line, the circular arc, the full
    circle (curves) and the plane and the sphere (surfaces).

  * a REFIT path, taken for everything else.  The input is sampled, each
    sample is displaced along the local offset direction, and a B-spline is
    least-squares fitted to the displaced cloud.  The true offset of a
    general NURBS curve or surface is not itself a NURBS, so this path is
    approximate by nature.  It is never presented as exact: the returned
    ``actual_max_deviation`` is the largest distance MEASURED between the
    fitted result and the displaced samples, and ``ok`` is true only when
    that measured value is within the caller's ``tol``.

Public entry points
-------------------

``offset_curve(curve, d, *, tol, plane_normal, num_samples)``
    Offsets a planar curve by the signed distance ``d`` along the right-hand
    normal ``R = unit(cross(plane_normal, T))`` of the tangent ``T``.  For
    the recognised circular families the reconstruction uses the radius
    convention ``r_new = r + d``, i.e. a positive ``d`` always grows the
    radius, independent of how the input happens to be parameterised.
    Returns ``{"ok", "curve", "actual_max_deviation", "reason",
    "analytic"}``, where ``analytic`` names the exact case taken
    (``"line"``, ``"arc"``, ``"circle"``) or is ``""`` on the refit path.

``offset_surface(surface, d, *, tol, grid_samples)``
    Offsets along the analytic unit normal.  Planes are offset by
    translating the control net, spheres by scaling it about the recognised
    centre; both are exact.  Everything else is grid-sampled, displaced and
    refitted by a two-pass (first across each row, then down the resulting
    columns) tensor-product least-squares fit, with the deviation measured
    on the sample grid.  Returns ``{"ok", "surface",
    "actual_max_deviation", "reason", "analytic"}`` with ``analytic`` in
    ``{"plane", "sphere", ""}``.

``offset_loop(curves, d, *, plane_normal, tol, num_samples)``
    Offsets a closed planar loop segment by segment and then repairs the
    corners so the result is again a connected closed loop.  A convex corner
    opens a gap between the two offset neighbours; it is bridged by an exact
    rational arc of radius ``|d|`` centred on the ORIGINAL corner vertex,
    which is by construction the point equidistant from both offset
    endpoints, so the bridge is tangent and gap-free.  A concave corner
    makes the neighbours overlap; the offset tangent lines are intersected
    and straight segments are trimmed or extended exactly to that point,
    while curved segments keep their body and are joined to it by straight
    connectors.  Returns ``{"ok", "curves", "perimeter", "reason"}``.

Refusals are structured, never exceptions: a request that cannot be honoured
(for instance an inward offset that collapses a circle to zero radius) comes
back with ``ok`` false and a human-readable ``reason``.  ``ValueError`` is
raised only for malformed input -- a non-finite distance, an empty loop or a
zero-length curve.

Supporting API: :func:`make_line_curve`, :func:`make_circle_curve` and
:func:`make_arc_curve` build the exact primitives; :func:`fit_curve_points`
and :func:`interp_curve_points` expose the underlying B-spline fitter.

Implementation notes
--------------------

  * Pure stdlib and deterministic; no numpy, no randomness.
  * Curves are the harness plain-data 4-tuple ``(control_points, weights,
    degree, knots)`` and surfaces the 6-tuple ``(poles, weights, deg_u,
    deg_v, knots_u, knots_v)``.  Evaluation is delegated to
    :mod:`harnesscad.domain.geometry.parametric.nurbs_curve` and
    :mod:`harnesscad.domain.geometry.parametric.nurbs_surface`.  Inputs may
    be 2-D (lifted to ``z = 0``); results are always 3-D.
  * Shape recognition is done by measurement, not by pattern-matching the
    control net: the curve or surface is densely sampled, a candidate
    primitive is solved for from the samples, and the candidate is accepted
    only if every sample lies on it to a relative tolerance of 1e-9.  A
    curve recognised as circular is reported as ``"circle"`` when the
    samples close up and as ``"arc"`` otherwise.  This keeps recognition
    independent of degree, knot vector and parameterisation.
  * Fitting solves the least-squares normal equations by Cholesky
    factorisation, which is well defined exactly when the sampled basis has
    full column rank; a breakdown is reported as a failed fit rather than
    silently producing a wild curve.  Interior knots for the fit follow the
    standard averaging placement (Piegl & Tiller eq. 9.68).
  * Curve tangents on the refit path come from the analytic NURBS
    derivative, falling back to a central difference where the derivative
    vanishes (cusps); surface normals likewise retry at a slightly interior
    parameter where the normal degenerates (poles of a revolution).
  * Offsetting a curve that lies ON a surface (a geodesic offset) is not
    offered: it needs a closest-point surface inversion the harness does not
    have.

Relation to ``parametric/path_offset.py``: that module offsets 2-D
*polylines* with mitered corners; this module offsets *NURBS* curves and
surfaces, recognises and exactly reconstructs the analytic families, inserts
true rational arc fillets rather than discretised ones, and reports a
measured deviation for every approximate result.

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

# Relative acceptance band for analytic shape recognition.
_RECOGNISE_REL = 1e-9
# Anything below this is treated as numerically zero length.
_TINY = 1e-14

_TAU = 2.0 * math.pi


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


# ===========================================================================
# 1. Vector arithmetic on plain 3-tuples
# ===========================================================================

def _vsub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vadd(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vmul(a: Sequence[float], k: float) -> Vec3:
    return (a[0] * k, a[1] * k, a[2] * k)


def _vdot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vcross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _vlen(a: Sequence[float]) -> float:
    return math.sqrt(_vdot(a, a))


def _vhat(a: Sequence[float]) -> Vec3:
    """Unit vector; raises on a zero-length input."""
    n = _vlen(a)
    if n < 1e-300:
        raise ValueError("zero-length vector cannot be normalised")
    return (a[0] / n, a[1] / n, a[2] / n)


def _vhat_or(a: Sequence[float], fallback: Vec3) -> Vec3:
    """Unit vector, or ``fallback`` when ``a`` is degenerate."""
    return _vhat(a) if _vlen(a) > _TINY else fallback


def _pt3(p: Sequence[float]) -> Vec3:
    """Coerce a 2-D or 3-D point to a float 3-tuple (2-D lifts to z = 0)."""
    k = len(p)
    if k == 3:
        return (float(p[0]), float(p[1]), float(p[2]))
    if k == 2:
        return (float(p[0]), float(p[1]), 0.0)
    raise ValueError("points must be 2-D or 3-D, got dimension %d" % k)


def _finite_distance(d) -> float:
    """Validate an offset distance."""
    d = float(d)
    if math.isnan(d) or math.isinf(d):
        raise ValueError("offset distance must be finite, got %r" % d)
    return d


def _plane_axis(plane_normal: Optional[Sequence[float]]) -> Vec3:
    """Unit plane normal; defaults to +z and tolerates short/zero input."""
    if plane_normal is None:
        return (0.0, 0.0, 1.0)
    comps = [float(c) for c in plane_normal][:3]
    comps += [0.0] * (3 - len(comps))
    return _vhat_or(comps, (0.0, 0.0, 1.0))


# ===========================================================================
# 2. Plain-data adapters and evaluation
# ===========================================================================

def _curve_of(curve) -> CurveData:
    """Normalise any plain-data curve 4-tuple into a 3-D :class:`CurveData`."""
    cps, wts, degree, knots = curve
    return CurveData(tuple(_pt3(p) for p in cps),
                     tuple(float(w) for w in wts),
                     int(degree),
                     tuple(float(k) for k in knots))


def _surface_of(surface) -> SurfaceData:
    """Normalise any plain-data surface 6-tuple into a :class:`SurfaceData`."""
    poles, wts, deg_u, deg_v, ku, kv = surface
    return SurfaceData(tuple(tuple(_pt3(p) for p in row) for row in poles),
                       tuple(tuple(float(w) for w in row) for row in wts),
                       int(deg_u), int(deg_v),
                       tuple(float(k) for k in ku),
                       tuple(float(k) for k in kv))


def _curve_domain(c: CurveData) -> Tuple[float, float]:
    last = len(c.control_points) - 1
    return float(c.knots[c.degree]), float(c.knots[last + 1])


def _surface_domain(s: SurfaceData) -> Tuple[float, float, float, float]:
    last_u = len(s.poles) - 1
    last_v = len(s.poles[0]) - 1
    return (float(s.knots_u[s.deg_u]), float(s.knots_u[last_u + 1]),
            float(s.knots_v[s.deg_v]), float(s.knots_v[last_v + 1]))


def _spread(lo: float, hi: float, count: int) -> List[float]:
    """``count`` parameters spread evenly over the closed span ``[lo, hi]``."""
    count = max(2, int(count))
    step = (hi - lo) / (count - 1)
    return [lo + step * i for i in range(count)]


def _at(c: CurveData, t: float) -> Vec3:
    return _pt3(curve_point(c.control_points, c.weights, c.degree, c.knots, t))


def _sample_curve_pts(c: CurveData, num: int) -> Tuple[List[float], List[Vec3]]:
    """Parameters and points at ``num`` evenly spread parameter values."""
    lo, hi = _curve_domain(c)
    ts = _spread(lo, hi, max(3, int(num)))
    return ts, [_at(c, t) for t in ts]


def _tangent_at(c: CurveData, t: float) -> Optional[Vec3]:
    """Unit tangent: analytic derivative, central difference at a cusp."""
    try:
        d1 = _pt3(curve_derivatives(c.control_points, c.weights, c.degree,
                                    c.knots, t, 1)[1])
        if _vlen(d1) > _TINY:
            return _vhat(d1)
    except (ValueError, IndexError):
        pass
    lo, hi = _curve_domain(c)
    step = (hi - lo) * 1e-4
    chord = _vsub(_at(c, min(t + step, hi)), _at(c, max(t - step, lo)))
    return _vhat(chord) if _vlen(chord) > _TINY else None


def _surface_pt(s: SurfaceData, u: float, v: float) -> Vec3:
    return _pt3(surface_point(s.poles, s.weights, s.deg_u, s.deg_v,
                              s.knots_u, s.knots_v, u, v))


def _surface_nrm(s: SurfaceData, u: float, v: float) -> Optional[Vec3]:
    """Unit surface normal, retried just inside the domain at a degeneracy.

    The normal is undefined where one partial derivative vanishes (the poles
    of a surface of revolution, for example).  Rather than fail there, the
    normal of a nearby interior parameter is used, which is the limit of the
    normal field approaching the degenerate point.
    """
    u_lo, u_hi, v_lo, v_hi = _surface_domain(s)
    du = (u_hi - u_lo) * 1e-6
    dv = (v_hi - v_lo) * 1e-6
    for pull in (0.0, 1.0, 1000.0):
        uu = min(max(u + du * pull * (1 if u <= (u_lo + u_hi) * 0.5 else -1),
                     u_lo), u_hi)
        vv = min(max(v + dv * pull * (1 if v <= (v_lo + v_hi) * 0.5 else -1),
                     v_lo), v_hi)
        try:
            n = _pt3(surface_normal(s.poles, s.weights, s.deg_u, s.deg_v,
                                    s.knots_u, s.knots_v, uu, vv))
        except (ValueError, ZeroDivisionError):
            continue
        if _vlen(n) > _TINY and all(math.isfinite(x) for x in n):
            return _vhat(n)
    return None


# ===========================================================================
# 3. Exact primitive constructors
# ===========================================================================

def make_line_curve(p1: Sequence[float], p2: Sequence[float]) -> CurveData:
    """Straight segment as a degree-1 NURBS with a clamped unit knot vector."""
    return CurveData((_pt3(p1), _pt3(p2)), (1.0, 1.0), 1, (0.0, 0.0, 1.0, 1.0))


def make_circle_curve(center: Sequence[float], radius: float,
                      x_axis: Optional[Sequence[float]] = None,
                      y_axis: Optional[Sequence[float]] = None) -> CurveData:
    """Exact full circle as the standard 9-point rational quadratic NURBS.

    The circle is four quadratic rational Bezier quadrants.  Each quadrant
    interpolates its two axis points and has the corner of the circumscribed
    square as its shoulder; the shoulder weight ``cos(45 deg) = sqrt(2)/2``
    is what makes the segment an exact conic arc rather than a parabola.
    """
    ctr = _pt3(center)
    ex = _vhat(_pt3(x_axis)) if x_axis is not None else (1.0, 0.0, 0.0)
    ey = _vhat(_pt3(y_axis)) if y_axis is not None else (0.0, 1.0, 0.0)
    r = float(radius)
    shoulder_w = math.cos(math.pi / 4.0)

    # (a, b) coefficients of the axis / square-corner control polygon.
    frame = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0),
             (-1, -1), (0, -1), (1, -1), (1, 0))
    cps = tuple(_vadd(ctr, _vadd(_vmul(ex, r * a), _vmul(ey, r * b)))
                for a, b in frame)
    weights = tuple(1.0 if i % 2 == 0 else shoulder_w for i in range(9))
    knots = (0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0)
    return CurveData(cps, weights, 2, knots)


def make_arc_curve(center: Sequence[float], radius: float,
                   start_angle: float, end_angle: float,
                   x_axis: Optional[Sequence[float]] = None,
                   y_axis: Optional[Sequence[float]] = None) -> CurveData:
    """Exact rational quadratic circular arc over a signed sweep.

    The sweep ``end_angle - start_angle`` is divided into the fewest equal
    pieces of at most 90 degrees.  Each piece is one rational quadratic
    Bezier: its outer control point is where the two end tangents meet, and
    its weight is ``cos(half the piece sweep)``.  Negative sweeps are
    supported and traverse the arc the other way.
    """
    ctr = _pt3(center)
    ex = _vhat(_pt3(x_axis)) if x_axis is not None else (1.0, 0.0, 0.0)
    ey = _vhat(_pt3(y_axis)) if y_axis is not None else (0.0, 1.0, 0.0)
    r = float(radius)
    total = float(end_angle) - float(start_angle)
    if abs(total) < _TINY:
        raise ValueError("arc sweep must be non-zero")

    pieces = max(1, int(math.ceil(abs(total) / (math.pi / 2.0) - 1e-12)))
    step = total / pieces
    shoulder_w = math.cos(abs(step) / 2.0)

    def on_arc(angle: float) -> Vec3:
        return _vadd(ctr, _vadd(_vmul(ex, r * math.cos(angle)),
                                _vmul(ey, r * math.sin(angle))))

    def dir_at(angle: float) -> Vec3:
        return _vadd(_vmul(ex, -math.sin(angle)), _vmul(ey, math.cos(angle)))

    cps: List[Vec3] = [on_arc(float(start_angle))]
    weights: List[float] = [1.0]
    angle = float(start_angle)
    for _ in range(pieces):
        nxt = angle + step
        a_pt, b_pt = on_arc(angle), on_arc(nxt)
        a_dir, b_dir = dir_at(angle), dir_at(nxt)
        # Shoulder = a_pt + s * a_dir = b_pt - u * b_dir, solved in (ex, ey).
        gap = _vsub(b_pt, a_pt)
        m = ((_vdot(a_dir, ex), -_vdot(b_dir, ex)),
             (_vdot(a_dir, ey), -_vdot(b_dir, ey)))
        rhs = (_vdot(gap, ex), _vdot(gap, ey))
        det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
        s = ((rhs[0] * m[1][1] - rhs[1] * m[0][1]) / det
             if abs(det) > _TINY else 0.0)
        cps.append(_vadd(a_pt, _vmul(a_dir, s)))
        weights.append(shoulder_w)
        cps.append(b_pt)
        weights.append(1.0)
        angle = nxt

    knots: List[float] = [0.0, 0.0, 0.0]
    for k in range(1, pieces):
        knots += [k / pieces, k / pieces]
    knots += [1.0, 1.0, 1.0]
    return CurveData(tuple(cps), tuple(weights), 2, tuple(knots))


# ===========================================================================
# 4. Least-squares B-spline fitting
# ===========================================================================

def _chord_params(points: Sequence[Vec3]) -> List[float]:
    """Chord-length parameters normalised onto ``[0, 1]``."""
    count = len(points)
    if count < 2:
        return [0.0] * count
    steps = [_vlen(_vsub(b, a)) for a, b in zip(points, points[1:])]
    total = math.fsum(steps)
    if total < _TINY:
        return [i / (count - 1) for i in range(count)]
    out = [0.0]
    walked = 0.0
    for step in steps:
        walked += step
        out.append(walked / total)
    out[-1] = 1.0
    return out


def _clamped_ends(num_ctrl: int, degree: int) -> List[float]:
    """Clamped knot vector with the interior knots left at zero."""
    knots = [0.0] * (num_ctrl + degree + 1)
    for i in range(num_ctrl, num_ctrl + degree + 1):
        knots[i] = 1.0
    return knots


def _averaged_fit_knots(ts: Sequence[float], num_ctrl: int,
                        degree: int) -> List[float]:
    """Interior knots for a least-squares fit (Piegl & Tiller eq. 9.68).

    Each interior knot is placed by linear interpolation between two data
    parameters, at a stride chosen so that every knot span receives roughly
    the same number of data points.  This keeps the normal equations
    non-singular whenever the data are reasonably distributed.
    """
    knots = _clamped_ends(num_ctrl, degree)
    interior = num_ctrl - 1 - degree
    if interior <= 0:
        return knots
    last = len(ts) - 1
    stride = (last + 1) / (interior + 1)
    for j in range(1, interior + 1):
        pos = j * stride
        left = int(pos)
        frac = pos - left
        left = min(max(left, 1), last)
        knots[degree + j] = (1.0 - frac) * ts[left - 1] + frac * ts[left]
    # Knots must be non-decreasing even if the data parameters bunch up.
    for i in range(degree + 1, num_ctrl):
        if knots[i] < knots[i - 1]:
            knots[i] = knots[i - 1]
    return knots


def _design_matrix(ts: Sequence[float], degree: int, knots: Sequence[float],
                   num_ctrl: int) -> List[List[float]]:
    """Rows of B-spline basis values, one row per parameter."""
    return [all_basis(num_ctrl - 1, degree, t, knots) for t in ts]


def _normal_equations(design: Sequence[Sequence[float]],
                      rhs: Sequence[Sequence[float]]
                      ) -> Tuple[List[List[float]], List[List[float]]]:
    """Accumulate ``(A^T A, A^T B)``, skipping the many zero basis values."""
    cols = len(design[0])
    width = len(rhs[0])
    gram = [[0.0] * cols for _ in range(cols)]
    proj = [[0.0] * width for _ in range(cols)]
    for row, target in zip(design, rhs):
        live = [i for i in range(cols) if row[i] != 0.0]
        for i in live:
            weight = row[i]
            gram_i = gram[i]
            for j in live:
                gram_i[j] += weight * row[j]
            proj_i = proj[i]
            for k in range(width):
                proj_i[k] += weight * target[k]
    return gram, proj


def _cholesky_solve(gram: List[List[float]], proj: List[List[float]]
                    ) -> List[List[float]]:
    """Solve a symmetric positive-definite system by Cholesky factorisation.

    ``A^T A`` is positive definite exactly when the sampled basis has full
    column rank, so a non-positive pivot is a faithful rank-deficiency
    signal.  It is reported as a ``ValueError`` so the caller can back off to
    fewer control points instead of returning an arbitrary curve.
    """
    n = len(gram)
    width = len(proj[0])
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        row_i = lower[i]
        for j in range(i):
            row_j = lower[j]
            acc = gram[i][j] - math.fsum(row_i[k] * row_j[k] for k in range(j))
            row_i[j] = acc / row_j[j]
        pivot = gram[i][i] - math.fsum(row_i[k] * row_i[k] for k in range(i))
        if not (pivot > 0.0) or not math.isfinite(pivot):
            raise ValueError("rank-deficient least-squares system "
                             "(no unique fit for %d control points)" % n)
        row_i[i] = math.sqrt(pivot)

    # Forward substitution L y = proj, then back substitution L^T x = y.
    y = [[0.0] * width for _ in range(n)]
    for i in range(n):
        for k in range(width):
            acc = proj[i][k] - math.fsum(lower[i][j] * y[j][k]
                                         for j in range(i))
            y[i][k] = acc / lower[i][i]
    x = [[0.0] * width for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for k in range(width):
            acc = y[i][k] - math.fsum(lower[j][i] * x[j][k]
                                      for j in range(i + 1, n))
            x[i][k] = acc / lower[i][i]
    return x


def _solve_least_squares(design: Sequence[Sequence[float]],
                         rhs: Sequence[Sequence[float]]) -> List[List[float]]:
    """Least-squares solution of ``design @ X = rhs`` via normal equations."""
    gram, proj = _normal_equations(design, rhs)
    return _cholesky_solve(gram, proj)


def _fit_ctrl_points(design: Sequence[Sequence[float]],
                     points: Sequence[Vec3]) -> List[Vec3]:
    """Least-squares control points for a point cloud."""
    return [(row[0], row[1], row[2])
            for row in _solve_least_squares(design, points)]


def _max_gap(curve: CurveData, ts: Sequence[float],
             points: Sequence[Vec3]) -> float:
    """Largest distance between ``curve`` at ``ts`` and the matching points."""
    return max(_vlen(_vsub(_at(curve, t), p)) for t, p in zip(ts, points))


def interp_curve_points(points: Sequence[Vec3], degree: int = 3) -> CurveData:
    """Interpolate a non-rational B-spline through every one of ``points``.

    Chord-length parameterisation with averaging knot placement (Piegl &
    Tiller 9.3.6), then a square collocation solve.  Raises ``ValueError``
    for fewer than two points or a rank-deficient system.
    """
    pts = [_pt3(p) for p in points]
    count = len(pts)
    if count < 2:
        raise ValueError("interp_curve_points requires at least 2 points")
    degree = min(int(degree), count - 1)
    ts = _chord_params(pts)
    knots = uniform_clamped_knots(count - 1, degree)
    for j in range(1, count - degree):
        knots[j + degree] = math.fsum(ts[j:j + degree]) / degree
    design = _design_matrix(ts, degree, knots, count)
    return CurveData(tuple(_fit_ctrl_points(design, pts)),
                     (1.0,) * count, degree, tuple(knots))


def fit_curve_points(points: Sequence[Vec3], degree: int = 3,
                     tolerance: float = 1e-3, max_ctrl: int = 64) -> dict:
    """Least-squares B-spline fit to ``points`` with a growing control net.

    The number of control points is raised one at a time from ``degree + 1``
    until the largest residual at the data parameters is within
    ``tolerance``, or until ``max_ctrl`` (or the number of points) is
    reached.  This routine never raises: every outcome, including a failure
    to reach the tolerance, is returned as
    ``{"ok", "curve", "deviation", "num_ctrl", "reason"}``.  When the
    tolerance is missed, ``curve`` still holds the best fit obtained and
    ``deviation`` its measured residual.
    """
    try:
        pts = [_pt3(p) for p in points]
        count = len(pts)
        if count < 2:
            return {"ok": False, "curve": None, "deviation": float("inf"),
                    "num_ctrl": 0, "reason": "need at least 2 points"}

        extent = max(_vlen(_vsub(p, pts[0])) for p in pts)
        if extent < _TINY:
            return {"ok": True, "curve": make_line_curve(pts[0], pts[0]),
                    "deviation": 0.0, "num_ctrl": 2,
                    "reason": "degenerate: all points identical"}

        degree = min(int(degree), count - 1)
        ts = _chord_params(pts)
        ceiling = min(int(max_ctrl), count)

        best_curve: Optional[CurveData] = None
        best_dev = float("inf")
        best_n = degree + 1
        for num_ctrl in range(degree + 1, ceiling + 1):
            knots = _averaged_fit_knots(ts, num_ctrl, degree)
            design = _design_matrix(ts, degree, knots, num_ctrl)
            try:
                ctrl = _fit_ctrl_points(design, pts)
            except ValueError:
                continue
            candidate = CurveData(tuple(ctrl), (1.0,) * num_ctrl,
                                  degree, tuple(knots))
            dev = _max_gap(candidate, ts, pts)
            if dev < best_dev:
                best_curve, best_dev, best_n = candidate, dev, num_ctrl
            if dev <= tolerance:
                return {"ok": True, "curve": candidate, "deviation": dev,
                        "num_ctrl": num_ctrl, "reason": ""}

        return {"ok": False, "curve": best_curve, "deviation": best_dev,
                "num_ctrl": best_n,
                "reason": "tolerance %g not achieved; best deviation %.4g"
                          % (tolerance, best_dev)}
    except Exception as exc:  # a fit failure is data, not an exception
        return {"ok": False, "curve": None, "deviation": float("inf"),
                "num_ctrl": 0, "reason": str(exc)}


# ===========================================================================
# 5. Analytic recognition of planar curves
# ===========================================================================

class _CircularFit(NamedTuple):
    """A curve recognised as lying on a circle, measured from its samples."""

    centre: Vec3
    radius: float
    x_axis: Vec3
    y_axis: Vec3
    sweep: float
    closed: bool


def _straight_direction(c: CurveData) -> Optional[Vec3]:
    """Unit direction if every control point is collinear, else ``None``.

    A collinear control net means the curve is a (possibly unevenly
    parameterised) straight segment, and translating the net translates the
    curve exactly whatever the degree or the weights.
    """
    cps = c.control_points
    span = _vsub(cps[-1], cps[0])
    if _vlen(span) < _TINY:
        return None
    axis = _vhat(span)
    for p in cps:
        if _vlen(_vcross(_vsub(p, cps[0]), axis)) > _RECOGNISE_REL:
            return None
    return axis


def _wrap_to_pi(angle: float) -> float:
    """Fold an angle into ``(-pi, pi]``."""
    folded = math.fmod(angle, _TAU)
    if folded > math.pi:
        folded -= _TAU
    elif folded <= -math.pi:
        folded += _TAU
    return folded


def _circle_through(a: Tuple[float, float], b: Tuple[float, float],
                    c: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    """Planar circumcentre of three 2-D points, or ``None`` if collinear."""
    (ax, ay), (bx, by), (cx, cy) = a, b, c
    twice_area = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    scale = max(abs(ax), abs(ay), abs(bx), abs(by), abs(cx), abs(cy), 1.0)
    if abs(twice_area) < 1e-12 * scale * scale:
        return None
    sa = ax * ax + ay * ay
    sb = bx * bx + by * by
    sc = cx * cx + cy * cy
    return ((sa * (by - cy) + sb * (cy - ay) + sc * (ay - by)) / twice_area,
            (sa * (cx - bx) + sb * (ax - cx) + sc * (bx - ax)) / twice_area)


def _recognise_circular(c: CurveData, normal: Vec3) -> Optional[_CircularFit]:
    """Recognise a curve as a circular arc or full circle by measurement.

    The curve is densely sampled, a candidate circle is solved from three
    well-separated samples (at the start and at the one- and two-third
    marks, so that a closed curve -- whose start and end coincide -- is
    handled exactly like an open one), and the candidate is accepted only if
    every sample lies on it, and in the given plane, to a relative tolerance
    of 1e-9.  The signed sweep is then accumulated along the samples so a
    reversed parameterisation is preserved.  Returns ``None`` for anything
    that is not circular.
    """
    count = 49
    _, pts = _sample_curve_pts(c, count)
    origin = pts[0]
    extent = max(_vlen(_vsub(p, origin)) for p in pts)
    if extent < _TINY:
        return None

    # Local 2-D frame of the requested plane, anchored at the first sample.
    probe = pts[count // 3]
    try:
        ex = _vhat(_vsub(probe, origin))
    except ValueError:
        return None
    span = _vcross(normal, ex)
    if _vlen(span) < _RECOGNISE_REL:
        return None
    ey = _vhat(span)

    def flatten(p: Vec3) -> Tuple[float, float]:
        rel = _vsub(p, origin)
        return _vdot(rel, ex), _vdot(rel, ey)

    solved = _circle_through(flatten(origin), flatten(probe),
                             flatten(pts[(2 * count) // 3]))
    if solved is None:
        return None  # three collinear samples: straight, not circular
    centre = _vadd(origin, _vadd(_vmul(ex, solved[0]), _vmul(ey, solved[1])))
    radius = _vlen(_vsub(origin, centre))
    if radius < _TINY or radius > 1e12 * max(extent, _TINY):
        return None

    band = _RECOGNISE_REL * max(1.0, radius)
    for p in pts:
        if abs(_vlen(_vsub(p, centre)) - radius) > band:
            return None
        if abs(_vdot(_vsub(p, origin), normal)) > band:
            return None  # off the requested plane

    x_axis = _vhat(_vsub(origin, centre))
    y_axis = _vhat_or(_vcross(normal, x_axis), (0.0, 1.0, 0.0))

    # Accumulate the signed sweep across the samples, unwrapping as we go.
    sweep = 0.0
    previous = 0.0
    for p in pts[1:]:
        rel = _vsub(p, centre)
        angle = math.atan2(_vdot(rel, y_axis), _vdot(rel, x_axis))
        sweep += _wrap_to_pi(angle - previous)
        previous = angle
    if abs(sweep) < 1e-10:
        return None

    closed = _vlen(_vsub(pts[-1], pts[0])) <= _RECOGNISE_REL * extent
    if closed:
        # Snap to a whole turn: the sampled sum is a whole turn up to the
        # rounding of 48 additions.
        sweep = math.copysign(_TAU, sweep)
    return _CircularFit(centre, radius, x_axis, y_axis, sweep, closed)


# ===========================================================================
# 6. offset_curve
# ===========================================================================

def _exact(curve, label: str, key: str = "curve") -> dict:
    return {"ok": True, key: curve, "actual_max_deviation": 0.0,
            "reason": "", "analytic": label}


def _decline(reason: str, label: str, key: str = "curve") -> dict:
    return {"ok": False, key: None, "actual_max_deviation": 0.0,
            "reason": reason, "analytic": label}


def _displace_curve(c: CurveData, d: float, normal: Vec3,
                    num: int) -> List[Vec3]:
    """Sample the curve and push each sample along its right-hand normal."""
    ts, pts = _sample_curve_pts(c, num)
    moved: List[Vec3] = []
    for t, p in zip(ts, pts):
        tangent = _tangent_at(c, t)
        side = _vcross(normal, tangent) if tangent is not None else None
        if side is None or _vlen(side) < _TINY:
            moved.append(p)  # cusp / tangent along the plane normal
        else:
            moved.append(_vadd(p, _vmul(side, d / _vlen(side))))
    return moved


def offset_curve(curve, d: float, *, tol: float = 1e-4,
                 plane_normal: Optional[Sequence[float]] = None,
                 num_samples: int = 200) -> dict:
    """Offset a planar curve by the signed distance ``d``.

    The offset direction at a point is the right-hand normal
    ``R = unit(cross(plane_normal, T))`` of the unit tangent ``T``; positive
    ``d`` moves along ``+R``.  ``plane_normal`` defaults to ``+z``.

    Straight and circular inputs are recognised and reconstructed exactly,
    with ``actual_max_deviation`` reported as ``0.0`` and ``analytic`` set to
    ``"line"``, ``"arc"`` or ``"circle"``.  For circular inputs the radius
    convention is ``r_new = r + d``, so a positive ``d`` always grows the
    radius; if that would take the radius to zero or below, the request is
    declined with a reason rather than approximated.

    Any other curve takes the refit path: ``num_samples`` samples are
    displaced and a B-spline is least-squares fitted to them.  The result is
    an approximation, and ``actual_max_deviation`` is the largest MEASURED
    distance from the fit to the displaced samples; ``ok`` is true only if
    that measurement is within ``tol``.  ``analytic`` is ``""``.

    Returns ``{"ok", "curve", "actual_max_deviation", "reason", "analytic"}``.
    Raises ``ValueError`` if ``d`` is not finite or the curve has no length.
    """
    c = _curve_of(curve)
    d = _finite_distance(d)
    normal = _plane_axis(plane_normal)

    lo, hi = _curve_domain(c)
    probes = [_at(c, t) for t in (lo, (lo + hi) * 0.5, hi)]
    if all(_vlen(_vsub(p, probes[0])) < _TINY for p in probes[1:]):
        raise ValueError("curve is degenerate (zero length)")

    # -- exact case 1: straight segment; translate the whole control net. --
    axis = _straight_direction(c)
    if axis is not None:
        side = _vcross(normal, axis)
        if _vlen(side) > _TINY:
            shift = _vmul(side, d / _vlen(side))
            shifted = tuple(_vadd(p, shift) for p in c.control_points)
            return _exact(CurveData(shifted, c.weights, c.degree, c.knots),
                          "line")

    # -- exact case 2: circular arc or full circle; rebuild concentrically. --
    circ = _recognise_circular(c, normal)
    if circ is not None:
        label = "circle" if circ.closed else "arc"
        grown = circ.radius + d
        if grown <= 0.0:
            return _decline(
                "offset distance %g collapses %s of radius %g"
                % (d, label, circ.radius), label)
        return _exact(make_arc_curve(circ.centre, grown, 0.0, circ.sweep,
                                     x_axis=circ.x_axis, y_axis=circ.y_axis),
                      label)

    # -- approximate case: displace samples and refit. --
    cloud = _displace_curve(c, d, normal, num_samples)
    fit_degree = min(3, c.degree)
    fit = fit_curve_points(cloud, degree=fit_degree, tolerance=tol,
                           max_ctrl=max(16, int(num_samples) // 4))

    approx = fit["curve"]
    deviation = float(fit["deviation"])
    if approx is None:
        # The whole ladder was rank-deficient; interpolation still gives a
        # curve through every displaced sample, whose residual we measure.
        approx = interp_curve_points(cloud, degree=fit_degree)
        lo_a, hi_a = _curve_domain(approx)
        ts = [lo_a + (hi_a - lo_a) * t for t in _chord_params(cloud)]
        deviation = _max_gap(approx, ts, cloud)

    ok = math.isfinite(deviation) and deviation <= tol
    return {"ok": ok, "curve": approx, "actual_max_deviation": deviation,
            "reason": "" if ok else "tolerance %g not achieved; best dev %.4g"
                                    % (tol, deviation),
            "analytic": ""}


# ===========================================================================
# 7. offset_surface
# ===========================================================================

class _Probe(NamedTuple):
    """One grid sample of a surface: position and (possibly missing) normal."""

    point: Vec3
    normal: Optional[Vec3]


def _probe_surface(s: SurfaceData, grid: int
                   ) -> Tuple[List[float], List[float], List[List[_Probe]]]:
    """Sample the surface on a ``grid x grid`` lattice of its own domain."""
    u_lo, u_hi, v_lo, v_hi = _surface_domain(s)
    us = _spread(u_lo, u_hi, grid)
    vs = _spread(v_lo, v_hi, grid)
    lattice = [[_Probe(_surface_pt(s, u, v), _surface_nrm(s, u, v))
                for v in vs] for u in us]
    return us, vs, lattice


def _flat_probes(lattice: Sequence[Sequence[_Probe]]) -> List[_Probe]:
    return [probe for row in lattice for probe in row]


def _recognise_plane(probes: Sequence[_Probe]) -> Optional[Vec3]:
    """Unit normal if every sample lies in one plane, else ``None``."""
    normals = [p.normal for p in probes if p.normal is not None]
    if not normals:
        return None
    axis = normals[0]
    if any(_vlen(_vcross(axis, n)) > _RECOGNISE_REL for n in normals):
        return None
    base = probes[0].point
    extent = max(_vlen(_vsub(p.point, base)) for p in probes)
    band = _RECOGNISE_REL * max(1.0, extent)
    if any(abs(_vdot(_vsub(p.point, base), axis)) > band for p in probes):
        return None
    return axis


def _recognise_sphere(probes: Sequence[_Probe]) -> Optional[Tuple[Vec3, float]]:
    """Centre and radius if every sample lies on one sphere, else ``None``.

    The centre is recovered by the standard linearisation of the sphere
    equation: writing ``|p - c|^2 = r^2`` as ``2 c.p + (r^2 - |c|^2) =
    |p|^2`` makes the four unknowns ``(c, r^2 - |c|^2)`` linear in the
    samples, so a least-squares solve locates the candidate centre.  The
    candidate is accepted only after checking every sample against it.
    """
    pts = [p.point for p in probes]
    if len(pts) < 4:
        return None
    design = [[2.0 * p[0], 2.0 * p[1], 2.0 * p[2], 1.0] for p in pts]
    rhs = [[_vdot(p, p)] for p in pts]
    try:
        solution = _solve_least_squares(design, rhs)
    except ValueError:
        return None  # degenerate (coplanar, collinear, ...)
    centre = (solution[0][0], solution[1][0], solution[2][0])
    squared = solution[3][0] + _vdot(centre, centre)
    if not math.isfinite(squared) or squared <= _TINY:
        return None
    radius = math.sqrt(squared)
    band = _RECOGNISE_REL * radius
    if any(abs(_vlen(_vsub(p, centre)) - radius) > band for p in pts):
        return None
    return centre, radius


def _radial_sign(probes: Sequence[_Probe], centre: Vec3) -> float:
    """+1 if the surface normals point away from ``centre``, -1 if toward."""
    for probe in probes:
        if probe.normal is None:
            continue
        radial = _vsub(probe.point, centre)
        if _vlen(radial) > _TINY:
            return 1.0 if _vdot(probe.normal, _vhat(radial)) >= 0.0 else -1.0
    return 1.0


def _fit_across(design: Sequence[Sequence[float]],
                rows: Sequence[Sequence[Vec3]]) -> List[List[Vec3]]:
    """Least-squares fit every row of points independently."""
    return [_fit_ctrl_points(design, row) for row in rows]


def _transpose(grid: Sequence[Sequence[Vec3]]) -> List[List[Vec3]]:
    return [list(col) for col in zip(*grid)]


def offset_surface(surface, d: float, *, tol: float = 1e-4,
                   grid_samples: int = 20) -> dict:
    """Offset a surface by the signed distance ``d`` along its unit normal.

    Planes and spheres are recognised from the sampled geometry and offset
    exactly -- a plane by translating its control net along the normal, a
    sphere by scaling its control net about the recognised centre -- with
    ``actual_max_deviation`` reported as ``0.0`` and ``analytic`` set to
    ``"plane"`` or ``"sphere"``.  For a sphere the new radius follows the
    orientation of the surface's own normals, and a distance that would
    collapse it is declined with a reason.

    Any other surface takes the refit path: a ``grid_samples`` square
    lattice is displaced along the analytic normal and a tensor-product
    B-spline is fitted to the displaced lattice, first across each row and
    then down the resulting columns.  That result is an approximation;
    ``actual_max_deviation`` is the largest MEASURED distance from the
    fitted surface to the displaced lattice and ``ok`` is true only if it is
    within ``tol``.  ``analytic`` is ``""``.

    Returns ``{"ok", "surface", "actual_max_deviation", "reason",
    "analytic"}``.  Raises ``ValueError`` if ``d`` is not finite.
    """
    s = _surface_of(surface)
    d = _finite_distance(d)
    grid = max(4, int(grid_samples))
    _, _, lattice = _probe_surface(s, grid)
    probes = _flat_probes(lattice)

    # -- exact case 1: plane; translate every pole along the normal. --
    plane_axis = _recognise_plane(probes)
    if plane_axis is not None:
        shift = _vmul(plane_axis, d)
        moved = tuple(tuple(_vadd(p, shift) for p in row) for row in s.poles)
        return _exact(SurfaceData(moved, s.weights, s.deg_u, s.deg_v,
                                  s.knots_u, s.knots_v), "plane", "surface")

    # -- exact case 2: sphere; scale every pole about the centre. --
    ball = _recognise_sphere(probes)
    if ball is not None:
        centre, radius = ball
        grown = radius + _radial_sign(probes, centre) * d
        if grown <= 0.0:
            return _decline("offset distance %g collapses sphere of radius %g"
                            % (d, radius), "sphere", "surface")
        factor = grown / radius
        moved = tuple(tuple(_vadd(centre, _vmul(_vsub(p, centre), factor))
                            for p in row) for row in s.poles)
        return _exact(SurfaceData(moved, s.weights, s.deg_u, s.deg_v,
                                  s.knots_u, s.knots_v), "sphere", "surface")

    # -- approximate case: displace the lattice and refit it. --
    target = [[_vadd(probe.point, _vmul(probe.normal, d))
               if probe.normal is not None else probe.point
               for probe in row] for row in lattice]

    deg_u = min(3, s.deg_u)
    deg_v = min(3, s.deg_v)
    n_u = max(deg_u + 1, min(grid, 16))
    n_v = max(deg_v + 1, min(grid, 16))

    # One consistent uniform parameterisation is used both to fit and to
    # evaluate the result, so the deviation reported below is the deviation
    # a caller evaluating the returned surface will actually observe.
    ts = _spread(0.0, 1.0, grid)
    knots_u = _averaged_fit_knots(ts, n_u, deg_u)
    knots_v = _averaged_fit_knots(ts, n_v, deg_v)
    design_u = _design_matrix(ts, deg_u, knots_u, n_u)
    design_v = _design_matrix(ts, deg_v, knots_v, n_v)

    try:
        # Pass 1: collapse each u-row of samples to n_v control points.
        partial = _fit_across(design_v, target)
        # Pass 2: collapse each of those v-columns to n_u control points.
        net = _transpose(_fit_across(design_u, _transpose(partial)))
    except ValueError as exc:
        return {"ok": False, "surface": None,
                "actual_max_deviation": float("inf"),
                "reason": "surface refit failed: %s" % exc, "analytic": ""}

    fitted = SurfaceData(tuple(tuple(row) for row in net),
                         tuple((1.0,) * n_v for _ in range(n_u)),
                         deg_u, deg_v, tuple(knots_u), tuple(knots_v))

    deviation = 0.0
    for i, u in enumerate(ts):
        for j, v in enumerate(ts):
            gap = _vlen(_vsub(_surface_pt(fitted, u, v), target[i][j]))
            if gap > deviation:
                deviation = gap

    ok = deviation <= tol
    return {"ok": ok, "surface": fitted, "actual_max_deviation": deviation,
            "reason": "" if ok else "tolerance %g not achieved; best dev %.4g"
                                    % (tol, deviation),
            "analytic": ""}


# ===========================================================================
# 8. offset_loop
# ===========================================================================

class _Joint(NamedTuple):
    """A corner of the offset loop between segment ``i`` and segment ``i+1``."""

    convex: bool
    pivot: Vec3                 # the ORIGINAL corner vertex
    meet: Optional[Vec3]        # concave only: where the offset lines cross


def _lines_meet(p_a: Vec3, dir_a: Vec3, p_b: Vec3, dir_b: Vec3,
                normal: Vec3) -> Optional[Vec3]:
    """Intersect two coplanar lines, or ``None`` if they do not cross once."""
    try:
        ex = _vhat(dir_a)
    except ValueError:
        return None
    span = _vcross(normal, ex)
    if _vlen(span) < 1e-12:
        return None
    ey = _vhat(span)
    m = ((_vdot(dir_a, ex), -_vdot(dir_b, ex)),
         (_vdot(dir_a, ey), -_vdot(dir_b, ey)))
    gap = _vsub(p_b, p_a)
    rhs = (_vdot(gap, ex), _vdot(gap, ey))
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    if abs(det) < _TINY:
        return None  # parallel offsets
    step = (rhs[0] * m[1][1] - rhs[1] * m[0][1]) / det
    return _vadd(p_a, _vmul(dir_a, step))


def _corner_arc(pivot: Vec3, p_from: Vec3, p_to: Vec3, radius: float,
                normal: Vec3) -> Optional[CurveData]:
    """Exact arc of ``radius`` about ``pivot`` bridging a convex corner.

    Both offset endpoints sit at distance ``radius`` from the original
    corner vertex by the definition of the offset, so the arc centred there
    meets each neighbour tangentially.  The bridge takes the short way
    round: the sweep is the signed angle between the two radii folded into
    ``(-pi, pi]``.
    """
    from_radius = _vsub(p_from, pivot)
    to_radius = _vsub(p_to, pivot)
    if _vlen(from_radius) < _TINY or _vlen(to_radius) < _TINY:
        return None
    x_axis = _vhat(from_radius)
    span = _vcross(normal, x_axis)
    if _vlen(span) < _TINY:
        return None
    y_axis = _vhat(span)
    sweep = _wrap_to_pi(math.atan2(_vdot(to_radius, y_axis),
                                   _vdot(to_radius, x_axis)))
    if abs(sweep) < 1e-12:
        return None
    try:
        return make_arc_curve(pivot, radius, 0.0, sweep,
                              x_axis=x_axis, y_axis=y_axis)
    except ValueError:
        return None


def _loop_length(segments: Sequence[CurveData]) -> float:
    """Chord-length perimeter, 50 samples per segment."""
    total = 0.0
    for seg in segments:
        pts = _sample_curve_pts(seg, 50)[1]
        total += math.fsum(_vlen(_vsub(b, a)) for a, b in zip(pts, pts[1:]))
    return total


def offset_loop(curves: Sequence, d: float, *,
                plane_normal: Optional[Sequence[float]] = None,
                tol: float = 1e-4, num_samples: int = 100) -> dict:
    """Offset a closed planar loop of curves, keeping it closed.

    Every segment is offset with :func:`offset_curve`; if any segment cannot
    be offset the whole request is declined with that segment's reason.  The
    corners are then repaired:

      * a CONVEX corner (``d * turn_sign < 0``, with ``turn_sign =
        dot(cross(T_end, T_next_start), plane_normal)``) leaves a gap, which
        is bridged by an exact rational arc of radius ``|d|`` centred on the
        original corner vertex;
      * a CONCAVE corner leaves an overlap, so the two offset tangent lines
        are intersected: straight segments are trimmed or extended exactly
        to that point, and curved segments are kept whole and joined to it
        by straight connectors.

    Returns ``{"ok", "curves", "perimeter", "reason"}`` where ``perimeter``
    is the chord-length total over 50 samples of each output segment.
    Raises ``ValueError`` on an empty loop or a non-finite ``d``.
    """
    if not curves:
        raise ValueError("curves list is empty")
    d = _finite_distance(d)
    normal = _plane_axis(plane_normal)

    sources = [_curve_of(crv) for crv in curves]
    offsets: List[CurveData] = []
    for crv in sources:
        res = offset_curve(crv, d, tol=tol, plane_normal=normal,
                           num_samples=num_samples)
        if not res["ok"] or res["curve"] is None:
            return {"ok": False, "curves": [], "perimeter": 0.0,
                    "reason": "segment offset failed: %s" % res["reason"]}
        offsets.append(res["curve"])

    n = len(offsets)
    radius = abs(d)
    if radius < _TINY:
        return {"ok": True, "curves": offsets,
                "perimeter": _loop_length(offsets), "reason": ""}

    # Endpoint and end-tangent data for every offset segment.
    heads: List[Vec3] = []
    tails: List[Vec3] = []
    head_dirs: List[Vec3] = []
    tail_dirs: List[Vec3] = []
    for seg in offsets:
        lo, hi = _curve_domain(seg)
        heads.append(_at(seg, lo))
        tails.append(_at(seg, hi))
        head_dirs.append(_tangent_at(seg, lo) or (0.0, 0.0, 0.0))
        tail_dirs.append(_tangent_at(seg, hi) or (0.0, 0.0, 0.0))

    # The original corner vertex is where the source segments met.
    joints: List[_Joint] = []
    for i in range(n):
        j = (i + 1) % n
        turn = _vdot(_vcross(tail_dirs[i], head_dirs[j]), normal)
        convex = d * turn < 0.0
        pivot = _at(sources[i], _curve_domain(sources[i])[1])
        meet = None if convex else _lines_meet(tails[i], tail_dirs[i],
                                               heads[j], head_dirs[j], normal)
        joints.append(_Joint(convex, pivot, meet))

    straight = [_straight_direction(seg) is not None for seg in offsets]

    out: List[CurveData] = []
    for i in range(n):
        entry = joints[(i - 1) % n].meet   # None at a convex corner
        exit_ = joints[i].meet

        if straight[i]:
            # A straight offset can absorb the trim/extend into itself.
            a = entry if entry is not None else heads[i]
            b = exit_ if exit_ is not None else tails[i]
            if _vlen(_vsub(b, a)) > _TINY:
                out.append(make_line_curve(a, b))
        else:
            if entry is not None and _vlen(_vsub(entry, heads[i])) > 1e-12:
                out.append(make_line_curve(entry, heads[i]))
            out.append(offsets[i])
            if exit_ is not None and _vlen(_vsub(tails[i], exit_)) > 1e-12:
                out.append(make_line_curve(tails[i], exit_))

        if joints[i].convex:
            j = (i + 1) % n
            bridge = _corner_arc(joints[i].pivot, tails[i], heads[j],
                                 radius, normal)
            if bridge is not None:
                out.append(bridge)
            elif _vlen(_vsub(heads[j], tails[i])) > 1e-12:
                # No usable arc: a chord still keeps the loop connected.
                out.append(make_line_curve(tails[i], heads[j]))

    if not out:
        return {"ok": False, "curves": [], "perimeter": 0.0,
                "reason": "no output segments generated"}
    return {"ok": True, "curves": out, "perimeter": _loop_length(out),
            "reason": ""}


# ===========================================================================
# 9. Selfcheck
# ===========================================================================

def _distance_to_polyline(p: Vec3, pts: Sequence[Vec3]) -> float:
    """Shortest distance from ``p`` to the polyline through ``pts``."""
    best = float("inf")
    for a, b in zip(pts, pts[1:]):
        ab = _vsub(b, a)
        denom = _vdot(ab, ab)
        t = 0.0 if denom < 1e-300 else min(1.0, max(0.0,
                                                    _vdot(_vsub(p, a), ab)
                                                    / denom))
        best = min(best, _vlen(_vsub(p, _vadd(a, _vmul(ab, t)))))
    return best


def _revolved_sphere(centre: Vec3, radius: float) -> SurfaceData:
    """Exact rational sphere built as a surface of revolution about +z.

    The meridian is a 180-degree rational arc in the xz half-plane; each of
    its control points is spun into a scaled copy of the unit circle's
    control polygon, and the weights multiply.  That is the standard
    revolution construction and reproduces the sphere exactly.
    """
    meridian = make_arc_curve((0.0, 0.0, 0.0), radius,
                              -math.pi / 2.0, math.pi / 2.0,
                              x_axis=(1.0, 0.0, 0.0), y_axis=(0.0, 0.0, 1.0))
    ring = make_circle_curve((0.0, 0.0, 0.0), 1.0)
    poles = tuple(
        tuple((centre[0] + m[0] * r[0], centre[1] + m[0] * r[1],
               centre[2] + m[2]) for m in meridian.control_points)
        for r in ring.control_points)
    weights = tuple(tuple(rw * mw for mw in meridian.weights)
                    for rw in ring.weights)
    return SurfaceData(poles, weights, 2, 2, ring.knots, meridian.knots)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.geometry.parametric.offset_nurbs",
        description="NURBS curve / surface / closed-loop offsetting: exact "
                    "reconstruction for the recognised analytic families, "
                    "least-squares refit with a measured deviation for "
                    "everything else.",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="offset a line, a circle, an arc, a square loop "
                             "(outward and inward), a freeform curve and "
                             "three surfaces on synthetic geometry, and "
                             "verify the exactness and honesty contract.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    # 1. Straight line -> exact, and every point is |d| from the original.
    a, b = (0.0, 0.0, 0.0), (3.0, 4.0, 0.0)
    res = offset_curve(make_line_curve(a, b), 1.0)
    assert res["ok"] and res["analytic"] == "line"
    assert res["actual_max_deviation"] == 0.0
    for t in (0.0, 0.25, 0.5, 1.0):
        gap = _distance_to_polyline(_at(res["curve"], t), [a, b])
        assert abs(gap - 1.0) < 1e-12, gap
    print("[selfcheck] line offset: analytic, deviation=0, distance exact")

    # 2. Full circle -> exact r + d; a collapsing distance is declined.
    centre = (1.0, 2.0, 0.0)
    res = offset_curve(make_circle_curve(centre, 2.0), 0.5)
    assert res["ok"] and res["analytic"] == "circle"
    assert res["actual_max_deviation"] == 0.0
    for p in _sample_curve_pts(res["curve"], 97)[1]:
        assert abs(_vlen(_vsub(p, centre)) - 2.5) < 1e-9
    shrunk = offset_curve(make_circle_curve((0.0, 0.0, 0.0), 1.0), -1.5)
    assert not shrunk["ok"] and "collapses" in shrunk["reason"]
    print("[selfcheck] circle offset: analytic, radius r+d within 1e-9, "
          "collapse declined")

    # 3. Circular arc -> exact r + d.
    res = offset_curve(make_arc_curve((0.0, 0.0, 0.0), 1.0, 0.0,
                                      math.pi / 2.0), 0.25)
    assert res["ok"] and res["analytic"] == "arc"
    assert res["actual_max_deviation"] == 0.0
    for p in _sample_curve_pts(res["curve"], 97)[1]:
        assert abs(_vlen(p) - 1.25) < 1e-9
    print("[selfcheck] arc offset: analytic, radius r+d within 1e-9")

    # 4. Unit square, outward: 4 sides kept, 4 convex arc bridges, and the
    #    perimeter grows by exactly one full circle of radius d.
    corners = [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0),
               (1.0, 0.0, 0.0)]  # clockwise, so cross(z, T) points outward
    loop = [make_line_curve(corners[k], corners[(k + 1) % 4])
            for k in range(4)]
    step = 0.25
    res = offset_loop(loop, step)
    assert res["ok"]
    assert len([s for s in res["curves"] if s.degree == 2]) == 4
    assert len([s for s in res["curves"] if s.degree == 1]) == 4
    grown = 4.0 + _TAU * step
    assert abs(res["perimeter"] - grown) < 5e-4, res["perimeter"]
    for k, seg in enumerate(res["curves"]):
        nxt = res["curves"][(k + 1) % len(res["curves"])]
        tail = _at(seg, _curve_domain(seg)[1])
        head = _at(nxt, _curve_domain(nxt)[0])
        assert _vlen(_vsub(head, tail)) < 1e-9, k
    print("[selfcheck] square loop outward: 4 convex arc bridges, perimeter "
          "%.6f ~ old + 2*pi*d = %.6f" % (res["perimeter"], grown))

    # 5. Unit square, inward: every corner concave, trimmed exactly.
    res = offset_loop(loop, -step)
    assert res["ok"]
    assert len(res["curves"]) == 4
    assert all(seg.degree == 1 for seg in res["curves"])
    assert abs(res["perimeter"] - (4.0 - 8.0 * step)) < 1e-9
    print("[selfcheck] square loop inward: concave corners trimmed, "
          "perimeter %.6f = old - 8*d" % res["perimeter"])

    # 6. Freeform cubic -> refit path, measured deviation inside tol.
    freeform = CurveData(((0.0, 0.0, 0.0), (1.0, 1.5, 0.0), (2.0, -1.0, 0.0),
                          (3.0, 0.5, 0.0), (4.0, 0.0, 0.0)), (1.0,) * 5, 3,
                         tuple(uniform_clamped_knots(4, 3)))
    tol = 1e-3
    res = offset_curve(freeform, 0.2, tol=tol)
    assert res["analytic"] == "", res["analytic"]
    assert res["ok"] and res["actual_max_deviation"] <= tol
    dense = _sample_curve_pts(freeform, 400)[1]
    for p in _sample_curve_pts(res["curve"], 21)[1][1:-1]:
        assert abs(_distance_to_polyline(p, dense) - 0.2) < 2e-2
    print("[selfcheck] freeform offset: refit deviation %.3g <= tol %.3g"
          % (res["actual_max_deviation"], tol))

    # 7. Plane -> exact translation along the normal.
    plane = SurfaceData((((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                         ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0))),
                        ((1.0, 1.0), (1.0, 1.0)), 1, 1,
                        (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0))
    res = offset_surface(plane, 1.0)
    assert res["ok"] and res["analytic"] == "plane"
    assert res["actual_max_deviation"] == 0.0
    assert abs(_surface_pt(res["surface"], 0.5, 0.5)[2] - 1.0) < 1e-12
    print("[selfcheck] plane surface offset: analytic, exact")

    # 8. Sphere -> exact concentric scale of the control net.
    ball_centre = (0.5, -1.0, 2.0)
    res = offset_surface(_revolved_sphere(ball_centre, 2.0), 0.5)
    assert res["ok"] and res["analytic"] == "sphere", res["analytic"]
    assert res["actual_max_deviation"] == 0.0
    for u in (0.0, 0.3, 0.6, 1.0):
        for v in (0.1, 0.5, 0.9):
            p = _surface_pt(res["surface"], u, v)
            assert abs(_vlen(_vsub(p, ball_centre)) - 2.5) < 1e-9
    print("[selfcheck] sphere surface offset: analytic, radius r+d "
          "within 1e-9")

    # 9. Freeform bump surface -> refit path, measured deviation inside tol.
    bump = SurfaceData(
        tuple(tuple((float(i), float(j),
                     0.15 if (1 <= i <= 2 and 1 <= j <= 2) else 0.0)
                    for j in range(4)) for i in range(4)),
        tuple((1.0,) * 4 for _ in range(4)), 2, 2,
        tuple(uniform_clamped_knots(3, 2)), tuple(uniform_clamped_knots(3, 2)))
    stol = 5e-3
    res = offset_surface(bump, 0.05, tol=stol)
    assert res["analytic"] == "", res["analytic"]
    assert res["ok"] and res["actual_max_deviation"] <= stol
    print("[selfcheck] bump surface offset: refit deviation %.3g <= tol %.3g"
          % (res["actual_max_deviation"], stol))

    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
