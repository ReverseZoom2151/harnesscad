"""Rational Bezier surface evaluation and control-net utilities (DreamCAD).

Implements the deterministic geometry primitives underlying DreamCAD's
"differentiable parametric surfaces" (Eq. 1):

  * Bernstein basis and its analytic derivative.
  * de Casteljau evaluation of a (non-rational) Bezier curve.
  * Rational Bezier *surface* evaluation S(u, v) from an (n+1) x (m+1)
    control-point grid and non-negative weights (bicubic n = m = 3 is the
    paper's choice, but any bidegree works here).
  * Analytic first partial derivatives S_u, S_v of the rational surface and
    the resulting unit surface normal.
  * The paper's initial-quad construction: a 4 x 4 control grid built by
    bilinear interpolation of a quad's four corners (Section 4.1).
  * The parameter transforms the decoder uses to keep evaluation valid:
    softplus for strictly positive weights and tanh-bounded control-point
    deformation (Section 4.1 / supplementary training details).

Everything is pure-Python stdlib and deterministic: no wall clock, no
randomness.  Points are ordinary tuples of floats; a control grid is a list
of rows ``grid[i][j]`` with ``i`` indexing the u-direction (0..n) and ``j``
the v-direction (0..m).
"""

from __future__ import annotations

from math import comb, exp, log1p, sqrt


def bernstein(n, i, t):
    """Bernstein basis polynomial B_{i}^{n}(t) for t in [0, 1]."""
    if n < 0:
        raise ValueError("degree must be non-negative")
    if not 0 <= i <= n:
        raise ValueError("index out of range")
    if not 0.0 <= t <= 1.0:
        raise ValueError("parameter must lie in [0, 1]")
    return comb(n, i) * t ** i * (1.0 - t) ** (n - i)


def bernstein_basis(n, t):
    """Return the full basis [B_0^n(t), ..., B_n^n(t)]."""
    return [bernstein(n, i, t) for i in range(n + 1)]


def bernstein_derivative(n, i, t):
    """Analytic derivative dB_{i}^{n}/dt = n (B_{i-1}^{n-1} - B_{i}^{n-1})."""
    if n < 0:
        raise ValueError("degree must be non-negative")
    if not 0 <= i <= n:
        raise ValueError("index out of range")
    if not 0.0 <= t <= 1.0:
        raise ValueError("parameter must lie in [0, 1]")
    if n == 0:
        return 0.0
    left = bernstein(n - 1, i - 1, t) if i - 1 >= 0 else 0.0
    right = bernstein(n - 1, i, t) if i <= n - 1 else 0.0
    return n * (left - right)


def bernstein_derivative_basis(n, t):
    """Return [dB_0^n/dt, ..., dB_n^n/dt]."""
    return [bernstein_derivative(n, i, t) for i in range(n + 1)]


def de_casteljau(control_points, t):
    """Evaluate a Bezier curve at ``t`` via the de Casteljau recursion.

    ``control_points`` is a sequence of equal-length point tuples.  This is a
    numerically stable alternative to the Bernstein sum and is used by the
    paper's tessellation of surface iso-curves.
    """
    if not control_points:
        raise ValueError("need at least one control point")
    if not 0.0 <= t <= 1.0:
        raise ValueError("parameter must lie in [0, 1]")
    dim = len(control_points[0])
    points = [tuple(float(c) for c in p) for p in control_points]
    while len(points) > 1:
        points = [
            tuple((1.0 - t) * points[k][d] + t * points[k + 1][d]
                  for d in range(dim))
            for k in range(len(points) - 1)
        ]
    return points[0]


def _grid_degrees(grid):
    n = len(grid) - 1
    if n < 0:
        raise ValueError("empty control grid")
    m = len(grid[0]) - 1
    if m < 0:
        raise ValueError("empty control row")
    for row in grid:
        if len(row) != m + 1:
            raise ValueError("ragged control grid")
    return n, m


def _validate_weights(grid, weights):
    n, m = _grid_degrees(grid)
    if len(weights) != n + 1 or any(len(row) != m + 1 for row in weights):
        raise ValueError("weights shape must match control grid")
    if any(w < 0.0 for row in weights for w in row):
        raise ValueError("weights must be non-negative")
    return n, m


def bezier_surface_point(grid, weights, u, v):
    """Rational Bezier surface S(u, v) (DreamCAD Eq. 1).

    ``grid[i][j]`` are control points, ``weights[i][j]`` non-negative weights.
    Returns the 3-tuple (or d-tuple) surface point.
    """
    n, m = _validate_weights(grid, weights)
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        raise ValueError("(u, v) must lie in the unit square")
    bu = bernstein_basis(n, u)
    bv = bernstein_basis(m, v)
    dim = len(grid[0][0])
    numer = [0.0] * dim
    denom = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            factor = bu[i] * bv[j] * weights[i][j]
            denom += factor
            point = grid[i][j]
            for d in range(dim):
                numer[d] += factor * point[d]
    if denom <= 0.0:
        raise ValueError("degenerate surface: zero denominator")
    return tuple(numer[d] / denom for d in range(dim))


def bezier_surface_derivatives(grid, weights, u, v):
    """Return the pair (S_u, S_v) of first partial derivatives at (u, v).

    Uses the quotient rule on the rational form S = N / D, with N and D the
    weighted-Bernstein numerator vector and scalar denominator.
    """
    n, m = _validate_weights(grid, weights)
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        raise ValueError("(u, v) must lie in the unit square")
    bu = bernstein_basis(n, u)
    bv = bernstein_basis(m, v)
    dbu = bernstein_derivative_basis(n, u)
    dbv = bernstein_derivative_basis(m, v)
    dim = len(grid[0][0])

    numer = [0.0] * dim
    numer_u = [0.0] * dim
    numer_v = [0.0] * dim
    denom = 0.0
    denom_u = 0.0
    denom_v = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            w = weights[i][j]
            b = bu[i] * bv[j] * w
            b_u = dbu[i] * bv[j] * w
            b_v = bu[i] * dbv[j] * w
            denom += b
            denom_u += b_u
            denom_v += b_v
            point = grid[i][j]
            for d in range(dim):
                numer[d] += b * point[d]
                numer_u[d] += b_u * point[d]
                numer_v[d] += b_v * point[d]
    if denom <= 0.0:
        raise ValueError("degenerate surface: zero denominator")
    inv = 1.0 / denom
    inv2 = inv * inv
    s_u = tuple((numer_u[d] * denom - numer[d] * denom_u) * inv2
                for d in range(dim))
    s_v = tuple((numer_v[d] * denom - numer[d] * denom_v) * inv2
                for d in range(dim))
    return s_u, s_v


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def bezier_surface_normal(grid, weights, u, v):
    """Unit surface normal = normalize(S_u x S_v) at (u, v) (3-D only)."""
    s_u, s_v = bezier_surface_derivatives(grid, weights, u, v)
    if len(s_u) != 3:
        raise ValueError("normal is only defined for 3-D surfaces")
    nx, ny, nz = _cross(s_u, s_v)
    length = sqrt(nx * nx + ny * ny + nz * nz)
    if length <= 0.0:
        raise ValueError("degenerate normal (parallel partials)")
    return (nx / length, ny / length, nz / length)


def bilinear_quad_grid(corners, n=3, m=3):
    """Build an (n+1) x (m+1) control grid from a quad's four corners.

    ``corners`` is (c00, c10, c11, c01), i.e. the points at parameter
    coordinates (0, 0), (1, 0), (1, 1), (0, 1).  Interior control points are
    placed by bilinear interpolation, matching DreamCAD's "convert each quad
    into a bicubic patch by uniformly sampling a 4 x 4 grid using bilinear
    interpolation of its four corners".  Returned as ``grid[i][j]`` with i
    along u and j along v.
    """
    if len(corners) != 4:
        raise ValueError("need exactly four corners")
    if n < 1 or m < 1:
        raise ValueError("degrees must be at least 1")
    c00, c10, c11, c01 = (tuple(float(x) for x in c) for c in corners)
    dim = len(c00)
    grid = []
    for i in range(n + 1):
        u = i / n
        row = []
        for j in range(m + 1):
            v = j / m
            point = tuple(
                (1 - u) * (1 - v) * c00[d] + u * (1 - v) * c10[d]
                + u * v * c11[d] + (1 - u) * v * c01[d]
                for d in range(dim)
            )
            row.append(point)
        grid.append(row)
    return grid


def unit_weight_grid(n=3, m=3):
    """Control-point weights initialised to 1 (the paper's unit weights)."""
    return [[1.0] * (m + 1) for _ in range(n + 1)]


def softplus_weight(raw):
    """Map an unconstrained value to a strictly positive weight, log(1+e^x).

    The paper stabilises training by predicting weights through softplus so
    that they stay positive (negative weights give degenerate surfaces).
    Computed in an overflow-safe form.
    """
    if raw > 0:
        return raw + log1p(exp(-raw))
    return log1p(exp(raw))


def bounded_deform(control_point, deformation):
    """Apply a tanh-bounded deformation: c <- c + tanh(d), per coordinate."""
    if len(control_point) != len(deformation):
        raise ValueError("dimension mismatch")
    return tuple(c + _tanh(d) for c, d in zip(control_point, deformation))


def _tanh(x):
    # math.tanh, but kept local so callers need not import math.
    if x > 20:
        return 1.0
    if x < -20:
        return -1.0
    e2 = exp(2.0 * x)
    return (e2 - 1.0) / (e2 + 1.0)
