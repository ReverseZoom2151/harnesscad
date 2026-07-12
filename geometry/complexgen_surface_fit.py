"""Least-squares fitting of 3D CAD surface primitives to point samples.

ComplexGen (Guo et al., "ComplexGen: CAD Reconstruction by B-Rep Chain Complex
Generation", SIGGRAPH 2022) classifies every generated patch into one of the CAD
primitive types (plane / cylinder / cone / sphere / torus / spline) and its
geometric-refinement stage re-fits the analytic primitive to the supporting
points before rebuilding the B-Rep.  The learned generator is out of scope, but
the fitting/distance layer it wraps is fully deterministic and closed-form; the
repository's ``src/primitives.py`` (``ComputePrimitiveDistance``) and the
``GeometricRefine`` quadric solver are exactly that layer.

This module rebuilds it in stdlib Python:

  * :func:`fit_plane`    -- total least squares (PCA) plane: unit normal + offset.
  * :func:`fit_sphere`   -- algebraic (linear) sphere fit: centre + radius.
  * :func:`fit_cylinder` -- axis from the point normals (the axis is the null
    direction of ``sum n n^T``) or, when normals are absent, a deterministic
    direction search refined by successive grid shrinking; radius/centre from a
    circle fit of the points projected onto the plane orthogonal to the axis.
  * :func:`fit_cone`     -- axis + half-angle from the homogeneous system
    ``n . d = sin(alpha)``, apex from the least-squares intersection of the
    tangent planes ``n . (apex - p) = 0``.
  * :func:`fit_best`     -- fit every supported type, return the one with the
    smallest RMS residual (the "primitive type classification" the network does
    probabilistically, done here by residual).

Distances (signed-free, in the same units as the input) mirror the
``distance_from_*`` routines of ``src/primitives.py``:
:func:`distance_to_plane`, :func:`distance_to_sphere`, :func:`distance_to_cylinder`,
:func:`distance_to_cone`, :func:`distance_to_torus`.

Everything is deterministic: the eigen-decomposition is a cyclic Jacobi sweep
with a fixed sweep count, and linear systems are solved with Gaussian
elimination with partial pivoting.
"""

from __future__ import annotations

import math

Point = tuple[float, float, float]

PLANE = "plane"
SPHERE = "sphere"
CYLINDER = "cylinder"
CONE = "cone"

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# small linear algebra (stdlib)
# --------------------------------------------------------------------------- #
def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a):
    return math.sqrt(_dot(a, a))


def _normalize(a):
    n = _norm(a)
    if n < _EPS:
        raise ValueError("cannot normalize a zero-length vector")
    return (a[0] / n, a[1] / n, a[2] / n)


def _centroid(points):
    n = len(points)
    if n == 0:
        raise ValueError("no points")
    sx = sy = sz = 0.0
    for p in points:
        sx += p[0]
        sy += p[1]
        sz += p[2]
    return (sx / n, sy / n, sz / n)


def jacobi_eigen(matrix, sweeps: int = 60):
    """Eigen-decomposition of a symmetric ``n x n`` matrix (cyclic Jacobi).

    Returns ``(eigenvalues, eigenvectors)`` sorted by ascending eigenvalue;
    ``eigenvectors[k]`` is the unit eigenvector for ``eigenvalues[k]``.  The sign
    of each eigenvector is canonicalised (first non-negligible component
    positive) so results are reproducible.
    """
    n = len(matrix)
    a = [[float(matrix[i][j]) for j in range(n)] for i in range(n)]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(sweeps):
        off = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                off += a[i][j] * a[i][j]
        if off < 1e-24:
            break
        for p in range(n):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-18:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp = a[k][p]
                    akq = a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk = a[p][k]
                    aqk = a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp = v[k][p]
                    vkq = v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    vals = [a[i][i] for i in range(n)]
    vecs = []
    for j in range(n):
        col = [v[i][j] for i in range(n)]
        for x in col:
            if abs(x) > 1e-9:
                if x < 0:
                    col = [-y for y in col]
                break
        vecs.append(tuple(col))
    order = sorted(range(n), key=lambda k: vals[k])
    return [vals[k] for k in order], [vecs[k] for k in order]


def solve_linear(matrix, rhs):
    """Solve ``matrix x = rhs`` by Gaussian elimination with partial pivoting."""
    n = len(matrix)
    m = [[float(matrix[i][j]) for j in range(n)] + [float(rhs[i])] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-14:
            raise ValueError("singular system")
        m[col], m[pivot] = m[pivot], m[col]
        pv = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= pv
        for r in range(n):
            if r == col:
                continue
            f = m[r][col]
            if f == 0.0:
                continue
            for j in range(col, n + 1):
                m[r][j] -= f * m[col][j]
    return [m[i][n] for i in range(n)]


def _covariance(points, centre):
    cov = [[0.0] * 3 for _ in range(3)]
    for p in points:
        d = _sub(p, centre)
        for i in range(3):
            for j in range(3):
                cov[i][j] += d[i] * d[j]
    n = float(len(points))
    for i in range(3):
        for j in range(3):
            cov[i][j] /= n
    return cov


def _rms(values):
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


# --------------------------------------------------------------------------- #
# distances (mirror src/primitives.py ComputePrimitiveDistance)
# --------------------------------------------------------------------------- #
def distance_to_plane(point, params) -> float:
    """``params = (normal, offset)`` with ``normal . x = offset``."""
    normal, offset = params
    return abs(_dot(point, normal) - offset)


def distance_to_sphere(point, params) -> float:
    """``params = (centre, radius)``."""
    centre, radius = params
    return abs(_norm(_sub(point, centre)) - radius)


def distance_to_cylinder(point, params) -> float:
    """``params = (axis_point, axis_dir, radius)``."""
    axis_point, axis_dir, radius = params
    d = _sub(point, axis_point)
    axial = _dot(d, axis_dir)
    radial = _sub(d, _scale(axis_dir, axial))
    return abs(_norm(radial) - radius)


def distance_to_cone(point, params) -> float:
    """``params = (apex, axis_dir, half_angle)``.

    Orthogonal distance from the (double) cone surface: with ``t`` the angle
    between ``point - apex`` and the axis, the distance is ``|v| * sin(t - a)``.
    """
    apex, axis_dir, half_angle = params
    v = _sub(point, apex)
    length = _norm(v)
    if length < _EPS:
        return 0.0
    cos_t = max(-1.0, min(1.0, _dot(v, axis_dir) / length))
    t = math.acos(cos_t)
    if t > math.pi / 2.0:          # the mirrored nappe
        t = math.pi - t
    return abs(length * math.sin(t - half_angle))


def distance_to_torus(point, params) -> float:
    """``params = (centre, axis_dir, major_radius, minor_radius)``."""
    centre, axis_dir, major, minor = params
    d = _sub(point, centre)
    axial = _dot(d, axis_dir)
    radial = _norm(_sub(d, _scale(axis_dir, axial)))
    return abs(math.sqrt((radial - major) ** 2 + axial * axial) - minor)


def residuals(points, kind: str, params) -> list[float]:
    fn = {
        PLANE: distance_to_plane,
        SPHERE: distance_to_sphere,
        CYLINDER: distance_to_cylinder,
        CONE: distance_to_cone,
    }[kind]
    return [fn(p, params) for p in points]


def rms_residual(points, kind: str, params) -> float:
    return _rms(residuals(points, kind, params))


# --------------------------------------------------------------------------- #
# plane
# --------------------------------------------------------------------------- #
def fit_plane(points):
    """Total-least-squares plane.  Returns ``((normal, offset), rms)``."""
    if len(points) < 3:
        raise ValueError("fit_plane needs at least 3 points")
    centre = _centroid(points)
    _, vecs = jacobi_eigen(_covariance(points, centre))
    normal = vecs[0]                       # smallest-variance direction
    offset = _dot(normal, centre)
    params = (normal, offset)
    return params, rms_residual(points, PLANE, params)


# --------------------------------------------------------------------------- #
# sphere
# --------------------------------------------------------------------------- #
def fit_sphere(points):
    """Algebraic sphere fit.  Returns ``((centre, radius), rms)``.

    Solves the linear system obtained from ``|p|^2 = 2 p . c + (r^2 - |c|^2)``.
    """
    if len(points) < 4:
        raise ValueError("fit_sphere needs at least 4 points")
    # normal equations for the 4 unknowns (2cx, 2cy, 2cz, r^2 - |c|^2)
    a = [[0.0] * 4 for _ in range(4)]
    b = [0.0] * 4
    for p in points:
        row = (p[0], p[1], p[2], 1.0)
        rhs = _dot(p, p)
        for i in range(4):
            for j in range(4):
                a[i][j] += row[i] * row[j]
            b[i] += row[i] * rhs
    sol = solve_linear(a, b)
    centre = (sol[0] / 2.0, sol[1] / 2.0, sol[2] / 2.0)
    r2 = sol[3] + _dot(centre, centre)
    if r2 <= 0.0:
        raise ValueError("degenerate sphere fit")
    params = (centre, math.sqrt(r2))
    return params, rms_residual(points, SPHERE, params)


# --------------------------------------------------------------------------- #
# cylinder
# --------------------------------------------------------------------------- #
def _circle_fit_2d(pts2):
    """Kasa circle fit in 2D.  Returns ``(cx, cy, r)``."""
    a = [[0.0] * 3 for _ in range(3)]
    b = [0.0] * 3
    for (x, y) in pts2:
        row = (x, y, 1.0)
        rhs = x * x + y * y
        for i in range(3):
            for j in range(3):
                a[i][j] += row[i] * row[j]
            b[i] += row[i] * rhs
    sol = solve_linear(a, b)
    cx = sol[0] / 2.0
    cy = sol[1] / 2.0
    r2 = sol[2] + cx * cx + cy * cy
    if r2 <= 0.0:
        raise ValueError("degenerate circle fit")
    return cx, cy, math.sqrt(r2)


def _basis_for(axis):
    """A deterministic orthonormal basis ``(u, v)`` of the plane normal to axis."""
    helper = (0.0, 0.0, 1.0) if abs(axis[2]) < 0.9 else (1.0, 0.0, 0.0)
    u = _normalize(_cross(axis, helper))
    v = _cross(axis, u)
    return u, v


def _cylinder_from_axis(points, axis):
    axis = _normalize(axis)
    u, v = _basis_for(axis)
    pts2 = [(_dot(p, u), _dot(p, v)) for p in points]
    cx, cy, r = _circle_fit_2d(pts2)
    axis_point = _add(_scale(u, cx), _scale(v, cy))
    params = (axis_point, axis, r)
    return params, rms_residual(points, CYLINDER, params)


def _direction_grid(steps: int):
    """Deterministic hemispherical direction grid (theta x phi)."""
    dirs = []
    for i in range(steps + 1):
        theta = math.pi * i / (2.0 * steps)        # 0 .. pi/2
        n_phi = 1 if i == 0 else 4 * steps
        for j in range(n_phi):
            phi = 2.0 * math.pi * j / n_phi
            dirs.append((math.sin(theta) * math.cos(phi),
                         math.sin(theta) * math.sin(phi),
                         math.cos(theta)))
    return dirs


def fit_cylinder(points, normals=None):
    """Cylinder fit.  Returns ``((axis_point, axis_dir, radius), rms)``.

    ``axis_point`` is the point of the axis closest to the origin.  If per-point
    surface ``normals`` are supplied the axis is recovered exactly (it is the
    null direction of ``sum n n^T``, since every cylinder normal is orthogonal to
    the axis); otherwise a deterministic direction search over a hemispherical
    grid, refined by three shrinking local passes, is used.
    """
    if len(points) < 5:
        raise ValueError("fit_cylinder needs at least 5 points")
    if normals is not None:
        if len(normals) != len(points):
            raise ValueError("normals must match points")
        m = [[0.0] * 3 for _ in range(3)]
        for n in normals:
            n = _normalize(n)
            for i in range(3):
                for j in range(3):
                    m[i][j] += n[i] * n[j]
        _, vecs = jacobi_eigen(m)
        return _cylinder_from_axis(points, vecs[0])

    best = None
    for axis in _direction_grid(12):
        try:
            params, rms = _cylinder_from_axis(points, axis)
        except ValueError:
            continue
        if best is None or rms < best[1]:
            best = (params, rms, axis)
    if best is None:
        raise ValueError("cylinder fit failed")
    axis = best[2]
    span = math.pi / 12.0
    for _ in range(6):                                  # local refinement
        u, v = _basis_for(_normalize(axis))
        improved = False
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                if du == 0 and dv == 0:
                    continue
                cand = _add(axis, _add(_scale(u, du * span), _scale(v, dv * span)))
                try:
                    params, rms = _cylinder_from_axis(points, cand)
                except ValueError:
                    continue
                if rms < best[1]:
                    best = (params, rms, _normalize(cand))
                    improved = True
        axis = best[2]
        if not improved:
            span *= 0.5
    return best[0], best[1]


# --------------------------------------------------------------------------- #
# cone
# --------------------------------------------------------------------------- #
def fit_cone(points, normals):
    """Cone fit from points + surface normals.

    Returns ``((apex, axis_dir, half_angle), rms)``.  Every cone normal satisfies
    ``n . d = sin(alpha)`` for the unit axis ``d`` (outward normals of one nappe),
    which is a homogeneous 4-unknown system solved by the smallest eigenvector of
    ``sum a a^T`` with ``a = (nx, ny, nz, -1)``.  The apex is the least-squares
    intersection of the tangent planes ``n . (apex - p) = 0``.
    """
    if len(points) < 5:
        raise ValueError("fit_cone needs at least 5 points")
    if normals is None or len(normals) != len(points):
        raise ValueError("fit_cone requires one normal per point")
    unit = [_normalize(n) for n in normals]

    m = [[0.0] * 4 for _ in range(4)]
    for n in unit:
        a = (n[0], n[1], n[2], -1.0)
        for i in range(4):
            for j in range(4):
                m[i][j] += a[i] * a[j]
    _, vecs = jacobi_eigen(m)
    sol = vecs[0]
    d = (sol[0], sol[1], sol[2])
    scale = _norm(d)
    if scale < _EPS:
        raise ValueError("degenerate cone fit (normals are parallel)")
    axis = _scale(d, 1.0 / scale)
    sin_a = sol[3] / scale
    if sin_a < 0.0:                       # flip so the half-angle is positive
        axis = _scale(axis, -1.0)
        sin_a = -sin_a
    sin_a = max(-1.0, min(1.0, sin_a))
    half_angle = math.asin(sin_a)

    ata = [[0.0] * 3 for _ in range(3)]
    atb = [0.0] * 3
    for p, n in zip(points, unit):
        rhs = _dot(n, p)
        for i in range(3):
            for j in range(3):
                ata[i][j] += n[i] * n[j]
            atb[i] += n[i] * rhs
    apex = tuple(solve_linear(ata, atb))
    params = (apex, axis, half_angle)
    return params, rms_residual(points, CONE, params)


# --------------------------------------------------------------------------- #
# type selection
# --------------------------------------------------------------------------- #
def fit_best(points, normals=None):
    """Fit every supported primitive; return ``(kind, params, rms)`` of the best.

    Ties are broken by the fixed order plane < sphere < cylinder < cone, so the
    simplest primitive wins an exact tie.
    """
    results = []
    for kind, fn in ((PLANE, lambda: fit_plane(points)),
                     (SPHERE, lambda: fit_sphere(points)),
                     (CYLINDER, lambda: fit_cylinder(points, normals)),
                     (CONE, lambda: fit_cone(points, normals) if normals else None)):
        try:
            out = fn()
        except (ValueError, ZeroDivisionError):
            continue
        if out is None:
            continue
        params, rms = out
        results.append((kind, params, rms))
    if not results:
        raise ValueError("no primitive could be fitted")
    order = {PLANE: 0, SPHERE: 1, CYLINDER: 2, CONE: 3}
    results.sort(key=lambda r: (round(r[2], 9), order[r[0]]))
    return results[0]
