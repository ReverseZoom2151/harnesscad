"""Tensor-product NURBS surface evaluation and mesh tessellation (NURBGen).

Usama, Khan, Stricker & Afzal, *NURBGen: High-Fidelity Text-to-CAD Generation
through LLM-Driven NURBS Modeling* (AAAI 2026), Eq. 3.

NURBGen's primary representation is the untrimmed NURBS *surface*: a 2-D grid of
control points (poles), a 2-D grid of weights, knot vectors ``U`` and ``V`` and
degrees ``p`` and ``q``.  The surface is the tensor product of two NURBS curves
(paper Eq. 3):

    S(u,v) = sum_i sum_j N_{i,p}(u) M_{j,q}(v) w_{ij} P_{ij}
             / sum_i sum_j N_{i,p}(u) M_{j,q}(v) w_{ij}

This module evaluates that surface, its partial derivatives S_u / S_v, the unit
normal, and tessellates the surface into a triangle mesh (vertices + triangle
index triples) -- NURBGen's "directly convert to B-rep / textureless triangular
mesh" step (paper Sec. Data Preparation, Multi-View Rendering).

Control grid convention: ``poles[i][j]`` is the pole at u-index ``i`` (0..n) and
v-index ``j`` (0..m); ``weights[i][j] > 0``.  Built on
:mod:`numeric.nurbs_basis`.  Pure-Python stdlib, deterministic.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from harnesscad.domain.numeric.nurbs_basis import basis_derivatives, basis_functions, find_span

Point = Tuple[float, ...]
Grid = Sequence[Sequence[Point]]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _check(poles: Grid, weights, deg_u: int, deg_v: int,
           knots_u: Sequence[float], knots_v: Sequence[float]):
    n = len(poles) - 1
    if n < 0:
        raise ValueError("empty control grid")
    m = len(poles[0]) - 1
    if m < 0:
        raise ValueError("empty control row")
    for row in poles:
        if len(row) != m + 1:
            raise ValueError("ragged control grid")
    if deg_u < 1 or deg_v < 1:
        raise ValueError("degrees must be >= 1")
    if n < deg_u or m < deg_v:
        raise ValueError("need degree + 1 control points in each direction")
    if len(weights) != n + 1 or any(len(r) != m + 1 for r in weights):
        raise ValueError("weights shape must match control grid")
    if any(w <= 0.0 for r in weights for w in r):
        raise ValueError("weights must be strictly positive")
    if len(knots_u) != n + deg_u + 2:
        raise ValueError("U knot vector has wrong length")
    if len(knots_v) != m + deg_v + 2:
        raise ValueError("V knot vector has wrong length")
    dim = len(poles[0][0])
    return n, m, dim


# ---------------------------------------------------------------------------
# Point evaluation (paper Eq. 3)
# ---------------------------------------------------------------------------

def surface_point(poles: Grid, weights, deg_u: int, deg_v: int,
                  knots_u: Sequence[float], knots_v: Sequence[float],
                  u: float, v: float) -> Point:
    """Evaluate the NURBS surface ``S(u, v)`` (paper Eq. 3)."""
    n, m, dim = _check(poles, weights, deg_u, deg_v, knots_u, knots_v)
    su = find_span(n, deg_u, u, knots_u)
    sv = find_span(m, deg_v, v, knots_v)
    Nu = basis_functions(su, u, deg_u, knots_u)
    Nv = basis_functions(sv, v, deg_v, knots_v)
    numer = [0.0] * dim
    denom = 0.0
    for a in range(deg_u + 1):
        i = su - deg_u + a
        for b in range(deg_v + 1):
            j = sv - deg_v + b
            wN = weights[i][j] * Nu[a] * Nv[b]
            denom += wN
            pt = poles[i][j]
            for c in range(dim):
                numer[c] += wN * pt[c]
    if abs(denom) < 1e-14:
        raise ValueError("degenerate weight sum at (u,v)=(%r,%r)" % (u, v))
    return tuple(numer[c] / denom for c in range(dim))


# ---------------------------------------------------------------------------
# First partial derivatives and normal
# ---------------------------------------------------------------------------

def surface_derivatives(poles: Grid, weights, deg_u: int, deg_v: int,
                        knots_u: Sequence[float], knots_v: Sequence[float],
                        u: float, v: float) -> Tuple[Point, Point, Point]:
    """Return ``(S, S_u, S_v)`` at ``(u, v)``.

    Differentiates the homogeneous numerator ``A(u,v)`` and weight ``w(u,v)`` in
    each parameter, then applies the rational quotient rule per direction:
    ``S_a = (A_a - w_a S) / w``.
    """
    n, m, dim = _check(poles, weights, deg_u, deg_v, knots_u, knots_v)
    su = find_span(n, deg_u, u, knots_u)
    sv = find_span(m, deg_v, v, knots_v)
    dNu = basis_derivatives(su, u, deg_u, knots_u, 1)
    dNv = basis_derivatives(sv, v, deg_v, knots_v, 1)

    A = [0.0] * dim
    Au = [0.0] * dim
    Av = [0.0] * dim
    w = wu = wv = 0.0
    for a in range(deg_u + 1):
        i = su - deg_u + a
        for b in range(deg_v + 1):
            j = sv - deg_v + b
            wij = weights[i][j]
            f = wij * dNu[0][a] * dNv[0][b]
            fu = wij * dNu[1][a] * dNv[0][b]
            fv = wij * dNu[0][a] * dNv[1][b]
            w += f
            wu += fu
            wv += fv
            pt = poles[i][j]
            for c in range(dim):
                A[c] += f * pt[c]
                Au[c] += fu * pt[c]
                Av[c] += fv * pt[c]
    if abs(w) < 1e-14:
        raise ValueError("degenerate weight sum at (u,v)")
    S = tuple(A[c] / w for c in range(dim))
    S_u = tuple((Au[c] - wu * S[c]) / w for c in range(dim))
    S_v = tuple((Av[c] - wv * S[c]) / w for c in range(dim))
    return S, S_u, S_v


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def surface_normal(poles: Grid, weights, deg_u: int, deg_v: int,
                   knots_u: Sequence[float], knots_v: Sequence[float],
                   u: float, v: float) -> Point:
    """Unit surface normal ``normalize(S_u x S_v)`` (3-D surfaces only)."""
    _, S_u, S_v = surface_derivatives(
        poles, weights, deg_u, deg_v, knots_u, knots_v, u, v)
    if len(S_u) != 3:
        raise ValueError("normal only defined for 3-D surfaces")
    nx, ny, nz = _cross(S_u, S_v)
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length < 1e-14:
        raise ValueError("degenerate normal (parallel partials)")
    return (nx / length, ny / length, nz / length)


# ---------------------------------------------------------------------------
# Tessellation: NURBS surface -> triangle mesh
# ---------------------------------------------------------------------------

def tessellate_surface(poles: Grid, weights, deg_u: int, deg_v: int,
                       knots_u: Sequence[float], knots_v: Sequence[float],
                       n_u: int = 8, n_v: int = 8
                       ) -> Tuple[List[Point], List[Tuple[int, int, int]]]:
    """Tessellate the surface into ``(vertices, triangles)``.

    Samples an ``(n_u + 1) x (n_v + 1)`` grid over the valid parameter domain
    and splits each quad into two triangles.  Vertices are flattened row-major
    (u outer, v inner); triangle entries are vertex indices.
    """
    n, m, _ = _check(poles, weights, deg_u, deg_v, knots_u, knots_v)
    if n_u < 1 or n_v < 1:
        raise ValueError("n_u and n_v must be >= 1")
    u0, u1 = knots_u[deg_u], knots_u[n + 1]
    v0, v1 = knots_v[deg_v], knots_v[m + 1]

    verts: List[Point] = []
    for a in range(n_u + 1):
        u = u0 + (u1 - u0) * (a / n_u)
        for b in range(n_v + 1):
            v = v0 + (v1 - v0) * (b / n_v)
            verts.append(surface_point(
                poles, weights, deg_u, deg_v, knots_u, knots_v, u, v))

    stride = n_v + 1
    tris: List[Tuple[int, int, int]] = []
    for a in range(n_u):
        for b in range(n_v):
            i00 = a * stride + b
            i01 = i00 + 1
            i10 = i00 + stride
            i11 = i10 + 1
            tris.append((i00, i10, i11))
            tris.append((i00, i11, i01))
    return verts, tris


def mesh_area(vertices: Sequence[Point],
              triangles: Sequence[Tuple[int, int, int]]) -> float:
    """Sum of triangle areas of a 3-D mesh (surface-area proxy)."""
    total = 0.0
    for i, j, k in triangles:
        a, b, c = vertices[i], vertices[j], vertices[k]
        ab = tuple(b[d] - a[d] for d in range(3))
        ac = tuple(c[d] - a[d] for d in range(3))
        cx, cy, cz = _cross(ab, ac)
        total += 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)
    return total


# ---------------------------------------------------------------------------
# Analytic ground truth: a NURBS cylinder patch (quarter cylinder)
# ---------------------------------------------------------------------------

def nurbs_cylinder_quadrant(radius: float = 1.0, height: float = 1.0):
    """A quarter-cylinder NURBS patch: rational-quadratic arc extruded in z.

    u-direction is the exact 90-degree circular arc (weights 1, 1/sqrt2, 1);
    v-direction is a linear extrusion in z.  Returns ``(poles, weights, p, q,
    U, V)`` -- an analytic surface whose points all satisfy ``x^2 + y^2 = r^2``.
    """
    r = float(radius)
    h = float(height)
    s = math.sqrt(2.0) / 2.0
    arc = [(r, 0.0), (r, r), (0.0, r)]
    poles = [[(x, y, 0.0) for (x, y) in arc],
             [(x, y, h) for (x, y) in arc]]
    weights = [[1.0, s, 1.0], [1.0, s, 1.0]]
    U = [0.0, 0.0, 1.0, 1.0]          # deg_u = 1 (extrusion direction)
    V = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]  # deg_v = 2 (arc direction)
    return poles, weights, 1, 2, U, V
