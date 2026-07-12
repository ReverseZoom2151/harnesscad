"""Dual contouring of an f-rep graph with QEF vertex placement, after libfive.

Marching-cubes-family algorithms place one vertex per grid *edge* crossing and
so cannot represent sharp features (a cube corner gets rounded off).  *Dual
contouring* instead places one vertex per grid *cell*, positioned by minimising a
**quadratic error function** (QEF) built from the Hermite data -- the surface
crossing points on the cell's edges together with the surface normal at each.
Where several planes meet, the QEF minimum lands exactly on their intersection,
so corners and creases stay sharp.  This is the algorithm behind libfive's mesher
(see Matt Keeter's QEF writeup).

This module implements the 2D (quadtree) case end-to-end -- fully testable and
structurally identical to the 3D octree:

1. **Octree/quadtree build with interval pruning** -- a cell whose interval is
   wholly inside or wholly outside (:mod:`numeric.libfive_interval`) is not
   subdivided; only surface-straddling cells refine to the target depth.
2. **Edge sign-change detection + Hermite data** -- for each refined cell edge
   that changes sign, the crossing point is found by bisection and its normal by
   exact forward-mode AD (:mod:`numeric.libfive_forward_ad`).
3. **QEF vertex placement** -- a truncated-eigenvalue pseudoinverse solves the
   3x3 (in 2D) normal equations, biased toward the cell's mass point so flat and
   under-determined cells stay well behaved.
4. **Dual connectivity** -- each finest grid edge with a sign change emits a
   contour segment joining the vertices of the two cells sharing it.

Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

from geometry import libfive_frep_ir as ir
from numeric import libfive_forward_ad as ad
from numeric import libfive_interval as liv

Vec2 = Tuple[float, float]


# ===========================================================================
# Small symmetric linear algebra: Jacobi eigen-decomposition + QEF solve
# ===========================================================================


def _jacobi_eigen(A: List[List[float]], iters: int = 60
                  ) -> Tuple[List[float], List[List[float]]]:
    """Eigenvalues/vectors of a small symmetric matrix (cyclic Jacobi).

    Returns ``(eigenvalues, eigenvectors)`` where ``eigenvectors[k]`` is the
    k-th (column) eigenvector.  Deterministic and adequate for the 2x2/3x3
    systems that arise in dual contouring.
    """
    n = len(A)
    a = [row[:] for row in A]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(iters):
        # find largest off-diagonal magnitude
        p, q, off = 0, 1, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > off:
                    off = abs(a[i][j])
                    p, q = i, j
        if off < 1e-18:
            break
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        phi = 0.5 * math.atan2(2.0 * apq, aqq - app) if (aqq - app) != 0.0 \
            else math.pi / 4.0
        c, s = math.cos(phi), math.sin(phi)
        for k in range(n):
            akp, akq = a[k][p], a[k][q]
            a[k][p] = c * akp - s * akq
            a[k][q] = s * akp + c * akq
        for k in range(n):
            apk, aqk = a[p][k], a[q][k]
            a[p][k] = c * apk - s * aqk
            a[q][k] = s * apk + c * aqk
        for k in range(n):
            vkp, vkq = v[k][p], v[k][q]
            v[k][p] = c * vkp - s * vkq
            v[k][q] = s * vkp + c * vkq
    eigvals = [a[i][i] for i in range(n)]
    eigvecs = [[v[i][k] for i in range(n)] for k in range(n)]  # columns
    return eigvals, eigvecs


class QEF:
    """Quadratic error function accumulator over N dimensions.

    Sums plane constraints ``n_i . (p - p_i) = 0`` and finds the point ``p``
    minimising ``sum (n_i . (p - p_i))^2`` via a truncated-eigenvalue
    pseudoinverse of the normal-equation matrix ``A = sum n_i n_i^T``, solving
    for an offset from the mass point so that under-determined directions are
    left at the mass point (the libfive/Ju placement).
    """

    def __init__(self, n_dims: int):
        self.n = n_dims
        self.AtA = [[0.0] * n_dims for _ in range(n_dims)]
        self.Atb = [0.0] * n_dims
        self.mass = [0.0] * n_dims
        self.count = 0

    def insert(self, point, normal) -> None:
        d = self.n
        nb = sum(normal[i] * point[i] for i in range(d))  # n . p_i
        for i in range(d):
            self.Atb[i] += normal[i] * nb
            self.mass[i] += point[i]
            for j in range(d):
                self.AtA[i][j] += normal[i] * normal[j]
        self.count += 1

    def solve(self, svd_tol: float = 1e-6) -> Tuple[List[float], int]:
        """Return ``(position, rank)``.  ``rank`` is the number of significant
        constraint directions (2 at a sharp corner in 2D)."""
        d = self.n
        if self.count == 0:
            return [0.0] * d, 0
        mass = [m / self.count for m in self.mass]
        # residual b' = Atb - AtA . mass  (solve for offset from mass point)
        bp = [self.Atb[i] - sum(self.AtA[i][j] * mass[j] for j in range(d))
              for i in range(d)]
        eigvals, eigvecs = _jacobi_eigen(self.AtA)
        max_ev = max((abs(e) for e in eigvals), default=0.0)
        cutoff = svd_tol * max_ev
        offset = [0.0] * d
        rank = 0
        for k in range(d):
            ev = eigvals[k]
            if abs(ev) <= cutoff:
                continue
            rank += 1
            vec = eigvecs[k]
            coeff = sum(vec[i] * bp[i] for i in range(d)) / ev
            for i in range(d):
                offset[i] += coeff * vec[i]
        return [mass[i] + offset[i] for i in range(d)], rank


# ===========================================================================
# Hermite data extraction on an edge
# ===========================================================================


def _bisect_edge(f: Callable[[float, float, float], float],
                 a: Vec2, b: Vec2, iters: int = 40) -> Vec2:
    """Find the surface crossing on segment ``a``->``b`` by bisection.

    Assumes ``f(a)`` and ``f(b)`` have opposite signs.
    """
    fa = f(a[0], a[1], 0.0)
    lo, hi = a, b
    for _ in range(iters):
        mid = (0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]))
        fm = f(mid[0], mid[1], 0.0)
        if fm == 0.0:
            return mid
        if (fm < 0.0) == (fa < 0.0):
            lo, fa = mid, fm
        else:
            hi = mid
    return (0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]))


# ===========================================================================
# Quadtree build with interval pruning
# ===========================================================================


class _DCState:
    def __init__(self, node: ir.Node, depth: int):
        self.node = node
        self.f = ir.make_callable(node)
        self.depth = depth
        self.res = 1 << depth  # cells per axis at full depth
        self.x0 = self.y0 = self.x1 = self.y1 = 0.0
        # active cells: (i, j) -> vertex position (Vec2)
        self.cells: Dict[Tuple[int, int], Vec2] = {}

    def cell_bounds(self, i: int, j: int) -> Tuple[float, float, float, float]:
        w = (self.x1 - self.x0) / self.res
        h = (self.y1 - self.y0) / self.res
        return (self.x0 + i * w, self.y0 + j * h,
                self.x0 + (i + 1) * w, self.y0 + (j + 1) * h)


def _refine(state: _DCState, i0: int, j0: int, size: int) -> None:
    """Recursively refine a square block of cells spanning ``size`` leaves.

    ``(i0, j0)`` is the lower-left leaf index; ``size`` a power of two.
    """
    # bounds of this block at full resolution
    w = (state.x1 - state.x0) / state.res
    h = (state.y1 - state.y0) / state.res
    bx0 = state.x0 + i0 * w
    by0 = state.y0 + j0 * h
    bx1 = state.x0 + (i0 + size) * w
    by1 = state.y0 + (j0 + size) * h

    interval = liv.eval_interval(state.node, (bx0, by0, 0.0), (bx1, by1, 0.0))
    if liv.classify(interval) != liv.AMBIGUOUS:
        return  # pruned: wholly inside or outside -> no surface here
    if size == 1:
        _make_leaf_vertex(state, i0, j0)
        return
    half = size // 2
    for di in (0, half):
        for dj in (0, half):
            _refine(state, i0 + di, j0 + dj, half)


def _make_leaf_vertex(state: _DCState, i: int, j: int) -> None:
    x0, y0, x1, y1 = state.cell_bounds(i, j)
    f = state.f
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    qef = QEF(2)
    crossings: List[Vec2] = []
    for (ca, cb) in edges:
        pa, pb = corners[ca], corners[cb]
        fa = f(pa[0], pa[1], 0.0)
        fb = f(pb[0], pb[1], 0.0)
        if (fa < 0.0) != (fb < 0.0):
            p = _bisect_edge(f, pa, pb)
            n = ad.normal(state.node, p[0], p[1], 0.0)
            qef.insert((p[0], p[1]), (n[0], n[1]))
            crossings.append(p)
    if not crossings:
        return
    pos, _rank = qef.solve()
    # clamp the vertex into the cell (QEF can wander for near-degenerate data)
    px = min(max(pos[0], x0), x1)
    py = min(max(pos[1], y0), y1)
    state.cells[(i, j)] = (px, py)


def dual_contour_2d(node: ir.Node,
                    bounds: Tuple[float, float, float, float],
                    depth: int = 6
                    ) -> Tuple[List[Vec2], List[Tuple[int, int]]]:
    """Contour the 2D level set ``node = 0`` (evaluated at ``z = 0``).

    ``bounds`` is ``(x0, y0, x1, y1)``; ``depth`` sets the quadtree resolution
    (``2**depth`` cells per axis).  Returns ``(vertices, segments)`` where each
    segment is a pair of vertex indices.
    """
    state = _DCState(node, depth)
    state.x0, state.y0, state.x1, state.y1 = bounds
    _refine(state, 0, 0, state.res)

    # assign vertex indices
    order = sorted(state.cells.keys())
    index_of = {cell: k for k, cell in enumerate(order)}
    vertices = [state.cells[c] for c in order]

    # dual connectivity: each finest grid edge with a sign change joins the two
    # cells sharing it.
    f = state.f
    res = state.res
    w = (state.x1 - state.x0) / res
    h = (state.y1 - state.y0) / res

    def corner_val(ci: int, cj: int) -> float:
        return f(state.x0 + ci * w, state.y0 + cj * h, 0.0)

    segments: List[Tuple[int, int]] = []
    seen = set()
    for (i, j) in order:
        # horizontal grid edge along the bottom of cell (i, j): corners
        # (i, j)-(i+1, j); shared by cell (i, j-1) and (i, j).
        if (corner_val(i, j) < 0.0) != (corner_val(i + 1, j) < 0.0):
            c1, c2 = (i, j - 1), (i, j)
            if c1 in index_of and c2 in index_of:
                key = (index_of[c1], index_of[c2])
                if key not in seen:
                    seen.add(key)
                    segments.append(key)
        # vertical grid edge along the left of cell (i, j): corners
        # (i, j)-(i, j+1); shared by cell (i-1, j) and (i, j).
        if (corner_val(i, j) < 0.0) != (corner_val(i, j + 1) < 0.0):
            c1, c2 = (i - 1, j), (i, j)
            if c1 in index_of and c2 in index_of:
                key = (index_of[c1], index_of[c2])
                if key not in seen:
                    seen.add(key)
                    segments.append(key)
    return vertices, segments
