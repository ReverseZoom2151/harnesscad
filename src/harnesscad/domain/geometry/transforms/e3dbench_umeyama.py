"""Umeyama Sim(3) / SE(3) point-set alignment (E3D-Bench, Sec. 3.3 / 3.4).

E3D-Bench aligns predicted trajectories and point maps to ground truth with the
Umeyama [Umeyama 1991] least-squares similarity transform before computing pose
and reconstruction metrics.  This module provides that shared primitive plus a
tiny stdlib-only linear-algebra core (symmetric Jacobi eigensolver + a 3x3-safe
SVD) so the rest of the benchmark can align point clouds deterministically
without numpy.

Given source points ``X`` and target points ``Y`` (same count), Umeyama finds a
scale ``c``, rotation ``R`` and translation ``t`` minimising

    sum_i || Y_i - (c R X_i + t) ||^2

with a reflection guard so ``R`` is always a proper rotation (det = +1).

Everything is deterministic: no wall clock, no randomness.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Vec3 = Sequence[float]
Mat = List[List[float]]


# ---------------------------------------------------------------------------
# tiny linear algebra (stdlib only)
# ---------------------------------------------------------------------------

def _zeros(n: int, m: int) -> Mat:
    return [[0.0 for _ in range(m)] for _ in range(n)]


def _identity(n: int) -> Mat:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def matmul(a: Mat, b: Mat) -> Mat:
    """Matrix product ``a @ b``."""
    n = len(a)
    k = len(b)
    m = len(b[0])
    if len(a[0]) != k:
        raise ValueError("incompatible shapes for matmul")
    out = _zeros(n, m)
    for i in range(n):
        ai = a[i]
        oi = out[i]
        for t in range(k):
            ait = ai[t]
            if ait == 0.0:
                continue
            bt = b[t]
            for j in range(m):
                oi[j] += ait * bt[j]
    return out


def transpose(a: Mat) -> Mat:
    return [list(col) for col in zip(*a)]


def det3(a: Mat) -> float:
    """Determinant of a 3x3 matrix."""
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def jacobi_eigen(a: Mat, max_sweeps: int = 100, tol: float = 1e-14
                 ) -> Tuple[List[float], Mat]:
    """Eigenvalues/vectors of a real symmetric matrix via cyclic Jacobi.

    Returns ``(eigenvalues, V)`` where ``V`` columns are the (unit) eigenvectors
    and ``a @ V[:,i] == eigenvalues[i] * V[:,i]``.  Results are sorted by
    descending eigenvalue for stable downstream use.
    """
    n = len(a)
    # working copy
    m = [list(row) for row in a]
    v = _identity(n)
    for _ in range(max_sweeps):
        # largest off-diagonal magnitude
        off = 0.0
        p = q = 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(m[i][j]) > off:
                    off = abs(m[i][j])
                    p, q = i, j
        if off < tol:
            break
        app = m[p][p]
        aqq = m[q][q]
        apq = m[p][q]
        if apq == 0.0:
            break
        phi = 0.5 * (aqq - app) / apq
        t = (1.0 if phi >= 0 else -1.0) / (abs(phi) + math.sqrt(phi * phi + 1.0))
        c = 1.0 / math.sqrt(t * t + 1.0)
        s = t * c
        for i in range(n):
            mip = m[i][p]
            miq = m[i][q]
            m[i][p] = c * mip - s * miq
            m[i][q] = s * mip + c * miq
        for i in range(n):
            mpi = m[p][i]
            mqi = m[q][i]
            m[p][i] = c * mpi - s * mqi
            m[q][i] = s * mpi + c * mqi
        for i in range(n):
            vip = v[i][p]
            viq = v[i][q]
            v[i][p] = c * vip - s * viq
            v[i][q] = s * vip + c * viq
    eigvals = [m[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: eigvals[i], reverse=True)
    eig_sorted = [eigvals[i] for i in order]
    v_sorted = [[v[r][order[c]] for c in range(n)] for r in range(n)]
    return eig_sorted, v_sorted


def svd3(a: Mat) -> Tuple[Mat, List[float], Mat]:
    """SVD ``a = U diag(S) V^T`` for a 3x3 matrix (S descending, >= 0).

    Built from the eigendecomposition of ``a^T a``.  Degenerate columns (near
    zero singular value) are completed with a right-handed orthonormal basis so
    ``U`` and ``V`` stay orthogonal.
    """
    at = transpose(a)
    ata = matmul(at, a)
    eigvals, v = jacobi_eigen(ata)
    s = [math.sqrt(max(0.0, ev)) for ev in eigvals]
    # columns of V
    vcols = [[v[r][c] for r in range(3)] for c in range(3)]
    ucols: List[List[float]] = []
    eps = 1e-12
    for i in range(3):
        if s[i] > eps:
            av = [
                a[r][0] * vcols[i][0] + a[r][1] * vcols[i][1] + a[r][2] * vcols[i][2]
                for r in range(3)
            ]
            ucols.append([x / s[i] for x in av])
        else:
            ucols.append(None)  # fill later
    # complete missing U columns via cross products / Gram-Schmidt
    known = [c for c in ucols if c is not None]
    for i in range(3):
        if ucols[i] is None:
            if len(known) == 2:
                c = _cross(known[0], known[1])
            elif len(known) == 1:
                c = _any_orthogonal(known[0])
            else:
                c = [1.0, 0.0, 0.0]
            c = _normalize(c)
            ucols[i] = c
            known.append(c)
    u = [[ucols[c][r] for c in range(3)] for r in range(3)]
    vmat = [[vcols[c][r] for c in range(3)] for r in range(3)]
    return u, s, vmat


def _cross(a: Vec3, b: Vec3) -> List[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _normalize(a: Vec3) -> List[float]:
    n = math.sqrt(sum(x * x for x in a))
    if n == 0.0:
        return [0.0, 0.0, 0.0]
    return [x / n for x in a]


def _any_orthogonal(a: Vec3) -> List[float]:
    ref = [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0]
    return _cross(a, ref)


# ---------------------------------------------------------------------------
# Umeyama alignment
# ---------------------------------------------------------------------------

class Sim3:
    """A similarity transform ``y = c R x + t``."""

    __slots__ = ("scale", "R", "t")

    def __init__(self, scale: float, R: Mat, t: Vec3) -> None:
        self.scale = float(scale)
        self.R = [list(row) for row in R]
        self.t = list(t)

    def apply(self, x: Vec3) -> List[float]:
        r = self.R
        c = self.scale
        return [
            c * (r[0][0] * x[0] + r[0][1] * x[1] + r[0][2] * x[2]) + self.t[0],
            c * (r[1][0] * x[0] + r[1][1] * x[1] + r[1][2] * x[2]) + self.t[1],
            c * (r[2][0] * x[0] + r[2][1] * x[1] + r[2][2] * x[2]) + self.t[2],
        ]

    def apply_all(self, xs: Sequence[Vec3]) -> List[List[float]]:
        return [self.apply(x) for x in xs]


def umeyama_alignment(source: Sequence[Vec3], target: Sequence[Vec3],
                      with_scale: bool = True) -> Sim3:
    """Least-squares Sim(3)/SE(3) alignment mapping ``source`` onto ``target``.

    Implements Umeyama (1991).  With ``with_scale=False`` the scale is fixed to
    1 (an SE(3) / rigid alignment, used when metric scale must be preserved).

    Raises ``ValueError`` for empty or mismatched inputs.
    """
    n = len(source)
    if n == 0 or n != len(target):
        raise ValueError("source and target must be non-empty and equal length")

    dim = 3
    mu_x = [sum(p[d] for p in source) / n for d in range(dim)]
    mu_y = [sum(p[d] for p in target) / n for d in range(dim)]

    # covariance Sigma = (1/n) sum (y-mu_y)(x-mu_x)^T  and source variance
    sigma = _zeros(dim, dim)
    var_x = 0.0
    for i in range(n):
        dx = [source[i][d] - mu_x[d] for d in range(dim)]
        dy = [target[i][d] - mu_y[d] for d in range(dim)]
        var_x += sum(v * v for v in dx)
        for r in range(dim):
            for c in range(dim):
                sigma[r][c] += dy[r] * dx[c]
    sigma = [[sigma[r][c] / n for c in range(dim)] for r in range(dim)]
    var_x /= n

    u, s, v = svd3(sigma)
    # reflection guard: S = diag(1,1, det(U)det(V))
    d = [1.0, 1.0, 1.0]
    if det3(u) * det3(v) < 0:
        d[2] = -1.0
    # R = U diag(d) V^T
    ud = [[u[r][c] * d[c] for c in range(dim)] for r in range(dim)]
    R = matmul(ud, transpose(v))

    if with_scale:
        if var_x <= 0.0:
            scale = 1.0
        else:
            scale = (s[0] * d[0] + s[1] * d[1] + s[2] * d[2]) / var_x
    else:
        scale = 1.0

    t = [mu_y[r] - scale * (R[r][0] * mu_x[0] + R[r][1] * mu_x[1] + R[r][2] * mu_x[2])
         for r in range(dim)]
    return Sim3(scale, R, t)


def alignment_rmse(source: Sequence[Vec3], target: Sequence[Vec3],
                   transform: Sim3) -> float:
    """Root-mean-square residual after applying ``transform`` to ``source``."""
    n = len(source)
    if n == 0:
        return 0.0
    total = 0.0
    for i in range(n):
        p = transform.apply(source[i])
        total += sum((p[d] - target[i][d]) ** 2 for d in range(3))
    return math.sqrt(total / n)
