"""Dual Contouring of Hermite Data in 3D, on a uniform grid.

Marching Cubes (Lorensen & Cline 1987) places its vertices on grid *edges*, at
the linearly-interpolated zero crossing.  A vertex constrained to an edge can
never land on a point where two or three surfaces meet, so **marching cubes
cannot represent a sharp feature**: every convex corner of a box is chamfered
off by up to half a cell, and every crease is faceted.  That is a systematic,
one-sided *loss* of material, and it is the whole of the volume error a
prismatic part shows under MC.

Dual Contouring (Ju, Losasso, Schaefer & Warren, "Dual Contouring of Hermite
Data", ACM TOG 21(3):339-346, 2002 -- https://www.cs.rice.edu/~jwarren/papers/dualcontour.pdf)
places one vertex per *cell* instead, and positions it by minimising the
**quadratic error function** built from the cell's Hermite data -- the surface
crossing point ``p_i`` on each sign-changing edge together with the surface
normal ``n_i`` there:

    E(x) = sum_i ( n_i . (x - p_i) )^2

Each term is the squared distance from ``x`` to the tangent plane of the
surface at ``p_i``.  Where three tangent planes meet -- a box corner -- the
unique minimiser of ``E`` is exactly their intersection point, so the corner is
reproduced **exactly**, not chamfered.  This is the algorithm behind libfive
and Fidget (Matt Keeter).

Numerics.  ``E(x) = (Ax - b)^T (Ax - b)`` with the normals as the rows of ``A``
and ``b_i = n_i . p_i``, so the minimiser solves the normal equations
``A^T A x = A^T b``.  ``A^T A`` is routinely rank-deficient (on a flat wall it
has rank 1, on a crease rank 2), so it is *never* inverted directly.  Following
Ju et al. -- and Keeter's QEF writeup, https://www.mattkeeter.com/projects/qef/ --
we take the symmetric eigen-decomposition of the 3x3 ``A^T A``, form the
**truncated pseudo-inverse** (reciprocate the eigenvalues, but send any
eigenvalue below a relative threshold to zero), and bias the solve toward the
cell's **mass point** ``c`` (the mean of the crossing points) so that the
undetermined directions of a rank-deficient system resolve to something inside
the cell rather than drifting to the origin:

    x = c + pinv(A^T A) . ( A^T b - A^T A c )

The result is finally clamped into the cell, which is what keeps the dual mesh
from self-intersecting when the QEF is near-degenerate.

Connectivity is the dual rule: every *minimal grid edge* that changes sign is
shared by exactly four cells, and the four vertices of those cells are joined
into a quad.  The quad is wound from the sign of the edge so that the mesh
comes out consistently outward-facing.

Known limitation, stated plainly (Ju et al. 2002, and Schaefer et al.,
"Manifold Dual Contouring", TVCG 2007): one vertex per cell means a cell
through which *two* separate sheets of the surface pass gets a single shared
vertex, which can make the output non-manifold.  It is not a hole -- the mesh
stays closed -- but the half-edge check can legitimately reject it.  Manifold
DC fixes this by splitting such cells' vertices; that is not implemented here.
Prismatic CAD parts at a sane resolution do not hit it.

Pure stdlib, deterministic, no numpy.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad.domain.geometry.volumes.surface_nets import ScalarGrid

Vec3 = Tuple[float, float, float]

#: Local corner offsets of a cell, in the standard marching-cubes numbering.
_CORNER: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
)

#: The 12 cell edges as (corner_a, corner_b) index pairs.
_EDGE_VERTS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

#: Eigenvalues of A^T A below this fraction of the largest are treated as zero.
#: 0.1 is the value libfive uses; it is deliberately blunt, because the whole
#: point is to refuse to trust near-degenerate directions.
QEF_EIGEN_CUTOFF = 0.1


# ---------------------------------------------------------------------------
# 3x3 symmetric eigen-decomposition (cyclic Jacobi) and the truncated solve
# ---------------------------------------------------------------------------
def _jacobi3(a_in: List[List[float]], iters: int = 24
             ) -> Tuple[List[float], List[List[float]]]:
    """Eigenvalues and eigenvectors of a symmetric 3x3 matrix (cyclic Jacobi).

    Returns ``(values, vectors)`` with ``vectors[k]`` the unit eigenvector for
    ``values[k]``.  Deterministic: a fixed sweep count, no tolerance-based exit
    that could depend on rounding.
    """
    a = [row[:] for row in a_in]
    v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    for _ in range(iters):
        # largest off-diagonal magnitude
        p, q, best = 0, 1, 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                m = abs(a[i][j])
                if m > best:
                    p, q, best = i, j, m
        if best <= 1e-300:
            break
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        theta = 0.5 * (aqq - app) / apq
        t = (1.0 if theta >= 0.0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
        c = 1.0 / math.sqrt(t * t + 1.0)
        s = t * c
        for k in range(3):
            akp, akq = a[k][p], a[k][q]
            a[k][p] = c * akp - s * akq
            a[k][q] = s * akp + c * akq
        for k in range(3):
            apk, aqk = a[p][k], a[q][k]
            a[p][k] = c * apk - s * aqk
            a[q][k] = s * apk + c * aqk
        for k in range(3):
            vkp, vkq = v[k][p], v[k][q]
            v[k][p] = c * vkp - s * vkq
            v[k][q] = s * vkp + c * vkq
    values = [a[0][0], a[1][1], a[2][2]]
    vectors = [[v[0][k], v[1][k], v[2][k]] for k in range(3)]
    return values, vectors


def solve_qef(points: Sequence[Vec3], normals: Sequence[Vec3],
              mass_point: Vec3) -> Vec3:
    """Minimise ``sum_i (n_i . (x - p_i))^2``, biased toward ``mass_point``.

    Ju et al. 2002 sec. 2.2 / Keeter's QEF notes: build ``A^T A`` and
    ``A^T b`` from the Hermite data, eigen-decompose the (symmetric, 3x3)
    ``A^T A``, and apply the pseudo-inverse with any eigenvalue below
    :data:`QEF_EIGEN_CUTOFF` of the largest truncated to zero.  Solving for the
    offset from the mass point rather than from the origin is what makes the
    rank-deficient cases (flat walls: rank 1; creases: rank 2) land somewhere
    sensible instead of shooting off to infinity.
    """
    ata = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    atb = [0.0, 0.0, 0.0]
    for p, n in zip(points, normals):
        d = n[0] * p[0] + n[1] * p[1] + n[2] * p[2]
        for i in range(3):
            atb[i] += n[i] * d
            for j in range(3):
                ata[i][j] += n[i] * n[j]

    # residual right-hand side about the mass point: A^T b - A^T A c
    c = (float(mass_point[0]), float(mass_point[1]), float(mass_point[2]))
    rhs = [atb[i] - (ata[i][0] * c[0] + ata[i][1] * c[1] + ata[i][2] * c[2])
           for i in range(3)]

    values, vectors = _jacobi3(ata)
    biggest = max(abs(x) for x in values)
    if biggest <= 0.0:
        return c
    cutoff = QEF_EIGEN_CUTOFF * biggest

    # x = c + (sum_k [ |lambda_k| > cutoff ] * (e_k . rhs) / lambda_k * e_k)
    out = [c[0], c[1], c[2]]
    for k in range(3):
        lam = values[k]
        if abs(lam) <= cutoff:
            continue                      # truncated: this direction is not determined
        e = vectors[k]
        coeff = (e[0] * rhs[0] + e[1] * rhs[1] + e[2] * rhs[2]) / lam
        out[0] += coeff * e[0]
        out[1] += coeff * e[1]
        out[2] += coeff * e[2]
    return (out[0], out[1], out[2])


# ---------------------------------------------------------------------------
# Hermite data
# ---------------------------------------------------------------------------
def _gradient(field: Callable[[Sequence[float]], float], p: Vec3, h: float) -> Vec3:
    """Central-difference gradient, normalised.

    Inigo Quilez, "normals for an SDF" (https://iquilezles.org/articles/normalsSDF/):
    the surface normal of a distance field is the normalised gradient, and the
    central difference is the correct estimator -- the forward difference is
    biased by O(h) and visibly skews the shading.  ``h`` is a fraction of the
    cell, not an absolute epsilon, so the estimate scales with the model.
    """
    gx = field((p[0] + h, p[1], p[2])) - field((p[0] - h, p[1], p[2]))
    gy = field((p[0], p[1] + h, p[2])) - field((p[0], p[1] - h, p[2]))
    gz = field((p[0], p[1], p[2] + h)) - field((p[0], p[1], p[2] - h))
    m = math.sqrt(gx * gx + gy * gy + gz * gz)
    if m == 0.0:
        return (0.0, 0.0, 1.0)
    return (gx / m, gy / m, gz / m)


def _crossing(field: Callable[[Sequence[float]], float],
              a: Vec3, b: Vec3, va: float, vb: float, iso: float,
              refine: int) -> Vec3:
    """The point on segment ``a->b`` where the field crosses ``iso``.

    Starts from the linear interpolant (which is all Ju et al. assume is
    available) and then optionally runs ``refine`` bisection steps against the
    *true* field.  We have the analytic field in hand, so we may as well use it:
    the Hermite crossing is only as good as the sample it came from, and a
    linear guess on a curved field is off by O(cell^2).
    """
    den = vb - va
    t = 0.5 if den == 0.0 else (iso - va) / den
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    lo, hi = 0.0, 1.0
    for _ in range(int(refine)):
        p = (a[0] + t * (b[0] - a[0]),
             a[1] + t * (b[1] - a[1]),
             a[2] + t * (b[2] - a[2]))
        v = field(p) - iso
        # keep the bracket that still straddles the iso-value
        if (v < 0.0) == (va - iso < 0.0):
            lo = t
        else:
            hi = t
        t = 0.5 * (lo + hi)
    return (a[0] + t * (b[0] - a[0]),
            a[1] + t * (b[1] - a[1]),
            a[2] + t * (b[2] - a[2]))


# ---------------------------------------------------------------------------
# the mesher
# ---------------------------------------------------------------------------
def dual_contour(grid: ScalarGrid,
                 field: Callable[[Sequence[float]], float],
                 iso: float = 0.0,
                 refine: int = 12,
                 clamp: bool = True) -> Tuple[List[Vec3], List[Tuple[int, int, int]]]:
    """Extract the ``iso`` level set of ``grid`` by dual contouring.

    ``grid`` supplies the sign configuration (exactly the same samples marching
    cubes would use); ``field`` is the analytic function behind it, used for the
    Hermite data -- the crossing points and the normals there.  Returns
    ``(vertices, triangles)``.

    ``clamp`` keeps each QEF solution inside its own cell.  Ju et al. do this
    too; without it a near-degenerate QEF can throw a vertex clear across the
    model and tangle the dual mesh.
    """
    nx, ny, nz = grid.shape
    sx, sy, sz = grid.spacing
    ox, oy, oz = grid.origin
    vals = grid.values
    cx, cy, cz = nx - 1, ny - 1, nz - 1          # cell counts
    if cx < 1 or cy < 1 or cz < 1:
        return [], []

    h = 0.5 * min(sx, sy, sz) * 1e-3             # gradient step: sub-cell, scale-aware

    def vidx(i: int, j: int, k: int) -> int:
        return i + nx * (j + ny * k)

    def value(i: int, j: int, k: int) -> float:
        return vals[vidx(i, j, k)] - iso

    def world(i: int, j: int, k: int) -> Vec3:
        return (ox + i * sx, oy + j * sy, oz + k * sz)

    # -- one QEF vertex per surface-straddling cell -------------------------
    verts: List[Vec3] = []
    cell_vertex: Dict[Tuple[int, int, int], int] = {}

    for k in range(cz):
        for j in range(cy):
            for i in range(cx):
                corner_v = [value(i + d[0], j + d[1], k + d[2]) for d in _CORNER]
                neg = [v < 0.0 for v in corner_v]
                if all(neg) or not any(neg):
                    continue                     # no crossing: no vertex

                pts: List[Vec3] = []
                nrm: List[Vec3] = []
                for (ca, cb) in _EDGE_VERTS:
                    va, vb = corner_v[ca], corner_v[cb]
                    if (va < 0.0) == (vb < 0.0):
                        continue
                    da, db = _CORNER[ca], _CORNER[cb]
                    pa = world(i + da[0], j + da[1], k + da[2])
                    pb = world(i + db[0], j + db[1], k + db[2])
                    p = _crossing(field, pa, pb, va + iso, vb + iso, iso, refine)
                    pts.append(p)
                    nrm.append(_gradient(field, p, h))

                if not pts:                      # pragma: no cover (sign says otherwise)
                    continue
                n = float(len(pts))
                mass = (sum(p[0] for p in pts) / n,
                        sum(p[1] for p in pts) / n,
                        sum(p[2] for p in pts) / n)
                x = solve_qef(pts, nrm, mass)

                if clamp:
                    lo = world(i, j, k)
                    hi = (lo[0] + sx, lo[1] + sy, lo[2] + sz)
                    x = (min(max(x[0], lo[0]), hi[0]),
                         min(max(x[1], lo[1]), hi[1]),
                         min(max(x[2], lo[2]), hi[2]))

                cell_vertex[(i, j, k)] = len(verts)
                verts.append(x)

    # -- dual connectivity: one quad per sign-changing minimal grid edge ----
    tris: List[Tuple[int, int, int]] = []

    def emit(cells: Sequence[Tuple[int, int, int]], flip: bool) -> None:
        try:
            q = [cell_vertex[c] for c in cells]
        except KeyError:                          # pragma: no cover
            return                                # a sharing cell had no vertex
        if flip:
            q = [q[3], q[2], q[1], q[0]]
        # split the quad along the shorter diagonal: it keeps the two triangles
        # from folding over each other when the QEF has pulled the four vertices
        # into a strongly non-planar quad (a sharp crease does exactly that).
        d02 = sum((verts[q[0]][t] - verts[q[2]][t]) ** 2 for t in range(3))
        d13 = sum((verts[q[1]][t] - verts[q[3]][t]) ** 2 for t in range(3))
        if d02 <= d13:
            tris.append((q[0], q[1], q[2]))
            tris.append((q[0], q[2], q[3]))
        else:
            tris.append((q[1], q[2], q[3]))
            tris.append((q[1], q[3], q[0]))

    # An edge along +x from grid point (i,j,k) is shared by the four cells whose
    # x-index is i and whose (y,z) indices are (j-1|j, k-1|k). Listing them in
    # the order below walks counter-clockwise about the +x axis, so when the
    # surface normal points along +x (inside at the low endpoint, outside at the
    # high one) the quad already faces outward.
    for k in range(1, cz):
        for j in range(1, cy):
            for i in range(cx):
                a, b = value(i, j, k), value(i + 1, j, k)
                if (a < 0.0) == (b < 0.0):
                    continue
                emit(((i, j - 1, k - 1), (i, j, k - 1), (i, j, k), (i, j - 1, k)),
                     flip=not (a < 0.0))

    # +y: right-handed about +y is the (z, x) plane.
    for k in range(1, cz):
        for j in range(cy):
            for i in range(1, cx):
                a, b = value(i, j, k), value(i, j + 1, k)
                if (a < 0.0) == (b < 0.0):
                    continue
                emit(((i - 1, j, k - 1), (i - 1, j, k), (i, j, k), (i, j, k - 1)),
                     flip=not (a < 0.0))

    # +z: right-handed about +z is the (x, y) plane.
    for k in range(cz):
        for j in range(1, cy):
            for i in range(1, cx):
                a, b = value(i, j, k), value(i, j, k + 1)
                if (a < 0.0) == (b < 0.0):
                    continue
                emit(((i - 1, j - 1, k), (i, j - 1, k), (i, j, k), (i - 1, j, k)),
                     flip=not (a < 0.0))

    return verts, tris
