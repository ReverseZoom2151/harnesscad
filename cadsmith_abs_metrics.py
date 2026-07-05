"""CADSmith absolute-millimetre-space evaluation metrics (CADSmith sec. III-G).

CADSmith deliberately departs from the normalized [0,1]^3 evaluation used in
prior work: normalized metrics erase dimensional accuracy, scoring a 10mm box
and a 100mm box identically if their shapes match. Because the benchmark
specifies explicit millimetre dimensions, all metrics are computed in absolute
millimetre space. Before scoring, the two point sets are co-registered by
translating their bounding-box centres to the origin, then aligned with
Iterative Closest Point (ICP) to resolve orientation mismatches from different
construction frames.

This module implements that deterministic pipeline, stdlib-only:

  * :func:`bbox_center` / :func:`center_to_origin` — bbox-centre co-registration,
  * :func:`kabsch` — the optimal rigid rotation between corresponded point sets,
  * :func:`icp` — deterministic point-to-point ICP (nearest-neighbour
    correspondences + Kabsch), returning the aligned points and per-iteration RMS,
  * :func:`f1_score` — F1 at a distance threshold tau (default 1.0mm) in absolute
    space,
  * :func:`voxel_iou` — volumetric IoU on 1.0mm occupancy grids with adaptive
    coarsening for parts exceeding a 100mm extent,
  * :func:`chamfer` — bidirectional mean nearest-neighbour distance.

Everything is pure Python (no numpy): 3x3 linear algebra done by hand, so the
results are exactly reproducible. Intended for modest point counts (tests and
diagnostics), not the 10k-point production sampling.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Mat3 = Tuple[Vec3, Vec3, Vec3]

_IDENTITY: Mat3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Small vector / matrix helpers
# --------------------------------------------------------------------------- #
def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _matvec(m: Mat3, v: Vec3) -> Vec3:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def _matmul(a: Mat3, b: Mat3) -> Mat3:
    return tuple(
        tuple(sum(a[r][k] * b[k][c] for k in range(3)) for c in range(3))
        for r in range(3)
    )  # type: ignore[return-value]


def _transpose(m: Mat3) -> Mat3:
    return tuple(tuple(m[r][c] for r in range(3)) for c in range(3))  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Co-registration by bounding-box centre
# --------------------------------------------------------------------------- #
def bbox_center(points: Sequence[Vec3]) -> Vec3:
    if not points:
        raise ValueError("empty point set")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    return (
        0.5 * (min(xs) + max(xs)),
        0.5 * (min(ys) + max(ys)),
        0.5 * (min(zs) + max(zs)),
    )


def bbox_extent(points: Sequence[Vec3]) -> Vec3:
    if not points:
        raise ValueError("empty point set")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    return (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))


def center_to_origin(points: Sequence[Vec3]) -> List[Vec3]:
    """Translate the point set so its bounding-box centre sits at the origin."""
    c = bbox_center(points)
    return [_sub(p, c) for p in points]


# --------------------------------------------------------------------------- #
# Kabsch — optimal rotation between corresponded point sets
# --------------------------------------------------------------------------- #
def kabsch(src: Sequence[Vec3], dst: Sequence[Vec3]) -> Mat3:
    """Return the proper rotation R minimising sum |R*src_i - dst_i|^2.

    Both sets must be equal length and are assumed already centred (or the
    rotation is about their common origin). Uses the cross-covariance matrix and
    an iterative symmetric-eigenproblem-free polar decomposition via Newton-style
    orthogonalisation — deterministic and numpy-free.
    """
    if len(src) != len(dst):
        raise ValueError("src and dst must have equal length")
    if not src:
        raise ValueError("empty point set")

    # Cross-covariance H = sum src_i * dst_i^T
    H = [[0.0] * 3 for _ in range(3)]
    for s, d in zip(src, dst):
        for r in range(3):
            for c in range(3):
                H[r][c] += s[r] * d[c]
    Hm: Mat3 = tuple(tuple(row) for row in H)  # type: ignore[assignment]

    # Polar decomposition H = R P via iteration: R_{k+1} = 1/2 (R_k + (R_k^-T)).
    # We approximate the orthogonal factor of H by iterating from H itself,
    # which converges to the rotation for a well-conditioned H.
    R = Hm
    for _ in range(64):
        Rinv_t = _inverse_transpose(R)
        if Rinv_t is None:
            break
        newR = tuple(
            tuple(0.5 * (R[r][c] + Rinv_t[r][c]) for c in range(3))
            for r in range(3)
        )  # type: ignore[assignment]
        if _frob_diff(newR, R) < 1e-15:
            R = newR
            break
        R = newR

    # The Newton iteration converges to the orthogonal polar factor Q of H,
    # where H = Q P. For the Kabsch problem R*src ~= dst the optimal rotation is
    # Q^T, so transpose before returning.
    R = _transpose(R)

    # Ensure a proper rotation (det = +1); if reflected, flip the least-aligned axis.
    if _det(R) < 0:
        R = tuple(
            tuple(-R[r][c] if r == 2 else R[r][c] for c in range(3))
            for r in range(3)
        )  # type: ignore[assignment]
    return R


def _det(m: Mat3) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _inverse_transpose(m: Mat3):
    det = _det(m)
    if abs(det) < 1e-18:
        return None
    # cofactor matrix
    c = [[0.0] * 3 for _ in range(3)]
    c[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1])
    c[0][1] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0])
    c[0][2] = (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    c[1][0] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1])
    c[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0])
    c[1][2] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0])
    c[2][0] = (m[0][1] * m[1][2] - m[0][2] * m[1][1])
    c[2][1] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0])
    c[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0])
    # inverse = adjugate^T / det = cofactor / det ; inverse-transpose = cofactor/det.
    inv_t = tuple(tuple(c[r][col] / det for col in range(3)) for r in range(3))
    return inv_t  # type: ignore[return-value]


def _frob_diff(a: Mat3, b: Mat3) -> float:
    return sum((a[r][c] - b[r][c]) ** 2 for r in range(3) for c in range(3))


# --------------------------------------------------------------------------- #
# ICP
# --------------------------------------------------------------------------- #
def _nearest(p: Vec3, cloud: Sequence[Vec3]) -> Tuple[int, float]:
    best_i, best_d = 0, float("inf")
    for i, q in enumerate(cloud):
        d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2
        if d < best_d:
            best_i, best_d = i, d
    return best_i, best_d


def _rms(src: Sequence[Vec3], dst: Sequence[Vec3]) -> float:
    n = len(src)
    if n == 0:
        return 0.0
    total = sum(
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
        for a, b in zip(src, dst)
    )
    return math.sqrt(total / n)


def icp(source: Sequence[Vec3], target: Sequence[Vec3],
        *, max_iter: int = 20, tol: float = 1e-9) -> Tuple[List[Vec3], List[float]]:
    """Deterministic point-to-point ICP.

    Both sets are first centred on their bbox centres (per CADSmith), then each
    iteration finds nearest-neighbour correspondences from source to target,
    solves the optimal rotation with Kabsch, and applies it. Returns the aligned
    source points and the RMS history (one entry per iteration).
    """
    src = center_to_origin(source)
    dst = center_to_origin(target)
    history: List[float] = []
    prev = float("inf")
    for _ in range(max_iter):
        corr = [dst[_nearest(p, dst)[0]] for p in src]
        R = kabsch(src, corr)
        src = [_matvec(R, p) for p in src]
        rms = _rms(src, [dst[_nearest(p, dst)[0]] for p in src])
        history.append(rms)
        if abs(prev - rms) < tol:
            break
        prev = rms
    return src, history


# --------------------------------------------------------------------------- #
# Chamfer / F1 in absolute space
# --------------------------------------------------------------------------- #
def chamfer(a: Sequence[Vec3], b: Sequence[Vec3]) -> float:
    """Bidirectional mean nearest-neighbour Euclidean distance (absolute mm)."""
    if not a or not b:
        raise ValueError("empty point set")
    fwd = sum(math.sqrt(_nearest(p, b)[1]) for p in a) / len(a)
    bwd = sum(math.sqrt(_nearest(p, a)[1]) for p in b) / len(b)
    return 0.5 * (fwd + bwd)


def f1_score(pred: Sequence[Vec3], target: Sequence[Vec3],
             *, tau: float = 1.0) -> float:
    """F1 at absolute distance threshold ``tau`` (mm).

    Precision = fraction of predicted points whose nearest target point is
    within tau; recall = fraction of target points covered within tau.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    if not pred or not target:
        raise ValueError("empty point set")
    t2 = tau * tau
    precision = sum(1 for p in pred if _nearest(p, target)[1] <= t2) / len(pred)
    recall = sum(1 for q in target if _nearest(q, pred)[1] <= t2) / len(target)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- #
# Volumetric IoU on occupancy grids (adaptive coarsening)
# --------------------------------------------------------------------------- #
def voxel_resolution(pred: Sequence[Vec3], target: Sequence[Vec3],
                     *, base_mm: float = 1.0, coarsen_extent_mm: float = 100.0
                     ) -> float:
    """Choose the voxel size: 1.0mm by default, coarsened for large parts.

    If the maximum bounding-box extent across both point sets exceeds
    ``coarsen_extent_mm``, the voxel size scales up proportionally so the grid
    stays bounded (prevents memory exhaustion, per the paper).
    """
    ext = max(max(bbox_extent(pred)), max(bbox_extent(target)))
    if ext <= coarsen_extent_mm:
        return base_mm
    return base_mm * (ext / coarsen_extent_mm)


def _occupancy(points: Sequence[Vec3], size: float, origin: Vec3):
    cells = set()
    for p in points:
        cells.add((
            math.floor((p[0] - origin[0]) / size),
            math.floor((p[1] - origin[1]) / size),
            math.floor((p[2] - origin[2]) / size),
        ))
    return cells


def voxel_iou(pred: Sequence[Vec3], target: Sequence[Vec3],
              *, base_mm: float = 1.0, coarsen_extent_mm: float = 100.0) -> float:
    """Intersection-over-union of voxelised occupancy for the two point sets.

    Both sets share one grid (common origin = min corner of their union) at the
    adaptively-chosen resolution, so occupancy cells are directly comparable.
    """
    if not pred or not target:
        raise ValueError("empty point set")
    size = voxel_resolution(pred, target,
                            base_mm=base_mm, coarsen_extent_mm=coarsen_extent_mm)
    allpts = list(pred) + list(target)
    origin = (min(p[0] for p in allpts),
              min(p[1] for p in allpts),
              min(p[2] for p in allpts))
    a = _occupancy(pred, size, origin)
    b = _occupancy(target, size, origin)
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# --------------------------------------------------------------------------- #
# Full absolute-space comparison (co-register -> ICP -> metrics)
# --------------------------------------------------------------------------- #
def absolute_space_metrics(pred: Sequence[Vec3], target: Sequence[Vec3],
                           *, tau: float = 1.0, icp_iters: int = 20) -> dict:
    """Run the full CADSmith comparison and return CD / F1 / IoU.

    Co-registers by bbox centre, aligns the prediction to the target with ICP,
    then scores Chamfer distance, F1@tau, and voxel IoU in absolute mm space.
    """
    aligned, hist = icp(pred, target, max_iter=icp_iters)
    dst = center_to_origin(target)
    return {
        "chamfer": chamfer(aligned, dst),
        "f1": f1_score(aligned, dst, tau=tau),
        "iou": voxel_iou(aligned, dst),
        "icp_rms": hist[-1] if hist else 0.0,
    }
