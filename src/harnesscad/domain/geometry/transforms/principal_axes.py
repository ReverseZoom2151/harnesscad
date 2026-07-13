"""Correspondence-free inertia / principal-axis shape alignment (CAD-Coder).

CAD-Coder (Doris, Alam, Nobari & Ahmed, 2025 -- arXiv:2505.14646) evaluates a
generated CAD solid against the ground-truth solid with an *alignment-invariant*
volumetric IoU. Its ``SolidAlign`` routine (``SolidAlign/_cq.py`` /
``scripts/compute_iou.py::cq_align_shapes``) registers the two solids WITHOUT any
point correspondences, using only their mass distribution:

  1. translate each solid to its own centre of mass,
  2. compute the moment-of-inertia matrix, eigendecompose it to get the three
     principal axes (the solid's own body frame),
  3. isotropically rescale each solid by ``sqrt(sum|eigenvalue| / volume)`` so
     the two are size-normalised,
  4. rotate the source body frame onto the target body frame -- but a principal
     frame is only defined up to the sign of each eigenvector, so CAD-Coder
     enumerates the FOUR proper-rotation sign combinations and keeps whichever
     maximises the overlap IoU.

The overlap in the paper is an exact OpenCascade boolean-volume IoU (external
kernel).  This module reimplements everything *around* that boolean -- the
deterministic geometry-to-alignment pipeline -- in pure stdlib Python, operating
on a sampled point cloud of each solid and scoring candidates with a voxel
occupancy IoU (reusing :mod:`bench.magic3d_voxel_metrics`).

Why this is not a duplicate:

  * :mod:`bench.solid_iou` normalises by inertia and enumerates the 24 proper
    axis-permutation rotations *abstractly*, delegating centroid, inertia,
    normalisation, alignment AND iou to an injected external adapter.  It never
    computes a principal frame from raw geometry.
  * :func:`geometry.e3dbench_umeyama.umeyama_alignment` needs known point
    CORRESPONDENCES; SolidAlign has none -- the two solids are different meshes.

This module supplies the missing concrete pieces: the covariance / inertia
tensor of a point set, the principal frame via the shared Jacobi eigensolver,
the exact CAD-Coder four-sign eigenvector-ambiguity rotation family, and the
best-of-four alignment with a correspondence-free voxel-IoU score.

Pure stdlib, deterministic (no wall clock, no RNG).
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from harnesscad.domain.geometry.transforms.umeyama import jacobi_eigen, matmul, transpose

Vec3 = Sequence[float]
Mat = List[List[float]]


# --------------------------------------------------------------------------- #
# mass-distribution descriptors
# --------------------------------------------------------------------------- #
def centroid(points: Sequence[Vec3]) -> Tuple[float, float, float]:
    """Arithmetic mean (centre of mass for equal point masses)."""
    n = len(points)
    if n == 0:
        raise ValueError("centroid needs at least one point")
    sx = sy = sz = 0.0
    for p in points:
        sx += p[0]
        sy += p[1]
        sz += p[2]
    return (sx / n, sy / n, sz / n)


def covariance(points: Sequence[Vec3]) -> Mat:
    """Symmetric 3x3 second-central-moment matrix about the centroid.

    Its eigenvectors are the principal axes of the point set; its eigenvalues
    are the variances along those axes.
    """
    n = len(points)
    if n == 0:
        raise ValueError("covariance needs at least one point")
    cx, cy, cz = centroid(points)
    xx = yy = zz = xy = xz = yz = 0.0
    for p in points:
        dx, dy, dz = p[0] - cx, p[1] - cy, p[2] - cz
        xx += dx * dx
        yy += dy * dy
        zz += dz * dz
        xy += dx * dy
        xz += dx * dz
        yz += dy * dz
    return [[xx / n, xy / n, xz / n],
            [xy / n, yy / n, yz / n],
            [xz / n, yz / n, zz / n]]


def inertia_tensor(points: Sequence[Vec3]) -> Mat:
    """Rigid-body moment-of-inertia matrix about the centroid (unit masses).

    ``I = sum_i (|r_i|^2 * Identity - r_i r_i^T)``.  This is the descriptor
    CAD-Coder uses (OpenCascade ``matrixOfInertia``).  ``I = trace(C)*Id - C``
    where ``C`` is the (unnormalised) covariance, so it shares its eigenvectors
    with :func:`covariance`; provided for parity with the paper.
    """
    n = len(points)
    if n == 0:
        raise ValueError("inertia_tensor needs at least one point")
    cx, cy, cz = centroid(points)
    ixx = iyy = izz = ixy = ixz = iyz = 0.0
    for p in points:
        dx, dy, dz = p[0] - cx, p[1] - cy, p[2] - cz
        ixx += dy * dy + dz * dz
        iyy += dx * dx + dz * dz
        izz += dx * dx + dy * dy
        ixy -= dx * dy
        ixz -= dx * dz
        iyz -= dy * dz
    return [[ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz]]


def principal_frame(points: Sequence[Vec3]
                    ) -> Tuple[Tuple[float, float, float], List[float], Mat]:
    """Return ``(centroid, eigenvalues, V)`` for the point set.

    ``V`` is a 3x3 matrix whose COLUMNS are the unit principal axes (body
    frame), sorted by descending eigenvalue (via the shared Jacobi solver).
    """
    return centroid(points), *jacobi_eigen(covariance(points))


def normalization_scale(eigenvalues: Sequence[float]) -> float:
    """Isotropic size normaliser ``sqrt(sum|eigenvalue|)`` (RMS extent).

    The point-cloud analogue of CAD-Coder's ``sqrt(sum|inertia eig| / volume)``:
    for a unit-mass point set the covariance eigenvalues are the per-axis
    variances, so this is the root-mean-square distance to the centroid.
    """
    total = sum(abs(e) for e in eigenvalues)
    if total <= 0.0:
        raise ValueError("degenerate point set: zero spatial extent")
    return math.sqrt(total)


# --------------------------------------------------------------------------- #
# eigenvector sign ambiguity -> proper-rotation family
# --------------------------------------------------------------------------- #
def reflection_signs() -> Tuple[Tuple[int, int, int], ...]:
    """The four eigenvector-sign combinations CAD-Coder enumerates.

    A principal frame is defined only up to the sign of each eigenvector.  Only
    an EVEN number of flips keeps the frame right-handed (det = +1), giving
    exactly four proper choices -- identity plus the three double-flips.  This
    reproduces ``align_shapes``'s ``alignment = 1 - 2*[i>0, (i+1)%2, i%3<=1]``.
    """
    return ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))


def _scale_columns(v: Mat, signs: Sequence[int]) -> Mat:
    return [[v[r][c] * signs[c] for c in range(3)] for r in range(3)]


def candidate_rotations(source_axes: Mat, target_axes: Mat) -> List[Mat]:
    """Four rotations mapping the source body frame onto the target body frame.

    ``R = target_axes @ (signs * source_axes)^T`` for each sign combination,
    exactly as CAD-Coder forms ``I_v_target @ (alignment * I_v_source).T``.
    Each is a proper rotation (det = +1).
    """
    return [matmul(target_axes, transpose(_scale_columns(source_axes, s)))
            for s in reflection_signs()]


def apply_rotation(points: Sequence[Vec3], rotation: Mat) -> List[List[float]]:
    """Rotate each point ``p`` to ``rotation @ p``."""
    out = []
    for p in points:
        out.append([
            rotation[0][0] * p[0] + rotation[0][1] * p[1] + rotation[0][2] * p[2],
            rotation[1][0] * p[0] + rotation[1][1] * p[1] + rotation[1][2] * p[2],
            rotation[2][0] * p[0] + rotation[2][1] * p[1] + rotation[2][2] * p[2],
        ])
    return out


def normalize_points(points: Sequence[Vec3], centre: Vec3,
                     scale: float) -> List[List[float]]:
    """Translate to ``centre`` origin and isotropically divide by ``scale``."""
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return [[(p[0] - centre[0]) / scale,
             (p[1] - centre[1]) / scale,
             (p[2] - centre[2]) / scale] for p in points]


# --------------------------------------------------------------------------- #
# correspondence-free voxel-IoU scoring
# --------------------------------------------------------------------------- #
def voxel_iou_score(a: Sequence[Vec3], b: Sequence[Vec3],
                    *, resolution: int = 32) -> float:
    """Occupancy IoU of two point clouds on a shared voxel grid.

    A common axis-aligned grid is derived from the combined bounding box and
    ``resolution`` cells along the longest axis, then both clouds are voxelised
    (via :mod:`bench.magic3d_voxel_metrics`) and scored by Jaccard index.
    """
    from harnesscad.eval.bench.geometry.voxel_iou import voxel_iou, voxelize_points

    if resolution < 1:
        raise ValueError("resolution must be >= 1")
    if not a and not b:
        return 1.0
    combined = list(a) + list(b)
    lo = [min(p[i] for p in combined) for i in range(3)]
    hi = [max(p[i] for p in combined) for i in range(3)]
    extent = max(hi[i] - lo[i] for i in range(3))
    if extent <= 0.0:
        return 1.0
    spacing = extent / resolution
    va = voxelize_points(a, origin=lo, spacing=spacing)
    vb = voxelize_points(b, origin=lo, spacing=spacing)
    return voxel_iou(va, vb)


def align_point_clouds(source: Sequence[Vec3], target: Sequence[Vec3],
                       *, resolution: int = 32) -> dict:
    """Register ``source`` to ``target`` by principal axes; keep the best of four.

    Returns a dict with:
      * ``iou``            -- best voxel IoU over the four sign candidates,
      * ``rotation``       -- the winning 3x3 rotation (applied to the
                              size-normalised, centred source),
      * ``candidate_ious`` -- the IoU of every sign candidate (length 4),
      * ``aligned``        -- source points mapped fully into the TARGET frame
                              (normalise, rotate, rescale by target scale,
                              translate to target centroid) -- mirroring
                              CAD-Coder's final ``.scale(s_target).translate(c_target)``.

    Deterministic: enumeration order is fixed by :func:`reflection_signs`, ties
    are broken toward the earlier candidate.
    """
    cs, evs, vs = principal_frame(source)
    ct, evt, vt = principal_frame(target)
    ss = normalization_scale(evs)
    st = normalization_scale(evt)

    norm_src = normalize_points(source, cs, ss)
    norm_tgt = normalize_points(target, ct, st)

    rotations = candidate_rotations(vs, vt)
    scores: List[float] = []
    best_iou = -1.0
    best_rot = rotations[0]
    best_rotated: List[List[float]] = apply_rotation(norm_src, rotations[0])
    for rotation in rotations:
        rotated = apply_rotation(norm_src, rotation)
        iou = voxel_iou_score(rotated, norm_tgt, resolution=resolution)
        scores.append(iou)
        if iou > best_iou:
            best_iou = iou
            best_rot = rotation
            best_rotated = rotated

    aligned = [[coord * st + ct[i] for i, coord in enumerate(p)]
               for p in best_rotated]
    return {
        "iou": best_iou,
        "rotation": best_rot,
        "candidate_ious": tuple(scores),
        "aligned": aligned,
    }
