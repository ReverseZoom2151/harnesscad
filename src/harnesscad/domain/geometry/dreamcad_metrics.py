"""Point-supervision metrics for parametric surfaces (DreamCAD, Sec. 4/5).

DreamCAD supervises differentiably tessellated surfaces against target point
clouds with a Chamfer-distance loss and reports Chamfer / Hausdorff geometric
fidelity.  The forward metrics are deterministic, so they are implemented here
(the gradient-based optimisation of control points is out of scope).

Provided:

  * ``sample_rational_bezier`` -- turn a patch into a point cloud via the
    tessellator.
  * ``chamfer_distance`` -- symmetric mean nearest-neighbour distance.
  * ``one_sided_residual`` -- mean nearest distance from a surface's samples to
    a target cloud (the fitting residual driving the Chamfer loss).
  * ``hausdorff_distance`` -- symmetric worst-case nearest-neighbour distance.
  * ``surface_consistency`` -- Chamfer distance between two sampled surfaces,
    used to check patch-vs-primitive or patch-vs-patch agreement.

All functions take/return plain point tuples and are O(|A| * |B|) brute-force
nearest-neighbour, which is exact and deterministic.
"""

from __future__ import annotations

from math import sqrt

from harnesscad.domain.geometry.dreamcad_tessellation import tessellate_patch


def _sq_dist(a, b):
    return sum((a[d] - b[d]) ** 2 for d in range(len(a)))


def _nearest_sq(point, cloud):
    best = None
    for other in cloud:
        d = _sq_dist(point, other)
        if best is None or d < best:
            best = d
    return best


def sample_rational_bezier(grid, weights, resolution=8):
    """Sample a rational Bezier patch into a point cloud (vertex list)."""
    vertices, _ = tessellate_patch(grid, weights, resolution)
    return vertices


def one_sided_residual(source, target):
    """Mean over ``source`` of the distance to its nearest ``target`` point."""
    if not source:
        return 0.0
    if not target:
        raise ValueError("target cloud is empty")
    total = 0.0
    for point in source:
        total += sqrt(_nearest_sq(point, target))
    return total / len(source)


def chamfer_distance(cloud_a, cloud_b):
    """Symmetric Chamfer distance: mean A->B plus mean B->A nearest distance."""
    if not cloud_a or not cloud_b:
        raise ValueError("both clouds must be non-empty")
    return one_sided_residual(cloud_a, cloud_b) + one_sided_residual(cloud_b, cloud_a)


def hausdorff_distance(cloud_a, cloud_b):
    """Symmetric Hausdorff distance (max over both directed distances)."""
    if not cloud_a or not cloud_b:
        raise ValueError("both clouds must be non-empty")
    forward = max(sqrt(_nearest_sq(p, cloud_b)) for p in cloud_a)
    backward = max(sqrt(_nearest_sq(p, cloud_a)) for p in cloud_b)
    return max(forward, backward)


def surface_consistency(patch_a, patch_b, resolution=8):
    """Chamfer distance between two rational-Bezier patches' sampled clouds.

    Each ``patch`` is a ``(grid, weights)`` pair.  A value near zero means the
    patches describe (approximately) the same surface region.
    """
    cloud_a = sample_rational_bezier(patch_a[0], patch_a[1], resolution)
    cloud_b = sample_rational_bezier(patch_b[0], patch_b[1], resolution)
    return chamfer_distance(cloud_a, cloud_b)
