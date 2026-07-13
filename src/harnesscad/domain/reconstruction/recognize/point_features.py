"""Deterministic geometric point-wise features for CAD point-cloud part segmentation.

Few-shot CAM/CAD part segmentation (Wang et al.) feeds each point an attribute
vector ``f = (x, y, z, X, Y, Z)`` -- coordinates plus a unit normal -- and lets a
learned DGCNN backbone aggregate local edge features ``pj - pi`` over the k
nearest neighbours (Eq. 4). The *training* of that backbone is external, but the
hand-crafted geometric descriptors it is meant to approximate are fully
deterministic and buildable: local-neighbourhood PCA (covariance) descriptors --
linearity, planarity, scattering, anisotropy and surface-variation curvature --
plus a max-pooled edge feature in the spirit of DGCNN's EdgeConv.

These descriptors are exactly what separates the paper's five CAD part classes
(Hole, Pocket, Chamfer, Fillet, Plane): planes are planar/low-curvature, edges
of chamfers are linear/high-curvature, and so on. Everything here is closed-form
and stdlib-only.
"""

from __future__ import annotations

import math


def knn_indices(points, k):
    """For each point return the indices of its ``k`` nearest neighbours.

    The query point itself is excluded. Ties are broken by point index so the
    ordering is fully deterministic. ``k`` is clamped to ``len(points) - 1``.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    pts = [tuple(map(float, p)) for p in points]
    n = len(pts)
    limit = min(k, n - 1) if n else 0
    result = []
    for i, pi in enumerate(pts):
        order = sorted(
            (j for j in range(n) if j != i),
            key=lambda j: (_sqdist(pi, pts[j]), j),
        )
        result.append(tuple(order[:limit]))
    return tuple(result)


def _sqdist(a, b):
    return sum((a[d] - b[d]) ** 2 for d in range(len(a)))


def _covariance(points, indices, center):
    """3x3 covariance of the neighbourhood (center + neighbours)."""
    members = [center] + [points[j] for j in indices]
    m = len(members)
    mean = tuple(sum(p[d] for p in members) / m for d in range(3))
    cov = [[0.0, 0.0, 0.0] for _ in range(3)]
    for p in members:
        diff = [p[d] - mean[d] for d in range(3)]
        for a in range(3):
            for b in range(3):
                cov[a][b] += diff[a] * diff[b]
    for a in range(3):
        for b in range(3):
            cov[a][b] /= m
    return cov


def symmetric_eigenvalues(cov):
    """Eigenvalues of a symmetric 3x3 matrix, descending.

    Closed-form solution (Smith 1961) -- no iterative solver required.
    """
    a = cov
    p1 = a[0][1] ** 2 + a[0][2] ** 2 + a[1][2] ** 2
    q = (a[0][0] + a[1][1] + a[2][2]) / 3.0
    if p1 <= 1e-18:
        # Already diagonal.
        return tuple(sorted((a[0][0], a[1][1], a[2][2]), reverse=True))
    p2 = ((a[0][0] - q) ** 2 + (a[1][1] - q) ** 2 + (a[2][2] - q) ** 2
          + 2.0 * p1)
    p = math.sqrt(p2 / 6.0)
    b = [[(a[i][j] - (q if i == j else 0.0)) / p for j in range(3)]
         for i in range(3)]
    detb = (b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1])
            - b[0][1] * (b[1][0] * b[2][2] - b[1][2] * b[2][0])
            + b[0][2] * (b[1][0] * b[2][1] - b[1][1] * b[2][0]))
    r = max(-1.0, min(1.0, detb / 2.0))
    phi = math.acos(r) / 3.0
    eig1 = q + 2.0 * p * math.cos(phi)
    eig3 = q + 2.0 * p * math.cos(phi + 2.0 * math.pi / 3.0)
    eig2 = 3.0 * q - eig1 - eig3
    return tuple(sorted((eig1, eig2, eig3), reverse=True))


def covariance_descriptors(eigenvalues, *, epsilon=1e-12):
    """Dimensionality/curvature descriptors from descending eigenvalues.

    Returns a dict with the classic PCA neighbourhood shape features
    (Weinmann/West conventions):

      linearity     = (l1 - l2) / l1
      planarity     = (l2 - l3) / l1
      scattering    = l3 / l1                (a.k.a. sphericity)
      anisotropy    = (l1 - l3) / l1
      curvature     = l3 / (l1 + l2 + l3)    (surface variation)
    """
    l1, l2, l3 = eigenvalues
    l1 = max(l1, 0.0)
    l2 = max(l2, 0.0)
    l3 = max(l3, 0.0)
    denom = l1 if l1 > epsilon else epsilon
    total = l1 + l2 + l3
    total = total if total > epsilon else epsilon
    return {
        "linearity": (l1 - l2) / denom,
        "planarity": (l2 - l3) / denom,
        "scattering": l3 / denom,
        "anisotropy": (l1 - l3) / denom,
        "curvature": l3 / total,
    }


def edge_feature(points, i, indices):
    """DGCNN-style max-pooled local edge feature ``max_j (pj - pi)``.

    Deterministic surrogate for EdgeConv's learned aggregation: per axis the
    maximum signed offset to a neighbour, plus the maximum neighbour distance.
    """
    pi = points[i]
    if not indices:
        return (0.0, 0.0, 0.0, 0.0)
    max_off = [-math.inf, -math.inf, -math.inf]
    max_dist = 0.0
    for j in indices:
        pj = points[j]
        for d in range(3):
            off = pj[d] - pi[d]
            if off > max_off[d]:
                max_off[d] = off
        max_dist = max(max_dist, math.sqrt(_sqdist(pi, pj)))
    return (max_off[0], max_off[1], max_off[2], max_dist)


def point_features(points, k=8):
    """Per-point deterministic geometric descriptor vectors.

    Each row is::

        (linearity, planarity, scattering, anisotropy, curvature,
         edge_dx, edge_dy, edge_dz, edge_dmax)

    a 9-dim vector combining local PCA shape descriptors with a DGCNN-style
    max-pooled edge feature. This is the descriptor a few-shot segmenter's
    prototypes and label-propagation graph are built over.
    """
    pts = [tuple(map(float, p))[:3] for p in points]
    neighbours = knn_indices(pts, k)
    rows = []
    for i, idx in enumerate(neighbours):
        cov = _covariance(pts, idx, pts[i])
        eig = symmetric_eigenvalues(cov)
        desc = covariance_descriptors(eig)
        ef = edge_feature(pts, i, idx)
        rows.append((
            desc["linearity"], desc["planarity"], desc["scattering"],
            desc["anisotropy"], desc["curvature"],
            ef[0], ef[1], ef[2], ef[3],
        ))
    return tuple(rows)
