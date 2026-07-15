"""Azimuth/depression viewing projection and GECM derivation (GeoDiff-SAR II).

Mined from *GeoDiff-SAR II: 3D Model-Guided SAR Image Generation with Explicit
Control of Key Imaging Parameters*. The paper's controllable-generation machinery
(ControlNet + LoRA + FLUX) is a trained diffusion pipeline, but the *conditioning
map* it renders from a 3D CAD model under specified imaging parameters is a fully
deterministic geometric construction -- the paper calls it the
Geometric-Electromagnetic Conditioning Map (GECM). This module ports that
deterministic core:

*   :func:`viewing_direction` -- the line-of-sight unit vector for an ``(azimuth,
    depression)`` pair (the "specified SAR parameters" of the forward engine).
*   :func:`project_points` -- orthographic projection of a 3D model onto the image
    plane perpendicular to the line of sight (the "2D pose" render).
*   :func:`principal_axis_2d` / :func:`pose_skeleton` -- the target *pose skeleton*
    via a closed-form 2D principal-axis (PCA) estimate plus axis-extreme keypoints.
*   :func:`dbscan` -- the density clustering used for *strong scatterer
    localization* (paper: "Mask-Gated Local Maxima -> DBSCAN Clustering").
*   :func:`scattering_centers` / :func:`build_gecm` -- dominant scattering-center
    localisation and the assembled GECM.

Everything is stdlib-only and deterministic: the same points and parameters always
yield the same GECM (DBSCAN visits points in index order and assigns the first
cluster id it reaches).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "viewing_direction",
    "camera_basis",
    "project_points",
    "principal_axis_2d",
    "pose_skeleton",
    "dbscan",
    "scattering_centers",
    "build_gecm",
    "GECM",
]

Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]


def viewing_direction(azimuth_deg: float, depression_deg: float) -> Vec3:
    """Unit line-of-sight vector for an ``(azimuth, depression)`` pair.

    Azimuth is measured counter-clockwise about ``+z`` from ``+x``; depression is
    the downward tilt from the horizontal (a sensor looking down at the target).
    The returned vector points *from the sensor toward the target*.
    """
    az = math.radians(azimuth_deg)
    dep = math.radians(depression_deg)
    cd = math.cos(dep)
    # sensor is above and off to the side; it looks toward the origin.
    return (-cd * math.cos(az), -cd * math.sin(az), -math.sin(dep))


def _normalise(v: Vec3) -> Vec3:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n == 0.0:
        raise ValueError("cannot normalise a zero vector")
    return (v[0] / n, v[1] / n, v[2] / n)


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def camera_basis(azimuth_deg: float, depression_deg: float) -> Tuple[Vec3, Vec3, Vec3]:
    """Right-handed ``(right, up, forward)`` image-plane basis for the view.

    ``forward`` is the line of sight; ``right`` and ``up`` span the image plane.
    """
    forward = _normalise(viewing_direction(azimuth_deg, depression_deg))
    world_up: Vec3 = (0.0, 0.0, 1.0)
    if abs(forward[2]) > 0.999:  # looking straight down: pick a stable reference
        world_up = (0.0, 1.0, 0.0)
    right = _normalise(_cross(forward, world_up))
    up = _normalise(_cross(right, forward))
    return right, up, forward


def project_points(
    points: Sequence[Vec3], azimuth_deg: float, depression_deg: float
) -> List[Vec2]:
    """Orthographic image-plane coordinates of a 3D model under the given view."""
    right, up, _ = camera_basis(azimuth_deg, depression_deg)
    out: List[Vec2] = []
    for p in points:
        u = p[0] * right[0] + p[1] * right[1] + p[2] * right[2]
        v = p[0] * up[0] + p[1] * up[1] + p[2] * up[2]
        out.append((u, v))
    return out


def principal_axis_2d(points: Sequence[Vec2]) -> Tuple[Vec2, float, Tuple[float, float]]:
    """Centroid, principal-axis angle (radians) and the two eigenvalues.

    Closed-form eigen-decomposition of the 2x2 covariance. The angle is that of
    the dominant eigenvector; eigenvalues are returned ``(major, minor)``.
    """
    if not points:
        raise ValueError("need at least one point")
    n = len(points)
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    sxx = syy = sxy = 0.0
    for x, y in points:
        dx, dy = x - cx, y - cy
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    sxx /= n
    syy /= n
    sxy /= n
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = max(0.0, tr * tr / 4.0 - det)
    root = math.sqrt(disc)
    lam1 = tr / 2.0 + root  # major
    lam2 = tr / 2.0 - root  # minor
    if abs(sxy) < 1e-12:
        angle = 0.0 if sxx >= syy else math.pi / 2.0
    else:
        angle = math.atan2(lam1 - sxx, sxy)
    return (cx, cy), angle, (lam1, lam2)


def pose_skeleton(points: Sequence[Vec2]) -> Dict[str, object]:
    """Target pose skeleton: principal axis endpoints + axis-extreme keypoints.

    The skeleton is the projection of ``points`` onto the principal axis, capped at
    the min/max extents. Keypoints are the two axis extremes (nose/tail of the
    target) plus the centroid.
    """
    (cx, cy), angle, eig = principal_axis_2d(points)
    ax = (math.cos(angle), math.sin(angle))
    ts = [(x - cx) * ax[0] + (y - cy) * ax[1] for x, y in points]
    tmin, tmax = min(ts), max(ts)
    p_min = (cx + tmin * ax[0], cy + tmin * ax[1])
    p_max = (cx + tmax * ax[0], cy + tmax * ax[1])
    return {
        "centroid": (cx, cy),
        "axis_angle": angle,
        "endpoints": (p_min, p_max),
        "keypoints": [p_min, (cx, cy), p_max],
        "eigenvalues": eig,
        "length": tmax - tmin,
    }


def dbscan(
    points: Sequence[Vec2], eps: float, min_samples: int
) -> List[int]:
    """Deterministic DBSCAN. Returns a cluster label per point (``-1`` = noise).

    Points are visited in index order, so cluster ids are assigned deterministically.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")
    n = len(points)
    labels = [-2] * n  # -2 = unvisited, -1 = noise, >=0 = cluster id
    eps2 = eps * eps

    def region(i: int) -> List[int]:
        xi, yi = points[i]
        out = []
        for j in range(n):
            dx = points[j][0] - xi
            dy = points[j][1] - yi
            if dx * dx + dy * dy <= eps2:
                out.append(j)
        return out

    cluster = 0
    for i in range(n):
        if labels[i] != -2:
            continue
        neigh = region(i)
        if len(neigh) < min_samples:
            labels[i] = -1
            continue
        labels[i] = cluster
        seeds = [j for j in neigh if j != i]
        k = 0
        while k < len(seeds):
            j = seeds[k]
            k += 1
            if labels[j] == -1:
                labels[j] = cluster
            if labels[j] != -2:
                continue
            labels[j] = cluster
            jneigh = region(j)
            if len(jneigh) >= min_samples:
                for q in jneigh:
                    if q not in seeds:
                        seeds.append(q)
        cluster += 1
    return labels


def scattering_centers(
    points: Sequence[Vec2],
    intensities: Sequence[float],
    intensity_threshold: float,
    eps: float,
    min_samples: int,
) -> List[Vec2]:
    """Dominant scattering centres: mask by intensity, DBSCAN, take centroids.

    Mirrors the paper's "Mask-Gated Local Maxima -> DBSCAN Clustering" stage.
    Returns one centroid per cluster, ordered by cluster id.
    """
    if len(points) != len(intensities):
        raise ValueError("points and intensities must be the same length")
    kept = [(p, idx) for idx, p in enumerate(points)
            if intensities[idx] >= intensity_threshold]
    if not kept:
        return []
    pts = [p for p, _ in kept]
    labels = dbscan(pts, eps, min_samples)
    groups: Dict[int, List[Vec2]] = {}
    for lab, p in zip(labels, pts):
        if lab < 0:
            continue
        groups.setdefault(lab, []).append(p)
    centres: List[Vec2] = []
    for lab in sorted(groups):
        g = groups[lab]
        cx = sum(q[0] for q in g) / len(g)
        cy = sum(q[1] for q in g) / len(g)
        centres.append((cx, cy))
    return centres


@dataclass(frozen=True)
class GECM:
    """A Geometric-Electromagnetic Conditioning Map (deterministic render)."""

    azimuth_deg: float
    depression_deg: float
    polarization: str
    pose: Dict[str, object]
    scatterers: Tuple[Vec2, ...]


def build_gecm(
    points: Sequence[Vec3],
    azimuth_deg: float,
    depression_deg: float,
    polarization: str = "HH",
    intensities: Optional[Sequence[float]] = None,
    intensity_threshold: float = 0.5,
    eps: float = 0.1,
    min_samples: int = 2,
) -> GECM:
    """Render a GECM from a 3D model under specified imaging parameters.

    ``intensities`` are optional per-vertex electromagnetic responses; when absent
    every projected point is treated as a unit-intensity scatterer candidate.
    """
    proj = project_points(points, azimuth_deg, depression_deg)
    pose = pose_skeleton(proj)
    inten = list(intensities) if intensities is not None else [1.0] * len(proj)
    centres = scattering_centers(proj, inten, intensity_threshold, eps, min_samples)
    return GECM(
        azimuth_deg=azimuth_deg,
        depression_deg=depression_deg,
        polarization=polarization,
        pose=pose,
        scatterers=tuple(centres),
    )
