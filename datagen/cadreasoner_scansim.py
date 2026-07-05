"""CADReasoner scan-simulation: deterministic scan-defect point-cloud pipeline.

Paper: "CADReasoner - Iterative Program Editing for CAD Reverse Engineering",
Sec. 3.4 ("Scan-Simulation").

Public benchmarks are synthetic — inputs are clean point sets sampled from ideal
B-Reps — so a model trained/evaluated on them never sees the *scan defects* of a
real reverse-engineering pipeline (incomplete surface coverage, smoothed edges,
missing regions from occlusion and scanning markers). CADReasoner therefore
virtually scans the model and evaluates under the same conditions.

The paper's protocol: sample surface points (keeping normals), move a virtual
camera along a spherical trajectory, extract the *visible* points at several
viewpoints via a visibility criterion, merge them, reconstruct a mesh (Screened
Poisson), and finally punch random holes for missing regions.

What is deterministic and locally buildable is implemented here on the point
cloud directly (no external mesh kernel):

  * **spherical viewpoint trajectory** — a deterministic set of camera positions
    on a sphere around the (normalized) object;
  * **occlusion-based visibility** — at each viewpoint a spherical depth buffer
    keeps only the nearest point per angular bin, so self-occluded surfaces drop
    out (incomplete coverage), directly modelling the paper's "visible points are
    extracted at distinct viewpoints" step (a simplified, dependency-free stand-in
    for the Mehra et al. visibility criterion the paper cites);
  * **merge** of visible points across viewpoints;
  * **sensor noise** — seeded Gaussian jitter along the point normal (or isotropic
    if no normals), producing smoothed/perturbed edges;
  * **random holes** — remove all points within a radius of randomly chosen seed
    points, simulating missing regions from scanning markers.

Screened Poisson surface reconstruction is an external mesh-kernel step and is
*not* reimplemented; the result documents it as a downstream option.

Determinism: a single ``random.Random(seed)`` drives every stochastic choice;
math-only otherwise.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float, float]
Vec = Tuple[float, float, float]


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class ScanSimResult:
    """A simulated scan of an input point cloud.

    - ``points``           : the final scan-defect point cloud.
    - ``visible_counts``   : visible-point count per viewpoint (before merge).
    - ``coverage``         : fraction of the input retained after visibility+holes.
    - ``removed_occluded`` : points dropped as self-occluded.
    - ``removed_holes``    : points dropped by random hole punching.
    - ``viewpoints``       : the camera positions used.
    - ``note``             : scope note (Poisson reconstruction is external).
    """

    points: List[Point] = field(default_factory=list)
    normals: List[Vec] = field(default_factory=list)
    visible_counts: List[int] = field(default_factory=list)
    coverage: float = 0.0
    removed_occluded: int = 0
    removed_holes: int = 0
    viewpoints: List[Point] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "n_points": len(self.points),
            "visible_counts": list(self.visible_counts),
            "coverage": self.coverage,
            "removed_occluded": self.removed_occluded,
            "removed_holes": self.removed_holes,
            "n_viewpoints": len(self.viewpoints),
            "note": self.note,
        }


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _sub(a: Sequence[float], b: Sequence[float]) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _unit(v: Sequence[float]) -> Vec:
    n = _norm(v)
    if n == 0.0:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _centroid(points: Sequence[Point]) -> Point:
    n = len(points)
    return (sum(p[0] for p in points) / n,
            sum(p[1] for p in points) / n,
            sum(p[2] for p in points) / n)


def spherical_viewpoints(radius: float, count: int) -> List[Point]:
    """Deterministic camera positions on a sphere (Fibonacci-sphere trajectory).

    Evenly distributes ``count`` points on a sphere of the given ``radius`` — a
    reproducible stand-in for the paper's spherical camera trajectory.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    pts: List[Point] = []
    golden = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
    for i in range(count):
        # y from 1 to -1
        y = 1.0 - (2.0 * i + 1.0) / count
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * i
        pts.append((radius * math.cos(theta) * r,
                    radius * y,
                    radius * math.sin(theta) * r))
    return pts


# --------------------------------------------------------------------------- #
# Occlusion-based visibility (spherical depth buffer)
# --------------------------------------------------------------------------- #
def visible_from(
    points: Sequence[Point],
    viewpoint: Point,
    *,
    angular_bins: int = 64,
) -> List[int]:
    """Indices of points visible from ``viewpoint`` via a spherical depth buffer.

    Each point is mapped to an (azimuth, elevation) angular bin as seen from the
    viewpoint; within a bin only the point nearest the viewpoint survives, so
    farther points along the same ray are occluded. ``angular_bins`` controls the
    angular resolution (higher = fewer points culled). Deterministic.
    """
    if angular_bins < 1:
        raise ValueError("angular_bins must be >= 1")
    best: dict = {}  # (az_bin, el_bin) -> (distance, index)
    for idx, p in enumerate(points):
        d = _sub(p, viewpoint)
        r = _norm(d)
        if r == 0.0:
            # Point coincides with the camera; always keep it.
            best[(idx, "self")] = (0.0, idx)
            continue
        az = math.atan2(d[1], d[0])            # -pi..pi
        el = math.asin(max(-1.0, min(1.0, d[2] / r)))  # -pi/2..pi/2
        az_bin = int((az + math.pi) / (2 * math.pi) * angular_bins)
        el_bin = int((el + math.pi / 2) / math.pi * angular_bins)
        az_bin = min(az_bin, angular_bins - 1)
        el_bin = min(el_bin, angular_bins - 1)
        key = (az_bin, el_bin)
        prev = best.get(key)
        if prev is None or r < prev[0]:
            best[key] = (r, idx)
    return sorted(v[1] for v in best.values())


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #
def simulate_scan(
    points: Sequence[Point],
    *,
    normals: Optional[Sequence[Vec]] = None,
    n_viewpoints: int = 5,
    angular_bins: int = 64,
    camera_radius: float = 3.0,
    noise_sigma: float = 0.01,
    n_holes: int = 2,
    hole_radius: float = 0.15,
    seed: int = 0,
) -> ScanSimResult:
    """Produce a scan-defect point cloud from a clean input point cloud.

    Stages (Sec. 3.4): spherical viewpoints -> per-view occlusion visibility ->
    merge -> Gaussian sensor noise -> random hole punching.

    Args:
        points: input surface points (assumed roughly centred; the camera radius
            is measured from their centroid).
        normals: optional per-point unit normals; noise is applied along them when
            present (isotropic otherwise).
        n_viewpoints: number of camera positions (5 in the paper).
        angular_bins: angular resolution of the visibility depth buffer.
        camera_radius: camera distance as a multiple of the object's bounding
            radius.
        noise_sigma: standard deviation of the sensor noise (object-space units,
            assuming a normalized cloud).
        n_holes / hole_radius: number and radius of random missing regions.
        seed: master seed for all randomness.

    Returns:
        ``ScanSimResult``.
    """
    pts = [tuple(map(float, p)) for p in points]
    if len(pts) < 2:
        raise ValueError("need at least 2 input points to simulate a scan")
    nrm = None
    if normals is not None:
        if len(normals) != len(pts):
            raise ValueError("normals must align with points")
        nrm = [tuple(map(float, v)) for v in normals]

    rng = random.Random(seed)
    center = _centroid(pts)
    bound_r = max(_norm(_sub(p, center)) for p in pts) or 1.0
    cam_r = camera_radius * bound_r
    viewpoints = [tuple(c + v for c, v in zip(center, vp))
                  for vp in spherical_viewpoints(cam_r, n_viewpoints)]

    # Per-view visibility, then merge (union of visible indices).
    visible_counts: List[int] = []
    visible_idx: set = set()
    for vp in viewpoints:
        vis = visible_from(pts, vp, angular_bins=angular_bins)
        visible_counts.append(len(vis))
        visible_idx.update(vis)
    removed_occluded = len(pts) - len(visible_idx)
    kept = sorted(visible_idx)

    # Sensor noise (along normal if available, else isotropic Gaussian).
    scan_pts: List[Point] = []
    scan_nrm: List[Vec] = []
    for i in kept:
        p = pts[i]
        if noise_sigma > 0.0:
            if nrm is not None:
                mag = rng.gauss(0.0, noise_sigma)
                d = _unit(nrm[i])
                p = (p[0] + d[0] * mag, p[1] + d[1] * mag, p[2] + d[2] * mag)
            else:
                p = (p[0] + rng.gauss(0.0, noise_sigma),
                     p[1] + rng.gauss(0.0, noise_sigma),
                     p[2] + rng.gauss(0.0, noise_sigma))
        scan_pts.append(p)
        if nrm is not None:
            scan_nrm.append(nrm[i])

    # Random holes: remove all points within hole_radius of chosen seeds.
    removed_holes = 0
    if n_holes > 0 and hole_radius > 0.0 and scan_pts:
        hole_centers = [scan_pts[rng.randrange(len(scan_pts))]
                        for _ in range(n_holes)]
        surviving_pts: List[Point] = []
        surviving_nrm: List[Vec] = []
        for j, p in enumerate(scan_pts):
            if any(_norm(_sub(p, hc)) <= hole_radius for hc in hole_centers):
                removed_holes += 1
                continue
            surviving_pts.append(p)
            if scan_nrm:
                surviving_nrm.append(scan_nrm[j])
        scan_pts, scan_nrm = surviving_pts, surviving_nrm

    coverage = len(scan_pts) / len(pts) if pts else 0.0
    return ScanSimResult(
        points=scan_pts,
        normals=scan_nrm,
        visible_counts=visible_counts,
        coverage=coverage,
        removed_occluded=removed_occluded,
        removed_holes=removed_holes,
        viewpoints=viewpoints,
        note=("point-cloud scan-defect simulation (occlusion + noise + holes); "
              "Screened Poisson surface reconstruction is an external downstream "
              "step and is not reimplemented here"),
    )
