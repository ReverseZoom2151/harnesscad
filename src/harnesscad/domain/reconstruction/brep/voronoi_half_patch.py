"""Voronoi Half-Patch decomposition of a B-Rep face's parametric domain.

**BrepGPT** (Li et al., 2025) introduces the *Voronoi Half-Patch* as its geometric
representation of a face: each face's UV parametric domain is partitioned into regions
"that are geometrically closest to their respective boundary curves" (their Fig. 1).
Formally, given a face bounded by a set of boundary curves, every point of the face is
assigned to the boundary curve nearest to it -- a Voronoi diagram in parametric space
whose sites are the boundary loops rather than points. The result exposes, per boundary
curve, the sub-region of the face it "owns", which BrepGPT uses to couple geometry and
topology in a single stage.

This module builds that decomposition deterministically: given a face's UV domain
(sampled on a grid) and its boundary curves (each a polyline of UV points), it labels
each UV sample with the index of the nearest boundary curve and reports per-curve cell
areas and the boundary-partition adjacency (which curves' cells touch). It is a
geometry-only analyser -- a companion to
:mod:`harnesscad.domain.reconstruction.brep.chain_complex` that describes *how a face's
interior is apportioned among its boundary*, useful for face segmentation and
boundary-aware sampling. Stdlib only, deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = [
    "HalfPatch",
    "VoronoiHalfPatches",
    "nearest_curve",
    "decompose_face",
    "distance_to_polyline",
]


def _dist_point_segment(px: float, py: float,
                        ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    if seg2 < 1e-18:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / seg2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def distance_to_polyline(point: Sequence[float], polyline: Sequence[Sequence[float]]) -> float:
    """Minimum distance from a UV point to a boundary curve given as a polyline."""
    if not polyline:
        raise ValueError("polyline must have at least one vertex")
    if len(polyline) == 1:
        return math.hypot(point[0] - polyline[0][0], point[1] - polyline[0][1])
    best = math.inf
    for i in range(len(polyline) - 1):
        a, b = polyline[i], polyline[i + 1]
        d = _dist_point_segment(point[0], point[1], a[0], a[1], b[0], b[1])
        if d < best:
            best = d
    return best


def nearest_curve(point: Sequence[float],
                  curves: Sequence[Sequence[Sequence[float]]]) -> int:
    """Index of the boundary curve nearest to ``point`` (ties -> lowest index)."""
    if not curves:
        raise ValueError("at least one boundary curve is required")
    best_idx = 0
    best_d = math.inf
    for idx, curve in enumerate(curves):
        d = distance_to_polyline(point, curve)
        if d < best_d - 1e-12:
            best_d = d
            best_idx = idx
    return best_idx


@dataclass(frozen=True)
class HalfPatch:
    """The region of a face owned by one boundary curve."""

    curve_index: int
    sample_count: int
    cell_area: float


@dataclass(frozen=True)
class VoronoiHalfPatches:
    """Full Voronoi-Half-Patch decomposition of a face's UV domain."""

    patches: tuple[HalfPatch, ...]
    labels: tuple[tuple[int, ...], ...]   # per-row nearest-curve index of each UV sample
    adjacency: frozenset[tuple[int, int]]  # pairs of curves whose cells are grid-adjacent

    def dominant_curve(self) -> int:
        """Index of the boundary curve owning the largest region."""
        return max(self.patches, key=lambda p: (p.cell_area, -p.curve_index)).curve_index


def decompose_face(
    curves: Sequence[Sequence[Sequence[float]]],
    u_range: tuple[float, float] = (0.0, 1.0),
    v_range: tuple[float, float] = (0.0, 1.0),
    *,
    resolution: int = 16,
) -> VoronoiHalfPatches:
    """Partition a rectangular UV domain by nearest boundary curve (BrepGPT half-patch).

    Samples the domain on a ``resolution x resolution`` grid, labels each sample with
    its nearest curve, and accumulates per-curve cell area (each cell carries the area
    of one grid pixel). ``adjacency`` records which curves own horizontally/vertically
    neighbouring cells -- the boundary between adjacent half-patches.
    """
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    if not curves:
        raise ValueError("at least one boundary curve is required")
    u0, u1 = u_range
    v0, v1 = v_range
    du = (u1 - u0) / (resolution - 1)
    dv = (v1 - v0) / (resolution - 1)
    pixel_area = abs(du * dv)

    labels: list[tuple[int, ...]] = []
    counts: dict[int, int] = {}
    for j in range(resolution):
        v = v0 + j * dv
        row: list[int] = []
        for i in range(resolution):
            u = u0 + i * du
            idx = nearest_curve((u, v), curves)
            row.append(idx)
            counts[idx] = counts.get(idx, 0) + 1
        labels.append(tuple(row))

    adjacency: set[tuple[int, int]] = set()
    for j in range(resolution):
        for i in range(resolution):
            here = labels[j][i]
            if i + 1 < resolution:
                right = labels[j][i + 1]
                if right != here:
                    adjacency.add((min(here, right), max(here, right)))
            if j + 1 < resolution:
                down = labels[j + 1][i]
                if down != here:
                    adjacency.add((min(here, down), max(here, down)))

    patches = tuple(
        HalfPatch(curve_index=idx, sample_count=counts[idx],
                  cell_area=counts[idx] * pixel_area)
        for idx in sorted(counts)
    )
    return VoronoiHalfPatches(
        patches=patches,
        labels=tuple(labels),
        adjacency=frozenset(adjacency),
    )
