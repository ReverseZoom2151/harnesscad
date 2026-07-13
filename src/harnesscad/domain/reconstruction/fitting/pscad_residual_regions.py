"""PS-CAD geometric-guidance computation: local-geometry-difference detector.

Implements the deterministic ``p_ref`` computation from PS-CAD (Yang et al. 2024,
Sec. 4 "Geometric Guidance Computation").  Given the target point cloud
``p_full`` (a complete input CAD model) and a point cloud ``p_prev`` sampled from
the current partial reconstruction ``O_{t-1}``, the module identifies the *distinct
regions* between the two clouds -- the local geometry where the current
reconstruction differs from the target -- and clusters the high-residual points
into local regions that still need work.

The paper trains a segmentation network to predict binary masks ``M`` on
``p_full`` and ``p_prev``.  This module realises the same geometric definition
deterministically with a bidirectional nearest-neighbour residual test:

  * a point of ``p_full`` is *distinct* (mask 1) when its nearest neighbour in
    ``p_prev`` is farther than a threshold -- geometry not yet covered by the
    current reconstruction;
  * a point of ``p_prev`` is *distinct* (mask 1) when its nearest neighbour in
    ``p_full`` is farther than a threshold -- auxiliary construction geometry
    that will not remain in the final model.

``p_ref`` is the concatenation ``M_full(p_full) + M_prev(p_prev)`` (Eq. in
Sec. 4).  The learned network is intentionally out of scope; everything here is
closed-form and stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import dist


def _nearest_distance(point, cloud):
    """Euclidean distance from ``point`` to its nearest neighbour in ``cloud``."""
    best = None
    for other in cloud:
        d = dist(point, other)
        if best is None or d < best:
            best = d
    return best


def distinct_mask(source, reference, *, threshold):
    """Binary mask: 1 for source points whose nearest reference point is far.

    Mirrors ``M`` applied to a single cloud.  With an empty reference every
    source point is distinct (nothing covers it).
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    mask = []
    for point in source:
        nearest = _nearest_distance(point, reference)
        mask.append(1 if nearest is None or nearest > threshold else 0)
    return tuple(mask)


def _masked(cloud, mask):
    return [point for point, flag in zip(cloud, mask) if flag]


@dataclass(frozen=True)
class ResidualGuidance:
    """The two-sided residual guidance for one reconstruction step."""

    full_mask: tuple
    prev_mask: tuple
    missing: tuple      # M_full(p_full): target geometry not yet reconstructed
    auxiliary: tuple    # M_prev(p_prev): construction geometry to be removed
    threshold: float

    @property
    def p_ref(self):
        """``p_ref`` = concatenation of the two distinct regions (Sec. 4)."""
        return self.missing + self.auxiliary

    @property
    def missing_ratio(self):
        """Fraction of target points still uncovered -- a progress signal."""
        if not self.full_mask:
            return 0.0
        return sum(self.full_mask) / len(self.full_mask)


def compute_pref(p_full, p_prev, *, threshold):
    """Compute the PS-CAD residual guidance ``p_ref`` between target and state.

    ``p_full`` is the target cloud, ``p_prev`` the current-reconstruction cloud.
    Returns a :class:`ResidualGuidance` holding both masks and both distinct
    regions.  When ``p_prev`` is empty the whole target is "missing", which is
    exactly the first-iteration behaviour in the paper.
    """
    full_mask = distinct_mask(p_full, p_prev, threshold=threshold)
    prev_mask = distinct_mask(p_prev, p_full, threshold=threshold)
    missing = tuple(tuple(p) for p in _masked(p_full, full_mask))
    auxiliary = tuple(tuple(p) for p in _masked(p_prev, prev_mask))
    return ResidualGuidance(full_mask, prev_mask, missing, auxiliary, float(threshold))


@dataclass(frozen=True)
class ResidualRegion:
    """A spatially-connected cluster of residual (distinct) points."""

    points: tuple
    size: int = field(default=0)

    @property
    def centroid(self):
        n = len(self.points)
        if not n:
            return None
        dims = len(self.points[0])
        return tuple(sum(p[k] for p in self.points) / n for k in range(dims))

    def bounding_box(self):
        if not self.points:
            return None
        dims = len(self.points[0])
        lo = tuple(min(p[k] for p in self.points) for k in range(dims))
        hi = tuple(max(p[k] for p in self.points) for k in range(dims))
        return lo, hi


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def cluster_residual_regions(points, *, radius):
    """Group residual points into connected local regions by a radius graph.

    Two points join the same region when they are within ``radius`` of each
    other (single-linkage / connected components via union-find).  Returns the
    regions sorted by descending size then lexicographic centroid, so the order
    is fully deterministic.  These regions are the "parts that still need the
    most work" highlighted in Fig. 1 of the paper.
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    pts = [tuple(p) for p in points]
    n = len(pts)
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if dist(pts[i], pts[j]) <= radius:
                uf.union(i, j)
    buckets = {}
    for i in range(n):
        buckets.setdefault(uf.find(i), []).append(pts[i])
    regions = [ResidualRegion(tuple(group), len(group)) for group in buckets.values()]
    regions.sort(key=lambda r: (-r.size, r.centroid))
    return tuple(regions)


def highest_residual_region(guidance, *, radius, side="missing"):
    """Return the largest connected residual region (the focus for the next step).

    ``side`` selects which distinct region to cluster: ``"missing"`` (target
    geometry to add), ``"auxiliary"`` (state geometry to remove), or ``"both"``
    (the full ``p_ref``).  Returns ``None`` when there is no residual geometry.
    """
    if side == "missing":
        cloud = guidance.missing
    elif side == "auxiliary":
        cloud = guidance.auxiliary
    elif side == "both":
        cloud = guidance.p_ref
    else:
        raise ValueError("side must be 'missing', 'auxiliary' or 'both'")
    if not cloud:
        return None
    regions = cluster_residual_regions(cloud, radius=radius)
    return regions[0] if regions else None
