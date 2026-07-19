"""Geometry-discrepancy encoding: nearest-surface cross-shape offsets.

Where a scalar Chamfer distance (already computed elsewhere in the benchmark
layer) gives only *how far* a candidate is from the target, this encoding feeds
the editor a directional per-point error field: for every sampled point it
attaches the vector toward the nearest location on the *opposite* shape.
Magnitude measures local discrepancy; direction indicates how the prediction
should move.

Given a target point set ``P_T`` (sampled on the target ``T``) and a render point
set ``P_S`` (sampled on the previous render ``S = R(C_{t-1})``) we form, per point:

    dp_i = NN_S(p_i) - p_i      for p_i in P_T   (target -> nearest render point)
    dq_j = NN_T(q_j) - q_j      for q_j in P_S   (render -> nearest target point)

so each point becomes ``(x, y, z, dx, dy, dz)``. We keep the ``k`` *most
discrepant* points from each side (largest distance to the other shape) -- a
farthest-point selection that concentrates supervision on the regions that still
disagree.

The t=1 case has no render yet: we adopt a null prediction at the origin. Target
offsets point back toward the origin (``dp_i = -p_i``) and the render cloud is a
bag of origin points paired to a permuted target sample (``dq_j = p_{pi(j)}``),
yielding informative, non-identical offsets without special tokens.

Everything here is stdlib-only and deterministic (a ``random.Random(seed)`` drives
the t=1 permutation; nothing else is stochastic).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from math import dist, sqrt
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, ...]
OffsetPoint = Tuple[float, ...]  # (x, y, z, dx, dy, dz)


# --------------------------------------------------------------------------- #
# Nearest-neighbour primitives
# --------------------------------------------------------------------------- #
def nearest(query: Point, cloud: Sequence[Point]) -> Tuple[int, float]:
    """Index of, and distance to, the nearest point of ``cloud`` (brute force).

    Ties resolve to the lowest index, keeping the result deterministic. Raises
    ``ValueError`` on an empty cloud.
    """
    if not cloud:
        raise ValueError("cannot query nearest point of an empty cloud")
    best_i, best_d = 0, dist(query, cloud[0])
    for i in range(1, len(cloud)):
        d = dist(query, cloud[i])
        if d < best_d:
            best_i, best_d = i, d
    return best_i, best_d


def _offset(a: Point, b: Point) -> Tuple[float, ...]:
    """Vector b - a."""
    return tuple(bi - ai for ai, bi in zip(a, b))


# --------------------------------------------------------------------------- #
# Encoding result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DiscrepancyEncoding:
    """A directional error field the editor consumes to propose the next edit.

    - ``target_offsets`` / ``render_offsets`` : the selected ``(x,y,z,dx,dy,dz)``
      points from each side (already trimmed to the ``k`` most discrepant).
    - ``max_discrepancy`` / ``mean_discrepancy`` : summary magnitudes over the
      *selected* points (the signal the loop watches for convergence).
    - ``symmetric_discrepancy`` : mean of the two directed means -- a scalar the
      edit loop can use as a stopping/selection metric without a separate CD call.
    - ``t1`` : whether this is the null-prediction t=1 encoding.
    """

    target_offsets: Tuple[OffsetPoint, ...] = ()
    render_offsets: Tuple[OffsetPoint, ...] = ()
    max_discrepancy: float = 0.0
    mean_discrepancy: float = 0.0
    symmetric_discrepancy: float = 0.0
    t1: bool = False
    note: str = ""

    @property
    def points(self) -> Tuple[OffsetPoint, ...]:
        """The union set the set-encoder consumes: {(p,dp)} u {(q,dq)}."""
        return self.target_offsets + self.render_offsets

    def to_dict(self) -> dict:
        return {
            "target_offsets": [list(p) for p in self.target_offsets],
            "render_offsets": [list(p) for p in self.render_offsets],
            "max_discrepancy": self.max_discrepancy,
            "mean_discrepancy": self.mean_discrepancy,
            "symmetric_discrepancy": self.symmetric_discrepancy,
            "t1": self.t1,
            "note": self.note,
        }


# --------------------------------------------------------------------------- #
# Directed offset field with farthest-point selection
# --------------------------------------------------------------------------- #
def _directed_offsets(
    source: Sequence[Point], other: Sequence[Point], k: int
) -> Tuple[List[OffsetPoint], List[float]]:
    """For each point in ``source`` attach the vector to its nearest ``other``
    point, then keep the ``k`` points with the largest such distance.

    Returns the selected offset points and their distances (parallel lists),
    both ordered by descending discrepancy then ascending original index (stable,
    deterministic).
    """
    scored = []
    for idx, p in enumerate(source):
        j, d = nearest(p, other)
        scored.append((d, idx, tuple(p) + _offset(p, other[j])))
    # Largest distance first; ties broken by original index for determinism.
    scored.sort(key=lambda t: (-t[0], t[1]))
    if k > 0:
        scored = scored[:k]
    offsets = [t[2] for t in scored]
    dists = [t[0] for t in scored]
    return offsets, dists


def encode_discrepancy(
    target_points: Sequence[Point],
    render_points: Sequence[Point],
    *,
    k: int = 128,
) -> DiscrepancyEncoding:
    """Build the cross-shape offset field between a target and a render cloud.

    ``k`` is the per-side farthest-point budget (128 by default); ``k <= 0``
    keeps every point. Both clouds must be non-empty.
    """
    if not target_points or not render_points:
        raise ValueError("both target_points and render_points must be non-empty")
    t_off, t_d = _directed_offsets(target_points, render_points, k)
    r_off, r_d = _directed_offsets(render_points, target_points, k)
    all_d = t_d + r_d
    max_d = max(all_d) if all_d else 0.0
    # Symmetric discrepancy: mean of the two directed means (Chamfer-like, but
    # over the *selected* farthest points, matching what the editor actually sees).
    t_mean = sum(t_d) / len(t_d) if t_d else 0.0
    r_mean = sum(r_d) / len(r_d) if r_d else 0.0
    mean_d = sum(all_d) / len(all_d) if all_d else 0.0
    return DiscrepancyEncoding(
        target_offsets=tuple(t_off),
        render_offsets=tuple(r_off),
        max_discrepancy=max_d,
        mean_discrepancy=mean_d,
        symmetric_discrepancy=(t_mean + r_mean) / 2.0,
        t1=False,
        note=f"cross-shape offsets: {len(t_off)} target + {len(r_off)} render points",
    )


def encode_null_init(
    target_points: Sequence[Point],
    *,
    k: int = 128,
    seed: int = 0,
) -> DiscrepancyEncoding:
    """The t=1 null-prediction encoding.

    No render exists, so the prediction is a null cloud at the origin. Target
    offsets point back to the origin (``dp_i = -p_i``); the render cloud is a bag
    of origin points, each paired to a *distinct* permuted target sample so its
    offset is ``dq_j = p_{pi(j)}`` -- informative and non-identical without special
    tokens. The permutation is deterministic under ``seed``.
    """
    if not target_points:
        raise ValueError("target_points must be non-empty")
    pts = [tuple(map(float, p)) for p in target_points]
    dims = len(pts[0])
    origin = (0.0,) * dims

    # Target side: offset toward the origin, distance = |p|. Keep k farthest.
    t_scored = []
    for idx, p in enumerate(pts):
        d = dist(p, origin)
        t_scored.append((d, idx, tuple(p) + tuple(-c for c in p)))
    t_scored.sort(key=lambda t: (-t[0], t[1]))
    if k > 0:
        t_scored = t_scored[:k]
    t_off = [t[2] for t in t_scored]
    t_d = [t[0] for t in t_scored]

    # Render side: q_j = 0, dq_j = p_{pi(j)} under a deterministic permutation.
    perm = list(range(len(pts)))
    random.Random(seed).shuffle(perm)
    r_scored = []
    for j, pi_j in enumerate(perm):
        tgt = pts[pi_j]
        d = dist(origin, tgt)
        r_scored.append((d, j, origin + tuple(tgt)))
    r_scored.sort(key=lambda t: (-t[0], t[1]))
    if k > 0:
        r_scored = r_scored[:k]
    r_off = [t[2] for t in r_scored]
    r_d = [t[0] for t in r_scored]

    all_d = t_d + r_d
    t_mean = sum(t_d) / len(t_d) if t_d else 0.0
    r_mean = sum(r_d) / len(r_d) if r_d else 0.0
    return DiscrepancyEncoding(
        target_offsets=tuple(t_off),
        render_offsets=tuple(r_off),
        max_discrepancy=max(all_d) if all_d else 0.0,
        mean_discrepancy=sum(all_d) / len(all_d) if all_d else 0.0,
        symmetric_discrepancy=(t_mean + r_mean) / 2.0,
        t1=True,
        note="t=1 null-prediction encoding (prediction at origin)",
    )
