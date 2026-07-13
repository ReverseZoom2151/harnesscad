"""FlexCAD appendix-specific deterministic procedures (ICLR 2025, App. A.1 / A.4).

This module implements two small, self-contained procedures that are unique to
the *appendix* of the FlexCAD paper and sit outside the core FlexCAD pipeline
(hierarchical CAD->text serialization, hierarchy-aware masking, masked-infill
pair construction and controllability metrics, all covered elsewhere):

1. Circle-representation variants (App. A.1 / Table 4).
   FlexCAD ablates three ways of encoding a circle inside a sketch loop:

     * ``center_radius`` -- centre coordinate plus a scalar radius,
     * ``diameter``      -- two uniformly opposed points on the circumference
                            that together define a diameter,
     * ``four_points``   -- four points uniformly distributed along the
                            circumference (the representation FlexCAD adopts,
                            reported as giving a slight edge in Table 4).

   :func:`encode_circle` / :func:`decode_circle` give an exact round-trip for
   each variant, and :func:`circle_repr_token_count` reports how many scalar
   tokens each variant costs -- the quantity the ablation trades off.

2. PV-constrained sampling-hyperparameter selection (App. A.4 / Table 6).
   FlexCAD observes a monotone trade-off: as the sampling temperature ``tau``
   or ``top_p`` rise, the diversity/quality metrics (COV, MMD, JSD, Novel,
   Unique) improve while Prediction Validity (PV) declines. The authors state
   they "made a trade-off by selecting the values of tau and top_p to guarantee
   that the PV value remains above 90%." :func:`select_sampling_config`
   reproduces that rule deterministically: among candidate configurations whose
   PV meets a floor (default 0.90), it returns the one maximizing a composite
   diversity objective, with fully specified tie-breaking.

Everything is stdlib-only, pure Python and deterministic.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]

# --------------------------------------------------------------------------- #
# 1. Circle representation variants (App. A.1 / Table 4)
# --------------------------------------------------------------------------- #
CIRCLE_VARIANTS = ("center_radius", "diameter", "four_points")

# Number of scalar values each variant serializes (Table 4 trade-off).
_CIRCLE_TOKENS: Dict[str, int] = {
    "center_radius": 3,   # cx, cy, r
    "diameter": 4,        # two points (x0, y0, x1, y1)
    "four_points": 8,     # four points on the circumference
}


def circle_repr_token_count(variant: str) -> int:
    """Return the number of scalar tokens used by a circle ``variant``."""
    if variant not in _CIRCLE_TOKENS:
        raise ValueError("unknown circle variant: %r" % (variant,))
    return _CIRCLE_TOKENS[variant]


def encode_circle(center: Point, radius: float, variant: str) -> List[float]:
    """Encode a circle (``center``, ``radius``) into a flat scalar list.

    The point ordering follows FlexCAD's convention of walking the
    circumference counter-clockwise starting at angle 0 (the +x direction):
    for ``diameter`` the two points are at 0 and 180 degrees; for
    ``four_points`` they are at 0, 90, 180 and 270 degrees.
    """
    if radius < 0.0:
        raise ValueError("radius must be non-negative")
    cx, cy = center
    if variant == "center_radius":
        return [cx, cy, radius]
    if variant == "diameter":
        return [cx + radius, cy, cx - radius, cy]
    if variant == "four_points":
        return [
            cx + radius, cy,
            cx, cy + radius,
            cx - radius, cy,
            cx, cy - radius,
        ]
    raise ValueError("unknown circle variant: %r" % (variant,))


def decode_circle(values: Sequence[float], variant: str) -> Tuple[Point, float]:
    """Invert :func:`encode_circle`, returning ``(center, radius)``.

    For the multi-point variants the centre is the centroid of the supplied
    points and the radius is the mean distance from that centroid, matching the
    uniform-distribution assumption the paper makes about the points.
    """
    vals = list(values)
    expected = circle_repr_token_count(variant)
    if len(vals) != expected:
        raise ValueError(
            "variant %r expects %d values, got %d" % (variant, expected, len(vals))
        )
    if variant == "center_radius":
        return (vals[0], vals[1]), vals[2]
    pts = [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    radius = sum(math.hypot(p[0] - cx, p[1] - cy) for p in pts) / len(pts)
    return (cx, cy), radius


def circle_points_on_circumference(center: Point, radius: float, n: int) -> List[Point]:
    """Return ``n`` points uniformly spaced on the circumference from angle 0.

    Generalizes the ``four_points`` layout; used to validate that the sampled
    points really are uniformly distributed (App. A.1).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    cx, cy = center
    return [
        (cx + radius * math.cos(2.0 * math.pi * k / n),
         cy + radius * math.sin(2.0 * math.pi * k / n))
        for k in range(n)
    ]


# --------------------------------------------------------------------------- #
# 2. PV-constrained sampling-hyperparameter selection (App. A.4 / Table 6)
# --------------------------------------------------------------------------- #
# Direction of improvement for each metric name. +1 => larger is better,
# -1 => smaller is better. Matches Table 1/6 arrow annotations.
_METRIC_DIRECTION: Dict[str, int] = {
    "cov": +1,      # Coverage, higher better
    "novel": +1,    # Novel, higher better
    "unique": +1,   # Unique, higher better
    "mmd": -1,      # Minimum Matching Distance, lower better
    "jsd": -1,      # Jensen-Shannon Divergence, lower better
}

DEFAULT_PV_FLOOR = 0.90


class SamplingConfig:
    """A candidate inference configuration and its measured metrics.

    ``metrics`` is a mapping over any subset of ``_METRIC_DIRECTION`` keys, in
    the same units reported in Table 6 (percentages as fractions in [0, 1] for
    COV/Novel/Unique; raw values for MMD/JSD). ``pv`` is Prediction Validity in
    [0, 1].
    """

    __slots__ = ("tau", "top_p", "metrics", "pv")

    def __init__(self, tau: float, top_p: float, metrics: Dict[str, float], pv: float):
        self.tau = float(tau)
        self.top_p = float(top_p)
        self.metrics = dict(metrics)
        self.pv = float(pv)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "SamplingConfig(tau=%s, top_p=%s, pv=%s)" % (self.tau, self.top_p, self.pv)


def diversity_score(
    metrics: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Composite diversity/quality objective (higher is better).

    Each metric contributes ``direction * weight * value``; lower-is-better
    metrics (MMD, JSD) therefore subtract. Unknown metric keys raise, so a
    typo cannot silently drop a term.
    """
    total = 0.0
    for name, value in metrics.items():
        if name not in _METRIC_DIRECTION:
            raise ValueError("unknown metric: %r" % (name,))
        w = 1.0 if weights is None else float(weights.get(name, 0.0))
        total += _METRIC_DIRECTION[name] * w * float(value)
    return total


def select_sampling_config(
    candidates: Sequence[SamplingConfig],
    pv_floor: float = DEFAULT_PV_FLOOR,
    weights: Optional[Dict[str, float]] = None,
) -> SamplingConfig:
    """Select the FlexCAD inference configuration per App. A.4.

    Rule: keep only candidates whose ``pv >= pv_floor``, then return the one
    with the greatest :func:`diversity_score`. Ties (equal diversity) break
    toward the *higher* PV, then higher ``tau``, then higher ``top_p`` -- a
    fully deterministic ordering. Raises ``ValueError`` if no candidate meets
    the floor.
    """
    if not candidates:
        raise ValueError("no candidate configurations supplied")
    feasible = [c for c in candidates if c.pv >= pv_floor]
    if not feasible:
        raise ValueError("no candidate meets the PV floor of %s" % (pv_floor,))

    def key(c: SamplingConfig) -> Tuple[float, float, float, float]:
        return (diversity_score(c.metrics, weights), c.pv, c.tau, c.top_p)

    return max(feasible, key=key)


def pv_frontier(candidates: Sequence[SamplingConfig]) -> List[SamplingConfig]:
    """Return candidates on the PV-vs-diversity Pareto frontier.

    A configuration is on the frontier if no other configuration has both a
    higher PV and a higher (or equal) diversity score (and strictly better in
    at least one). Result is sorted by descending PV. This exposes the exact
    monotone trade-off surface Table 6 describes, letting a caller pick a
    different PV floor without re-measuring.
    """
    scored = [(c, diversity_score(c.metrics)) for c in candidates]
    frontier: List[SamplingConfig] = []
    for c, s in scored:
        dominated = False
        for other, so in scored:
            if other is c:
                continue
            if other.pv >= c.pv and so >= s and (other.pv > c.pv or so > s):
                dominated = True
                break
        if not dominated:
            frontier.append(c)
    frontier.sort(key=lambda c: c.pv, reverse=True)
    return frontier
