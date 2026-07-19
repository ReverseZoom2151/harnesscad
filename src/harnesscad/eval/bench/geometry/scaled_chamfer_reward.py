"""Scaled Chamfer Distance reward with a fail-stage taxonomy.

Deterministic re-implementation of a reward function for RL-trained STEP
generation against a *scale-invariant* geometric reward. The transferable,
model-free core is the reward math and its honest fail-stage classification --
this implements the underlying reward formulation's three equations exactly:

* **Eq. 1 -- bidirectional Chamfer distance** between two point clouds P, Q::

      CD(P,Q) = mean_p min_q ||p-q||^2 + mean_q min_p ||p-q||^2

* **Eq. 2 -- Scaled Chamfer Distance**: centre GT at its centroid, divide the
  Chamfer distance by ``scale^2`` where ``scale`` is the RMS distance of GT
  points from their centroid. This makes the metric invariant to translation and
  scale. (A full such pipeline also aligns rotation via FPFH+RANSAC+ICP;
  that needs Open3D and is out of scope here -- this module keeps the
  translation+scale-invariant metric, which is deterministic and stdlib-only.)

* **Eq. 3 -- piecewise-linear reward** ``R_geo``::

      R_geo(scd) = 1                          if scd <= delta_low
      R_geo(scd) = 0                          if scd >= delta_high
      R_geo(scd) = (delta_high - scd)/(delta_high - delta_low)  otherwise

The reward pipeline never raises and returns an explicit **fail-stage** --
``ok`` / ``pred_empty`` / ``gt_empty`` / ``pred_degenerate`` / ``gt_degenerate``
/ ``scd_nonfinite`` -- which is the checkable, auditable signal a harness wants:
a zero reward that says *why* (parse failed vs geometry wrong) rather than an
opaque 0.

Distinct from the harness's existing plain Chamfer metrics
(:mod:`harnesscad.eval.bench.geometry.chamfer` and the unit-shape variants):
those measure a raw distance; this adds the scale-normalization above, the
piecewise reward gate with thresholds, and the fail-stage classifier.

Point clouds are plain sequences of ``(x, y, z)`` tuples. Pure Python math, no
numpy. Deterministic, absolute imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

__all__ = [
    "Point",
    "chamfer_distance",
    "scaled_chamfer_distance",
    "r_geo",
    "RewardConfig",
    "RewardResult",
    "compute_reward",
]

Point = Tuple[float, float, float]

_MIN_UNIQUE_POINTS = 4


def _sq_dist(a: Point, b: Point) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return dx * dx + dy * dy + dz * dz


def _nearest_sq(p: Point, cloud: Sequence[Point]) -> float:
    best = float("inf")
    for q in cloud:
        d = _sq_dist(p, q)
        if d < best:
            best = d
    return best


def chamfer_distance(
    P: Sequence[Point], Q: Sequence[Point], *, bidirectional: bool = True
) -> float:
    """Chamfer distance (Eq. 1 of the reward formulation).

    Forward term is ``mean_p min_q ||p-q||^2``. With ``bidirectional=True`` the
    symmetric ``mean_q min_p`` term is added (the stated method);
    ``False`` matches a note that the reference eval used forward-only.
    Empty inputs return ``inf``.
    """
    if not P or not Q:
        return float("inf")
    d_pq = sum(_nearest_sq(p, Q) for p in P) / len(P)
    if not bidirectional:
        return d_pq
    d_qp = sum(_nearest_sq(q, P) for q in Q) / len(Q)
    return d_pq + d_qp


def _centroid(cloud: Sequence[Point]) -> Point:
    n = len(cloud)
    sx = sum(p[0] for p in cloud)
    sy = sum(p[1] for p in cloud)
    sz = sum(p[2] for p in cloud)
    return (sx / n, sy / n, sz / n)


def _centered(cloud: Sequence[Point], c: Point) -> List[Point]:
    return [(p[0] - c[0], p[1] - c[1], p[2] - c[2]) for p in cloud]


def _rms_radius(centered: Sequence[Point]) -> float:
    n = len(centered)
    return (sum(_sq_dist(p, (0.0, 0.0, 0.0)) for p in centered) / n) ** 0.5


def scaled_chamfer_distance(
    pred: Sequence[Point], gt: Sequence[Point], *, bidirectional: bool = True
) -> float:
    """Scaled Chamfer Distance (Eq. 2 of the reward formulation), translation+scale invariant.

    Asymmetric by design: both the scale factor and the centring target derive
    from ``gt`` -- do not swap the arguments. ``scale`` is the RMS distance of GT
    points from their centroid; returns ``inf`` on a degenerate (zero-scale) GT.
    """
    if not pred or not gt:
        return float("inf")
    gt_c = _centroid(gt)
    gt_centered = _centered(gt, gt_c)
    scale = _rms_radius(gt_centered)
    if scale < 1e-9:
        return float("inf")
    pred_centered = _centered(pred, _centroid(pred))
    # Scale pre-normalization (``scale_prenorm``): bring pred to GT's
    # RMS scale before comparison. With rotation alignment dropped, this is what
    # makes the metric scale-invariant (Eq. 2 relies on the alignment stage to
    # pre-scale pred; the division by ``scale^2`` then normalizes GT's own size).
    pred_scale = _rms_radius(pred_centered)
    if pred_scale > 1e-9:
        factor = scale / pred_scale
        pred_centered = [(p[0] * factor, p[1] * factor, p[2] * factor) for p in pred_centered]
    cd = chamfer_distance(pred_centered, gt_centered, bidirectional=bidirectional)
    return cd / (scale * scale)


def r_geo(scd: float, *, delta_low: float = 0.01, delta_high: float = 0.50) -> float:
    """Piecewise-linear geometric reward (Eq. 3 of the reward formulation), in [0, 1]."""
    if scd <= delta_low:
        return 1.0
    if scd >= delta_high:
        return 0.0
    return (delta_high - scd) / (delta_high - delta_low)


@dataclass(frozen=True)
class RewardConfig:
    """Reward-shape parameters (the frozen reward config)."""

    delta_low: float = 0.01
    delta_high: float = 0.50
    bidirectional: bool = True
    min_unique_points: int = _MIN_UNIQUE_POINTS


@dataclass(frozen=True)
class RewardResult:
    """Reward outcome with an explicit fail-stage.

    ``reward`` is ``r_geo(scd)`` on success, ``0.0`` on a prediction failure, and
    ``nan`` on a ground-truth failure (a broken GT must not reward-hack the
    policy toward 0). ``fail_stage`` is one of ``ok`` / ``pred_empty`` /
    ``gt_empty`` / ``pred_degenerate`` / ``gt_degenerate`` / ``scd_nonfinite``.
    """

    reward: float
    scd: float
    fail_stage: str


def _unique_count(cloud: Sequence[Point]) -> int:
    return len(set(cloud))


def compute_reward(
    pred: Sequence[Point],
    gt: Sequence[Point],
    *,
    cfg: RewardConfig = RewardConfig(),
) -> RewardResult:
    """Full reward pipeline. Never raises; returns a :class:`RewardResult`.

    Guards degenerate clouds (too few unique points -- a reward-hacking
    guard) before scoring, and distinguishes a prediction failure (reward 0) from
    a ground-truth failure (reward nan).
    """
    nan = float("nan")
    if not pred:
        return RewardResult(0.0, nan, "pred_empty")
    if not gt:
        return RewardResult(nan, nan, "gt_empty")
    if _unique_count(pred) < cfg.min_unique_points:
        return RewardResult(0.0, nan, "pred_degenerate")
    if _unique_count(gt) < cfg.min_unique_points:
        return RewardResult(nan, nan, "gt_degenerate")

    scd = scaled_chamfer_distance(pred, gt, bidirectional=cfg.bidirectional)
    if scd != scd or scd == float("inf"):  # nan or inf
        return RewardResult(nan, nan, "scd_nonfinite")

    reward = r_geo(scd, delta_low=cfg.delta_low, delta_high=cfg.delta_high)
    return RewardResult(reward, scd, "ok")
