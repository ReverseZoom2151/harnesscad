"""The paper's exact ``Design.design_distance`` (mrCAD ``design.py``).

This is a SECOND, distinct distance from the one in :mod:`bench.mrcad_metrics`.
Both descend from Sec. 2.2, but the reference implementation
(``Design.__design_distance_asymmetric`` / ``Design.design_distance`` in
``mrcad/design.py``) differs from :mod:`bench.mrcad_metrics.chamfer_asymmetric`
in three concrete, verifiable ways:

1. **Point-to-CURVE vs point-to-sampled-point.** ``bench.mrcad_metrics`` samples
   points on *both* designs and takes each source point's nearest *sampled*
   point on the target. This module samples points on the source only, and for
   each measures the *exact analytic* distance to the nearest *curve* of the
   target (:func:`geometry.mrcad2_curve_relations.point_to_curve_distance`).
   The exact distance is never larger than the nearest-sample distance, so this
   metric is tighter.

2. **Mean-normalised [0, 1] vs summed.** ``bench.mrcad_metrics`` divides each
   capped term by the cap and *sums* them (range roughly ``[0, n_points]``). The
   reference divides the summed capped distances by ``max_point_distance *
   n_sampled_points``, giving a *mean* proportional distance in ``[0, 1]``.

3. **Empty-design handling.** For an empty source, ``bench.mrcad_metrics``
   returns ``0.0`` (no points to penalise); the reference returns ``1.0`` (a
   maximally-distant design). NaN results also collapse to ``1.0`` here.

The cap value coincides: the reference uses ``max_point_distance =
grid_size * max_distance_proportion = 40 * 0.25 = 10``, and
``bench.mrcad_metrics`` caps at ``0.25 * canvas_size = 0.25 * 40 = 10``.

Sampling uses :func:`bench.mrcad_metrics.sample_points` for exact parity with the
existing metric's per-curve sampling. Pure stdlib, deterministic.
"""
from __future__ import annotations

import math
from typing import Sequence

from harnesscad.eval.bench.mrcad_metrics import sample_points
from harnesscad.domain.editing.mrcad_schema import Design
from harnesscad.domain.geometry.mrcad2_curve_relations import point_to_curve_distance

#: grid_size (40) * max_distance_proportion (0.25) from the reference RenderConfig.
DEFAULT_MAX_POINT_DISTANCE = 10.0
#: The paper samples 10 points per curve.
DEFAULT_SAMPLES = 10


def _sample_all(design: Design, n: int):
    for c in design.curves:
        for p in sample_points(c, n):
            yield p


def design_distance_asymmetric(
    d_from: Design,
    d_to: Design,
    max_point_distance: float = DEFAULT_MAX_POINT_DISTANCE,
    n: int = DEFAULT_SAMPLES,
) -> float:
    """Mean capped point-to-curve distance from ``d_from`` to ``d_to`` in [0, 1].

    Returns ``1.0`` when ``d_from`` has no curves (or the result is NaN).
    """
    src = list(_sample_all(d_from, n))
    if not src:
        return 1.0
    total = 0.0
    for p in src:
        best = max_point_distance
        for c in d_to.curves:
            best = min(best, point_to_curve_distance(c, p))
        total += best
    result = total / (max_point_distance * len(src))
    if math.isnan(result):
        return 1.0
    return result


def design_distance(
    a: Design,
    b: Design,
    max_point_distance: float = DEFAULT_MAX_POINT_DISTANCE,
    n: int = DEFAULT_SAMPLES,
) -> float:
    """Symmetric design distance: mean of both asymmetric directions (Sec. 2.2)."""
    return 0.5 * (
        design_distance_asymmetric(a, b, max_point_distance, n)
        + design_distance_asymmetric(b, a, max_point_distance, n)
    )


def proportional_improvement(
    before: Design,
    after: Design,
    target: Design,
    max_point_distance: float = DEFAULT_MAX_POINT_DISTANCE,
    n: int = DEFAULT_SAMPLES,
) -> float:
    """PI (Sec. 6.1) computed with the exact design_distance rather than chamfer.

    ``(dist(before, target) - dist(after, target)) / dist(before, target)``;
    ``0.0`` when already at zero distance.
    """
    db = design_distance(before, target, max_point_distance, n)
    if db == 0.0:
        return 0.0
    da = design_distance(after, target, max_point_distance, n)
    return (db - da) / db
