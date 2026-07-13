"""img2cadsvg_loi_align -- Joint-Decoupled Line-of-Interest Aligning (JD LOIAlign).

After binding (see :mod:`reconstruction.img2cadsvg_binding`), Img2CAD refines the
wireframe with **Line-of-Interest (LOI) Pooling** and the **Joint-Decoupled
Line-of-Interest Aligning (JD LOIAlign)** module (paper, Sec. IV), which
"filters out false positive proposals through interest point alignment" and
"captures the co-occurrence between the endpoint proposals and the HAT field".

The deterministic geometry the paper specifies:

* the sampling function ``Psi_t(X) = (1 - t) * x1 + t * x2`` with ``t in [0, 1]``
  maps a background point to a point on the segment;
* LOI Pooling samples a segment at evenly spaced parameters ``t`` to validate a
  proposal against data evidence;
* the *joint-decoupled* design keeps **three** sets of sampling points: the two
  endpoints, and two decoupled *middle* points -- so the model gains geometric
  awareness by separating endpoint evidence from mid-segment evidence.

This module implements LOI sampling (:func:`loi_sample`), the joint-decoupled
grouping (:func:`decoupled_groups`), and an alignment/validation score
(:func:`loi_align_score`) that measures the co-occurrence between a line proposal
and the node-deduced endpoints -- used to filter false positives.  The learned
pooling features are external; the sampling geometry and the alignment score are
deterministic.  Pure stdlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


Point = tuple[float, float]
Seg = tuple[Point, Point]


def psi(t: float, x1: Point, x2: Point) -> Point:
    """Sampling function ``Psi_t(X) = (1 - t) * x1 + t * x2`` for ``t in [0,1]``."""
    if not (0.0 <= t <= 1.0):
        raise ValueError("t must lie in [0, 1]")
    return (
        (1.0 - t) * x1[0] + t * x2[0],
        (1.0 - t) * x1[1] + t * x2[1],
    )


def loi_sample(seg: Seg, n: int) -> list[Point]:
    """Sample a segment at ``n`` evenly spaced parameters ``t in [0, 1]``.

    ``n >= 2``; endpoints are included (``t = 0`` and ``t = 1``).
    """
    if n < 2:
        raise ValueError("need at least 2 sample points")
    x1, x2 = seg
    return [psi(i / (n - 1), x1, x2) for i in range(n)]


@dataclass(frozen=True)
class DecoupledGroups:
    """The three joint-decoupled sampling sets of JD LOIAlign."""

    endpoints: tuple[Point, Point]
    mid_x: Point  # centre of x1 and its bound proposal x1'
    mid_y: Point  # centre of y1 and its bound proposal y1'


def decoupled_groups(
    x1: Point, x1_bound: Point, y1: Point, y1_bound: Point
) -> DecoupledGroups:
    """Build the joint-decoupled sampling sets.

    Per the paper the model maintains: (1) the two endpoints ``y1, y2``;
    (2) the centre ``Psi_t(X)`` of ``x1`` and its bound proposal ``x1'``;
    (3) the centre ``Psi_t(Y)`` of ``y1`` and its bound proposal ``y1'``.  The
    centres use ``t = 0.5`` (the midpoint) as the decoupling reference.
    """
    return DecoupledGroups(
        endpoints=(x1, y1),
        mid_x=psi(0.5, x1, x1_bound),
        mid_y=psi(0.5, y1, y1_bound),
    )


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def loi_align_score(
    proposal: Seg, node_endpoints: tuple[Point, Point], scale: float = 1.0
) -> float:
    """Alignment score between a line proposal and node-deduced endpoints.

    Measures the co-occurrence between the proposal's endpoints and the
    junction (node) proposals that would deduce the same segment.  Returns a
    value in ``(0, 1]``: ``1`` iff the proposal endpoints coincide (in either
    orientation) with the node endpoints, decaying with total displacement.
    ``scale`` sets the distance at which the score reaches ``1/(1+1)``.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    p1, p2 = proposal
    n1, n2 = node_endpoints
    direct = _dist(p1, n1) + _dist(p2, n2)
    swapped = _dist(p1, n2) + _dist(p2, n1)
    cost = min(direct, swapped) / scale
    return 1.0 / (1.0 + cost)


def filter_false_positives(
    proposals: list[Seg],
    node_endpoints: list[tuple[Point, Point]],
    threshold: float,
    scale: float = 1.0,
) -> list[Seg]:
    """Keep proposals whose best LOIAlign score over any node pair >= threshold.

    Reproduces "filters out false positive proposals through interest point
    alignment": a proposal survives only if some node-deduced endpoint pair
    aligns with it above ``threshold``.  Deterministic, order-preserving.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must lie in [0, 1]")
    kept: list[Seg] = []
    for prop in proposals:
        best = 0.0
        for nodes in node_endpoints:
            s = loi_align_score(prop, nodes, scale=scale)
            if s > best:
                best = s
        if best >= threshold:
            kept.append(prop)
    return kept
