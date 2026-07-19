"""Bind dense line proposals to sparse endpoint proposals.

An initial wireframe is built by *binding* the dense line-segment proposals
``L_hat_n`` produced from the HAT field to the sparse **endpoint proposals**
``P_hat_m``:

    "Specifically, in ``L_hat_n`` we find the nearest endpoint proposals
    ``P_hat_m`` for the two endpoints of the line segment proposal ... denoted as
    ``x1`` and ``x2`` ... find the nearest endpoint proposal ``x1'`` for the
    endpoint proposal ``y1`` ... likewise ``x2'`` for ``y2``.  We calculate the
    squared Euclidean distances between them as ``delta1`` and ``delta2``.  The
    maximum distance is defined as ``delta = Max(delta1, delta2)``.  Smaller
    distances indicate higher quality for the line segment ``L_hat_n`` (a higher
    likelihood of collinearity).  Use a threshold ``epsilon`` to select
    high-quality line segment proposals whose binding cost ``delta`` is less than
    this threshold.  Finally, generate a new set of endpoint-enhanced line segment
    proposals ``L_hat_n = (x1, x2, y1, y2)``."

Everything above the learned heat-map regression is deterministic nearest-point
geometry, implemented here.  This is the construction step that turns raw
proposals into the Structured Visual Geometry wireframe.  Pure stdlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


Point = tuple[float, float]
Seg = tuple[Point, Point]


def _dist2(a: Point, b: Point) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def nearest_endpoint(p: Point, endpoints: list[Point]) -> tuple[int, float]:
    """Index and squared distance of the endpoint proposal nearest to ``p``.

    Ties resolve to the lowest index (deterministic).
    """
    if not endpoints:
        raise ValueError("no endpoint proposals to bind against")
    best_i, best_d = 0, math.inf
    for i, q in enumerate(endpoints):
        dd = _dist2(p, q)
        if dd < best_d:
            best_d, best_i = dd, i
    return best_i, best_d


@dataclass(frozen=True)
class BoundSegment:
    """An endpoint-enhanced proposal ``L_hat_n = (x1, x2, y1, y2)``.

    ``x1, x2`` are the original line-proposal endpoints; ``y1, y2`` are the bound
    (snapped) endpoint proposals; ``delta`` is the binding cost
    ``max(delta1, delta2)``.
    """

    x1: Point
    x2: Point
    y1: Point
    y2: Point
    i1: int
    i2: int
    delta1: float
    delta2: float

    @property
    def delta(self) -> float:
        return max(self.delta1, self.delta2)

    def snapped(self) -> Seg:
        """The segment using the bound endpoint proposals ``(y1, y2)``."""
        return (self.y1, self.y2)


def bind_segment(seg: Seg, endpoints: list[Point]) -> BoundSegment:
    """Bind one line proposal to its nearest endpoint proposals."""
    x1, x2 = seg
    i1, d1 = nearest_endpoint(x1, endpoints)
    i2, d2 = nearest_endpoint(x2, endpoints)
    return BoundSegment(
        x1=x1,
        x2=x2,
        y1=endpoints[i1],
        y2=endpoints[i2],
        i1=i1,
        i2=i2,
        delta1=d1,
        delta2=d2,
    )


def bind_and_select(
    segments: list[Seg], endpoints: list[Point], epsilon: float
) -> list[BoundSegment]:
    """Bind every proposal and keep those with binding cost ``delta < epsilon``.

    Also rejects proposals whose two endpoints snap to the *same* endpoint
    proposal (``i1 == i2``), which would be a degenerate zero-length segment.
    Output preserves input order (deterministic).
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    out: list[BoundSegment] = []
    for seg in segments:
        b = bind_segment(seg, endpoints)
        if b.i1 == b.i2:
            continue
        if b.delta < epsilon:
            out.append(b)
    return out


def collinearity_quality(bound: BoundSegment) -> float:
    """A monotone quality score in ``(0, 1]``: higher = lower binding cost.

    ``1 / (1 + delta)``; a smaller ``delta`` means higher quality /
    higher likelihood of collinearity.
    """
    return 1.0 / (1.0 + bound.delta)
