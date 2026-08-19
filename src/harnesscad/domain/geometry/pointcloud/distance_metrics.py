"""Point-cloud and sampled-curve distances -- plain geometry, no scoring.

Chamfer distance between two clouds, and the point-to-point distance between two
equally-sampled curves (open, and closed with its cyclic-shift minimisation).
These are geometric primitives: a reconstruction pipeline uses them to decide
whether two candidate curves are the SAME curve, and a benchmark uses them to
score a prediction against a ground truth.  The benchmark is the consumer, not
the owner, so they live in the geometry layer and
``harnesscad.eval.bench.geometry.complex_matching`` re-exports them.

Points are indexable sequences of coordinates of equal length (2-D or 3-D).
Stdlib-only and deterministic.
"""

from __future__ import annotations

import math

__all__ = [
    "BIG",
    "chamfer_distance",
    "curve_distance",
    "closed_curve_distance",
    "sampled_curve_distance",
]

#: The distance reported between two curves that are not even comparable (one
#: closed, one open).  Large enough that no matcher will ever pair them.
BIG = 1e6


def _dist(a, b) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


def chamfer_distance(x, y, direction: str = "bi") -> float:
    """Chamfer distance between two point clouds (mean of nearest-neighbour L2).

    ``direction``: ``'x_to_y'``, ``'y_to_x'`` or ``'bi'`` (the average of both).
    """
    if not x or not y:
        raise ValueError("chamfer_distance needs two non-empty clouds")
    if direction == "x_to_y":
        return sum(min(_dist(p, q) for q in y) for p in x) / len(x)
    if direction == "y_to_x":
        return sum(min(_dist(q, p) for p in x) for q in y) / len(y)
    if direction == "bi":
        return 0.5 * (chamfer_distance(x, y, "x_to_y") + chamfer_distance(x, y, "y_to_x"))
    raise ValueError("direction must be 'x_to_y', 'y_to_x' or 'bi'")


def curve_distance(pts0, pts1) -> float:
    """Mean point-to-point distance of two equally-sampled open curves.

    Minimised over the two traversal orientations (a curve and its reverse are the
    same curve).
    """
    if len(pts0) != len(pts1):
        raise ValueError("curves must have the same number of samples")
    n = len(pts0)
    forward = sum(_dist(pts0[i], pts1[i]) for i in range(n)) / n
    backward = sum(_dist(pts0[n - 1 - i], pts1[i]) for i in range(n)) / n
    return min(forward, backward)


def closed_curve_distance(pts0, pts1) -> float:
    """Mean point-to-point distance of two closed curves, minimised over every
    cyclic shift and both orientations (reference ``closed_curve_distance``)."""
    if len(pts0) != len(pts1):
        raise ValueError("curves must have the same number of samples")
    n = len(pts0)
    best = float("inf")
    reversed0 = list(reversed(pts0))
    for base in (list(pts0), reversed0):
        for shift in range(n):
            total = 0.0
            for i in range(n):
                total += _dist(base[(i + shift) % n], pts1[i])
            best = min(best, total / n)
    return best


def sampled_curve_distance(curve0, curve1) -> float:
    """Distance between two sampled curve cells (closed-aware).

    ``curve0``/``curve1`` are anything carrying ``.closed`` and ``.points`` --
    a chain-complex ``Curve`` cell, for instance.  Two curves of different
    closedness are incomparable and score :data:`BIG`.
    """
    if curve0.closed != curve1.closed:
        return BIG
    if curve0.closed:
        return closed_curve_distance(curve0.points, curve1.points)
    return curve_distance(curve0.points, curve1.points)
