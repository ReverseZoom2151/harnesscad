"""Evaluation metrics for mrCAD refinement (Sec. 2.2 distance metric, Sec. 6).

Deterministic, stdlib-only reimplementations of the paper's programmatic
metrics:

  * :func:`sample_points` -- sample points along a line/circle/arc (the paper
    samples 10 points per curve, Fig. 2).
  * :func:`chamfer_asymmetric` / :func:`chamfer_symmetric` -- the vector-aware
    chamfer distance ``Delta`` (Sec. 2.2, App. B.2). Each point-to-design
    distance is capped at (and normalised by) a quarter of the canvas size --
    "beyond half a quadrant away, two points are not likely to be related".
  * :func:`proportional_improvement` -- the PI benchmark metric of Sec. 6.1,
    ``(Delta(D_i,D*) - Delta(A(D_i),D*)) / Delta(D_i,D*)``.
  * :func:`edit_accuracy` / :func:`exact_match` -- how well a predicted action
    sequence matches the gold maker actions.
  * :func:`convergence` -- distance-to-target trajectory across refinement
    rounds, per-round PI, monotonicity, and rounds-to-win.

No learned metric (no CLIP), no rendering, no model. Imports the CAD types from
:mod:`editing.mrcad_schema`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from harnesscad.domain.editing.sketch_edit_schema import Curve, Design

Point = Tuple[float, float]

#: The paper samples 10 points on every curve (Fig. 2).
DEFAULT_SAMPLES = 10
#: Default canvas extent: control points range over [-20, 20] (Sec. C, prompt).
DEFAULT_CANVAS_SIZE = 40.0


def _lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _sample_polyline(vertices: Sequence[Point], n: int) -> Tuple[Point, ...]:
    """Sample ``n`` points evenly by arc length along a polyline."""
    if n <= 1 or len(vertices) == 1:
        return (tuple(vertices[0]),)
    seglens = [
        math.dist(vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)
    ]
    total = sum(seglens)
    if total == 0.0:
        return tuple(tuple(vertices[0]) for _ in range(n))
    out = []
    for k in range(n):
        target = total * k / (n - 1)
        acc = 0.0
        for i, sl in enumerate(seglens):
            if sl == 0.0:
                continue
            if acc + sl >= target or i == len(seglens) - 1:
                out.append(_lerp(vertices[i], vertices[i + 1], (target - acc) / sl))
                break
            acc += sl
    return tuple(out)


def _circumcenter(p0: Point, p1: Point, p2: Point) -> Optional[Point]:
    ax, ay = p0
    bx, by = p1
    cx, cy = p2
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return (ux, uy)


def sample_points(curve: Curve, n: int = DEFAULT_SAMPLES) -> Tuple[Point, ...]:
    """Sample ``n`` points along ``curve`` deterministically."""
    pts = curve.points
    if curve.kind == "line":
        return _sample_polyline(pts, n)
    if curve.kind == "circle":
        a, b = pts
        cx, cy = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
        r = math.dist(a, b) / 2.0
        return tuple(
            (cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n))
            for k in range(n)
        )
    # arc: circle through (start, mid, end); sweep start->end passing mid.
    a, m, b = pts
    center = _circumcenter(a, m, b)
    if center is None:
        return _sample_polyline((a, m, b), n)
    cx, cy = center
    r = math.dist(center, a)
    a0 = math.atan2(a[1] - cy, a[0] - cx)
    a1 = math.atan2(m[1] - cy, m[0] - cx)
    a2 = math.atan2(b[1] - cy, b[0] - cx)
    two_pi = 2 * math.pi
    ccw_sweep = (a2 - a0) % two_pi
    mid_ccw = (a1 - a0) % two_pi
    sweep = ccw_sweep if mid_ccw <= ccw_sweep else ccw_sweep - two_pi
    if n <= 1:
        return (a,)
    return tuple(
        (cx + r * math.cos(a0 + sweep * k / (n - 1)),
         cy + r * math.sin(a0 + sweep * k / (n - 1)))
        for k in range(n)
    )


def _design_sample(design: Design, n: int) -> Tuple[Point, ...]:
    out: list[Point] = []
    for c in design.curves:
        out.extend(sample_points(c, n))
    return tuple(out)


def chamfer_asymmetric(
    d_from: Design,
    d_to: Design,
    n: int = DEFAULT_SAMPLES,
    canvas_size: float = DEFAULT_CANVAS_SIZE,
) -> float:
    """Asymmetric chamfer distance from ``d_from`` to ``d_to`` (Sec. 2.2, Fig. 2).

    Samples points on every curve of ``d_from`` and sums the (capped, normalised)
    minimum distance from each to ``d_to``. The cap/normaliser is a quarter of
    the canvas size.
    """
    cap = 0.25 * canvas_size
    src = _design_sample(d_from, n)
    dst = _design_sample(d_to, n)
    total = 0.0
    for p in src:
        if dst:
            best = min(math.dist(p, q) for q in dst)
        else:
            best = cap
        total += min(best, cap) / cap
    return total


def chamfer_symmetric(
    a: Design,
    b: Design,
    n: int = DEFAULT_SAMPLES,
    canvas_size: float = DEFAULT_CANVAS_SIZE,
) -> float:
    """Symmetric chamfer distance ``Delta`` = mean of both directions (Sec. 2.2)."""
    return 0.5 * (
        chamfer_asymmetric(a, b, n, canvas_size)
        + chamfer_asymmetric(b, a, n, canvas_size)
    )


def proportional_improvement_from_distances(before: float, after: float) -> float:
    """PI given pre/post distances-to-target (Sec. 6.1). Zero if already at 0."""
    if before == 0.0:
        return 0.0
    return (before - after) / before


def proportional_improvement(
    before: Design,
    after: Design,
    target: Design,
    n: int = DEFAULT_SAMPLES,
    canvas_size: float = DEFAULT_CANVAS_SIZE,
) -> float:
    """PI metric of Sec. 6.1 measuring how much closer to ``target`` a round got."""
    db = chamfer_symmetric(before, target, n, canvas_size)
    da = chamfer_symmetric(after, target, n, canvas_size)
    return proportional_improvement_from_distances(db, da)


def _multiset(seq: Sequence) -> dict:
    counts: dict = {}
    for x in seq:
        counts[x] = counts.get(x, 0) + 1
    return counts


def exact_match(predicted: Sequence, gold: Sequence) -> bool:
    """True if the predicted action sequence equals the gold sequence exactly."""
    return tuple(predicted) == tuple(gold)


def edit_accuracy(predicted: Sequence, gold: Sequence) -> float:
    """Fraction of gold actions recovered (order-insensitive multiset recall).

    Returns 1.0 when both are empty, 0.0 when only ``gold`` is empty.
    """
    gcounts = _multiset(gold)
    if not gcounts:
        return 1.0 if len(tuple(predicted)) == 0 else 0.0
    pcounts = _multiset(predicted)
    overlap = sum(min(c, pcounts.get(k, 0)) for k, c in gcounts.items())
    return overlap / sum(gcounts.values())


@dataclass(frozen=True)
class ConvergenceReport:
    """Distance-to-target trajectory across refinement rounds (Sec. 3.2, 5.2)."""

    distances: Tuple[float, ...]
    per_round_pi: Tuple[float, ...]
    total_reduction: float
    monotone_nonincreasing: bool
    rounds_to_win: Optional[int]


def convergence(
    designs: Sequence[Design],
    target: Design,
    initial: Optional[Design] = None,
    threshold: Optional[float] = None,
    n: int = DEFAULT_SAMPLES,
    canvas_size: float = DEFAULT_CANVAS_SIZE,
) -> ConvergenceReport:
    """Summarise convergence of a rollout's resulting designs toward ``target``.

    ``designs`` is the sequence ``[D'_1, ..., D'_n]``. ``initial`` (default empty)
    is the pre-rollout state used only to compute round 1's PI. ``rounds_to_win``
    is the 1-based index of the first round with ``Delta < threshold``.
    """
    start = initial if initial is not None else Design.empty()
    dist = lambda d: chamfer_symmetric(d, target, n, canvas_size)
    distances = tuple(dist(d) for d in designs)
    prefix = (dist(start),) + distances
    per_round_pi = tuple(
        proportional_improvement_from_distances(prefix[i], prefix[i + 1])
        for i in range(len(designs))
    )
    total_reduction = prefix[0] - prefix[-1] if designs else 0.0
    monotone = all(distances[i] <= distances[i - 1] + 1e-12 for i in range(1, len(distances)))
    rounds_to_win: Optional[int] = None
    if threshold is not None:
        for i, dd in enumerate(distances, start=1):
            if dd < threshold:
                rounds_to_win = i
                break
    return ConvergenceReport(
        distances=distances,
        per_round_pi=per_round_pi,
        total_reduction=total_reduction,
        monotone_nonincreasing=monotone,
        rounds_to_win=rounds_to_win,
    )
