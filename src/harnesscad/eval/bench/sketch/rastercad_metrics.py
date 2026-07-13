"""rastercad_metrics -- raster-sketch generation & vectorisation metrics for RECAD.

RECAD (Li et al., "Revisiting CAD Model Generation by Learning Raster Sketch")
generates a CAD sketch as a **binary raster image** (1 = extrudable region,
0 = empty).  Evaluating such a representation calls for *raster-space* measures --
how well a generated sketch canvas overlaps a reference, how much of the
reference stroke is covered -- and, once the raster is vectorised (see
:mod:`vision.rastercad_vectorize`), how accurately the recovered primitives match
the ground-truth primitives.

This module provides those deterministic evaluation pieces:

* :func:`raster_iou` -- intersection-over-union of two binary sketch canvases.
* :func:`raster_precision_recall_f1` -- pixel precision / recall / F1.
* :func:`stroke_coverage` -- fraction of ground-truth ink covered by the
  prediction (with an optional dilation tolerance for near-misses).
* :func:`vectorization_accuracy` -- greedy one-to-one matching of predicted vs
  ground-truth primitives by type and geometry, yielding type accuracy, matched
  fraction, mean geometric error, and F1 at a distance threshold.

Primitives are plain dicts so this module stays independent of any producer:
``{"type": "line", "start": (x, y), "end": (x, y)}``,
``{"type": "circle", "center": (x, y), "radius": r}``,
``{"type": "arc", "start": (x, y), "mid": (x, y), "end": (x, y)}``.
Coordinates are in the normalised ``[0, 1]`` canvas.  Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


Grid = list[list[int]]
Point = tuple[float, float]


# ---------------------------------------------------------------------------
# Raster overlap metrics.
# ---------------------------------------------------------------------------


def _check_same_shape(a: Grid, b: Grid) -> tuple[int, int]:
    if not a or not a[0] or not b or not b[0]:
        raise ValueError("grids must be non-empty")
    h, w = len(a), len(a[0])
    if len(b) != h or len(b[0]) != w:
        raise ValueError("grids must have the same shape")
    for row in a:
        if len(row) != w:
            raise ValueError("grid a is ragged")
    for row in b:
        if len(row) != w:
            raise ValueError("grid b is ragged")
    return h, w


def raster_iou(pred: Grid, gt: Grid) -> float:
    """Intersection-over-union of two binary sketch canvases.

    Two empty canvases are defined to have IoU ``1.0`` (perfect agreement on
    "nothing").
    """

    h, w = _check_same_shape(pred, gt)
    inter = 0
    union = 0
    for r in range(h):
        pr, gr = pred[r], gt[r]
        for c in range(w):
            p = 1 if pr[c] else 0
            g = 1 if gr[c] else 0
            if p or g:
                union += 1
                if p and g:
                    inter += 1
    if union == 0:
        return 1.0
    return inter / union


@dataclass(frozen=True)
class PRF:
    """Precision / recall / F1 triple."""

    precision: float
    recall: float
    f1: float


def raster_precision_recall_f1(pred: Grid, gt: Grid) -> PRF:
    """Pixel-level precision, recall and F1 of ``pred`` against ``gt`` ink."""

    h, w = _check_same_shape(pred, gt)
    tp = fp = fn = 0
    for r in range(h):
        pr, gr = pred[r], gt[r]
        for c in range(w):
            p = 1 if pr[c] else 0
            g = 1 if gr[c] else 0
            if p and g:
                tp += 1
            elif p and not g:
                fp += 1
            elif g and not p:
                fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    denom = precision + recall
    f1 = 2.0 * precision * recall / denom if denom else 0.0
    return PRF(precision=precision, recall=recall, f1=f1)


def _dilate_positions(gt: Grid, tol: int) -> set[tuple[int, int]]:
    """Set of pixels within Chebyshev distance ``tol`` of any prediction ink."""

    h, w = len(gt), len(gt[0])
    out: set[tuple[int, int]] = set()
    for r in range(h):
        row = gt[r]
        for c in range(w):
            if row[c]:
                for dr in range(-tol, tol + 1):
                    for dc in range(-tol, tol + 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            out.add((nr, nc))
    return out


def stroke_coverage(pred: Grid, gt: Grid, tolerance: int = 0) -> float:
    """Fraction of ground-truth ink pixels covered by the prediction.

    With ``tolerance > 0``, a GT ink pixel counts as covered if any prediction
    ink pixel lies within Chebyshev distance ``tolerance`` -- forgiving small
    rasterisation offsets.  Empty GT returns ``1.0``.
    """

    h, w = _check_same_shape(pred, gt)
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    gt_total = sum(1 for row in gt for v in row if v)
    if gt_total == 0:
        return 1.0
    if tolerance == 0:
        covered = 0
        for r in range(h):
            pr, gr = pred[r], gt[r]
            for c in range(w):
                if gr[c] and pr[c]:
                    covered += 1
        return covered / gt_total
    reach = _dilate_positions(pred, tolerance)
    covered = 0
    for r in range(h):
        gr = gt[r]
        for c in range(w):
            if gr[c] and (r, c) in reach:
                covered += 1
    return covered / gt_total


# ---------------------------------------------------------------------------
# Primitive matching / vectorisation accuracy.
# ---------------------------------------------------------------------------


def _line_endpoints(prim: dict) -> tuple[Point, Point]:
    return tuple(prim["start"]), tuple(prim["end"])


def primitive_distance(a: dict, b: dict) -> float:
    """Symmetric geometric distance between two primitives.

    Returns ``inf`` when types differ.  Lines are compared by best endpoint
    pairing (orientation-agnostic); circles by centre distance plus radius
    difference; arcs by their three ordered control points (with reversal).
    """

    ta, tb = a.get("type"), b.get("type")
    if ta != tb:
        return math.inf
    if ta == "line":
        (a0, a1) = _line_endpoints(a)
        (b0, b1) = _line_endpoints(b)
        straight = _pt(a0, b0) + _pt(a1, b1)
        swapped = _pt(a0, b1) + _pt(a1, b0)
        return min(straight, swapped) / 2.0
    if ta == "circle":
        return _pt(tuple(a["center"]), tuple(b["center"])) + abs(
            a["radius"] - b["radius"]
        )
    if ta == "arc":
        a0, am, a1 = tuple(a["start"]), tuple(a["mid"]), tuple(a["end"])
        b0, bm, b1 = tuple(b["start"]), tuple(b["mid"]), tuple(b["end"])
        forward = _pt(a0, b0) + _pt(am, bm) + _pt(a1, b1)
        reverse = _pt(a0, b1) + _pt(am, bm) + _pt(a1, b0)
        return min(forward, reverse) / 3.0
    raise ValueError(f"unknown primitive type: {ta!r}")


def _pt(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def match_primitives(
    pred: list[dict], gt: list[dict], threshold: float
) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching of ``pred`` to ``gt`` primitives.

    Considers all valid (same-type, distance <= ``threshold``) pairs in ascending
    distance order and greedily accepts them if neither side is already matched.
    Returns a list of ``(pred_index, gt_index, distance)`` in acceptance order.
    Deterministic: ties break by ``(pred_index, gt_index)``.
    """

    candidates: list[tuple[float, int, int]] = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            d = primitive_distance(p, g)
            if d <= threshold:
                candidates.append((d, i, j))
    candidates.sort()
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for d, i, j in candidates:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matches.append((i, j, d))
    return matches


@dataclass(frozen=True)
class VectorizationAccuracy:
    """Summary of primitive-level vectorisation accuracy."""

    num_pred: int
    num_gt: int
    num_matched: int
    type_accuracy: float
    mean_matched_distance: float
    precision: float
    recall: float
    f1: float


def vectorization_accuracy(
    pred: list[dict], gt: list[dict], threshold: float = 0.05
) -> VectorizationAccuracy:
    """Accuracy of extracted primitives ``pred`` against ground truth ``gt``.

    * ``type_accuracy`` -- fraction of GT primitives whose *type* multiset is met
      by the prediction (min per-type counts over union, normalised by GT count).
    * matching-based ``precision`` / ``recall`` / ``f1`` at the distance
      ``threshold`` using :func:`match_primitives`.
    * ``mean_matched_distance`` -- average geometric error over matched pairs
      (``0.0`` when nothing matched).
    """

    if threshold < 0.0:
        raise ValueError("threshold must be >= 0")
    n_pred, n_gt = len(pred), len(gt)
    # Type-multiset accuracy.
    if n_gt == 0:
        type_accuracy = 1.0 if n_pred == 0 else 0.0
    else:
        from collections import Counter

        cp = Counter(p.get("type") for p in pred)
        cg = Counter(g.get("type") for g in gt)
        overlap = sum(min(cp[t], cg[t]) for t in cg)
        type_accuracy = overlap / n_gt
    matches = match_primitives(pred, gt, threshold)
    n_match = len(matches)
    precision = n_match / n_pred if n_pred else (1.0 if n_gt == 0 else 0.0)
    recall = n_match / n_gt if n_gt else (1.0 if n_pred == 0 else 0.0)
    denom = precision + recall
    f1 = 2.0 * precision * recall / denom if denom else 0.0
    mean_dist = sum(d for _, _, d in matches) / n_match if n_match else 0.0
    return VectorizationAccuracy(
        num_pred=n_pred,
        num_gt=n_gt,
        num_matched=n_match,
        type_accuracy=type_accuracy,
        mean_matched_distance=mean_dist,
        precision=precision,
        recall=recall,
        f1=f1,
    )
