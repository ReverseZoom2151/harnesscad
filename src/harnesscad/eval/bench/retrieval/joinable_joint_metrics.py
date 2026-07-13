"""Evaluation metrics for B-Rep joint prediction (JoinABLe, CVPR 2022).

Joint prediction ranks every candidate pair of B-Rep entities (one entity from
each body) by a score; the ground truth is a set of entity pairs -- the labelled
joint plus its *equivalents* (entities that produce an identical joint, e.g. the
other side of a symmetric hole).  A prediction is a hit when any ground-truth
pair appears in the top-k of the ranking.

This module provides:

* :func:`hit_at_top_k` -- hit or miss within the top-k of a flat score matrix;
* :func:`precision_at_k_sequence` / :func:`aggregate_precision_at_k` -- the
  precision-at-k curve over a dataset, over the standard k sequence
  1..5, 10, 20 ... 100;
* :func:`rank_of_first_hit`, :func:`mean_reciprocal_rank`;
* :func:`joint_axis_hit` / :func:`joint_axis_error_stats` -- whether a predicted
  joint axis is colinear with the ground-truth axis, and summary statistics of
  the angle/distance errors.

Scores are given as a dense list of rows (body-one entities) by columns
(body-two entities), or as a flat list; labels are the matching 0/1 structure.
Ties are broken by ascending flat index, which makes every metric deterministic.
Stdlib only.
"""

import math

from harnesscad.domain.geometry.kinematics.joinable_joint_axis import joint_axis_error

__all__ = [
    "DEFAULT_K_SEQUENCE",
    "k_sequence",
    "flatten",
    "ranked_indices",
    "hit_at_top_k",
    "precision_at_k_sequence",
    "aggregate_precision_at_k",
    "rank_of_first_hit",
    "mean_reciprocal_rank",
    "joint_axis_hit",
    "joint_axis_error_stats",
]


def k_sequence():
    """The standard k values to report: 1..5, then 10, 20 ... 100."""
    return list(range(1, 6)) + list(range(10, 110, 10))


DEFAULT_K_SEQUENCE = tuple(k_sequence())


def flatten(matrix):
    """Flatten a list of rows (or pass a flat list through) to a flat list."""
    if not matrix:
        return []
    if isinstance(matrix[0], (list, tuple)):
        out = []
        for row in matrix:
            out.extend(row)
        return out
    return list(matrix)


def _check(scores, labels):
    flat_scores = [float(s) for s in flatten(scores)]
    flat_labels = [int(bool(v)) for v in flatten(labels)]
    if len(flat_scores) != len(flat_labels):
        raise ValueError("scores and labels must have the same number of "
                         "entity pairs")
    if not flat_scores:
        raise ValueError("no candidate entity pairs")
    return flat_scores, flat_labels


def ranked_indices(scores):
    """Flat candidate indices in descending score order, ties by index."""
    flat = [float(s) for s in flatten(scores)]
    return sorted(range(len(flat)), key=lambda i: (-flat[i], i))


def hit_at_top_k(scores, labels, k=1):
    """True when a positive label lies in the top-``k`` scored candidates.

    ``k`` is clamped to the number of candidates.
    """
    flat_scores, flat_labels = _check(scores, labels)
    if k < 1:
        raise ValueError("k must be >= 1")
    limit = min(int(k), len(flat_scores))
    order = ranked_indices(flat_scores)
    return any(flat_labels[i] == 1 for i in order[:limit])


def precision_at_k_sequence(scores, labels, ks=None):
    """Hit (1) / miss (0) for each k in ``ks`` for a single sample."""
    ks = list(DEFAULT_K_SEQUENCE) if ks is None else list(ks)
    flat_scores, flat_labels = _check(scores, labels)
    order = ranked_indices(flat_scores)
    n = len(flat_scores)
    rank = None
    for position, index in enumerate(order):
        if flat_labels[index] == 1:
            rank = position + 1
            break
    results = []
    for k in ks:
        if k < 1:
            raise ValueError("k must be >= 1")
        limit = min(int(k), n)
        results.append(1 if (rank is not None and rank <= limit) else 0)
    return results


def aggregate_precision_at_k(per_sample_hits, use_percent=True):
    """Column-wise mean of the per-sample hit vectors (the precision-at-k curve)."""
    rows = [list(r) for r in per_sample_hits]
    if not rows:
        raise ValueError("no samples")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("all samples must report the same number of k values")
    scale = 100.0 if use_percent else 1.0
    return [sum(r[j] for r in rows) / len(rows) * scale for j in range(width)]


def rank_of_first_hit(scores, labels):
    """1-based rank of the highest-scored ground-truth pair; ``None`` if absent."""
    flat_scores, flat_labels = _check(scores, labels)
    for position, index in enumerate(ranked_indices(flat_scores)):
        if flat_labels[index] == 1:
            return position + 1
    return None


def mean_reciprocal_rank(samples):
    """Mean of ``1 / rank`` over ``(scores, labels)`` samples; misses score 0."""
    samples = list(samples)
    if not samples:
        raise ValueError("no samples")
    total = 0.0
    for scores, labels in samples:
        rank = rank_of_first_hit(scores, labels)
        if rank is not None:
            total += 1.0 / rank
    return total / len(samples)


def joint_axis_hit(predicted_axis, ground_truth_axes, angle_tol_degs=10.0,
                   distance_tol=1e-2):
    """True when the predicted joint axis matches any ground-truth axis.

    ``ground_truth_axes`` is the set of equivalent joint axes -- matching any
    one of them counts as a hit, mirroring JoinABLe's joint-equivalent labels.
    """
    for axis in ground_truth_axes:
        angle, distance = joint_axis_error(predicted_axis, axis)
        if angle < angle_tol_degs and distance < distance_tol:
            return True
    return False


def _is_axis_line(value):
    """True when ``value`` looks like a single ``(origin, direction)`` axis."""
    if len(value) != 2:
        return False
    first = value[0]
    if isinstance(first, dict):
        return True
    try:
        components = list(first)
    except TypeError:
        return False
    return len(components) == 3 and all(
        isinstance(c, (int, float)) for c in components)


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _std(values):
    if not values:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def joint_axis_error_stats(pairs, angle_tol_degs=10.0, distance_tol=1e-2):
    """Summary statistics of the joint-axis error over ``(pred, gt)`` pairs.

    Each ``gt`` may be a single axis line or a list of equivalent axis lines;
    the error reported is that of the closest equivalent (smallest angle, then
    smallest distance).
    """
    pairs = list(pairs)
    if not pairs:
        raise ValueError("no axis pairs")
    angles = []
    distances = []
    hits = 0
    for predicted, truth in pairs:
        candidates = [truth] if _is_axis_line(truth) else list(truth)
        if not candidates:
            raise ValueError("no ground-truth axis for a prediction")
        best = min((joint_axis_error(predicted, axis) for axis in candidates),
                   key=lambda e: (e[0], e[1]))
        angles.append(best[0])
        distances.append(best[1])
        if best[0] < angle_tol_degs and best[1] < distance_tol:
            hits += 1
    return {
        "count": len(pairs),
        "hit_count": hits,
        "hit_rate": hits / len(pairs),
        "mean_angle_degs": _mean(angles),
        "median_angle_degs": _median(angles),
        "std_angle_degs": _std(angles),
        "mean_distance": _mean(distances),
        "median_distance": _median(distances),
        "std_distance": _std(distances),
    }
