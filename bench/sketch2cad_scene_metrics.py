"""Scene-reconstruction evaluation metrics for Sketch2CAD (Yang, EPFL 2023).

The paper evaluates a predicted scene descriptor against ground truth with the
metrics reported in Table I (simple dataset) and Table II (complex dataset):

  * **Camera Pose Estimation (Acc)** -- the fraction of scenes whose predicted
    camera-pose ID matches the ground-truth pose ID (a discrete classification).
  * **Object Classification (F1-score)** -- shape-type classification quality over
    the objects in the scene.
  * **Position error** -- per-axis mean absolute error, reported as a triple
    ``(ex, ey, ez)`` in world units (paper: "world size = 20" / "200").
  * **Rotation error** -- per-angle mean absolute error ``(e_yaw, e_pitch)`` in
    degrees (complex dataset only).
  * **Size error** -- per-axis mean absolute error ``(ex, ey, ez)`` (paper: max 20 /
    60).

Because a scene is an *unordered* set of objects, the per-object errors require
matching predicted objects to ground-truth objects first. This module uses a
deterministic greedy nearest-position assignment (no external solver), then
accumulates errors over matched pairs. Object classification F1 is computed as a
macro-F1 over shape types across matched pairs (unmatched predictions / ground
truths count as errors for the relevant class).

Pure stdlib. Consumes
:class:`reconstruction.sketch2cad_scene_descriptor.SceneObject` instances.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def pose_accuracy(pred_ids, gt_ids) -> float:
    """Fraction of scenes with a correctly predicted camera-pose ID."""
    pred_ids = list(pred_ids)
    gt_ids = list(gt_ids)
    if len(pred_ids) != len(gt_ids):
        raise ValueError("pred_ids and gt_ids differ in length")
    if not gt_ids:
        return 0.0
    correct = sum(1 for p, g in zip(pred_ids, gt_ids) if p == g)
    return correct / len(gt_ids)


def _dist2(a, b) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def match_objects(pred, gt):
    """Greedy nearest-position matching of predicted -> ground-truth objects.

    Returns ``(pairs, unmatched_pred, unmatched_gt)`` where ``pairs`` is a list of
    ``(pred_obj, gt_obj)``. Deterministic: candidate pairs are sorted by squared
    position distance, then by (pred index, gt index) to break ties.
    """
    pred = list(pred)
    gt = list(gt)
    candidates = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            candidates.append((_dist2(p.position, g.position), i, j))
    candidates.sort()
    used_p: set[int] = set()
    used_g: set[int] = set()
    pairs = []
    for _, i, j in candidates:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        pairs.append((pred[i], gt[j]))
    unmatched_pred = [p for i, p in enumerate(pred) if i not in used_p]
    unmatched_gt = [g for j, g in enumerate(gt) if j not in used_g]
    return pairs, unmatched_pred, unmatched_gt


def _axiswise_mae(pairs, attr, n):
    sums = [0.0] * n
    for p, g in pairs:
        pv = getattr(p, attr)
        gv = getattr(g, attr)
        for k in range(n):
            sums[k] += abs(pv[k] - gv[k])
    m = len(pairs)
    if m == 0:
        return tuple(0.0 for _ in range(n))
    return tuple(s / m for s in sums)


def position_error(pairs) -> tuple[float, float, float]:
    """Per-axis mean absolute position error ``(ex, ey, ez)`` over matched pairs."""
    return _axiswise_mae(pairs, "position", 3)


def size_error(pairs) -> tuple[float, float, float]:
    """Per-axis mean absolute size error ``(ex, ey, ez)`` over matched pairs."""
    return _axiswise_mae(pairs, "size", 3)


def rotation_error(pairs) -> tuple[float, float]:
    """Per-angle mean absolute rotation error ``(e_yaw, e_pitch)`` in degrees.

    Angular difference is taken on the circle (wrapped to ``[0, 180]``).
    """
    sums = [0.0, 0.0]
    for p, g in pairs:
        for k in range(2):
            sums[k] += _angle_diff(p.rotation[k], g.rotation[k])
    m = len(pairs)
    if m == 0:
        return (0.0, 0.0)
    return (sums[0] / m, sums[1] / m)


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def classification_f1(pairs, unmatched_pred, unmatched_gt, shape_types) -> float:
    """Macro-F1 of shape-type classification over matched + unmatched objects.

    Matched pairs contribute a (pred_shape -> gt_shape) prediction. Each unmatched
    ground-truth object counts as a false negative for its class; each unmatched
    prediction counts as a false positive for its class.
    """
    tp = {t: 0 for t in shape_types}
    fp = {t: 0 for t in shape_types}
    fn = {t: 0 for t in shape_types}
    for p, g in pairs:
        if p.shape == g.shape:
            tp[g.shape] += 1
        else:
            fp[p.shape] += 1
            fn[g.shape] += 1
    for g in unmatched_gt:
        fn[g.shape] += 1
    for p in unmatched_pred:
        fp[p.shape] += 1

    f1s = []
    for t in shape_types:
        denom = 2 * tp[t] + fp[t] + fn[t]
        if denom == 0:
            continue  # class absent from both -> excluded from macro average
        f1s.append(2 * tp[t] / denom)
    if not f1s:
        return 0.0
    return sum(f1s) / len(f1s)


@dataclass(frozen=True)
class SceneReport:
    """Aggregated per-scene metrics matching the paper's result tables."""

    pose_acc: float
    classification_f1: float
    position_error: tuple[float, float, float]
    rotation_error: tuple[float, float]
    size_error: tuple[float, float, float]
    matched: int
    unmatched_pred: int
    unmatched_gt: int


def evaluate_scene(pred_pose, gt_pose, pred_objs, gt_objs, shape_types) -> SceneReport:
    """Evaluate one scene reconstruction into a :class:`SceneReport`."""
    pairs, up, ug = match_objects(pred_objs, gt_objs)
    return SceneReport(
        pose_acc=1.0 if pred_pose == gt_pose else 0.0,
        classification_f1=classification_f1(pairs, up, ug, shape_types),
        position_error=position_error(pairs),
        rotation_error=rotation_error(pairs),
        size_error=size_error(pairs),
        matched=len(pairs),
        unmatched_pred=len(up),
        unmatched_gt=len(ug),
    )
