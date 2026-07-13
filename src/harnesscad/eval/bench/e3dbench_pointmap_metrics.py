"""Point-map / point-cloud reconstruction metrics (E3D-Bench, Sec. 3.4).

E3D-Bench evaluates multi-view 3D reconstruction with directional distances,
after aligning the predicted point cloud to ground truth with Umeyama:

* Accuracy (Acc): mean distance from each *predicted* point to the nearest
  ground-truth point (how clean the prediction is).
* Completeness (Comp): mean distance from each *ground-truth* point to the
  nearest predicted point (how much of the surface is covered).
* Normal Consistency (NC): mean absolute cosine similarity between a predicted
  point's normal and its nearest ground-truth point's normal.

We additionally provide the threshold-based Precision / Recall / F-score that is
standard for point-cloud reconstruction (points within a distance ``tau`` count
as matches), which the E3D-Bench toolkit family builds on.

Distances are Euclidean; nearest-neighbour search is exact (brute force).  All
functions are stdlib-only and deterministic.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from harnesscad.domain.geometry.e3dbench_umeyama import Sim3, umeyama_alignment

Vec3 = Sequence[float]
Cloud = Sequence[Vec3]


def _dist2(a: Vec3, b: Vec3) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def nearest_index(point: Vec3, cloud: Cloud) -> int:
    best = 0
    best_d = _dist2(point, cloud[0])
    for j in range(1, len(cloud)):
        d = _dist2(point, cloud[j])
        if d < best_d:
            best_d = d
            best = j
    return best


def _nn_distances(source: Cloud, target: Cloud) -> List[float]:
    """For each point in ``source``, distance to its nearest point in ``target``."""
    out = []
    for p in source:
        j = nearest_index(p, target)
        out.append(math.sqrt(_dist2(p, target[j])))
    return out


def accuracy(pred: Cloud, gt: Cloud) -> float:
    """Mean predicted->ground-truth nearest-neighbour distance."""
    if not pred or not gt:
        raise ValueError("clouds must be non-empty")
    d = _nn_distances(pred, gt)
    return sum(d) / len(d)


def completeness(pred: Cloud, gt: Cloud) -> float:
    """Mean ground-truth->predicted nearest-neighbour distance."""
    if not pred or not gt:
        raise ValueError("clouds must be non-empty")
    d = _nn_distances(gt, pred)
    return sum(d) / len(d)


def chamfer_l1(pred: Cloud, gt: Cloud) -> float:
    """Symmetric mean directional distance (Acc + Comp) / 2."""
    return 0.5 * (accuracy(pred, gt) + completeness(pred, gt))


def precision_recall_fscore(pred: Cloud, gt: Cloud, tau: float
                            ) -> Tuple[float, float, float]:
    """Threshold F-score at distance ``tau``.

    precision = fraction of predicted points within ``tau`` of some gt point;
    recall    = fraction of gt points within ``tau`` of some predicted point;
    F = 2 P R / (P + R).  All in [0, 1].
    """
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    dp = _nn_distances(pred, gt)
    dg = _nn_distances(gt, pred)
    precision = sum(1 for d in dp if d <= tau) / len(dp)
    recall = sum(1 for d in dg if d <= tau) / len(dg)
    if precision + recall == 0.0:
        f = 0.0
    else:
        f = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f


def _cos(a: Vec3, b: Vec3) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    return dot / (na * nb)


def normal_consistency(pred: Cloud, gt: Cloud,
                       pred_normals: Cloud, gt_normals: Cloud) -> float:
    """Mean |cos| between each predicted normal and its nearest gt normal.

    Absolute value makes it orientation-agnostic (surface normals may flip).
    """
    if len(pred) != len(pred_normals) or len(gt) != len(gt_normals):
        raise ValueError("points and normals must align")
    if not pred or not gt:
        raise ValueError("clouds must be non-empty")
    total = 0.0
    for i, p in enumerate(pred):
        j = nearest_index(p, gt)
        total += abs(_cos(pred_normals[i], gt_normals[j]))
    return total / len(pred)


def evaluate_reconstruction(pred: Cloud, gt: Cloud, tau: Optional[float] = None,
                            align: bool = True, with_scale: bool = True,
                            pred_normals: Optional[Cloud] = None,
                            gt_normals: Optional[Cloud] = None) -> dict:
    """Full reconstruction report; aligns pred to gt with Umeyama by default.

    When ``align`` is set, Umeyama Sim(3) (or SE(3) if ``with_scale=False``)
    aligns predicted points to ground truth before scoring, matching the paper's
    protocol.  ``tau`` enables the threshold F-score; normals enable NC.
    """
    p = list(pred)
    pn = list(pred_normals) if pred_normals is not None else None
    if align:
        n = min(len(pred), len(gt))
        xform: Sim3 = umeyama_alignment(pred[:n], gt[:n], with_scale=with_scale)
        p = xform.apply_all(pred)
        if pn is not None:
            # rotate normals only (scale/translation do not affect direction)
            R = xform.R
            pn = [[R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
                   R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
                   R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2]] for v in pn]
    report = {
        "accuracy": accuracy(p, gt),
        "completeness": completeness(p, gt),
        "chamfer_l1": chamfer_l1(p, gt),
    }
    if tau is not None:
        precision, recall, f = precision_recall_fscore(p, gt, tau)
        report["precision"] = precision
        report["recall"] = recall
        report["fscore"] = f
    if pn is not None and gt_normals is not None:
        report["normal_consistency"] = normal_consistency(p, gt, pn, gt_normals)
    return report
