"""cad2program_metrics — reconstruction / retrieval / parameter accuracy metrics.

Evaluation protocol of CAD2PROGRAM (Wang et al., AAAI 2025, Sec. 4.1).  Given a
predicted assembly of primitive instances and a ground-truth assembly, the paper
reports three accuracies, all of which are deterministic set/geometry operations:

  * **3D reconstruction** (common parameters): match predicted primitives to
    ground truth with **Hungarian matching** on the 3D bounding boxes, count a
    prediction as a true positive iff its 3D intersection-over-union (IoU) with
    its match exceeds 0.5, and report precision / recall / F1.
  * **model retrieval** (model ID): over the matched pairs, the fraction whose
    predicted model ID equals the ground-truth model ID.
  * **parameter estimation** (model-specific params): over the *correctly
    retrieved* pairs, the fraction for which **all** model-specific parameters
    are correct.

The boxes are the 7-parameter :class:`~reconstruction.cad2program_shape_program.
Bbox` (center position, size, z-rotation).  IoU is computed for axis-aligned
boxes (the paper's cabinet primitives use ``angle_z = 0``); a box with a non-zero
angle that is a multiple of 90 degrees is normalized by swapping its x/y extents,
and any other angle mismatch makes the pair non-overlapping (conservative).

The Hungarian assignment is a self-contained O(n^3) Kuhn-Munkres implementation
(stdlib only, deterministic).  The VLM that produces the predictions is external;
this module only scores whatever pair of programs it is given.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from harnesscad.domain.reconstruction.cad2program_shape_program import (
    Bbox, PrimitiveInstance, ShapeProgram,
)

_INF = float("inf")


# --------------------------------------------------------------------------- #
# 3D IoU of oriented-but-axis-aligned boxes
# --------------------------------------------------------------------------- #

def _axis_extents(box: Bbox) -> Optional[Tuple[Tuple[float, float],
                                               Tuple[float, float],
                                               Tuple[float, float]]]:
    """Return per-axis (min, max) intervals, folding 90-degree rotations.

    Returns ``None`` when the rotation is not an axis-aligned multiple of 90
    degrees (such a box cannot be compared as an axis-aligned interval product).
    """
    angle = box.angle_z % 360
    sx, sy, sz = box.scale_x, box.scale_y, box.scale_z
    if angle in (0, 180):
        ex, ey = sx, sy
    elif angle in (90, 270):
        ex, ey = sy, sx
    else:
        return None
    px, py, pz = box.position
    return ((px - ex / 2.0, px + ex / 2.0),
            (py - ey / 2.0, py + ey / 2.0),
            (pz - sz / 2.0, pz + sz / 2.0))


def _overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def box_iou_3d(a: Bbox, b: Bbox) -> float:
    """Volumetric IoU of two axis-aligned 3D boxes (0.0 on any angle mismatch)."""
    ea, eb = _axis_extents(a), _axis_extents(b)
    if ea is None or eb is None:
        return 0.0
    inter = 1.0
    for i in range(3):
        inter *= _overlap(ea[i], eb[i])
    if inter <= 0.0:
        return 0.0
    va = abs(a.scale_x * a.scale_y * a.scale_z)
    vb = abs(b.scale_x * b.scale_y * b.scale_z)
    union = va + vb - inter
    if union <= 0.0:
        return 0.0
    return inter / union


# --------------------------------------------------------------------------- #
# Hungarian (Kuhn-Munkres) minimum-cost assignment
# --------------------------------------------------------------------------- #

def hungarian(cost: Sequence[Sequence[float]]) -> List[Tuple[int, int]]:
    """Minimum-cost perfect assignment on a rectangular cost matrix.

    Returns a list of ``(row, col)`` pairs of length ``min(rows, cols)``.  Rows
    and columns beyond the square core are left unassigned.  Deterministic; ties
    resolve to the lowest column index.
    """
    if not cost or not cost[0]:
        return []
    n_rows, n_cols = len(cost), len(cost[0])
    n = max(n_rows, n_cols)
    # Pad to a square matrix with a large constant.
    big = 0.0
    for row in cost:
        for v in row:
            big = max(big, abs(v))
    pad = big * (n + 1) + 1.0
    mat = [[cost[r][c] if r < n_rows and c < n_cols else pad
            for c in range(n)] for r in range(n)]

    # Kuhn-Munkres with potentials (O(n^3)); classic 1-indexed formulation.
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)   # p[j] = row assigned to column j
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [_INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = _INF
            j1 = -1
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = mat[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    pairs = []
    for j in range(1, n + 1):
        r, c = p[j] - 1, j - 1
        if r < n_rows and c < n_cols:
            pairs.append((r, c))
    pairs.sort()
    return pairs


# --------------------------------------------------------------------------- #
# Reconstruction / retrieval / parameter accuracy
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MatchResult:
    """Outcome of matching a predicted program to a ground-truth program."""

    matches: Tuple[Tuple[int, int, float], ...]   # (pred_idx, gt_idx, iou)
    true_positives: int
    n_pred: int
    n_gt: int

    @property
    def precision(self) -> float:
        return self.true_positives / self.n_pred if self.n_pred else (
            1.0 if self.n_gt == 0 else 0.0)

    @property
    def recall(self) -> float:
        return self.true_positives / self.n_gt if self.n_gt else (
            1.0 if self.n_pred == 0 else 0.0)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def _instances(program) -> List[PrimitiveInstance]:
    if isinstance(program, ShapeProgram):
        return list(program.instances)
    return list(program)


def match_primitives(pred, gt, iou_threshold: float = 0.5) -> MatchResult:
    """Hungarian-match predicted to ground-truth primitives by 3D-box IoU.

    A matched pair counts as a true positive only when its IoU exceeds
    ``iou_threshold`` (the paper uses 0.5).  Every predicted/gt box is matched at
    most once.
    """
    preds, gts = _instances(pred), _instances(gt)
    if not preds or not gts:
        return MatchResult((), 0, len(preds), len(gts))
    iou = [[box_iou_3d(pi.bbox, gj.bbox) for gj in gts] for pi in preds]
    cost = [[1.0 - iou[r][c] for c in range(len(gts))]
            for r in range(len(preds))]
    matches: List[Tuple[int, int, float]] = []
    tp = 0
    for r, c in hungarian(cost):
        value = iou[r][c]
        matches.append((r, c, value))
        if value > iou_threshold:
            tp += 1
    matches.sort()
    return MatchResult(tuple(matches), tp, len(preds), len(gts))


def reconstruction_prf(pred, gt, iou_threshold: float = 0.5) -> Dict[str, float]:
    """Precision / recall / F1 for 3D reconstruction (common parameters)."""
    res = match_primitives(pred, gt, iou_threshold)
    return {"precision": res.precision, "recall": res.recall, "f1": res.f1,
            "true_positives": res.true_positives,
            "n_pred": res.n_pred, "n_gt": res.n_gt}


def model_retrieval_accuracy(pred, gt, iou_threshold: float = 0.5
                             ) -> Dict[str, float]:
    """Fraction of true-positive matches whose model ID is correct.

    Retrieval is scored only over primitives whose box was correctly recovered
    (IoU above threshold), following the paper's "correctly retrieved models"
    denominator.
    """
    preds, gts = _instances(pred), _instances(gt)
    res = match_primitives(pred, gt, iou_threshold)
    correct = 0
    considered = 0
    for pi, gj, value in res.matches:
        if value <= iou_threshold:
            continue
        considered += 1
        if preds[pi].model_id == gts[gj].model_id:
            correct += 1
    acc = correct / considered if considered else (1.0 if res.n_gt == 0 else 0.0)
    return {"accuracy": acc, "correct": correct, "considered": considered}


def parameter_estimation_accuracy(pred, gt, iou_threshold: float = 0.5
                                  ) -> Dict[str, float]:
    """All-or-nothing model-specific parameter accuracy over retrieved models.

    A prediction is correct iff its model ID matches (correctly retrieved) *and*
    every model-specific ``key=value`` equals the ground truth (Sec. 4.1: "a
    successful estimation means that all parameters are correct").  Primitives
    that have no model-specific parameters in the ground truth still count as a
    trivially-correct estimation, matching the paper's per-primitive accounting.
    """
    preds, gts = _instances(pred), _instances(gt)
    res = match_primitives(pred, gt, iou_threshold)
    correct = 0
    considered = 0
    for pi, gj, value in res.matches:
        if value <= iou_threshold:
            continue
        if preds[pi].model_id != gts[gj].model_id:
            continue
        considered += 1
        if preds[pi].param_dict() == gts[gj].param_dict():
            correct += 1
    acc = correct / considered if considered else (1.0 if res.n_gt == 0 else 0.0)
    return {"accuracy": acc, "correct": correct, "considered": considered}


def evaluate(pred, gt, iou_threshold: float = 0.5) -> Dict[str, object]:
    """Full evaluation bundle: reconstruction PRF + retrieval + param accuracy."""
    return {
        "reconstruction": reconstruction_prf(pred, gt, iou_threshold),
        "retrieval": model_retrieval_accuracy(pred, gt, iou_threshold),
        "parameter": parameter_estimation_accuracy(pred, gt, iou_threshold),
    }
