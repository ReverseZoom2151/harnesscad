"""Query-to-instance decoding for point-based symbol spotting.

The network emits a fixed set of ``Q`` *queries*, each a
(class-logit vector, per-point mask-logit vector) pair.  Turning those into a
panoptic result is entirely deterministic, and it is the piece implemented here
-- the network itself is out of scope.

Two decoders:

* :func:`semantic_inference` -- the per-point class scores are the *mask-weighted
  mixture* of the query class distributions,
  ``semseg[g][c] = sum_q softmax(cls[q])[c] * sigmoid(mask[q][g])``.  No query is
  selected; every query votes on every point in proportion to how much it claims
  it.
* :func:`instance_inference` -- **winner-takes-all grouping**, which replaces the
  IoU-NMS of box detectors and the centroid clustering of offset methods
  (``reconstruction.cadtransformer_instance_offsets``):

    1. keep queries whose arg-max class is not the no-object class and whose
       confidence reaches ``object_score`` (0.1);
    2. build ``prob[q][g] = score_q * sigmoid(mask[q][g])`` and give every point
       to the single query with the highest value -- so a point belongs to at
       most one instance *by construction*, no NMS pass needed;
    3. keep a query only if it retains a fraction ``overlap_threshold`` (0.8) of
       the points it originally claimed (``mask >= 0.5``).  A query that loses
       most of its mask to a more confident query is dropped whole rather than
       being emitted as a fragment.

Deterministic, stdlib-only.  Ties in the arg-max resolve to the lowest query
index (documented, not incidental).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

OBJECT_SCORE = 0.1
OVERLAP_THRESHOLD = 0.8
MASK_THRESHOLD = 0.5


def sigmoid(x: float) -> float:
    """Numerically stable logistic function."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def softmax(logits: Sequence[float]) -> List[float]:
    """Softmax of one logit vector."""
    if not logits:
        raise ValueError("logits must be non-empty")
    top = max(logits)
    exps = [math.exp(float(v) - top) for v in logits]
    total = sum(exps)
    return [v / total for v in exps]


def _check(class_logits: Sequence[Sequence[float]],
           mask_logits: Sequence[Sequence[float]]) -> Tuple[int, int, int]:
    q = len(class_logits)
    if q != len(mask_logits):
        raise ValueError("class_logits and mask_logits must have the same number of queries")
    if q == 0:
        return 0, 0, 0
    c = len(class_logits[0])
    g = len(mask_logits[0])
    for row in class_logits:
        if len(row) != c:
            raise ValueError("ragged class_logits")
    for row in mask_logits:
        if len(row) != g:
            raise ValueError("ragged mask_logits")
    return q, c, g


def semantic_inference(class_logits: Sequence[Sequence[float]],
                       mask_logits: Sequence[Sequence[float]]) -> List[List[float]]:
    """Per-point class scores: mask-probability-weighted mixture of query classes.

    ``class_logits`` is ``Q x (C + 1)`` -- the last column is the no-object class
    and is dropped.  Returns a ``G x C`` score table.
    """
    q, c, g = _check(class_logits, mask_logits)
    num_classes = c - 1
    if num_classes < 1:
        raise ValueError("class_logits must have at least one real class plus no-object")
    probs = [softmax(row)[:num_classes] for row in class_logits]
    masks = [[sigmoid(float(v)) for v in row] for row in mask_logits]
    out = [[0.0] * num_classes for _ in range(g)]
    for qi in range(q):
        for gi in range(g):
            m = masks[qi][gi]
            if m == 0.0:
                continue
            for ci in range(num_classes):
                out[gi][ci] += probs[qi][ci] * m
    return out


def semantic_labels(semseg: Sequence[Sequence[float]]) -> List[int]:
    """Arg-max class of every point (ties -> lowest class id)."""
    labels = []
    for row in semseg:
        best, best_v = 0, None
        for ci, v in enumerate(row):
            if best_v is None or v > best_v:
                best, best_v = ci, v
        labels.append(best)
    return labels


def instance_inference(class_logits: Sequence[Sequence[float]],
                       mask_logits: Sequence[Sequence[float]],
                       object_score: float = OBJECT_SCORE,
                       overlap_threshold: float = OVERLAP_THRESHOLD,
                       mask_threshold: float = MASK_THRESHOLD) -> List[Dict[str, object]]:
    """Winner-takes-all decoding of queries into disjoint symbol instances.

    Returns instances ``{"label", "score", "points"}`` ordered by query index;
    ``points`` are the (sorted, disjoint) primitive indices owned by the query.
    """
    q, c, g = _check(class_logits, mask_logits)
    if q == 0 or g == 0:
        return []
    num_classes = c - 1

    kept: List[Tuple[int, int, float, List[float]]] = []  # (query, label, score, mask_prob)
    for qi in range(q):
        probs = softmax(class_logits[qi])
        label, score = 0, probs[0]
        for ci in range(1, c):
            if probs[ci] > score:
                label, score = ci, probs[ci]
        if label == num_classes:  # no-object class
            continue
        if score < object_score:
            continue
        kept.append((qi, label, score, [sigmoid(float(v)) for v in mask_logits[qi]]))
    if not kept:
        return []

    # winner-takes-all: every point goes to the highest score * mask-probability
    owner: List[int] = []
    for gi in range(g):
        best_k, best_v = 0, None
        for k, (_qi, _label, score, mask) in enumerate(kept):
            v = score * mask[gi]
            if best_v is None or v > best_v:
                best_k, best_v = k, v
        owner.append(best_k)

    results: List[Dict[str, object]] = []
    for k, (_qi, label, score, mask) in enumerate(kept):
        original = [gi for gi in range(g) if mask[gi] >= mask_threshold]
        final = [gi for gi in original if owner[gi] == k]
        won = [gi for gi in range(g) if owner[gi] == k]
        if not won or not original or not final:
            continue
        if len(won) / len(original) < overlap_threshold:
            continue
        results.append({"label": label, "score": score, "points": final})
    return results
