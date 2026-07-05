"""CADTalk evaluation metrics — block accuracy and semantic IoU.

CADTalk (Sec. 4.2, Sec. 9.2) proposes two metrics to score any semantic
CAD-program-commenting algorithm against ground-truth comments:

  * **Block accuracy** ``Bacc = m / n`` (Eq. 2), where ``n`` is the number of
    commentable blocks and ``m`` the number that received the correct label.
    Ground truth may assign *several* valid labels to one block (a primitive that
    spans several semantic parts); the block is correct if the predicted label is
    among the ground-truth set (Sec. 4.1, "either of the labels predicted is
    counted as correct").

  * **Semantic IoU** ``SIoU = (1/K) * sum_k |{l_k} ∩ {l*_k}| / |{l_k} ∪ {l*_k}|``
    (Eq. 3), the per-label Intersection-over-Union between the set of blocks
    *predicted* to be label ``k`` and the set with ``k`` as ground truth,
    averaged over all ``K`` labels. Unlike block accuracy, it is sensitive to the
    long-tail problem where some labels cover only a few blocks.

Because a predicting algorithm may output synonymous but differently-worded
labels, CADTalk applies a **synonym mapping** (predicted-label -> ground-truth
label) *before* computing the metrics (Sec. 4.2, "Evaluation with synonyms").
The mapping itself is produced externally (ChatGPT), but *applying* a supplied
mapping is deterministic and lives here.

Pure stdlib; deterministic.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Set, Tuple, Union

# A ground-truth entry may be one label or a set/list of acceptable labels.
GTLabels = Union[str, Iterable[str]]


def _as_set(labels: GTLabels) -> Set[str]:
    if isinstance(labels, str):
        return {labels}
    return set(labels)


def apply_synonyms(
    predicted: Mapping[int, str],
    mapping: Mapping[str, str],
) -> Dict[int, str]:
    """Rewrite predicted labels through a synonym ``mapping`` (predicted -> GT).

    Labels absent from ``mapping`` are left unchanged (Sec. 4.2: the mapping is
    applied "if any")."""
    return {b: mapping.get(lab, lab) for b, lab in predicted.items()}


def block_accuracy(
    predicted: Mapping[int, str],
    ground_truth: Mapping[int, GTLabels],
) -> float:
    """Block accuracy ``Bacc = m / n`` (Eq. 2).

    ``n`` is the number of ground-truth blocks. A block counts as correct when
    its predicted label is in the block's ground-truth label set. Blocks present
    in ``ground_truth`` but missing from ``predicted`` count as incorrect."""
    n = len(ground_truth)
    if n == 0:
        return 0.0
    m = 0
    for b, gt in ground_truth.items():
        pred = predicted.get(b)
        if pred is not None and pred in _as_set(gt):
            m += 1
    return m / n


def _blocks_by_label_pred(
    predicted: Mapping[int, str],
) -> Dict[str, Set[int]]:
    out: Dict[str, Set[int]] = {}
    for b, lab in predicted.items():
        out.setdefault(lab, set()).add(b)
    return out


def _blocks_by_label_gt(
    ground_truth: Mapping[int, GTLabels],
) -> Dict[str, Set[int]]:
    out: Dict[str, Set[int]] = {}
    for b, labs in ground_truth.items():
        for lab in _as_set(labs):
            out.setdefault(lab, set()).add(b)
    return out


def semantic_iou(
    predicted: Mapping[int, str],
    ground_truth: Mapping[int, GTLabels],
    per_label: bool = False,
):
    """Semantic IoU (Eq. 3): per-label block-set IoU, averaged over all labels.

    The label universe ``K`` is the union of labels appearing in either the
    predictions or the ground truth. If ``per_label`` is true, returns a
    ``(mean, {label: iou})`` tuple; otherwise returns the mean IoU."""
    pred_sets = _blocks_by_label_pred(predicted)
    gt_sets = _blocks_by_label_gt(ground_truth)
    labels = sorted(set(pred_sets) | set(gt_sets))
    if not labels:
        return (0.0, {}) if per_label else 0.0
    ious: Dict[str, float] = {}
    for lab in labels:
        p = pred_sets.get(lab, set())
        g = gt_sets.get(lab, set())
        union = p | g
        if not union:
            ious[lab] = 0.0
        else:
            ious[lab] = len(p & g) / len(union)
    mean = sum(ious.values()) / len(labels)
    return (mean, ious) if per_label else mean


def evaluate(
    predicted: Mapping[int, str],
    ground_truth: Mapping[int, GTLabels],
    synonyms: Mapping[str, str] = None,
) -> Dict[str, object]:
    """Full CADTalk evaluation report for one program.

    Optionally applies a ``synonyms`` mapping (predicted -> GT) before scoring.
    Returns ``{"block_accuracy", "semantic_iou", "per_label_iou", "n_blocks",
    "n_correct"}``."""
    if synonyms:
        predicted = apply_synonyms(predicted, synonyms)
    bacc = block_accuracy(predicted, ground_truth)
    mean_iou, per = semantic_iou(predicted, ground_truth, per_label=True)
    n = len(ground_truth)
    return {
        "block_accuracy": bacc,
        "semantic_iou": mean_iou,
        "per_label_iou": per,
        "n_blocks": n,
        "n_correct": round(bacc * n),
    }


def aggregate(reports: Iterable[Dict[str, object]]) -> Dict[str, float]:
    """Average block accuracy and semantic IoU over a set of per-program reports
    (e.g. an entire CADTalk track), giving each program equal weight."""
    reports = list(reports)
    if not reports:
        return {"block_accuracy": 0.0, "semantic_iou": 0.0, "n_programs": 0}
    ba = sum(float(r["block_accuracy"]) for r in reports) / len(reports)
    si = sum(float(r["semantic_iou"]) for r in reports) / len(reports)
    return {
        "block_accuracy": ba,
        "semantic_iou": si,
        "n_programs": len(reports),
    }
