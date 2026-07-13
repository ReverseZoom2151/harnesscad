"""CADTalk part-label voting — multi-view confidence aggregation (Sec. 3.3).

The last stage of CADTalker aggregates per-image semantic evidence and transfers
it to code blocks. For every image ``i`` (rendered from view ``v``), block ``b``
and candidate label ``l`` it forms a confidence

    Ci(b, l) = CDINO(i, l) * IoU(M_vb, S_il)                         (Eq. 1)

where ``CDINO`` is the open-vocabulary detector's label confidence and ``IoU`` is
the overlap between the block's rendered binary mask ``M_vb`` and the segmenter's
predicted mask ``S_il``. CADTalker then fills a matrix ``C(b, l)`` by accumulating
these confidences over all 40 images (10 views x 4 image variants) **in three
steps, with intermediate thresholding to filter poor labels** (Sec. 3.3, and the
threshold schedule 0.001 / 0.01 / 0.02 from the supplement, Sec. 8.1):

  1. per-image confidence ``Ci(b, l)``               -> threshold t1
  2. sum over the 4 images of each view -> ``Cv(b, l)`` -> threshold t2
  3. sum over the 10 views -> ``C(b, l)``               -> threshold t3

Finally each block ``b`` is assigned ``argmax_l C(b, l)``.

The upstream vision confidences (CDINO, mask IoU) come from foundation models and
are *inputs* here; the deterministic aggregation / progressive-thresholding /
argmax scheme is what this module implements. Callers pass either raw
``(cdino, iou)`` pairs or precomputed ``ci`` values.

Pure stdlib; deterministic (ties broken by label sort order).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_THRESHOLDS: Tuple[float, float, float] = (0.001, 0.01, 0.02)


@dataclass(frozen=True)
class Evidence:
    """One piece of visual evidence for a (view, image, block, label) tuple.

    Supply either ``ci`` directly, or ``cdino`` and ``iou`` (Eq. 1 multiplies
    them). ``image`` indexes the variant within a view (e.g. 0..3)."""

    view: int
    image: int
    block: int
    label: str
    cdino: float = 1.0
    iou: float = 1.0
    ci: Optional[float] = None

    def confidence(self) -> float:
        return self.ci if self.ci is not None else self.cdino * self.iou


def _threshold(mat: Dict[Tuple[int, str], float], t: float) -> Dict[Tuple[int, str], float]:
    """Zero out (drop) entries whose confidence is below ``t``."""
    return {k: v for k, v in mat.items() if v >= t}


def accumulate(
    evidence: Iterable[Evidence],
    thresholds: Tuple[float, float, float] = DEFAULT_THRESHOLDS,
) -> Dict[Tuple[int, str], float]:
    """Build the confidence matrix ``C(b, l)`` via the 3-step aggregation.

    Returns a dict keyed by ``(block, label)``. Entries filtered out by a
    threshold at any step do not contribute to later steps."""
    t1, t2, t3 = thresholds
    evidence = list(evidence)

    # Step 1: per-image confidences Ci(b,l), keyed (view,image,block,label).
    per_image: Dict[Tuple[int, int, int, str], float] = {}
    for e in evidence:
        key = (e.view, e.image, e.block, e.label)
        per_image[key] = per_image.get(key, 0.0) + e.confidence()
    # threshold t1 acts on (block,label) view of each image entry
    per_image = {k: v for k, v in per_image.items() if v >= t1}

    # Step 2: sum the 4 images of each view -> Cv(b,l), keyed (view,block,label).
    per_view: Dict[Tuple[int, int, str], float] = defaultdict(float)
    for (view, _image, block, label), v in per_image.items():
        per_view[(view, block, label)] += v
    per_view = {k: v for k, v in per_view.items() if v >= t2}

    # Step 3: sum over all views -> C(b,l).
    matrix: Dict[Tuple[int, str], float] = defaultdict(float)
    for (_view, block, label), v in per_view.items():
        matrix[(block, label)] += v
    matrix = {k: v for k, v in matrix.items() if v >= t3}
    return dict(matrix)


def assign_labels(
    matrix: Dict[Tuple[int, str], float],
) -> Dict[int, str]:
    """Assign each block the label of highest cumulative confidence
    (``argmax_l C(b, l)``). Ties are broken by label sort order for
    determinism."""
    best: Dict[int, Tuple[float, str]] = {}
    for (block, label), conf in matrix.items():
        cur = best.get(block)
        if cur is None or conf > cur[0] or (conf == cur[0] and label < cur[1]):
            best[block] = (conf, label)
    return {b: lab for b, (conf, lab) in best.items()}


def vote(
    evidence: Iterable[Evidence],
    thresholds: Tuple[float, float, float] = DEFAULT_THRESHOLDS,
) -> Dict[int, str]:
    """End-to-end: aggregate ``evidence`` and return ``{block: label}``."""
    return assign_labels(accumulate(evidence, thresholds))


def confidence_matrix_dense(
    matrix: Dict[Tuple[int, str], float],
) -> Tuple[List[int], List[str], List[List[float]]]:
    """Expand the sparse ``C(b, l)`` into ``(blocks, labels, rows)`` for
    inspection / debugging. ``rows[i][j]`` is the confidence of ``blocks[i]``
    for ``labels[j]`` (0.0 where absent)."""
    blocks = sorted({b for (b, _l) in matrix})
    labels = sorted({l for (_b, l) in matrix})
    lab_idx = {l: j for j, l in enumerate(labels)}
    rows = [[0.0] * len(labels) for _ in blocks]
    blk_idx = {b: i for i, b in enumerate(blocks)}
    for (b, l), v in matrix.items():
        rows[blk_idx[b]][lab_idx[l]] = v
    return blocks, labels, rows
