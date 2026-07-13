"""Binary-free evaluation of generated OpenSCAD against a reference program.

ScadLM scores a generation by rendering it with the ``openscad`` binary and
asking a vision model "does it look correct?" -- an external, non-deterministic
judge. Its own README concedes the loop "didn't work that well".

Given the local front end (:mod:`programs.scadlm_ast`), gate
(:mod:`programs.scadlm_check`) and evaluator
(:mod:`geometry.scadlm_csg_eval`), the same question can be answered
deterministically at the *geometry* level, with no binary and no model:

  * :func:`compile_rate` -- the fraction of candidate programs that pass the
    static gate (the local stand-in for ScadLM's compile check), plus
    :func:`pass_at_k`-style best-of-k over a candidate list;
  * :func:`voxel_iou` -- intersection-over-union of two programs' solids on a
    shared cell-centre lattice covering the union of their bounds;
  * :func:`volume_ratio` / :func:`bbox_iou` / :func:`centroid_offset` -- cheap
    coarse agreement signals that catch the failure modes an LLM actually makes
    (right shape, wrong size; right size, wrong place);
  * :func:`score` -- a single :class:`MatchReport` bundling all of the above,
    including the *reason* a candidate scored zero (syntax error, unsupported
    construct, empty geometry).

All metrics are fixed-lattice and order-independent: repeated calls on the same
inputs return bit-identical numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from harnesscad.domain.geometry.sdf.scadlm_csg_eval import (
    ScadEvalError,
    bounds,
    contains,
    evaluate_source,
    volume,
)
from harnesscad.domain.programs.ast.scadlm_ast import ScadSyntaxError
from harnesscad.domain.programs.validate.scadlm_check import check, is_valid

__all__ = [
    "MatchReport",
    "compile_rate",
    "best_of_k",
    "voxel_iou",
    "volume_ratio",
    "bbox_iou",
    "centroid_offset",
    "score",
]

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class MatchReport:
    compiles: bool
    reason: str            # "" when the candidate evaluated cleanly
    voxel_iou: float
    volume_ratio: float
    bbox_iou: float
    centroid_offset: float

    @property
    def failed(self) -> bool:
        return not self.compiles or bool(self.reason)


# ---------------------------------------------------------------------------
# Gate-level metrics
# ---------------------------------------------------------------------------

def compile_rate(sources: Sequence[str]) -> float:
    """Fraction of ``sources`` that pass the static validity gate."""
    if not sources:
        return 0.0
    return sum(1 for s in sources if is_valid(s)) / float(len(sources))


def best_of_k(candidates: Sequence[str], reference: str,
              resolution: int = 16) -> Tuple[int, MatchReport]:
    """Index and report of the candidate with the highest voxel IoU.

    Ties break toward the earliest candidate (deterministic). Returns
    ``(-1, report)`` with a zero report when ``candidates`` is empty.
    """
    best_index = -1
    best: Optional[MatchReport] = None
    for i, cand in enumerate(candidates):
        report = score(cand, reference, resolution)
        if best is None or report.voxel_iou > best.voxel_iou:
            best_index, best = i, report
    if best is None:
        best = MatchReport(False, "no candidates", 0.0, 0.0, 0.0, float("inf"))
    return best_index, best


# ---------------------------------------------------------------------------
# Geometry metrics
# ---------------------------------------------------------------------------

def _union_box(a, b) -> Optional[Tuple[Vec3, Vec3]]:
    boxes = [x for x in (a, b) if x is not None]
    if not boxes:
        return None
    lo = tuple(min(box[0][i] for box in boxes) for i in range(3))
    hi = tuple(max(box[1][i] for box in boxes) for i in range(3))
    pad = [max((hi[i] - lo[i]) * 0.02, 1e-6) for i in range(3)]
    lo = tuple(lo[i] - pad[i] for i in range(3))
    hi = tuple(hi[i] + pad[i] for i in range(3))
    return (lo, hi)


def voxel_iou(tree_a, tree_b, resolution: int = 16) -> float:
    """IoU of two CSG trees on a shared lattice over the union of their bounds."""
    if resolution < 1:
        raise ValueError("resolution must be >= 1")
    box = _union_box(bounds(tree_a), bounds(tree_b))
    if box is None:
        return 1.0 if tree_a is None and tree_b is None else 0.0
    lo, hi = box
    step = [(hi[i] - lo[i]) / resolution for i in range(3)]
    inter = 0
    union = 0
    for i in range(resolution):
        x = lo[0] + (i + 0.5) * step[0]
        for j in range(resolution):
            y = lo[1] + (j + 0.5) * step[1]
            for k in range(resolution):
                z = lo[2] + (k + 0.5) * step[2]
                p = (x, y, z)
                in_a = contains(tree_a, p)
                in_b = contains(tree_b, p)
                if in_a and in_b:
                    inter += 1
                if in_a or in_b:
                    union += 1
    if union == 0:
        # The lattice missed both solids (they are thin relative to the shared
        # box). Fall back to the bounding-box IoU rather than claiming a match.
        return bbox_iou(tree_a, tree_b)
    return inter / float(union)


def volume_ratio(tree_a, tree_b, resolution: int = 16) -> float:
    """``min/max`` of the two volumes: 1.0 identical, 0.0 when one is empty."""
    va = volume(tree_a, resolution)
    vb = volume(tree_b, resolution)
    if va == 0.0 and vb == 0.0:
        return 1.0
    if va == 0.0 or vb == 0.0:
        return 0.0
    return min(va, vb) / max(va, vb)


def bbox_iou(tree_a, tree_b) -> float:
    """IoU of the two axis-aligned bounding boxes."""
    ba, bb = bounds(tree_a), bounds(tree_b)
    if ba is None and bb is None:
        return 1.0
    if ba is None or bb is None:
        return 0.0
    inter = 1.0
    for i in range(3):
        overlap = min(ba[1][i], bb[1][i]) - max(ba[0][i], bb[0][i])
        if overlap <= 0:
            inter = 0.0
            break
        inter *= overlap
    va = 1.0
    vb = 1.0
    for i in range(3):
        va *= ba[1][i] - ba[0][i]
        vb *= bb[1][i] - bb[0][i]
    denom = va + vb - inter
    if denom <= 0:
        return 1.0 if inter > 0 else 0.0
    return inter / denom


def centroid_offset(tree_a, tree_b) -> float:
    """Distance between the two bounding-box centres (``inf`` if one is empty)."""
    ba, bb = bounds(tree_a), bounds(tree_b)
    if ba is None and bb is None:
        return 0.0
    if ba is None or bb is None:
        return float("inf")
    total = 0.0
    for i in range(3):
        ca = (ba[0][i] + ba[1][i]) / 2.0
        cb = (bb[0][i] + bb[1][i]) / 2.0
        total += (ca - cb) ** 2
    return total ** 0.5


# ---------------------------------------------------------------------------
# Combined scoring
# ---------------------------------------------------------------------------

def _zero(reason: str) -> MatchReport:
    return MatchReport(False, reason, 0.0, 0.0, 0.0, float("inf"))


def score(candidate: str, reference: str, resolution: int = 16) -> MatchReport:
    """Score a candidate program against a reference program.

    The candidate is first put through the static gate; a candidate that fails
    the gate, uses an unevaluable construct, or produces no geometry scores zero
    on every geometry metric and carries the reason why.
    """
    try:
        ref_tree = evaluate_source(reference)
    except (ScadSyntaxError, ScadEvalError) as exc:
        raise ValueError("reference program is not evaluable: %s" % exc)

    issues = check(candidate)
    hard = [i for i in issues if i.severity == "error"]
    if hard:
        return _zero(hard[0].render())
    try:
        cand_tree = evaluate_source(candidate)
    except ScadSyntaxError as exc:
        return _zero("syntax error: %s" % exc)
    except ScadEvalError as exc:
        return _zero("not evaluable: %s" % exc)
    if cand_tree is None:
        return _zero("candidate produces no geometry")

    return MatchReport(
        compiles=True,
        reason="",
        voxel_iou=voxel_iou(cand_tree, ref_tree, resolution),
        volume_ratio=volume_ratio(cand_tree, ref_tree, resolution),
        bbox_iou=bbox_iou(cand_tree, ref_tree),
        centroid_offset=centroid_offset(cand_tree, ref_tree),
    )
