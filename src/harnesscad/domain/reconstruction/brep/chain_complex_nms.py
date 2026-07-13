"""Turning a *probabilistic* B-Rep complex into a definite one (ComplexGen post-process).

ComplexGen's network emits, for a fixed number of queries, a *probabilistic*
chain complex: per-primitive validity probabilities, per-curve closedness
probabilities, and soft incidence ("similarity") matrices.  Its post-process
(``PostProcess/complex_extraction.py``) then makes it definite:

  1. **similarity gating** -- every incidence probability is multiplied by the
     validity of the two cells it joins (and, for curve-corner, by the curve's
     open probability, since a closed curve has no corners);
  2. **NMS** -- duplicate primitives (near-identical geometry *and* identical
     rounded topology) are suppressed: corners by distance, curves by the
     closed-aware curve distance, patches by two-sided Chamfer;
  3. **duplicate-corner merging** -- when a curve claims more than 2 corners, the
     two corner columns whose incidence signatures are most alike are duplicates;
     the later one is suppressed;
  4. **extraction** -- threshold what survives into a
     :class:`reconstruction.complexgen_chain_complex.ChainComplex`.

The paper's step between 3 and 4 is a global ILP (Gurobi/MOSEK) maximising
likelihood under the structural constraints; that solver is external.  What is
implemented here is the deterministic, solver-free part, plus
:func:`repair_extraction`, a greedy fallback that drops the incidences the
constraints forbid so the result is at least structurally checkable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harnesscad.eval.bench.geometry.complex_matching import chamfer_distance, sampled_curve_distance
from harnesscad.domain.reconstruction.brep.chain_complex import (
    ChainComplex, Curve, Patch, make_complex)

Point = tuple[float, float, float]


@dataclass
class ProbabilisticComplex:
    """The network's raw output: cells with validity probabilities + soft incidences."""
    corners: list[Point]
    corner_valid: list[float]
    curves: list[Curve]
    curve_valid: list[float]
    curve_closed_prob: list[float]
    patches: list[Patch]
    patch_valid: list[float]
    curve_corner_prob: list[list[float]]      # n_curves x n_corners
    patch_curve_prob: list[list[float]]       # n_patches x n_curves
    suppressed_corners: set = field(default_factory=set)
    suppressed_curves: set = field(default_factory=set)
    suppressed_patches: set = field(default_factory=set)

    def __post_init__(self):
        if not (len(self.corners) == len(self.corner_valid)):
            raise ValueError("corner_valid must match corners")
        if not (len(self.curves) == len(self.curve_valid) == len(self.curve_closed_prob)):
            raise ValueError("curve_valid / curve_closed_prob must match curves")
        if len(self.patches) != len(self.patch_valid):
            raise ValueError("patch_valid must match patches")
        if len(self.curve_corner_prob) != len(self.curves):
            raise ValueError("curve_corner_prob must have one row per curve")
        if len(self.patch_curve_prob) != len(self.patches):
            raise ValueError("patch_curve_prob must have one row per patch")


def gate_similarities(pc: ProbabilisticComplex) -> ProbabilisticComplex:
    """Multiply each incidence probability by the validity of the cells it joins.

    Curve-corner entries are additionally scaled by the curve's *open* probability
    (``1 - closed_prob``).  Suppressed cells are treated as invalid.  Returns a new
    object; the input is not modified.
    """
    def cv(i):
        return 0.0 if i in pc.suppressed_curves else pc.curve_valid[i]

    def kv(j):
        return 0.0 if j in pc.suppressed_corners else pc.corner_valid[j]

    def pv(k):
        return 0.0 if k in pc.suppressed_patches else pc.patch_valid[k]

    ev = [[pc.curve_corner_prob[i][j] * cv(i) * kv(j) * (1.0 - pc.curve_closed_prob[i])
           for j in range(len(pc.corners))] for i in range(len(pc.curves))]
    fe = [[pc.patch_curve_prob[k][i] * pv(k) * cv(i)
           for i in range(len(pc.curves))] for k in range(len(pc.patches))]
    return ProbabilisticComplex(
        corners=list(pc.corners), corner_valid=list(pc.corner_valid),
        curves=list(pc.curves), curve_valid=list(pc.curve_valid),
        curve_closed_prob=list(pc.curve_closed_prob),
        patches=list(pc.patches), patch_valid=list(pc.patch_valid),
        curve_corner_prob=ev, patch_curve_prob=fe,
        suppressed_corners=set(pc.suppressed_corners),
        suppressed_curves=set(pc.suppressed_curves),
        suppressed_patches=set(pc.suppressed_patches))


# --------------------------------------------------------------------------- #
# non-maximum suppression
# --------------------------------------------------------------------------- #
def _round_row(row, threshold=0.5):
    return tuple(1 if v > threshold else 0 for v in row)


def _distance(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5


def nms_corners(pc: ProbabilisticComplex, valid_th: float = 0.5,
                dist_th: float = 0.05) -> set:
    """Suppress a later corner that duplicates an earlier one (distance < ``dist_th``
    and identical rounded curve-incidence column)."""
    live = [j for j in range(len(pc.corners))
            if pc.corner_valid[j] > valid_th and j not in pc.suppressed_corners]
    cols = [tuple(_round_row([pc.curve_corner_prob[i][j]])[0]
                  for i in range(len(pc.curves))) for j in range(len(pc.corners))]
    out = set()
    for a_idx, a in enumerate(live):
        if a in out:
            continue
        for b in live[a_idx + 1:]:
            if b in out:
                continue
            if _distance(pc.corners[a], pc.corners[b]) < dist_th and cols[a] == cols[b]:
                out.add(b)
    return out


def nms_curves(pc: ProbabilisticComplex, valid_th: float = 0.5,
               dist_th: float = 0.05) -> set:
    """Suppress duplicate curves (closed-aware curve distance < ``dist_th`` and the
    same rounded corner incidences)."""
    live = [i for i in range(len(pc.curves))
            if pc.curve_valid[i] > valid_th and i not in pc.suppressed_curves]
    out = set()
    for a_idx, a in enumerate(live):
        if a in out:
            continue
        for b in live[a_idx + 1:]:
            if b in out:
                continue
            if len(pc.curves[a].points) != len(pc.curves[b].points):
                continue
            if sampled_curve_distance(pc.curves[a], pc.curves[b]) >= dist_th:
                continue
            if _round_row(pc.curve_corner_prob[a]) == _round_row(pc.curve_corner_prob[b]):
                out.add(b)
    return out


def nms_patches(pc: ProbabilisticComplex, valid_th: float = 0.5,
                dist_th: float = 0.05) -> set:
    """Suppress duplicate patches (both one-sided Chamfer distances < ``dist_th``
    and the same rounded curve incidences)."""
    live = [k for k in range(len(pc.patches))
            if pc.patch_valid[k] > valid_th and k not in pc.suppressed_patches]
    out = set()
    for a_idx, a in enumerate(live):
        if a in out:
            continue
        for b in live[a_idx + 1:]:
            if b in out:
                continue
            pa = pc.patches[a].points
            pb = pc.patches[b].points
            if not pa or not pb:
                continue
            if chamfer_distance(pa, pb, "x_to_y") >= dist_th:
                continue
            if chamfer_distance(pa, pb, "y_to_x") >= dist_th:
                continue
            if _round_row(pc.patch_curve_prob[a]) == _round_row(pc.patch_curve_prob[b]):
                out.add(b)
    return out


def nms(pc: ProbabilisticComplex, valid_th: float = 0.5,
        dist_th: float = 0.05) -> ProbabilisticComplex:
    """Run corner / curve / patch NMS and return a copy with the duplicates suppressed."""
    out = gate_similarities(pc)
    out.suppressed_corners |= nms_corners(pc, valid_th, dist_th)
    out.suppressed_curves |= nms_curves(pc, valid_th, dist_th)
    out.suppressed_patches |= nms_patches(pc, valid_th, dist_th)
    return gate_similarities(out)


def merge_duplicated_corners(pc: ProbabilisticComplex, threshold: float = 0.5,
                             max_iterations: int = 100) -> set:
    """Suppress duplicate corners revealed by an over-connected curve.

    A curve claiming more than 2 corners means two of them are the same vertex.
    The pair whose incidence columns differ least (sum of squared differences over
    all curves) is the duplicate; the higher index is dropped.  Repeats until no
    curve claims more than 2 corners.
    """
    ev = [list(row) for row in pc.curve_corner_prob]
    n_corners = len(pc.corners)
    dropped: set = set()
    for _ in range(max_iterations):
        conflict = None
        for i, row in enumerate(ev):
            claimed = [j for j in range(n_corners) if row[j] > threshold]
            if len(claimed) > 2:
                conflict = claimed
                break
        if conflict is None:
            break
        best_pair = None
        best_diff = float("inf")
        for a_idx in range(len(conflict)):
            for b_idx in range(a_idx + 1, len(conflict)):
                a = conflict[a_idx]
                b = conflict[b_idx]
                diff = sum((ev[i][a] - ev[i][b]) ** 2 for i in range(len(ev)))
                if diff < best_diff:
                    best_diff = diff
                    best_pair = (a, b)
        drop = max(best_pair)
        dropped.add(drop)
        for i in range(len(ev)):
            ev[i][drop] = 0.0
    return dropped


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Extraction:
    complex: ChainComplex
    corner_ids: tuple[int, ...]      # original index of every kept corner
    curve_ids: tuple[int, ...]
    patch_ids: tuple[int, ...]


def extract(pc: ProbabilisticComplex, valid_th: float = 0.5,
            incidence_th: float = 0.5) -> Extraction:
    """Threshold a (gated, NMS'd) probabilistic complex into a definite one."""
    gated = gate_similarities(pc)
    corner_ids = tuple(j for j in range(len(pc.corners))
                       if pc.corner_valid[j] > valid_th and j not in pc.suppressed_corners)
    curve_ids = tuple(i for i in range(len(pc.curves))
                      if pc.curve_valid[i] > valid_th and i not in pc.suppressed_curves)
    patch_ids = tuple(k for k in range(len(pc.patches))
                      if pc.patch_valid[k] > valid_th and k not in pc.suppressed_patches)

    curves = []
    for i in curve_ids:
        curve = pc.curves[i]
        curves.append(Curve(curve.points, pc.curve_closed_prob[i] > 0.5))
    ev = [[1 if gated.curve_corner_prob[i][j] > incidence_th else 0 for j in corner_ids]
          for i in curve_ids]
    fe = [[1 if gated.patch_curve_prob[k][i] > incidence_th else 0 for i in curve_ids]
          for k in patch_ids]
    cx = make_complex([pc.corners[j] for j in corner_ids], curves,
                      [Patch(pc.patches[k].points) for k in patch_ids], ev, fe)
    return Extraction(cx, corner_ids, curve_ids, patch_ids)


def repair_extraction(cx: ChainComplex) -> ChainComplex:
    """Greedy structural repair of an extracted complex (the solver-free fallback).

    * a closed curve loses all its corners;
    * an open curve keeps only its 2 nearest corners (by endpoint distance);
    * a curve claimed by more than 2 patches keeps the 2 whose sample clouds are
      nearest to it.
    """
    ev = [list(row) for row in cx.curve_corner]
    fe = [list(row) for row in cx.patch_curve]

    for i, curve in enumerate(cx.curves):
        claimed = [j for j in range(cx.n_corners) if ev[i][j]]
        if curve.closed:
            for j in claimed:
                ev[i][j] = 0
            continue
        if len(claimed) <= 2:
            continue
        e1, e2 = curve.endpoints()
        ranked = sorted(claimed,
                        key=lambda j: (min(_distance(e1, cx.corners[j]),
                                           _distance(e2, cx.corners[j])), j))
        for j in ranked[2:]:
            ev[i][j] = 0

    for i in range(cx.n_curves):
        claimed = [k for k in range(cx.n_patches) if fe[k][i]]
        if len(claimed) <= 2:
            continue
        curve_pts = cx.curves[i].points
        ranked = sorted(claimed,
                        key=lambda k: (chamfer_distance(curve_pts, cx.patches[k].points,
                                                        "x_to_y")
                                       if cx.patches[k].points else float("inf"), k))
        for k in ranked[2:]:
            fe[k][i] = 0

    return make_complex(cx.corners, cx.curves, cx.patches, ev, fe)
