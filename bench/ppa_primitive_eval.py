"""Primitive-prediction evaluation protocol for PPA CAD-sketch analysis
(Wang et al., "Parametric Primitive Analysis of CAD Sketches with Vision
Transformer", IEEE T-II 2024, Sec. IV-A).

Because a predicted primitive set ``P_hat`` has no explicit correspondence to the
ground-truth set ``P`` (both are unordered), the paper builds a cost matrix and
solves it with Hungarian matching to obtain an index map ``sigma_p`` (Eq. 5-6),
then reports three accuracies over the matched pairs:

  * **Primitive Type Accuracy** ``ACC_ptype`` (Eq. 21) -- fraction of matched pairs
    with equal primitive type;
  * **Boolean Flag Accuracy** ``ACC_flag`` (Eq. 22) -- fraction with equal flag;
  * **Primitive Parameters Accuracy** ``ACC_ppar`` (Eq. 23) -- fraction whose
    quantised coordinate parameters agree within a threshold ``eta`` levels (paper:
    ``eta = 1`` out of 64), following DeepCAD's tolerance rule.

It also exposes a **Chamfer distance** between the predicted and GT sketch (Table
XV "prediction error"): each primitive is sampled to points (via
:func:`reconstruction.ppa_primitive.sample_primitive`) and the two clouds compared.

The Hungarian solver is reused from :mod:`reconstruction.gaussiancad_emd` and the
Chamfer from :mod:`bench.lasdiff_sketch_metrics`; this module contributes the
sketch-specific cost function, the padded/rectangular matching, and the paper's three
accuracy metrics + Chamfer wrapper.
"""

from __future__ import annotations

import math

from reconstruction import ppa_primitive as pp
from reconstruction import ppa_quantization as pq
from reconstruction.gaussiancad_emd import hungarian
from bench.lasdiff_sketch_metrics import chamfer_2d

# Default cost weights (paper's w_pt, w_f, w_pp; Eq. 6). Parameter term is scaled up.
W_TYPE = 1.0
W_FLAG = 1.0
W_PARAM = 5.0

_BIG = 1e6  # cost for matching to a padding (non-existent) slot


def _param_distance(a: pp.Primitive, b: pp.Primitive) -> float:
    """Mean absolute distance over the union of meaningful parameter slots.

    Different types compare over all 7 slots (their padding differs, which correctly
    inflates the distance). Same-type primitives compare only their meaningful slots.
    """
    if a.ptype == b.ptype:
        if a.ptype == pp.CIRCLE:
            idxs = (0, 1, 6)
        else:
            idxs = tuple(range(a.meaningful))
    else:
        idxs = tuple(range(pp.PARAM_SLOTS))
    return sum(abs(a.params[i] - b.params[i]) for i in idxs) / len(idxs)


def match_cost(gt: pp.Primitive, pred: pp.Primitive, *, w_type=W_TYPE, w_flag=W_FLAG,
               w_param=W_PARAM) -> float:
    """L_match(P_i, P_hat_j): weighted type + flag + parameter cost (Eq. 6)."""
    c_type = 0.0 if gt.ptype == pred.ptype else 1.0
    c_flag = 0.0 if gt.flag == pred.flag else 1.0
    c_param = _param_distance(gt, pred)
    return w_type * c_type + w_flag * c_flag + w_param * c_param


def match_primitives(gt_sketch: pp.Sketch, pred_sketch: pp.Sketch, **weights):
    """Hungarian match GT -> predicted primitives; return ``sigma`` (dict gt_i->pred_j).

    Pads the cost matrix to a square with a large constant so unequal set sizes are
    handled (unmatched GT / predictions map to padding slots and are dropped from
    ``sigma``). Deterministic (fixed cost function + O(n^3) Kuhn-Munkres).
    """
    gt = list(gt_sketch)
    pred = list(pred_sketch)
    kp, npred = len(gt), len(pred)
    if kp == 0:
        return {}
    n = max(kp, npred)
    cost = [[_BIG] * n for _ in range(n)]
    for i in range(kp):
        for j in range(npred):
            cost[i][j] = match_cost(gt[i], pred[j], **weights)
    assign = hungarian(cost)  # assign[i] = column matched to row i
    sigma = {}
    for i in range(kp):
        j = assign[i]
        if j < npred:
            sigma[i] = j
    return sigma


def _matched_pairs(gt_sketch, pred_sketch, sigma):
    gt = list(gt_sketch)
    pred = list(pred_sketch)
    return [(gt[i], pred[j]) for i, j in sigma.items()]


def primitive_type_accuracy(gt_sketch, pred_sketch, sigma=None, **weights) -> float:
    """ACC_ptype (Eq. 21): fraction of matched pairs with equal primitive type.

    Denominator is ``K_p`` (the number of GT primitives), matching the paper.
    """
    kp = len(gt_sketch)
    if kp == 0:
        return 1.0
    if sigma is None:
        sigma = match_primitives(gt_sketch, pred_sketch, **weights)
    hits = sum(1 for g, p in _matched_pairs(gt_sketch, pred_sketch, sigma)
               if g.ptype == p.ptype)
    return hits / kp


def boolean_flag_accuracy(gt_sketch, pred_sketch, sigma=None, **weights) -> float:
    """ACC_flag (Eq. 22): fraction of matched pairs with equal boolean flag."""
    kp = len(gt_sketch)
    if kp == 0:
        return 1.0
    if sigma is None:
        sigma = match_primitives(gt_sketch, pred_sketch, **weights)
    hits = sum(1 for g, p in _matched_pairs(gt_sketch, pred_sketch, sigma)
               if g.flag == p.flag)
    return hits / kp


def parameter_accuracy(gt_sketch, pred_sketch, sigma=None, *, eta=1,
                       bits=pq.DEFAULT_BITS, **weights) -> float:
    """ACC_ppar (Eq. 23): fraction of matched pairs whose quantised meaningful
    coordinates all agree within ``eta`` levels.

    Both sketches are assumed already normalised to ``[0, 1]`` (see
    :func:`reconstruction.ppa_quantization.normalize_sketch`); coordinates are
    quantised to ``bits``-bit integer levels and compared with DeepCAD's tolerance
    ``eta`` (paper default 1 out of 64). A type mismatch counts as a parameter miss.
    """
    kp = len(gt_sketch)
    if kp == 0:
        return 1.0
    if sigma is None:
        sigma = match_primitives(gt_sketch, pred_sketch, **weights)
    hits = 0
    for g, p in _matched_pairs(gt_sketch, pred_sketch, sigma):
        if g.ptype != p.ptype:
            continue
        gq = pq.quantize_primitive(g, bits)
        pqn = pq.quantize_primitive(p, bits)
        if g.ptype == pp.CIRCLE:
            idxs = (0, 1, 6)
        else:
            idxs = tuple(range(g.meaningful))
        if all(abs(gq[i] - pqn[i]) <= eta for i in idxs):
            hits += 1
    return hits / kp


def sketch_chamfer(gt_sketch, pred_sketch, *, samples=16) -> float:
    """2D Chamfer distance between the predicted and GT sketch point clouds.

    Each primitive is sampled to ``samples`` points and the two aggregate clouds are
    compared with the symmetric Chamfer distance (Table XV "prediction error"). Raises
    if either sketch is empty.
    """
    def cloud(sketch):
        pts = []
        for prim in sketch:
            pts.extend(pp.sample_primitive(prim, samples))
        return pts
    a, b = cloud(gt_sketch), cloud(pred_sketch)
    if not a or not b:
        raise ValueError("both sketches must contain at least one primitive")
    return chamfer_2d(a, b)


def evaluate(gt_sketch, pred_sketch, *, eta=1, bits=pq.DEFAULT_BITS, samples=16,
             **weights) -> dict:
    """Full protocol: match once, then report all three accuracies + Chamfer."""
    sigma = match_primitives(gt_sketch, pred_sketch, **weights)
    result = {
        "matched": len(sigma),
        "num_gt": len(gt_sketch),
        "num_pred": len(pred_sketch),
        "ACC_ptype": primitive_type_accuracy(gt_sketch, pred_sketch, sigma),
        "ACC_flag": boolean_flag_accuracy(gt_sketch, pred_sketch, sigma),
        "ACC_ppar": parameter_accuracy(gt_sketch, pred_sketch, sigma, eta=eta,
                                       bits=bits),
    }
    try:
        result["chamfer"] = sketch_chamfer(gt_sketch, pred_sketch, samples=samples)
    except ValueError:
        result["chamfer"] = math.inf
    return result
