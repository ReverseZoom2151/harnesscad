"""Set-based DAVINCI evaluation metrics for constrained CAD sketch inference.

DAVINCI (Karadeniz et al., 2024, Sec. 5) is a **set-based** predictor: it emits an
unordered set of primitive slots, so before any metric can be computed a one-to-one
correspondence between predicted and ground-truth primitives must be recovered by
**optimal bipartite (Hungarian) matching** (Sec. 3.3, following DETR). On top of
that correspondence the paper reports:

* **Accuracy** -- token-level accuracy w.r.t. the ground-truth token sequence
  (Sec. 5, Evaluation);
* **Primitive F1 (PF1)** -- a predicted primitive is a true positive iff its type is
  correct *and every parameter is within 5 quantization units* of the ground truth;
* **Constraint F1 (CF1)** -- a constraint is a true positive only if *all involved
  primitives are themselves true positives* and the (index-remapped, undirected)
  constraint exists in the ground truth;
* **bidirectional Chamfer Distance (CD)** -- points sampled uniformly on predicted
  and ground-truth primitives.

Everything here is deterministic and stdlib-only, including a self-contained
Hungarian solver (the corpus had none). Primitives are the 8-token blocks produced
by ``ingest.davinci_primitive_tokens``; constraints are ``(kind, i, si, j, sj)``
tuples as in ``ingest.davinci_cpt``. Distinct from ``bench.sketch_metrics`` (which
does continuous-tolerance F1 with a greedy match and no token accuracy / Chamfer /
Hungarian).
"""

from __future__ import annotations

import math

from harnesscad.io.ingest.primitive_tokens import N_TOKENS, TOKEN_TYPES, dequantize, PARAM_COUNT

PRIMITIVE_TOLERANCE = 5     # "within 5 quantization units" (Sec. 5, PF1)


# --------------------------------------------------------------------------- #
# Hungarian (Kuhn-Munkres) optimal assignment, minimisation, stdlib-only.
# --------------------------------------------------------------------------- #
def hungarian(cost) -> list:
    """Minimum-cost assignment of rows to columns.

    ``cost`` is an ``n x m`` matrix (``n <= m``). Returns a list ``a`` of length
    ``n`` with ``a[row] = col``. Uses the O(n^2 m) shortest-augmenting-path form.
    """
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])
    if n > m:
        raise ValueError("hungarian requires n <= m; pad the cost matrix")
    inf = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)      # p[col] = row assigned to col (1-indexed)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = -1
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
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
    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


# --------------------------------------------------------------------------- #
# Primitive correspondence.
# --------------------------------------------------------------------------- #
def token_cost(pred_tokens, gt_tokens) -> int:
    """Number of differing tokens between two 8-token primitive blocks."""
    a, b = tuple(pred_tokens), tuple(gt_tokens)
    if len(a) != N_TOKENS or len(b) != N_TOKENS:
        raise ValueError("primitives must be 8-token blocks")
    return sum(1 for x, y in zip(a, b) if x != y)


def match_primitives(pred, gt) -> dict:
    """Hungarian correspondence ``pred_index -> gt_index`` on token cost.

    Pads the cost matrix to square so predicted/ground-truth counts may differ; only
    real-to-real assignments are returned.
    """
    pred, gt = list(pred), list(gt)
    n, m = len(pred), len(gt)
    if n == 0 or m == 0:
        return {}
    size = max(n, m)
    big = N_TOKENS + 1
    cost = [[big] * size for _ in range(size)]
    for i in range(n):
        for j in range(m):
            cost[i][j] = token_cost(pred[i], gt[j])
    assignment = hungarian(cost)
    return {i: assignment[i] for i in range(n) if assignment[i] < m}


# --------------------------------------------------------------------------- #
# Metrics.
# --------------------------------------------------------------------------- #
def token_accuracy(pred, gt, mapping=None) -> float:
    """Token-level accuracy w.r.t. the ground-truth sequence (Sec. 5)."""
    pred, gt = list(pred), list(gt)
    if not gt:
        return 1.0 if not pred else 0.0
    if mapping is None:
        mapping = match_primitives(pred, gt)
    inv = {gj: pi for pi, gj in mapping.items()}
    correct = 0
    for gj, g in enumerate(gt):
        pi = inv.get(gj)
        if pi is None:
            continue
        correct += sum(1 for x, y in zip(pred[pi], g) if x == y)
    return correct / (N_TOKENS * len(gt))


def _params_within(pred_tokens, gt_tokens, tol) -> bool:
    ptype = TOKEN_TYPES.get(tuple(gt_tokens)[0])
    if ptype is None:
        return False
    used = PARAM_COUNT[ptype]
    return all(abs(pred_tokens[1 + k] - gt_tokens[1 + k]) <= tol for k in range(used))


def primitive_true_positives(pred, gt, mapping=None, *, tol=PRIMITIVE_TOLERANCE) -> dict:
    """Map ``pred_index -> gt_index`` for matched pairs that are primitive TPs.

    A pair is a true positive iff the type token is correct and every used parameter
    is within ``tol`` quantization units (Sec. 5, PF1).
    """
    pred, gt = list(pred), list(gt)
    if mapping is None:
        mapping = match_primitives(pred, gt)
    tps = {}
    for pi, gj in mapping.items():
        pt, gtt = tuple(pred[pi]), tuple(gt[gj])
        if pt[0] == gtt[0] and _params_within(pt, gtt, tol):
            tps[pi] = gj
    return tps


def _prf(tp, n_pred, n_gt) -> dict:
    precision = tp / n_pred if n_pred else (1.0 if n_gt == 0 else 0.0)
    recall = tp / n_gt if n_gt else (1.0 if n_pred == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp}


def primitive_f1(pred, gt, mapping=None, *, tol=PRIMITIVE_TOLERANCE) -> dict:
    """Primitive F1 (PF1): type correct and all params within ``tol`` units."""
    pred, gt = list(pred), list(gt)
    tps = primitive_true_positives(pred, gt, mapping, tol=tol)
    return _prf(len(tps), len(pred), len(gt))


def _canon(kind, i, si, j, sj) -> tuple:
    a, b = (i, si), (j, sj)
    return (kind, a, b) if a <= b else (kind, b, a)


def constraint_f1(pred_constraints, gt_constraints, prim_tps) -> dict:
    """Constraint F1 (CF1).

    ``prim_tps`` maps ``pred_index -> gt_index`` for primitive true positives
    (from :func:`primitive_true_positives`). A predicted constraint is a true
    positive iff both endpoints are primitive TPs and, after remapping its indices to
    ground-truth space, the undirected constraint exists in the ground truth.
    """
    gt_set = {_canon(*c) for c in gt_constraints}
    tp = 0
    for (kind, i, si, j, sj) in pred_constraints:
        if i in prim_tps and j in prim_tps:
            remapped = _canon(kind, prim_tps[i], si, prim_tps[j], sj)
            if remapped in gt_set:
                tp += 1
    return _prf(tp, len(list(pred_constraints)), len(gt_set))


# --------------------------------------------------------------------------- #
# Chamfer distance over primitive samples.
# --------------------------------------------------------------------------- #
def sample_primitive(tokens, samples: int = 8) -> tuple:
    """Uniformly sample ``(x, y)`` points on a primitive's geometry.

    Lines are sampled along the segment; arcs along the ``start->mid->end``
    polyline; circles around the circumference; points return themselves. ``none``
    slots return no points.
    """
    tok = tuple(tokens)
    ptype = TOKEN_TYPES.get(tok[0])
    coords = [dequantize(t) for t in tok[1:7]]
    if ptype == "none":
        return ()
    if ptype == "point":
        return ((coords[0], coords[1]),)
    if ptype == "line":
        xs, ys, xe, ye = coords[:4]
        return tuple((xs + (xe - xs) * t / (samples - 1),
                      ys + (ye - ys) * t / (samples - 1))
                     for t in range(samples)) if samples > 1 else ((xs, ys),)
    if ptype == "arc":
        xs, ys, xm, ym, xe, ye = coords[:6]
        half = max(1, samples // 2)
        pts = [(xs + (xm - xs) * t / half, ys + (ym - ys) * t / half)
               for t in range(half)]
        pts += [(xm + (xe - xm) * t / half, ym + (ye - ym) * t / half)
                for t in range(half + 1)]
        return tuple(pts)
    if ptype == "circle":
        xc, yc, r = coords[:3]
        return tuple((xc + r * math.cos(2 * math.pi * t / samples),
                      yc + r * math.sin(2 * math.pi * t / samples))
                     for t in range(samples))
    raise ValueError(f"unknown primitive type token: {tok[0]}")


def _points(prims, samples) -> list:
    return [pt for p in prims for pt in sample_primitive(p, samples)]


def chamfer_distance(pred, gt, *, samples: int = 8) -> float:
    """Bidirectional Chamfer distance between two sketches' sampled points."""
    a = _points(pred, samples)
    b = _points(gt, samples)
    if not a and not b:
        return 0.0
    if not a or not b:
        return float("inf")

    def one_way(src, dst):
        return sum(min(math.dist(p, q) for q in dst) for p in src) / len(src)

    return one_way(a, b) + one_way(b, a)


def evaluate(pred, gt, pred_constraints=(), gt_constraints=(), *,
             tol=PRIMITIVE_TOLERANCE, samples: int = 8) -> dict:
    """Full DAVINCI evaluation bundle (Acc, PF1, CF1, CD) under one matching."""
    pred, gt = list(pred), list(gt)
    mapping = match_primitives(pred, gt)
    tps = primitive_true_positives(pred, gt, mapping, tol=tol)
    return {
        "accuracy": token_accuracy(pred, gt, mapping),
        "primitive": _prf(len(tps), len(pred), len(gt)),
        "constraint": constraint_f1(pred_constraints, gt_constraints, tps),
        "chamfer": chamfer_distance(pred, gt, samples=samples),
        "mapping": mapping,
    }
