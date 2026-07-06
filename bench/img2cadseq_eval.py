"""Multi-level evaluation framework for image-to-CAD-sequence prediction
(Li & Sha, *Image2CADSeq*, 2024, Sec. 4.5 & Table 4(b)).

The paper argues that assessing predicted CAD *programs* against the ground
truth has no established metric, and introduces a system organised into three
**hierarchies** and two **layers** over the ``Nc x 7`` feature-matrix
representation ``P = [o_1, ..., o_Nc]`` where ``o_i = (t_i, p_i)`` (op type +
parameters):

* **H1 sequence evaluation** — accuracy over the whole program and its op-type
  order: ``ACP`` (accuracy of CAD programs, Eq. 1), ``ASOT`` (accuracy of the
  op-type sequence, Eq. 2), ``EDSOT`` (Levenshtein edit distance of the op-type
  sequence, Eq. 3).
* **H2 sequence-based op-type evaluation** — per-position within the sequence:
  ``AOT`` (accuracy of op types, order-aware, Eq. 4) and ``AP1`` (accuracy of
  parameters given a correctly predicted op type, order-aware, Eq. 5).
* **H3 set-based op-type evaluation** — order-agnostic multiset similarity
  ``MSOT`` via the Tanimoto coefficient (Eq. 6) and cosine similarity (Eq. 7),
  plus ``AP2`` (parameter accuracy without order).

**Layers**: L1 = operation-type layer (``AOT``); L2 = parameter layer
(``AP1``/``AP2``), evaluated only where the op type is correct.

Parameter comparison uses an 8-bit tolerance ``eta in [0, 255]``: a quantised
prediction ``z_hat`` matches ground truth ``z`` iff ``|z_hat - z| <= eta`` and
``z_hat in [0, 255]``. A closed-form random-guess baseline for ``AP1`` is also
provided (Eq. 8).

Inputs are quantised feature matrices as produced by
:mod:`reconstruction.img2cadseq_gallery_dsl` (lists of 7-tuples of ints, with
``-1`` marking unused parameter slots). Pure and deterministic.
"""

from __future__ import annotations

import math

# Feature-vector layout indices (mirrors gallery DSL; kept local to avoid an
# import-time coupling on the reconstruction package).
_T = 0                       # op-type column P[:, 0]
_PARAM_SLOTS = (2, 3, 4, 5, 6)  # x, y, alpha, r, d
UNUSED_LEVEL = -1


# --- helpers ---------------------------------------------------------------
def op_types(matrix) -> tuple[int, ...]:
    """The op-type sequence ``P[:, 0]`` (all rows, including SOP/EOP markers)."""
    return tuple(int(row[_T]) for row in matrix)


def levenshtein(a, b) -> int:
    """Levenshtein edit distance between sequences ``a`` and ``b`` (paper Eq. 3).

    Dynamic-programming distance with unit insertion/deletion/substitution
    costs, matching the paper's ``L(a, b) = M[m, n]`` formulation.
    """
    a, b = list(a), list(b)
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


# --- H1: sequence evaluation ----------------------------------------------
def _program_equal(pred, gt, eta: int) -> bool:
    """Whole-program equality: op types identical and every used parameter
    within tolerance ``eta`` (op types are compared exactly, no tolerance)."""
    if len(pred) != len(gt):
        return False
    for pr, gr in zip(pred, gt):
        if int(pr[_T]) != int(gr[_T]):
            return False
        for k in _PARAM_SLOTS:
            if gr[k] == UNUSED_LEVEL and pr[k] == UNUSED_LEVEL:
                continue
            if gr[k] == UNUSED_LEVEL or pr[k] == UNUSED_LEVEL:
                return False
            if not (0 <= pr[k] <= 255 and abs(pr[k] - gr[k]) <= eta):
                return False
    return True


def accuracy_cad_programs(preds, gts, eta: int = 3) -> float:
    """ACP (Eq. 1): fraction of programs matching ground truth exactly."""
    if not gts:
        return 1.0
    return sum(_program_equal(p, g, eta) for p, g in zip(preds, gts)) / len(gts)


def accuracy_seq_op_types(preds, gts) -> float:
    """ASOT (Eq. 2): fraction of programs whose op-type sequence matches exactly."""
    if not gts:
        return 1.0
    return sum(op_types(p) == op_types(g) for p, g in zip(preds, gts)) / len(gts)


def edit_distance_seq_op_types(preds, gts) -> float:
    """EDSOT (Eq. 3): mean Levenshtein distance over op-type sequences
    (lower is better)."""
    if not gts:
        return 0.0
    return sum(levenshtein(op_types(p), op_types(g))
               for p, g in zip(preds, gts)) / len(gts)


# --- H2 / L1: op-type accuracy ---------------------------------------------
def accuracy_op_types(preds, gts) -> float:
    """AOT (Eq. 4): order-aware proportion of correctly predicted op types.

    Compared position-by-position over ``l_i = min(len(pred), len(gt))`` rows;
    normalised by the total number of ground-truth op types.
    """
    num = den = 0
    for p, g in zip(preds, gts):
        pt, gt = op_types(p), op_types(g)
        li = min(len(pt), len(gt))
        num += sum(pt[j] == gt[j] for j in range(li))
        den += len(gt)
    return num / den if den else 1.0


# --- H2 / L2: ordered parameter accuracy (AP1) -----------------------------
def _param_slots_present(row):
    return [k for k in _PARAM_SLOTS if row[k] != UNUSED_LEVEL]


def accuracy_parameter_ordered(preds, gts, eta: int = 3) -> float:
    """AP1 (Eq. 5): order-aware parameter accuracy, scored only where the op
    type is correctly predicted (L2 sits beneath L1).

    A parameter counts as correct when (c1) the op type at that position
    matches, (c2) ``0 <= z_hat <= 255``, and (c3) ``|z - z_hat| <= eta``.
    """
    num = den = 0
    for p, g in zip(preds, gts):
        pt, gt = op_types(p), op_types(g)
        li = min(len(pt), len(gt))
        for j in range(li):
            for k in _PARAM_SLOTS:
                if g[j][k] == UNUSED_LEVEL:
                    continue
                den += 1
                if pt[j] != gt[j]:            # c1
                    continue
                zhat = p[j][k]
                if 0 <= zhat <= 255 and abs(g[j][k] - zhat) <= eta:  # c2, c3
                    num += 1
    return num / den if den else 1.0


# --- H3: set-based op-type similarity (MSOT) & AP2 --------------------------
def _multiset_counts(seq, universe) -> list[int]:
    counts = {u: 0 for u in universe}
    for s in seq:
        if s in counts:
            counts[s] += 1
    return [counts[u] for u in universe]


def tanimoto(a, b) -> float:
    """Tanimoto coefficient (Eq. 6): ``a.b / (|a|^2 + |b|^2 - a.b)``."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a)
    nb = sum(y * y for y in b)
    denom = na + nb - dot
    return dot / denom if denom else 1.0


def cosine_similarity(a, b) -> float:
    """Cosine similarity (Eq. 7): ``a.b / (||a|| ||b||)``."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0 if na == nb else 0.0
    return dot / (na * nb)


def multiset_similarity_op_types(preds, gts, universe=range(7)) -> dict:
    """MSOT: mean Tanimoto (``TC``) and cosine (``CS``) similarity of op-type
    multisets, ignoring order (paper Sec. 4.5, H3)."""
    universe = tuple(universe)
    tcs, css = [], []
    for p, g in zip(preds, gts):
        va = _multiset_counts(op_types(p), universe)
        vb = _multiset_counts(op_types(g), universe)
        tcs.append(tanimoto(va, vb))
        css.append(cosine_similarity(va, vb))
    n = len(tcs)
    return {"tc": sum(tcs) / n if n else 1.0,
            "cs": sum(css) / n if n else 1.0}


def accuracy_parameter_unordered(preds, gts, eta: int = 3) -> float:
    """AP2 (Eq. 5, order-agnostic): parameter accuracy where op types are
    matched by first-available instance rather than by position.

    For each ground-truth op, the first not-yet-consumed predicted op of the
    same type is used for the parameter comparison (paper's caveat: exact only
    when op types are not repeated).
    """
    num = den = 0
    for p, g in zip(preds, gts):
        available = list(range(len(p)))
        for grow in g:
            gt_t = int(grow[_T])
            match = None
            for idx in available:
                if int(p[idx][_T]) == gt_t:
                    match = idx
                    break
            if match is not None:
                available.remove(match)
            prow = p[match] if match is not None else None
            for k in _PARAM_SLOTS:
                if grow[k] == UNUSED_LEVEL:
                    continue
                den += 1
                if prow is None:
                    continue
                zhat = prow[k]
                if 0 <= zhat <= 255 and abs(grow[k] - zhat) <= eta:
                    num += 1
    return num / den if den else 1.0


# --- random-guess baseline for AP1 (paper Eq. 8) ---------------------------
def random_baseline_ap1(eta: int) -> float:
    """Closed-form AP1 for a random parameter guesser, ignoring the discrete
    sketch-plane parameter (paper Eq. 8, simplified form)::

        AP1 = (-eta^2 + 511*eta + 256) / 65536
    """
    return (-eta * eta + 511 * eta + 256) / 65536.0


# --- top-level report ------------------------------------------------------
def evaluate(preds, gts, eta: int = 3) -> dict:
    """Full multi-level report organised by hierarchy and layer."""
    if len(preds) != len(gts):
        raise ValueError("preds and gts must be the same length")
    msot = multiset_similarity_op_types(preds, gts)
    return {
        "H1": {
            "ACP": accuracy_cad_programs(preds, gts, eta),
            "ASOT": accuracy_seq_op_types(preds, gts),
            "EDSOT": edit_distance_seq_op_types(preds, gts),
        },
        "H2": {
            "AOT": accuracy_op_types(preds, gts),      # L1
            "AP1": accuracy_parameter_ordered(preds, gts, eta),   # L2
        },
        "H3": {
            "MSOT_TC": msot["tc"],
            "MSOT_CS": msot["cs"],
            "AP2": accuracy_parameter_unordered(preds, gts, eta),  # L2
        },
        "baseline": {"AP1_random": random_baseline_ap1(eta)},
        "eta": eta,
    }
