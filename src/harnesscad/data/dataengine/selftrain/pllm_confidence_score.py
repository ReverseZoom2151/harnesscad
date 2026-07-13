"""PLLM pseudo-label confidence / quality scoring.

PLLM (Section 5.4) lists what a good pseudo program-shape pair should ideally
satisfy: (1) use high-quality programs, (2) provide executable program-shape
consistency, (3) reflect the target shape distribution, and (4) introduce
informative program variations. In practice no pairing strategy satisfies all
four simultaneously, so PLLM trades them off.

This module turns those qualitative criteria into a deterministic confidence
score for accepting / ranking a pseudo-label, plus the fidelity mapping from
Chamfer Distance the paper uses to decide "high-fidelity executions".

Distinct from ``gift_threshold_selection`` (an IoU CDF band): here we score an
*individual* candidate/label on an absolute [0, 1] confidence combining a
Chamfer-derived fidelity, an execution-validity flag, a candidate-agreement
term (how decisively the best-of-k beat its runner-up), and a conciseness
term. Deterministic, stdlib-only.
"""

from __future__ import annotations

from math import isfinite


def fidelity_from_chamfer(chamfer, scale=1.0):
    """Map Chamfer Distance (>= 0, lower better) to a fidelity in (0, 1].

    Uses fidelity = scale / (scale + chamfer): 0 distance -> 1.0, and fidelity
    decays smoothly as distance grows. ``scale`` sets the half-fidelity
    distance (fidelity = 0.5 when chamfer == scale). Monotonically decreasing
    in chamfer, so lower distance always scores higher.
    """
    if scale <= 0:
        raise ValueError("scale must be > 0")
    if chamfer < 0 or not isfinite(chamfer):
        raise ValueError("chamfer must be finite and >= 0")
    return scale / (scale + float(chamfer))


def agreement_margin(best_chamfer, runner_up_chamfer, scale=1.0):
    """Confidence that the best-of-k winner is decisively better than #2.

    A large gap between the best and second-best Chamfer means the selection is
    unambiguous (high confidence); a tiny gap means the winner is nearly
    interchangeable with alternatives. Returns 1 - exp(-gap/scale) in [0, 1);
    equal distances -> 0, large gaps -> ~1. When there is no runner-up (k == 1)
    the caller should pass ``runner_up_chamfer=None`` -> returns 1.0.
    """
    if scale <= 0:
        raise ValueError("scale must be > 0")
    if runner_up_chamfer is None:
        return 1.0
    gap = float(runner_up_chamfer) - float(best_chamfer)
    if gap <= 0:
        return 0.0
    from math import exp
    return 1.0 - exp(-gap / scale)


def conciseness(length, ref_length):
    """Conciseness term rewarding programs no longer than a reference length.

    Returns min(1, ref_length / length) so programs at or under ``ref_length``
    score 1.0 and longer programs are gently penalised. Length 0 -> 1.0.
    """
    if ref_length <= 0:
        raise ValueError("ref_length must be > 0")
    if length < 0:
        raise ValueError("length must be >= 0")
    if length == 0:
        return 1.0
    return min(1.0, ref_length / float(length))


# Default weights for the four terms (quality/fidelity, validity, agreement,
# conciseness). Fidelity dominates, matching PLLM's Chamfer-centric objective.
DEFAULT_WEIGHTS = {"fidelity": 0.5, "validity": 0.2, "agreement": 0.2,
                   "conciseness": 0.1}


def confidence_score(chamfer, executable, runner_up_chamfer=None, length=0,
                     ref_length=100, scale=1.0, weights=None):
    """Combined pseudo-label confidence in [0, 1].

    Non-executable labels short-circuit to 0.0 (Criterion 2 hard fails). Valid
    labels combine the fidelity (Criterion 1), execution validity, the best-of-k
    agreement margin, and conciseness as a weighted average with ``weights``
    (defaults to :data:`DEFAULT_WEIGHTS`). Higher = more trustworthy label.
    """
    if not executable:
        return 0.0
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    total = sum(w.values())
    if total <= 0:
        raise ValueError("weights must sum to > 0")
    fid = fidelity_from_chamfer(chamfer, scale)
    agr = agreement_margin(chamfer, runner_up_chamfer, scale)
    con = conciseness(length, ref_length)
    parts = {"fidelity": fid, "validity": 1.0, "agreement": agr,
             "conciseness": con}
    return sum(w.get(k, 0.0) * parts[k] for k in parts) / total


def rank_candidates(records, **kwargs):
    """Sort label records by descending confidence (deterministic tie-break).

    Each record is a dict with keys ``chamfer``, ``executable`` and optionally
    ``runner_up_chamfer``/``length``/``program``. Extra kwargs pass through to
    :func:`confidence_score`. Returns a new list of (record, score) pairs
    sorted best-first; ties break by lower chamfer then program identity.
    """
    scored = []
    for r in records:
        s = confidence_score(
            r["chamfer"], r["executable"],
            r.get("runner_up_chamfer"), r.get("length", 0),
            **kwargs)
        scored.append((r, s))
    scored.sort(key=lambda rs: (-rs[1], rs[0]["chamfer"],
                                str(rs[0].get("program", ""))))
    return scored
