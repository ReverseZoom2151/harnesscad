"""Open-set recognition metrics for CAD retrieval (OSCAR, open-set protocol).

Pulli et al., *OSCAR: Open-Set CAD Retrieval from a Language Prompt and a Single
Image* (2024). OSCAR's distinct contribution is *open-set* retrieval: query
objects may belong to classes never seen during onboarding, so the system must
decide whether a query is matched by *some* database model at all (known) or is
novel / out-of-gallery (unknown). The paper's threshold on the CLIP-text cosine
similarity (``tau_text = 0.37``) is exactly such an accept/reject gate.

Closed-set ranked-retrieval metrics (:mod:`bench.ranked_retrieval_metrics`,
:mod:`bench.gencad_retrieval`) assume every query has a true match in the
gallery and only score *ordering*. This module scores the orthogonal open-set
question -- *known vs unknown* separability of a similarity/confidence score --
which none of the existing bench modules provide:

* **AUROC** -- threshold-free separability of the known-query scores from the
  unknown-query scores (probability a random known scores above a random
  unknown), computed exactly via the Mann-Whitney U statistic with tie handling.
* **Open-set F-measure** -- precision/recall/F1 of accepting a query as *known*
  at a given rejection threshold ``tau``.
* **Rejection accuracy / balanced accuracy** -- correct known-accept and
  unknown-reject rates at ``tau``.
* **Best-threshold sweep** -- the ``tau`` over observed scores maximising F1 or
  balanced accuracy (deterministic tie-break to the lower threshold).
* **Rank-based novelty score** -- normalised gap between a query's top-1 gallery
  similarity and the mean of its next ``m`` neighbours; a large gap signals a
  confident (known) match, a flat profile signals novelty.

Pure stdlib, deterministic. Scores are "higher = more confidently known".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

__all__ = [
    "auroc",
    "OpenSetCounts",
    "openset_confusion",
    "openset_f_measure",
    "rejection_accuracy",
    "balanced_rejection_accuracy",
    "best_threshold",
    "novelty_score",
]


def auroc(known_scores: Sequence[float], unknown_scores: Sequence[float]) -> float:
    """Area under the ROC curve for separating known from unknown queries.

    Equal to ``P(known_score > unknown_score) + 0.5 * P(tie)`` -- the normalised
    Mann-Whitney U statistic. ``1.0`` = perfect separation (all knowns score
    above all unknowns), ``0.5`` = chance. Returns ``0.5`` when either group is
    empty (no information to separate). Exact and deterministic; no sampling.
    """
    n_pos = len(known_scores)
    n_neg = len(unknown_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    greater = 0
    equal = 0
    for kp in known_scores:
        for kn in unknown_scores:
            if kp > kn:
                greater += 1
            elif kp == kn:
                equal += 1
    return (greater + 0.5 * equal) / (n_pos * n_neg)


@dataclass(frozen=True)
class OpenSetCounts:
    """Confusion counts for the accept-as-known decision at a threshold.

    ``tp`` -- known query correctly accepted (score >= tau).
    ``fn`` -- known query wrongly rejected.
    ``fp`` -- unknown query wrongly accepted.
    ``tn`` -- unknown query correctly rejected.
    """

    tp: int
    fn: int
    fp: int
    tn: int


def openset_confusion(known_scores: Sequence[float],
                      unknown_scores: Sequence[float],
                      tau: float) -> OpenSetCounts:
    """Confusion counts when accepting queries with ``score >= tau`` as known."""
    tp = sum(1 for s in known_scores if s >= tau)
    fn = len(known_scores) - tp
    fp = sum(1 for s in unknown_scores if s >= tau)
    tn = len(unknown_scores) - fp
    return OpenSetCounts(tp=tp, fn=fn, fp=fp, tn=tn)


def openset_f_measure(known_scores: Sequence[float],
                      unknown_scores: Sequence[float],
                      tau: float,
                      beta: float = 1.0) -> Tuple[float, float, float]:
    """Precision, recall, and F-beta of accepting queries as *known* at ``tau``.

    Precision = tp / (tp + fp), recall = tp / (tp + fn). Precision is ``0.0``
    when nothing is accepted; recall is ``0.0`` when there are no known queries.
    Returns ``(precision, recall, f_beta)``. Raises ``ValueError`` for
    non-positive ``beta``.
    """
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    c = openset_confusion(known_scores, unknown_scores, tau)
    precision = c.tp / (c.tp + c.fp) if (c.tp + c.fp) > 0 else 0.0
    recall = c.tp / (c.tp + c.fn) if (c.tp + c.fn) > 0 else 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    f = (1.0 + b2) * precision * recall / denom if denom > 0 else 0.0
    return precision, recall, f


def rejection_accuracy(known_scores: Sequence[float],
                       unknown_scores: Sequence[float],
                       tau: float) -> float:
    """Fraction of all queries decided correctly at ``tau`` (accuracy).

    ``(tp + tn) / total``. Returns ``0.0`` when there are no queries.
    """
    c = openset_confusion(known_scores, unknown_scores, tau)
    total = c.tp + c.fn + c.fp + c.tn
    if total == 0:
        return 0.0
    return (c.tp + c.tn) / total


def balanced_rejection_accuracy(known_scores: Sequence[float],
                                unknown_scores: Sequence[float],
                                tau: float) -> float:
    """Mean of known-accept rate (recall) and unknown-reject rate (specificity).

    Robust to class imbalance between known and unknown query counts. A group
    with no members contributes its own perfect rate is undefined, so it is
    dropped from the average; returns ``0.0`` if both groups are empty.
    """
    c = openset_confusion(known_scores, unknown_scores, tau)
    rates = []
    if c.tp + c.fn > 0:
        rates.append(c.tp / (c.tp + c.fn))
    if c.tn + c.fp > 0:
        rates.append(c.tn / (c.tn + c.fp))
    if not rates:
        return 0.0
    return sum(rates) / len(rates)


def best_threshold(known_scores: Sequence[float],
                   unknown_scores: Sequence[float],
                   objective: str = "f1",
                   beta: float = 1.0) -> Tuple[float, float]:
    """Sweep observed scores for the ``tau`` maximising an open-set objective.

    ``objective`` is ``"f1"`` (F-beta of known-acceptance) or ``"balanced"``
    (balanced rejection accuracy). Candidate thresholds are the distinct
    observed scores plus one just above the maximum (reject-all). Ties in the
    objective break to the *lower* threshold for determinism. Returns
    ``(tau, objective_value)``. Raises ``ValueError`` for an unknown objective or
    when there are no scores at all.
    """
    all_scores = list(known_scores) + list(unknown_scores)
    if not all_scores:
        raise ValueError("need at least one score")
    candidates = sorted(set(all_scores))
    candidates.append(max(all_scores) + 1.0)  # reject-everything threshold

    def score_at(tau):
        if objective == "f1":
            return openset_f_measure(known_scores, unknown_scores, tau, beta)[2]
        if objective == "balanced":
            return balanced_rejection_accuracy(known_scores, unknown_scores, tau)
        raise ValueError("objective must be 'f1' or 'balanced'")

    best_tau = candidates[0]
    best_val = score_at(best_tau)
    for tau in candidates[1:]:
        val = score_at(tau)
        if val > best_val:
            best_val = val
            best_tau = tau
    return best_tau, best_val


def novelty_score(sorted_similarities: Sequence[float], m: int = 5) -> float:
    """Rank-based novelty: gap between the top-1 similarity and the next ``m``.

    ``sorted_similarities`` are a query's gallery similarities in *descending*
    order (best first). A confident known match sits well above its neighbours
    (large positive gap); a novel/out-of-gallery query yields a flat profile
    (gap near zero). Defined as ``top1 - mean(next m)``, so *higher = more
    confidently known* (consistent with the score direction used by the other
    metrics here). Returns ``0.0`` when fewer than two similarities are given or
    ``m <= 0``.
    """
    n = len(sorted_similarities)
    if n < 2 or m <= 0:
        return 0.0
    top1 = sorted_similarities[0]
    neighbours = sorted_similarities[1:1 + m]
    mean_neighbours = sum(neighbours) / len(neighbours)
    return top1 - mean_neighbours
