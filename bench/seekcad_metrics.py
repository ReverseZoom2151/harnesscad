"""Seek-CAD evaluation metrics: Novelty, complexity binning, G-Score, VLM
feedback accounting.

Deterministic re-implementation of the bespoke metrics from "Seek-CAD" (Li et
al., ICLR 2026), Section 5.1(1) and 5.2(4-5).  The geometric metrics used by the
paper (Chamfer/Hausdorff/IoGT) already exist elsewhere in the suite and are not
duplicated here; this module covers only the metrics unique to Seek-CAD.

  * Novel_Pn (Sec 5.1(1)): a generated model counts as *novel* when it is
    dissimilar (below threshold tau) from a sufficiently large fraction (>= rho)
    of the local corpus renderings::

        novel(I_A) = 1  iff  (1/n) * sum_i I[ s(I_A, I_B_i) < tau ] >= rho

    with tau = rho = 0.8 by default and s a similarity in [0, 1].  The corpus
    metric is the fraction of generated models judged novel.

  * Complexity bands (Sec 5.2(5)): Low [0,30], Medium [31,70], High [71, inf)
    by total SSR command count.

  * G-Score (Sec 5.1(1)): mean of VLM alignment scores in [1, 5].

  * VLM feedback accounting (Sec 5.2(4), Table 4): categorise verdicts into
    Yes/No (Helpful) and Unsure (Useless), reporting the helpful/useless split.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

# Feedback verdict labels (Table 4).
YES, NO, UNSURE = "Yes", "No", "Unsure"


def is_novel(
    similarities: Sequence[float], *, tau: float = 0.8, rho: float = 0.8
) -> bool:
    """Novel_Pn indicator for one generated model.

    ``similarities`` is s(I_A, I_B_i) for each corpus image I_B_i, in [0, 1].
    Returns True iff the fraction below ``tau`` is at least ``rho``.
    """
    if not similarities:
        raise ValueError("need at least one corpus similarity")
    if not (0.0 <= tau <= 1.0) or not (0.0 <= rho <= 1.0):
        raise ValueError("tau and rho must lie in [0, 1]")
    for s in similarities:
        if not (0.0 <= s <= 1.0):
            raise ValueError("similarities must lie in [0, 1]")
    below = sum(1 for s in similarities if s < tau)
    return (below / len(similarities)) >= rho


def novelty_rate(
    corpus_similarities: Sequence[Sequence[float]],
    *,
    tau: float = 0.8,
    rho: float = 0.8,
) -> float:
    """Fraction of generated models judged novel (the reported Novel metric)."""
    if not corpus_similarities:
        raise ValueError("need at least one generated model")
    novel = sum(
        1 for sims in corpus_similarities if is_novel(sims, tau=tau, rho=rho)
    )
    return novel / len(corpus_similarities)


def complexity_band(command_count: int) -> str:
    """Bucket a model by SSR command count (Sec 5.2(5), Table 5)."""
    if command_count < 0:
        raise ValueError("command_count must be non-negative")
    if command_count <= 30:
        return "Low"
    if command_count <= 70:
        return "Medium"
    return "High"


def band_histogram(command_counts: Sequence[int]) -> Dict[str, int]:
    """Count models per complexity band."""
    hist = {"Low": 0, "Medium": 0, "High": 0}
    for c in command_counts:
        hist[complexity_band(c)] += 1
    return hist


def g_score(scores: Sequence[float]) -> float:
    """Mean G-Score over models; each score in [1, 5] (Sec 5.1(1))."""
    if not scores:
        raise ValueError("need at least one score")
    for s in scores:
        if not (1.0 <= s <= 5.0):
            raise ValueError("G-Score values must lie in [1, 5]")
    return sum(scores) / len(scores)


def feedback_accounting(verdicts: Sequence[str]) -> Dict[str, object]:
    """Summarise VLM feedback into Helpful (Yes/No) vs Useless (Unsure).

    Returns raw counts plus fractions (Table 4 reports helpful ~88.2% and
    useless ~11.8% at N=1).
    """
    counts = {YES: 0, NO: 0, UNSURE: 0}
    for v in verdicts:
        if v not in counts:
            raise ValueError("verdict must be one of Yes/No/Unsure, got %r" % (v,))
        counts[v] += 1
    total = len(verdicts)
    if total == 0:
        raise ValueError("need at least one verdict")
    helpful = counts[YES] + counts[NO]
    useless = counts[UNSURE]
    return {
        "counts": counts,
        "helpful": helpful,
        "useless": useless,
        "helpful_fraction": helpful / total,
        "useless_fraction": useless / total,
    }


def per_band_means(
    records: Sequence[Dict[str, float]], value_key: str
) -> Dict[str, float]:
    """Mean of ``value_key`` grouped by each record's ``command_count`` band
    (the per-complexity result rows of Table 5)."""
    sums: Dict[str, float] = {"Low": 0.0, "Medium": 0.0, "High": 0.0}
    nums: Dict[str, int] = {"Low": 0, "Medium": 0, "High": 0}
    for r in records:
        band = complexity_band(int(r["command_count"]))
        sums[band] += float(r[value_key])
        nums[band] += 1
    return {b: (sums[b] / nums[b]) for b in sums if nums[b] > 0}


__all__ = [
    "is_novel",
    "novelty_rate",
    "complexity_band",
    "band_histogram",
    "g_score",
    "feedback_accounting",
    "per_band_means",
    "YES",
    "NO",
    "UNSURE",
]
