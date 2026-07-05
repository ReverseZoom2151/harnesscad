"""Active-learning sample selection (Scale-AI data-engine playbook).

Not every synthetic candidate is worth the expensive label. The solver-in-the-
loop (datagen/pipeline.py) and the human/verifier queue are the scarce resource;
active learning spends them on the candidates that will teach the model the most.
:func:`select_informative` ranks a bag of candidate parts by *informativeness*
and returns the top-``k`` to route onward.

Informativeness here blends the two signals the playbook names:

  * **model uncertainty** — a per-candidate uncertainty proxy (verifier
    disagreement, a solver's low-confidence score, or a distribution-gap weight
    from :mod:`dataengine.distribution_audit`). Read from the candidate when it
    carries one; otherwise a neutral default.
  * **novelty** — how different a candidate's op-signature is from the ones
    already selected. Selection is *greedy and diverse*: once a part is picked,
    its near-duplicates lose novelty, so a redundant copy never crowds out a
    genuinely new region of the space.

Scorers are pluggable: pass any ``scorer(candidate) -> float`` for a pure static
ranking, or leave ``scorer=None`` for the default uncertainty+novelty greedy
selector. Deterministic: ties break by original index; no randomness, no clock.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional, Sequence

from dataengine.distribution_audit import op_tags, family_of

Scorer = Callable[[Any], float]


# =====================================================================
# Signal extraction
# =====================================================================

def uncertainty_of(candidate: Any, default: float = 0.5) -> float:
    """The candidate's model-uncertainty proxy in ``[0, 1]`` (higher = less sure).

    Looked up, in order, from an ``uncertainty`` attribute, an ``uncertainty`` /
    ``score`` key on the object or its ``summary`` dict; else ``default``. This is
    the plug point for a verifier-disagreement rate or a solver confidence.
    """
    for src in (candidate, getattr(candidate, "summary", None)):
        if src is None:
            continue
        val = getattr(src, "uncertainty", None)
        if val is not None:
            return float(val)
        if isinstance(src, dict):
            for key in ("uncertainty", "score"):
                if src.get(key) is not None:
                    return float(src[key])
    if isinstance(candidate, dict):
        for key in ("uncertainty", "score"):
            if candidate.get(key) is not None:
                return float(candidate[key])
        summ = candidate.get("summary")
        if isinstance(summ, dict) and summ.get("uncertainty") is not None:
            return float(summ["uncertainty"])
    return float(default)


def signature(candidate: Any) -> frozenset:
    """A candidate's op-signature: its set of op tags + op bigrams + family.

    Used for novelty scoring; two candidates built from the same op stream (and
    family) share a signature and are treated as duplicates.
    """
    tags = op_tags(candidate)
    toks = set(tags)
    for i in range(len(tags) - 1):
        toks.add(f"{tags[i]}>{tags[i + 1]}")
    fam = family_of(candidate)
    if fam is not None:
        toks.add(f"family:{fam}")
    return frozenset(toks)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def novelty_of(sig: frozenset, seen: Sequence[frozenset]) -> float:
    """Novelty of ``sig`` vs already-selected signatures: ``1 - max similarity``.

    An exact duplicate of a selected candidate scores 0; a signature sharing
    nothing with the seen set scores 1. The first pick (empty ``seen``) is
    maximally novel.
    """
    if not seen:
        return 1.0
    return 1.0 - max(_jaccard(sig, s) for s in seen)


# =====================================================================
# select_informative
# =====================================================================

def select_informative(candidates: Iterable[Any],
                       k: int,
                       *,
                       scorer: Optional[Scorer] = None,
                       uncertainty_weight: float = 1.0,
                       novelty_weight: float = 1.0) -> List[Any]:
    """Return the ``k`` most informative candidates to route to the solver/human.

    With a custom ``scorer`` the ranking is a plain descending sort by
    ``scorer(candidate)`` (ties -> original order). With ``scorer=None`` (the
    default) selection is greedy: at each step pick the remaining candidate
    maximising ``uncertainty_weight*uncertainty + novelty_weight*novelty``, where
    novelty is measured against the already-picked set — so a high-uncertainty,
    novel part outranks a duplicate, and the batch stays diverse.
    """
    cands = list(candidates)
    if k <= 0 or not cands:
        return []
    k = min(k, len(cands))

    if scorer is not None:
        order = sorted(range(len(cands)), key=lambda i: (-float(scorer(cands[i])), i))
        return [cands[i] for i in order[:k]]

    # Greedy uncertainty + novelty.
    sigs = [signature(c) for c in cands]
    uncs = [uncertainty_of(c) for c in cands]
    remaining = list(range(len(cands)))
    selected_sigs: List[frozenset] = []
    picked: List[Any] = []

    while remaining and len(picked) < k:
        best_i = None
        best_score = None
        for i in remaining:
            nov = novelty_of(sigs[i], selected_sigs)
            score = uncertainty_weight * uncs[i] + novelty_weight * nov
            if best_score is None or score > best_score:
                best_score = score
                best_i = i
        picked.append(cands[best_i])
        selected_sigs.append(sigs[best_i])
        remaining.remove(best_i)

    return picked


# A trivial pluggable scorer: rank by uncertainty alone (no diversity).
def uncertainty_scorer(candidate: Any) -> float:
    return uncertainty_of(candidate)
