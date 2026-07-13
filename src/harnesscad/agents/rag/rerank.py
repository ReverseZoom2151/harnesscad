"""partretr_rerank — training-free re-ranking of part-retrieval candidates.

Paper: "Error Notebook-Guided, Training-Free Part Retrieval ...". The Error
Notebook conditions inference by supplying similar past *mistakes* as
exemplars. This module operationalises the corrective signal deterministically
on the ranking side: given candidate answer-sets (each with an injected base
score — the VLM inference itself is external/skipped, per the campaign rules),
re-rank them by *down-weighting* answers that specification-similar past
queries are recorded as having gotten wrong.

Two candidate granularities are supported:

  - **answer-set candidates** — each candidate is a full subset of filenames
    (matches the paper's ``P*`` prediction); a candidate is penalised if it
    equals (or overlaps) a known-wrong answer for a similar past spec.
  - **per-part candidates** — each candidate is a single filename with a score;
    penalise filenames that appear in similar past queries' wrong answers.

Everything is deterministic: no wall-clock, no RNG, stable tie-breaking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from harnesscad.agents.memory.error_notebook import ErrorNotebook, _normalize_answer


@dataclass
class RankedCandidate:
    """A candidate answer after Error-Notebook re-ranking."""

    answer: Tuple[str, ...]
    base_score: float
    penalty: float
    final_score: float
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "answer": list(self.answer),
            "base_score": round(self.base_score, 6),
            "penalty": round(self.penalty, 6),
            "final_score": round(self.final_score, 6),
            "reason": self.reason,
        }


def _overlap_fraction(a: Tuple[str, ...], b: Tuple[str, ...]) -> float:
    if not a:
        return 0.0
    sb = set(b)
    return sum(1 for x in a if x in sb) / len(a)


def rerank_answer_sets(
    specification: str,
    candidates: Sequence[Tuple[Sequence[str], float]],
    notebook: ErrorNotebook,
    n_errors: int = 5,
    penalty_weight: float = 1.0,
    min_similarity: float = 0.0,
    exclude_id: Optional[str] = None,
) -> List[RankedCandidate]:
    """Re-rank whole-answer-set candidates using the Error Notebook.

    Each candidate is ``(answer_filenames, base_score)`` where ``base_score`` is
    the injected (VLM/retriever) confidence. For the query specification we pull
    known-wrong answer-sets from similar past entries; a candidate is penalised
    in proportion to how much it overlaps a known-wrong set, scaled by the
    similarity of the past query that flagged it and by ``penalty_weight``:

        final = base_score - penalty_weight * max_over_flags(sim * overlap)

    Exact repeats of a known-wrong set (overlap 1.0) get the full penalty.
    Results are sorted by descending final score; ties break by the normalised
    answer tuple for determinism.
    """
    known_wrong = notebook.known_wrong_for(
        specification, n=n_errors, min_similarity=min_similarity,
        exclude_id=exclude_id,
    )
    ranked: List[RankedCandidate] = []
    for ans_raw, base in candidates:
        ans = _normalize_answer(ans_raw)
        penalty = 0.0
        reason = ""
        for kw, sim in known_wrong.items():
            overlap = _overlap_fraction(ans, kw)
            if overlap <= 0.0:
                continue
            contrib = sim * overlap
            if contrib > penalty:
                penalty = contrib
                if overlap >= 1.0 and ans == kw:
                    reason = f"exact known-wrong for similar spec (sim={sim:.3f})"
                else:
                    reason = (f"overlaps known-wrong {list(kw)} "
                              f"by {overlap:.2f} (sim={sim:.3f})")
        final = base - penalty_weight * penalty
        ranked.append(RankedCandidate(
            answer=ans, base_score=float(base),
            penalty=penalty_weight * penalty, final_score=final, reason=reason,
        ))
    ranked.sort(key=lambda c: (-c.final_score, c.answer))
    return ranked


def rerank_parts(
    specification: str,
    candidates: Sequence[Tuple[str, float]],
    notebook: ErrorNotebook,
    n_errors: int = 5,
    penalty_weight: float = 1.0,
    min_similarity: float = 0.0,
    exclude_id: Optional[str] = None,
) -> List[RankedCandidate]:
    """Re-rank per-part (single-filename) candidates using the Error Notebook.

    A filename is penalised if it appears among the known-wrong answers of
    similar past queries, scaled by the flagging query's similarity. Useful for
    producing a re-ordered shortlist rather than a single subset.
    """
    known_wrong = notebook.known_wrong_for(
        specification, n=n_errors, min_similarity=min_similarity,
        exclude_id=exclude_id,
    )
    # Collapse to a per-filename max flag-strength.
    wrong_strength: Dict[str, float] = {}
    for kw, sim in known_wrong.items():
        for fn in kw:
            if sim > wrong_strength.get(fn, 0.0):
                wrong_strength[fn] = sim

    ranked: List[RankedCandidate] = []
    for fn, base in candidates:
        fn = str(fn).strip()
        sim = wrong_strength.get(fn, 0.0)
        penalty = penalty_weight * sim
        reason = (f"flagged wrong by similar spec (sim={sim:.3f})"
                  if sim > 0.0 else "")
        ranked.append(RankedCandidate(
            answer=(fn,), base_score=float(base), penalty=penalty,
            final_score=base - penalty, reason=reason,
        ))
    ranked.sort(key=lambda c: (-c.final_score, c.answer))
    return ranked


def select_top(ranked: Sequence[RankedCandidate], k: int = 1) -> List[Tuple[str, ...]]:
    """Return the answers of the top-``k`` re-ranked candidates."""
    return [c.answer for c in list(ranked)[:k]]
