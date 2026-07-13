"""VQAScore stopping criterion for Query2CAD (Badagabettu et al., 2024, sec. 3).

Query2CAD stops refining when the rendered isometric view of the generated CAD
model is judged to match the user query. It measures that match with the
*VQAScore* of Lin et al. 2024 (Eq. 1): the probability a VQA model assigns to
the answer "Yes" for the question

    "Does this figure show {user_query}? Please answer yes or no."

The VQA model (Clip-FlanT5-XL) is external and produces the probability; this
module owns everything deterministic around it that the paper specifies:

  * the exact question template (Eq. 1);
  * validation of a probability into a VQAScore;
  * the threshold gate (default 0.9, sec. 4) that decides whether to stop;
  * selecting the best candidate render across a refinement trajectory;
  * an averaged/aggregate VQAScore when several views are scored.

Stdlib only, deterministic, no VQA model, no wall clock.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

DEFAULT_THRESHOLD = 0.9

# The Eq.-1 question template. ``{user_query}`` is substituted verbatim.
QUESTION_TEMPLATE = ("Does this figure show {user_query}? "
                     "Please answer yes or no.")


def format_vqa_question(user_query: str) -> str:
    """Build the exact Eq.-1 VQA question for a user query."""
    q = str(user_query).strip()
    if not q:
        raise ValueError("user_query must be non-empty")
    return QUESTION_TEMPLATE.format(user_query=q)


def vqascore(prob_yes: float) -> float:
    """Validate a P("Yes") probability into a VQAScore in [0, 1] (Eq. 1)."""
    p = float(prob_yes)
    if not (0.0 <= p <= 1.0):
        raise ValueError("prob_yes must be in [0, 1], got %r" % (prob_yes,))
    return p


def meets_threshold(score: float, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """Stop-criterion gate: has the VQAScore reached the threshold?

    The paper stops "if the VQAScore exceeds a user-defined threshold (default is
    0.9)"; we treat reaching the threshold (>=) as satisfying it.
    """
    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError("threshold must be in [0, 1]")
    return vqascore(score) >= float(threshold)


def aggregate_vqascore(scores: Iterable[float]) -> float:
    """Mean VQAScore over several scored views of one model."""
    vals = [vqascore(s) for s in scores]
    if not vals:
        raise ValueError("no scores")
    return sum(vals) / len(vals)


def best_candidate(scores: Sequence[float]) -> Tuple[int, float]:
    """Index and VQAScore of the best render across a refinement trajectory.

    Ties resolve to the earliest index (the least-refined model that already
    achieves the best score, matching the paper's finding that most gains come
    from the first refinement).
    """
    vals = [vqascore(s) for s in scores]
    if not vals:
        raise ValueError("no scores")
    best_i = 0
    best_v = vals[0]
    for i, v in enumerate(vals):
        if v > best_v:
            best_i, best_v = i, v
    return best_i, best_v


def stopping_trajectory(scores: Sequence[float],
                        threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Replay a refinement trajectory's VQAScores against the threshold.

    Returns the index at which refinement would stop (first score to reach the
    threshold), whether it ever stopped, and the number of refinement rounds
    consumed. Mirrors the loop "run for a maximum of 3 times" (Figure 1).
    """
    vals = [vqascore(s) for s in scores]
    if not vals:
        raise ValueError("no scores")
    for i, v in enumerate(vals):
        if v >= float(threshold):
            return {"stopped": True, "stop_index": i, "rounds": i,
                    "final_score": v}
    return {"stopped": False, "stop_index": None, "rounds": len(vals) - 1,
            "final_score": vals[-1]}
