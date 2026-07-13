"""Success-rate metrics and query-benchmark structure for Query2CAD (sec. 5-6).

Query2CAD reports its results with a specific set of accounting schemes that this
module re-implements deterministically:

  * Difficulty-segmented benchmark (sec. 5.1): a dataset of natural-language
    queries split into easy / medium / hard, with the paper's exact bin sizes
    (21 / 20 / 16, total 57). :func:`benchmark_composition` reproduces it.
  * Per-difficulty success rate (Table 1): the fraction of exactly-correct models
    per difficulty bin. An output is correct only if it reproduces the *exact*
    model -- any deviation is a failure (sec. 5.2), so scoring is strict binary.
  * Per-iteration success curve (Table 2): y0 (direct generation) and y1/y2/y3
    (after the 1st/2nd/3rd refinement); success is monotone non-decreasing across
    refinements because a solved query stays solved.
  * Refinement improvement deltas (Figure 4): delta(y_{t-1} -> y_t), which
    quantify how much each refinement iteration adds -- the paper's key finding is
    that the first refinement dominates.
  * Failure taxonomy (Figure 5): every failure is either a non-executable-code
    failure or a wrong-structure failure; :func:`failure_breakdown` reports the
    split matching the paper's 69/31 and 65.4/34.6 percentages.

Stdlib only, deterministic, no model.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

DIFFICULTIES = ("easy", "medium", "hard")

# The paper's dataset bin sizes (sec. 5.1).
PAPER_BINS = {"easy": 21, "medium": 20, "hard": 16}

# The two failure modes (Figure 5).
FAILURE_MODES = ("non_executable", "wrong_structure")


def benchmark_composition(bins: Dict[str, int] = None) -> Dict[str, object]:
    """Reproduce the difficulty composition of the query benchmark (sec. 5.1)."""
    b = dict(PAPER_BINS if bins is None else bins)
    for k in b:
        if k not in DIFFICULTIES:
            raise ValueError("unknown difficulty: %r" % (k,))
        if int(b[k]) < 0:
            raise ValueError("bin size must be non-negative")
    total = sum(b.get(d, 0) for d in DIFFICULTIES)
    if total == 0:
        raise ValueError("benchmark is empty")
    return {
        "counts": {d: b.get(d, 0) for d in DIFFICULTIES},
        "total": total,
        "fractions": {d: b.get(d, 0) / total for d in DIFFICULTIES},
    }


def success_rate(correct: int, total: int) -> float:
    """Strict binary success rate = correct / total (sec. 5.2)."""
    if total <= 0:
        raise ValueError("total must be positive")
    if not (0 <= correct <= total):
        raise ValueError("correct must be in [0, total]")
    return correct / total


def per_difficulty_success(results: Dict[str, Sequence[bool]]) -> Dict[str, object]:
    """Table-1 accounting: success rate within each difficulty bin.

    ``results`` maps a difficulty to a sequence of per-query correctness flags.
    Reports per-bin rate and the pooled overall rate.
    """
    per: Dict[str, float] = {}
    tot_correct = 0
    tot_n = 0
    for d, flags in results.items():
        if d not in DIFFICULTIES:
            raise ValueError("unknown difficulty: %r" % (d,))
        flags = list(flags)
        if not flags:
            raise ValueError("empty result list for %r" % (d,))
        c = sum(1 for f in flags if f)
        per[d] = c / len(flags)
        tot_correct += c
        tot_n += len(flags)
    return {"per_difficulty": per,
            "overall": tot_correct / tot_n if tot_n else None}


def refinement_curve(y: Sequence[float]) -> Dict[str, object]:
    """Validate and describe a per-iteration success curve y0..yT (Table 2).

    Requires monotone non-decreasing values in [0, 1] (a solved query stays
    solved across refinements). Returns the curve, total gain y_T - y0, and the
    iteration count.
    """
    vals = [float(v) for v in y]
    if len(vals) < 1:
        raise ValueError("need at least y0")
    for v in vals:
        if not (0.0 <= v <= 1.0):
            raise ValueError("success rate must be in [0, 1]")
    for a, b in zip(vals, vals[1:]):
        if b < a - 1e-12:
            raise ValueError("success curve must be non-decreasing")
    return {"curve": vals, "y0": vals[0], "final": vals[-1],
            "total_gain": vals[-1] - vals[0], "iterations": len(vals) - 1}


def improvement_deltas(y: Sequence[float]) -> List[float]:
    """Figure-4 per-iteration improvements delta(y_{t-1} -> y_t)."""
    refinement_curve(y)  # validates monotonicity/bounds
    vals = [float(v) for v in y]
    return [round(vals[i] - vals[i - 1], 12) for i in range(1, len(vals))]


def first_refinement_dominates(y: Sequence[float]) -> bool:
    """Paper's finding: the first refinement's gain is the largest (Figure 4).

    True iff delta(y0->y1) is >= every later delta. With a single refinement it
    trivially holds; with no refinements it is False (no gain to compare).
    """
    deltas = improvement_deltas(y)
    if not deltas:
        return False
    return all(deltas[0] >= d - 1e-12 for d in deltas[1:])


def failure_breakdown(non_executable: int, wrong_structure: int) -> Dict[str, object]:
    """Figure-5 failure taxonomy split.

    Every failure is exactly one of the two modes; reports counts and fractions.
    """
    if non_executable < 0 or wrong_structure < 0:
        raise ValueError("counts must be non-negative")
    total = non_executable + wrong_structure
    if total == 0:
        raise ValueError("no failures")
    return {
        "counts": {"non_executable": non_executable,
                   "wrong_structure": wrong_structure},
        "total": total,
        "fractions": {"non_executable": non_executable / total,
                      "wrong_structure": wrong_structure / total},
    }


if __name__ == "__main__":  # pragma: no cover
    pass
