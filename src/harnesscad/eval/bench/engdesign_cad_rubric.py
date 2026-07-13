"""Engineering-drawing analysis and CAD-generation rubric (Section 4.2, Table 6).

Deterministic scorers for the seven-prompt CAD experiment:
  * P1 part description (1 pt) - reward "block with a (blind) hole", penalise
    any mention of a "through" hole (wrong geometry understanding).
  * P2 dimension extraction (10 pts) - 1 pt per drawing number extracted, 1 pt
    per correctly named dimension, minus 1 per extrapolated extra dimension.
  * P3-P7 CAD feature rubric (6 pts) - 1 pt for a runnable script and 1 pt each
    for five geometric features, minus 1 per extra incorrect feature.
  * iteration trajectory - track whether visual feedback iterations improve or
    (as the paper found) degrade the CAD score.

VLM/CAD outputs are injected; nothing renders or runs code here.
"""

from __future__ import annotations


def score_part_description(text, *, accept=("blind hole", "block with a hole"),
                           reject=("through",)):
    """1 point if the description shows correct geometry, else 0.

    A mention of any reject keyword (default: "through") zeroes the score
    regardless of accepted phrases, matching the paper's rule that calling the
    blind hole a through hole shows incorrect understanding.
    """
    low = text.lower()
    if any(r.lower() in low for r in reject):
        return 0
    return 1 if any(a.lower() in low for a in accept) else 0


def score_dimension_extraction(extracted, expected):
    """Score dimension extraction against expected dimensions (10 pts default).

    expected: sequence of dicts {"value": float, "labels": (acceptable names)}.
    extracted: sequence of dicts {"value": float, "label": str}.

    Awards 1 pt per expected value that appears in extracted values, 1 pt per
    extracted dimension whose value matches and whose label is acceptable, and
    subtracts 1 pt per extracted dimension whose value is not among expected
    (extrapolated / instruction violation). Returns dict with breakdown.
    """
    expected = list(expected)
    exp_values = [e["value"] for e in expected]
    ex_values = [x["value"] for x in extracted]

    value_points = 0
    remaining = list(ex_values)
    for v in exp_values:
        if v in remaining:
            value_points += 1
            remaining.remove(v)

    label_points = 0
    extra = 0
    used = list(expected)
    for x in extracted:
        match = None
        for e in used:
            if e["value"] == x["value"]:
                match = e
                break
        if match is None:
            extra += 1
            continue
        used.remove(match)
        if x.get("label", "").lower() in {lbl.lower() for lbl in match["labels"]}:
            label_points += 1
    total = value_points + label_points - extra
    return {"value_points": value_points, "label_points": label_points,
            "extra_penalty": extra, "score": total,
            "max": 2 * len(expected)}


def score_cad_features(features, *, extra_incorrect=0):
    """Score generated CAD (6 pts): runnable + five geometric features.

    features: dict of booleans among {runs_no_errors, correct_dimensions,
    hole_on_largest_face, hole_centered, correct_depth, correct_diameter}.
    extra_incorrect: number of additional wrong features present (each -1).
    """
    keys = ("runs_no_errors", "correct_dimensions", "hole_on_largest_face",
            "hole_centered", "correct_depth", "correct_diameter")
    earned = sum(1 for k in keys if features.get(k))
    score = earned - extra_incorrect
    return {"earned": earned, "extra_penalty": extra_incorrect,
            "score": score, "max": len(keys)}


def iteration_trajectory(scores):
    """Analyse CAD scores across the P3..P7 iterations.

    scores: ordered sequence of per-iteration CAD scores. Returns first, final,
    best, whether iterations improved over the first attempt, and whether the
    final attempt is worse than the first (the paper's key negative finding).
    """
    scores = tuple(scores)
    if not scores:
        raise ValueError("no iteration scores")
    first, final, best = scores[0], scores[-1], max(scores)
    return {"first": first, "final": final, "best": best,
            "improved": final > first,
            "final_worse_than_first": final < first,
            "peaked_at_first": best == first}
