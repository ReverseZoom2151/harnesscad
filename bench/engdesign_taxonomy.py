"""Engineering-design VLM benchmark taxonomy and aggregate scorecard.

Encodes the four-area task taxonomy and the cross-model aggregate scorecard
(Table 20) from "From Concept to Manufacturing: Evaluating Vision-Language
Models for Engineering Design" (Picard, Edwards et al.).

The VLM inference itself is research-heavy/external; this module is the
deterministic bench structure: the stage/dimension taxonomy with per-experiment
maximum scores, and a scorecard that aggregates injected per-experiment scores
per model, computes stage subtotals, normalised fractions, and per-experiment
leaders. No wall clock, no randomness.
"""

from __future__ import annotations

# Canonical taxonomy: the four main engineering-design areas, each with its
# quantitative experiments and maximum achievable score (from Table 20).
_TAXONOMY = (
    ("Conceptual Design", (
        ("Design description: with text", 30),
        ("Design description: no text", 30),
        ("Design description: no text, no N/A", 30),
    )),
    ("System-Level and Detailed Design", (
        ("Engineering drawing analysis", 99),
        ("CAD generation (1st try)", 54),
        ("Topology optimization", 90),
    )),
    ("Manufacturing and Inspection", (
        ("Design for additive manufacturing", 90),
        ("Machining feature recognition", 60),
        ("Crack/defect inspection", 345),
    )),
    ("Engineering Education Tasks", (
        ("Textbook questions", 135),
        ("Spatial reasoning: rotation", 100),
        ("Spatial reasoning: packing", 50),
    )),
)


def benchmark_taxonomy():
    """Return the ordered taxonomy as a tuple of (stage, (experiment, max)...)."""
    return _TAXONOMY


def experiment_index():
    """Map experiment name -> (stage, max_score)."""
    out = {}
    for stage, exps in _TAXONOMY:
        for name, mx in exps:
            out[name] = (stage, mx)
    return out


def total_max_score():
    """Maximum total score across all experiments (1113 in the paper)."""
    return sum(mx for _, exps in _TAXONOMY for _, mx in exps)


def stage_max_scores():
    """Map stage -> maximum achievable score for that stage."""
    return {stage: sum(mx for _, mx in exps) for stage, exps in _TAXONOMY}


def aggregate_scorecard(model_scores):
    """Aggregate injected per-experiment scores across models.

    model_scores: dict model_name -> dict experiment_name -> score (float) or
    None for not-applicable / not-run experiments.

    Returns a dict with per-model totals, per-stage subtotals, normalised
    fraction-of-max, applicable maxima, and the overall maximum.
    """
    idx = experiment_index()
    stage_of = {name: stg for name, (stg, _) in idx.items()}
    stage_names = [stage for stage, _ in _TAXONOMY]
    result = {"max_total": total_max_score(), "models": {}}
    for model, scores in model_scores.items():
        for exp in scores:
            if exp not in idx:
                raise KeyError("unknown experiment: %r" % (exp,))
        total = 0.0
        applicable_max = 0
        stage_totals = {s: 0.0 for s in stage_names}
        for exp, (stg, mx) in idx.items():
            val = scores.get(exp)
            if val is None:
                continue
            total += val
            applicable_max += mx
            stage_totals[stg] += val
        frac = (total / applicable_max) if applicable_max else None
        result["models"][model] = {
            "total": total,
            "applicable_max": applicable_max,
            "fraction_of_max": frac,
            "stage_totals": stage_totals,
        }
    return result


def experiment_leaders(model_scores):
    """For each experiment, name the model(s) with the highest injected score.

    Experiments where a model injected None do not count that model. Returns
    dict experiment -> {"best": score, "leaders": sorted tuple of model names}.
    Experiments with no applicable model map to None.
    """
    idx = experiment_index()
    leaders = {}
    for exp in idx:
        best = None
        winners = []
        for model, scores in model_scores.items():
            val = scores.get(exp)
            if val is None:
                continue
            if best is None or val > best:
                best = val
                winners = [model]
            elif val == best:
                winners.append(model)
        leaders[exp] = (None if best is None
                        else {"best": best, "leaders": tuple(sorted(winners))})
    return leaders


def model_win_counts(model_scores):
    """Count experiments each model uniquely leads (ties excluded from a win)."""
    leaders = experiment_leaders(model_scores)
    counts = {m: 0 for m in model_scores}
    for info in leaders.values():
        if info is None:
            continue
        if len(info["leaders"]) == 1:
            counts[info["leaders"][0]] += 1
    return counts
