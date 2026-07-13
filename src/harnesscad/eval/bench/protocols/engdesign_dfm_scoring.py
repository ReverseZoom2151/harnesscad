"""Design-for-manufacturing scorers (Section 5.1, Tables 9 and machining features).

Two deterministic protocols:

Additive manufacturing (DfAM, Table 9): for each design a VLM predicts
manufacturability and, for problematic designs, the violated design rule. Each
design scores Manufacturable? (0/1) + Correct Rule (0/1) + #Incorrect Rules
(negative), giving a signed total that matches the paper's "Score" column.

Subtractive manufacturing (machining feature recognition, Section 5.1.2):
predicted machining-feature sets are compared against ground-truth sets over a
fixed 15-feature taxonomy using precision, recall, F1, Jaccard, and the
"at least one correct" indicator.

VLM answers are injected. No randomness.
"""

from __future__ import annotations

# The 15 machining features tested (excludes the stock-material block).
MACHINING_FEATURES = (
    "rectangular through slot", "triangular through slot", "rectangular passage",
    "triangular passage", "6 sided passage", "rectangular through step",
    "2 sided through step", "slanted through step", "rectangular blind step",
    "triangular blind step", "rectangular blind slot", "rectangular pocket",
    "triangular pocket", "6 sided pocket", "chamfer",
)


def score_additive_design(manufacturable_correct, correct_rule,
                          num_incorrect_rules):
    """Signed score for one DfAM design trial (Table 9).

    manufacturable_correct: bool - did the model correctly answer if the design
        was manufacturable?
    correct_rule: bool - was the truly-violated rule named? (problematic set)
    num_incorrect_rules: int >= 0 - count of rules wrongly claimed violated.

    Score = manufacturable(0/1) + correct_rule(0/1) - num_incorrect_rules.
    """
    if num_incorrect_rules < 0:
        raise ValueError("num_incorrect_rules must be non-negative")
    m = 1 if manufacturable_correct else 0
    r = 1 if correct_rule else 0
    return {"manufacturable": m, "correct_rule": r,
            "incorrect_rules": -num_incorrect_rules,
            "score": m + r - num_incorrect_rules}


def additive_scorecard(trials):
    """Aggregate DfAM trials.

    trials: iterable of dicts accepted by score_additive_design (keys
    manufacturable_correct, correct_rule, num_incorrect_rules). Returns totals
    and averages across trials.
    """
    rows = [score_additive_design(t["manufacturable_correct"],
                                  t["correct_rule"], t["num_incorrect_rules"])
            for t in trials]
    if not rows:
        raise ValueError("no trials")
    n = len(rows)
    return {
        "n": n,
        "manufacturable_rate": sum(r["manufacturable"] for r in rows) / n,
        "correct_rule_rate": sum(r["correct_rule"] for r in rows) / n,
        "total_score": sum(r["score"] for r in rows),
        "mean_score": sum(r["score"] for r in rows) / n,
        "rows": tuple(rows),
    }


def _norm(feature):
    return feature.strip().lower()


def feature_recognition(predicted, ground_truth, *, taxonomy=MACHINING_FEATURES):
    """Set-based machining-feature recognition metrics for one CAD image.

    predicted / ground_truth: iterables of feature names (case-insensitive).
    Unknown feature names (outside taxonomy) raise ValueError. Returns
    precision, recall, f1, jaccard, exact-match, and at-least-one-correct.
    """
    allowed = {_norm(f) for f in taxonomy}
    pred = {_norm(f) for f in predicted}
    gt = {_norm(f) for f in ground_truth}
    for f in pred | gt:
        if f not in allowed:
            raise ValueError("feature outside taxonomy: %r" % (f,))
    inter = pred & gt
    union = pred | gt
    precision = (len(inter) / len(pred)) if pred else (1.0 if not gt else 0.0)
    recall = (len(inter) / len(gt)) if gt else (1.0 if not pred else 0.0)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    jaccard = (len(inter) / len(union)) if union else 1.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "jaccard": jaccard, "exact": pred == gt,
            "at_least_one_correct": bool(inter)}


def feature_scorecard(samples, *, taxonomy=MACHINING_FEATURES):
    """Aggregate feature recognition over samples.

    samples: iterable of (predicted, ground_truth) pairs. Returns mean
    precision/recall/f1/jaccard, exact-match rate, and at-least-one rate.
    """
    rows = [feature_recognition(p, g, taxonomy=taxonomy) for p, g in samples]
    if not rows:
        raise ValueError("no samples")
    n = len(rows)
    keys = ("precision", "recall", "f1", "jaccard")
    out = {"n": n}
    for k in keys:
        out["mean_" + k] = sum(r[k] for r in rows) / n
    out["exact_rate"] = sum(1 for r in rows if r["exact"]) / n
    out["at_least_one_rate"] = sum(1 for r in rows
                                   if r["at_least_one_correct"]) / n
    return out
