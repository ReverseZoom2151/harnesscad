"""Multiple-choice design-description matching scorer (Section 3.2.1, Table 2).

Given an early-stage design sketch and a set of textual description options, a
VLM must pick the correct description. The paper runs three cases (with text,
no text, no text + no "None of the above") across three trials of ten
multiple-choice questions each. This module turns that protocol into a
deterministic scorer: per-trial score, per-case average, random-chance
baseline, and a full scorecard. VLM answers are injected.
"""

from __future__ import annotations


def score_trial(predicted, key):
    """Number of correct multiple-choice answers in one trial.

    predicted / key are equal-length sequences of chosen / correct option
    labels. Returns (correct, total).
    """
    predicted = tuple(predicted)
    key = tuple(key)
    if len(predicted) != len(key):
        raise ValueError("predicted/key length mismatch")
    correct = sum(1 for p, k in zip(predicted, key) if p == k)
    return correct, len(key)


def random_baseline(num_options, num_questions):
    """Expected correct-count for uniformly random guessing."""
    if num_options <= 0:
        raise ValueError("num_options must be positive")
    return num_questions / num_options


def case_scorecard(trials, num_options):
    """Aggregate one case (list of (predicted, key) trials).

    Returns per-trial scores, average correct, average accuracy, question total,
    and the random-chance baseline for that case.
    """
    if not trials:
        raise ValueError("no trials")
    per_trial = []
    total_questions = None
    for predicted, key in trials:
        correct, total = score_trial(predicted, key)
        if total_questions is None:
            total_questions = total
        per_trial.append({"correct": correct, "total": total,
                          "accuracy": (correct / total) if total else None})
    avg_correct = sum(t["correct"] for t in per_trial) / len(per_trial)
    return {
        "per_trial": tuple(per_trial),
        "avg_correct": avg_correct,
        "avg_accuracy": (avg_correct / total_questions) if total_questions else None,
        "questions": total_questions,
        "random_baseline": random_baseline(num_options, total_questions or 0),
    }


def match_scorecard(cases):
    """Full Table-2 style scorecard across named cases.

    cases: mapping case_name -> {"trials": [(predicted, key), ...],
                                 "num_options": int}.
    Returns dict case_name -> case_scorecard(...).
    """
    return {name: case_scorecard(spec["trials"], spec["num_options"])
            for name, spec in cases.items()}
