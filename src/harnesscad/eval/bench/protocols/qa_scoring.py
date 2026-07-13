"""Textbook-problem and spatial-reasoning QA scorers (Sections 6 and 7).

Deterministic scorers for the education-task benchmarks:
  * textbook problems - each (sub-)question is repeated three times; a question
    counts as correct when at least two of three repeats are correct. Scores are
    grouped by image type and question format, and errors are tallied into the
    reasoning / inference / imprecise taxonomy.
  * spatial-reasoning multiple-choice tests (packing, MechE rotation) - per-run
    score against a bold answer key, average across runs, random-chance
    baseline (1/num_options), a "consistent" check (questions answered correctly
    by at least two runs), and a Run-H vs Run-P comparison.

VLM answers are injected. No randomness.
"""

from __future__ import annotations

from math import ceil

ERROR_TYPES = ("reasoning", "inference", "imprecise")


def majority_correct(repeats, *, threshold=None):
    """True if enough repeats are correct (default: strict majority)."""
    repeats = tuple(bool(r) for r in repeats)
    if not repeats:
        raise ValueError("no repeats")
    need = threshold if threshold is not None else (len(repeats) // 2 + 1)
    return sum(repeats) >= need


def qa_scorecard(questions, *, threshold=None):
    """Score textbook questions grouped by image type and format.

    questions: iterable of dicts {"id", "image_type", "format",
    "repeats": [bool,...], optional "error" in ERROR_TYPES}. Returns overall
    accuracy, per-image-type and per-format correct/total, and an error tally.
    """
    questions = list(questions)
    if not questions:
        raise ValueError("no questions")
    by_image = {}
    by_format = {}
    errors = {e: 0 for e in ERROR_TYPES}
    correct_total = 0
    for q in questions:
        ok = majority_correct(q["repeats"], threshold=threshold)
        correct_total += ok
        for key, table in (("image_type", by_image), ("format", by_format)):
            bucket = table.setdefault(q[key], {"correct": 0, "total": 0})
            bucket["total"] += 1
            bucket["correct"] += ok
        if not ok and q.get("error") is not None:
            err = q["error"]
            if err not in errors:
                raise ValueError("unknown error type: %r" % (err,))
            errors[err] += 1
    n = len(questions)

    def _acc(table):
        return {k: {"correct": v["correct"], "total": v["total"],
                    "accuracy": v["correct"] / v["total"] if v["total"] else None}
                for k, v in table.items()}

    return {"correct": correct_total, "total": n,
            "accuracy": correct_total / n,
            "by_image_type": _acc(by_image), "by_format": _acc(by_format),
            "errors": errors}


def spatial_run_scores(runs, key, num_options):
    """Score a spatial-reasoning multiple-choice test across runs.

    runs: mapping run_name -> sequence of chosen answers (one per question).
    key: sequence of correct answers. num_options: choices per question (for the
    random baseline). Returns per-run accuracy, average, random baseline, and
    the count/list of questions answered correctly by at least two runs.
    """
    key = tuple(key)
    q = len(key)
    per_run = {}
    correct_counts = [0] * q
    for name, answers in runs.items():
        answers = tuple(answers)
        if len(answers) != q:
            raise ValueError("run %r length mismatch" % (name,))
        correct = 0
        for i, (a, k) in enumerate(zip(answers, key)):
            if a == k:
                correct += 1
                correct_counts[i] += 1
        per_run[name] = {"correct": correct, "total": q,
                         "accuracy": correct / q if q else None}
    n_runs = len(per_run)
    avg = (sum(r["accuracy"] for r in per_run.values()) / n_runs
           if n_runs else None)
    consistent = tuple(i for i, c in enumerate(correct_counts)
                       if c >= (n_runs // 2 + 1)) if n_runs else ()
    return {"per_run": per_run, "average_accuracy": avg,
            "random_baseline": 1.0 / num_options if num_options else None,
            "consistent_questions": consistent,
            "consistent_count": len(consistent)}


def run_comparison(run_a, run_b):
    """Compare the average accuracies of two spatial-run scorecards.

    run_a / run_b are outputs of spatial_run_scores. Returns each average and
    the signed delta (b - a) plus which run performed better.
    """
    a = run_a["average_accuracy"]
    b = run_b["average_accuracy"]
    delta = None if (a is None or b is None) else b - a
    better = None
    if delta is not None:
        better = "b" if delta > 0 else ("a" if delta < 0 else "tie")
    return {"a": a, "b": b, "delta": delta, "better": better}
