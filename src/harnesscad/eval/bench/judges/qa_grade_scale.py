"""CAD-Q&A evaluation metrics (Kienle et al., "QueryCAD: Grounded Question
Answering for CAD Models", Sec. IV-A "Evaluation" and Table I).

The paper evaluates QueryCAD on a 111-question benchmark and reports, per answer,
one of three outcomes -- Correct, Partial (the answer "overlaps with the
solution"), or Wrong -- and categorises every wrong answer into one of four error
types (Table I): Syntax, Reasoning, Masks, CAD-Interface. Measurement answers are
numeric, so grading needs a tolerance rather than string equality.

This module implements that DETERMINISTIC grading and aggregation:

  * :func:`grade` -- score one predicted answer against the expected one, with
    per-answer-kind logic: exact for counts / booleans / categorical strings,
    absolute+relative numeric tolerance for measurements, per-component
    tolerance for position vectors, and set overlap (=> "partial") for list
    answers such as "the diameters of all holes".
  * :data:`ERROR_CATEGORIES` + :func:`aggregate` -- roll a batch of graded
    results into the paper's Table-I summary: correct / partial / wrong counts,
    accuracy, partial-inclusive accuracy, per-category error tallies, and the
    mean absolute error over the numeric questions (a measurement-quality signal
    the paper's binary scoring omits).

Pure, deterministic, stdlib-only. Works on the :class:`Answer` value produced by
:mod:`reconstruction.querycad_answer_engine`, but accepts plain values too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

# The four wrong-answer categories of Table I (plus a generic fallback).
ERROR_CATEGORIES = ("syntax", "reasoning", "masks", "cad_interface")

# Grading outcomes.
CORRECT = "correct"
PARTIAL = "partial"
WRONG = "wrong"


@dataclass(frozen=True)
class Grade:
    """Result of grading one predicted answer."""
    outcome: str                       # correct | partial | wrong
    abs_error: Optional[float] = None  # |pred - expected| for numeric answers
    error_category: Optional[str] = None  # only when outcome == wrong

    @property
    def is_correct(self):
        return self.outcome == CORRECT

    @property
    def is_partial(self):
        return self.outcome == PARTIAL


def _as_value(answer):
    """Extract a plain value from an engine ``Answer`` or a raw value."""
    if hasattr(answer, "value"):
        return answer.value, bool(getattr(answer, "abstained", False))
    return answer, False


def _num_close(pred, expected, abs_tol, rel_tol):
    diff = abs(float(pred) - float(expected))
    return diff <= max(abs_tol, rel_tol * abs(float(expected))), diff


def _grade_numeric(pred, expected, abs_tol, rel_tol, error_category):
    ok, diff = _num_close(pred, expected, abs_tol, rel_tol)
    if ok:
        return Grade(CORRECT, abs_error=diff)
    return Grade(WRONG, abs_error=diff, error_category=error_category)


def _grade_vector(pred, expected, abs_tol, rel_tol, error_category):
    p = tuple(pred)
    e = tuple(expected)
    if len(p) != len(e):
        return Grade(WRONG, error_category=error_category)
    max_diff = 0.0
    for pi, ei in zip(p, e):
        ok, diff = _num_close(pi, ei, abs_tol, rel_tol)
        max_diff = max(max_diff, diff)
        if not ok:
            return Grade(WRONG, abs_error=max_diff,
                         error_category=error_category)
    return Grade(CORRECT, abs_error=max_diff)


def _grade_set(pred, expected, error_category):
    """List / set answers: exact => correct, non-empty overlap => partial."""
    ps = list(pred)
    es = list(expected)
    if ps == es:
        return Grade(CORRECT)
    overlap = set(_hashable(ps)) & set(_hashable(es))
    if overlap:
        return Grade(PARTIAL)
    return Grade(WRONG, error_category=error_category)


def _hashable(seq):
    out = []
    for x in seq:
        out.append(round(float(x), 9) if isinstance(x, (int, float)) else x)
    return tuple(out)


def grade(pred, expected, *, kind=None, abs_tol=1e-6, rel_tol=0.0,
          error_category=None):
    """Grade ``pred`` against ``expected``.

    ``pred``            an engine :class:`Answer` or a raw value.
    ``expected``        the ground-truth value.
    ``kind``            optional answer kind ("number"/"int"/"bool"/"vector"/
                        "part_property"); inferred from the values if omitted.
    ``abs_tol``/``rel_tol``  numeric tolerances (mm) for measurement/position.
    ``error_category``  the Table-I category to tag a wrong answer with
                        (one of :data:`ERROR_CATEGORIES`); defaults to
                        "reasoning".

    Returns a :class:`Grade`. An abstained prediction is always wrong.
    """
    if error_category is None:
        error_category = "reasoning"
    if error_category not in ERROR_CATEGORIES:
        raise ValueError("unknown error category: %r" % (error_category,))

    value, abstained = _as_value(pred)
    if abstained or value is None:
        # CAD interface could not answer -> a CAD-interface failure by default,
        # unless the caller overrode the category.
        cat = error_category if error_category != "reasoning" else "cad_interface"
        return Grade(WRONG, error_category=cat)

    # comparison answers are (part_id, value) tuples.
    if kind == "part_property" or (
            kind is None and isinstance(value, tuple) and len(value) == 2
            and isinstance(value[1], (int, float))
            and isinstance(expected, tuple) and len(expected) == 2):
        pid_ok = value[0] == expected[0]
        num_ok, diff = _num_close(value[1], expected[1], abs_tol, rel_tol)
        if pid_ok and num_ok:
            return Grade(CORRECT, abs_error=diff)
        if pid_ok or num_ok:
            return Grade(PARTIAL, abs_error=diff)
        return Grade(WRONG, abs_error=diff, error_category=error_category)

    if kind == "bool" or (kind is None and isinstance(value, bool)):
        if bool(value) == bool(expected):
            return Grade(CORRECT)
        return Grade(WRONG, error_category=error_category)

    if kind == "vector" or (kind is None and isinstance(value, (tuple, list))
                            and isinstance(expected, (tuple, list))
                            and _is_vector(value) and _is_vector(expected)):
        return _grade_vector(value, expected, abs_tol, rel_tol, error_category)

    if kind == "int" or (kind is None and isinstance(value, int)
                         and not isinstance(value, bool)):
        if int(value) == int(expected):
            return Grade(CORRECT, abs_error=0.0)
        return Grade(WRONG, abs_error=abs(int(value) - int(expected)),
                     error_category=error_category)

    if isinstance(value, (list, tuple)):
        return _grade_set(value, expected, error_category)

    if isinstance(value, (int, float)):
        return _grade_numeric(value, expected, abs_tol, rel_tol,
                              error_category)

    # Fallback: categorical string equality.
    if str(value).strip().casefold() == str(expected).strip().casefold():
        return Grade(CORRECT)
    return Grade(WRONG, error_category=error_category)


def _is_vector(v):
    return (isinstance(v, (list, tuple)) and len(v) > 0
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    for x in v))


def aggregate(grades):
    """Roll a batch of :class:`Grade` into the Table-I summary dict.

    Returns keys: ``total``, ``correct``, ``partial``, ``wrong``,
    ``accuracy`` (correct/total), ``partial_accuracy`` ((correct+partial)/total),
    ``errors`` ({category: count}), and ``mae`` (mean absolute error over graded
    answers that carry one, or None).
    """
    grades = tuple(grades)
    total = len(grades)
    correct = sum(g.outcome == CORRECT for g in grades)
    partial = sum(g.outcome == PARTIAL for g in grades)
    wrong = sum(g.outcome == WRONG for g in grades)

    errors = {c: 0 for c in ERROR_CATEGORIES}
    for g in grades:
        if g.outcome == WRONG and g.error_category in errors:
            errors[g.error_category] += 1

    diffs = [g.abs_error for g in grades if g.abs_error is not None]
    mae = (sum(diffs) / len(diffs)) if diffs else None

    return {
        "total": total,
        "correct": correct,
        "partial": partial,
        "wrong": wrong,
        "accuracy": (correct / total) if total else None,
        "partial_accuracy": ((correct + partial) / total) if total else None,
        "errors": errors,
        "mae": mae,
    }
