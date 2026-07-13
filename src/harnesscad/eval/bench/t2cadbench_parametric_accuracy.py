"""Text2CAD-Bench parameter-level accuracy metric.

Deterministic parametric-CAD accuracy scorer for Text2CAD-Bench (Wang et al.,
"Text2CAD-Bench"). The benchmark's first design principle is *Unambiguous Ground
Truth*: "each text description corresponds to a unique, fully-specified CAD model
with all geometric parameters explicitly defined" (Section 3.1). This module
scores how well a model's recovered parameters (dimensions such as length,
width, height, radius, diameter, wall thickness, chamfer size, and categorical
choices such as the workplane) match the fully-specified ground truth.

This is a *parameter-level* metric complementing the paper's geometric metrics
(Chamfer Distance / IoU) and the executability metric (Invalidity Rate). Unlike
Chamfer Distance, it diagnoses *which* named parameters a model got right, and
unlike IR it credits partial correctness. It is DISTINCT from
``bench/cadtests_assertions`` (boolean unit-test assertions, paper 169): here we
compute a graded per-parameter match with relative/absolute tolerance over a
parameter dictionary, plus a parametric-validity check (all required parameters
present and inside declared valid ranges).

Parameters and model outputs are injected. No wall clock, no randomness.
"""

from __future__ import annotations

DEFAULT_REL_TOL = 0.02   # 2% relative tolerance for numeric dimensions
DEFAULT_ABS_TOL = 1e-6


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def match_parameter(predicted, truth, rel_tol=DEFAULT_REL_TOL,
                    abs_tol=DEFAULT_ABS_TOL):
    """Whether a single predicted parameter matches the ground truth.

    Numeric parameters match within max(rel_tol * |truth|, abs_tol). Non-numeric
    (categorical, e.g. "XY") match by case-insensitive string equality. A
    missing prediction (None) never matches.
    """
    if predicted is None:
        return False
    if _is_number(truth) and _is_number(predicted):
        tol = max(rel_tol * abs(float(truth)), abs_tol)
        return abs(float(predicted) - float(truth)) <= tol
    return str(predicted).strip().lower() == str(truth).strip().lower()


def parameter_accuracy(predicted, truth, rel_tol=DEFAULT_REL_TOL,
                       abs_tol=DEFAULT_ABS_TOL):
    """Per-parameter and overall accuracy of a predicted parameter dict.

    predicted : mapping name -> value (a missing key counts as unpredicted).
    truth     : mapping name -> value (the fully-specified ground truth; its key
                set defines the parameters that must be recovered).

    Returns a dict:
      per_parameter : name -> {predicted, truth, matched} for every truth key.
      matched       : number of matched parameters.
      total         : number of truth parameters.
      accuracy      : matched / total (1.0 for an empty ground truth).
      missing       : sorted tuple of truth params absent from the prediction.
      wrong         : sorted tuple of present-but-mismatched params.
      extra         : sorted tuple of predicted params not in the ground truth.
    """
    per = {}
    matched = 0
    missing = []
    wrong = []
    for name, tval in truth.items():
        present = name in predicted
        pval = predicted.get(name)
        ok = present and match_parameter(pval, tval, rel_tol, abs_tol)
        if ok:
            matched += 1
        elif not present:
            missing.append(name)
        else:
            wrong.append(name)
        per[name] = {"predicted": pval if present else None,
                     "truth": tval, "matched": ok}
    total = len(truth)
    extra = sorted(set(predicted) - set(truth))
    return {
        "per_parameter": per,
        "matched": matched,
        "total": total,
        "accuracy": matched / total if total else 1.0,
        "missing": tuple(sorted(missing)),
        "wrong": tuple(sorted(wrong)),
        "extra": tuple(extra),
    }


def parametric_validity(predicted, required, ranges=None):
    """Check the *fully-specified* + *in-range* validity of a prediction.

    Mirrors the benchmark's authoring constraint that all geometric parameters
    are fully specified and physically sensible.

    predicted : mapping name -> value.
    required  : iterable of parameter names that must be present (non-None).
    ranges    : optional mapping name -> (lo, hi) inclusive numeric bounds.

    Returns a dict:
      missing_required : sorted tuple of required params absent/None.
      out_of_range     : sorted tuple of numeric params outside their bounds.
      fully_specified  : True iff no required param is missing.
      valid            : True iff fully specified and nothing out of range.
    """
    ranges = ranges or {}
    missing = []
    for name in required:
        if predicted.get(name) is None:
            missing.append(name)
    oor = []
    for name, bound in ranges.items():
        val = predicted.get(name)
        if val is None or not _is_number(val):
            continue
        lo, hi = bound
        if float(val) < float(lo) or float(val) > float(hi):
            oor.append(name)
    fully = not missing
    return {
        "missing_required": tuple(sorted(missing)),
        "out_of_range": tuple(sorted(oor)),
        "fully_specified": fully,
        "valid": fully and not oor,
    }


def mean_parameter_accuracy(examples, rel_tol=DEFAULT_REL_TOL,
                            abs_tol=DEFAULT_ABS_TOL):
    """Mean parameter accuracy across a set of (predicted, truth) examples.

    examples : iterable of (predicted_dict, truth_dict) pairs.

    Returns a dict: n, mean_accuracy, micro_accuracy (total matched over total
    parameters, weighting examples by their parameter count).
    """
    rows = [parameter_accuracy(p, t, rel_tol, abs_tol) for p, t in examples]
    n = len(rows)
    if n == 0:
        raise ValueError("no examples")
    mean_acc = sum(r["accuracy"] for r in rows) / n
    tot_matched = sum(r["matched"] for r in rows)
    tot_params = sum(r["total"] for r in rows)
    micro = tot_matched / tot_params if tot_params else 1.0
    return {"n": n, "mean_accuracy": mean_acc, "micro_accuracy": micro,
            "matched": tot_matched, "total": tot_params}
