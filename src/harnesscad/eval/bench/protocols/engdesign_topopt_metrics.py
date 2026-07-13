"""Topology-optimization analysis metrics (Section 4.3, Tables 7 and 8).

Deterministic scorers for the VLM topology-optimization experiments:
  * Volume Fraction Error (VFE) - percent error of predicted volume fraction.
  * Floating Material Error (FME) - classification error rate for detecting
    disconnected components, with a 50% random-chance reference.
  * prompting-strategy comparison (w/o Expertise, w/ Expertise, w/ CoT).
  * technical caption analysis scorer (Table 8) - binary per-field scoring of
    caption reading, load/boundary positioning, and code validation.

VLM outputs are injected. No randomness, no wall clock.
"""

from __future__ import annotations


def _mean_std(values):
    values = tuple(values)
    if not values:
        return None, None
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return m, var ** 0.5


def volume_fraction_error(predicted, true):
    """Percent absolute error of a single volume-fraction estimate."""
    if true == 0:
        raise ValueError("true volume fraction is zero")
    return abs(predicted - true) / abs(true) * 100.0


def mean_volume_fraction_error(pairs):
    """Mean +/- std percent VFE over (predicted, true) pairs."""
    errs = [volume_fraction_error(p, t) for p, t in pairs]
    m, s = _mean_std(errs)
    return {"vfe_mean": m, "vfe_std": s, "n": len(errs)}


def floating_material_error(predictions, truths):
    """Classification error rate (%) for floating-material detection.

    predictions / truths are equal-length sequences of booleans (True = design
    has floating material). Returns the error percentage plus the 50% baseline.
    """
    predictions = tuple(predictions)
    truths = tuple(truths)
    if len(predictions) != len(truths):
        raise ValueError("prediction/truth length mismatch")
    if not predictions:
        raise ValueError("no samples")
    wrong = sum(1 for p, t in zip(predictions, truths) if bool(p) != bool(t))
    return {"fme_percent": wrong / len(truths) * 100.0,
            "errors": wrong, "n": len(truths), "random_baseline": 50.0}


def prompting_strategy_table(strategies):
    """Compare prompting strategies on VFE and FME.

    strategies: mapping name -> {"vf_pairs": [(pred, true)...],
                                 "fm_pred": [...], "fm_true": [...]}.
    Returns per-strategy VFE/FME summaries.
    """
    out = {}
    for name, spec in strategies.items():
        row = {}
        if "vf_pairs" in spec:
            row.update(mean_volume_fraction_error(spec["vf_pairs"]))
        if "fm_pred" in spec:
            row.update(floating_material_error(spec["fm_pred"], spec["fm_true"]))
        out[name] = row
    return out


_CAPTION_FIELDS = ("nelx", "nely", "F", "VF", "R", "phi")
_POSITION_FIELDS = ("load", "bc")


def caption_analysis_score(rows):
    """Table-8 style binary scorer over topology-optimization diagrams.

    rows: list of dicts with 0/1 flags for the caption fields
    (nelx, nely, F, VF, R, phi), position fields (load, bc) and "code" for code
    validation. Returns per-column averages plus caption/position/overall means.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("no rows")
    cols = _CAPTION_FIELDS + _POSITION_FIELDS + ("code",)
    col_avg = {}
    for c in cols:
        col_avg[c] = sum(float(r[c]) for r in rows) / len(rows)
    caption_mean = sum(col_avg[c] for c in _CAPTION_FIELDS) / len(_CAPTION_FIELDS)
    position_mean = sum(col_avg[c] for c in _POSITION_FIELDS) / len(_POSITION_FIELDS)
    return {"column_avg": col_avg, "caption_mean": caption_mean,
            "position_mean": position_mean, "code_mean": col_avg["code"],
            "n": len(rows)}
