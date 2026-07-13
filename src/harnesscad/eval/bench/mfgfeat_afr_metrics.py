"""AFR evaluation protocol for VLM manufacturing-feature recognition
(Khan et al., "Leveraging Vision-Language Models for Manufacturing Feature
Recognition in CAD Designs", Sec. 3.4).

The paper proposes four QUANTITY-aware metrics that compare a predicted feature
*count* dictionary against an expert ground-truth count dictionary for one CAD
design:

    FNA  Feature Name Accuracy (Eq. 1)
         = correctly identified feature names / total ground-truth names.
    FQA  Feature Quantity Accuracy (Eq. 2)
         = true-positive quantity / ground-truth quantity.
    HR   Hallucination Rate (Eq. 3)
         = hallucinated quantity / total predicted quantity.
    MAE  Mean Absolute Error (Eq. 4)
         = mean over evaluated features of |ground_truth_qty - predicted_qty|.

These are genuinely DISTINCT from ``bench/engdesign_dfm_scoring`` (paper 85),
which is *set-based* (presence-only precision / recall / F1 / Jaccard over a flat
machining-feature set). Here every feature carries a MULTIPLICITY (e.g. "3
holes, 2 slots") and the metrics are count-sensitive: a design with 114 blind
holes vs. a prediction of 100 is scored on the count, not merely on the presence
of "hole".

Inputs are ``{feature_name: quantity}`` dicts. Names are optionally normalised
through :mod:`fabrication.mfgfeat_taxonomy` so aliases ("blind hole",
"through hole") collapse onto canonical leaf labels before scoring.

stdlib-only, deterministic, no randomness.
"""

from __future__ import annotations

try:  # normalisation is optional; metrics work on raw names too.
    from harnesscad.domain.fabrication.mfgfeat_taxonomy import normalize_feature as _normalize
except Exception:  # pragma: no cover - defensive
    _normalize = None


# --------------------------------------------------------------------------- #
# Count-dict normalisation
# --------------------------------------------------------------------------- #
def _clean_counts(counts, *, normalize):
    """Return a {name: int_quantity} dict, dropping zero/negative entries and
    (optionally) folding names onto canonical taxonomy leaves. Quantities for
    names that collapse together are summed."""
    out = {}
    for name, qty in dict(counts).items():
        q = int(qty)
        if q < 0:
            raise ValueError("negative quantity for %r" % (name,))
        if q == 0:
            continue
        key = name
        if normalize:
            if _normalize is None:  # pragma: no cover - defensive
                raise RuntimeError("taxonomy normalisation unavailable")
            key = _normalize(name)
        out[key] = out.get(key, 0) + q
    return out


# --------------------------------------------------------------------------- #
# The four metrics (single design)
# --------------------------------------------------------------------------- #
def feature_name_accuracy(predicted, ground_truth, *, normalize=False):
    """FNA (Eq. 1): fraction of ground-truth feature *names* also predicted.

    Presence-based over names (quantity ignored). If the design has no
    ground-truth features, returns 1.0 when nothing is predicted else 0.0.
    """
    pred = _clean_counts(predicted, normalize=normalize)
    gt = _clean_counts(ground_truth, normalize=normalize)
    if not gt:
        return 1.0 if not pred else 0.0
    correct = len(set(pred) & set(gt))
    return correct / len(gt)


def feature_quantity_accuracy(predicted, ground_truth, *, normalize=False):
    """FQA (Eq. 2): true-positive quantity / ground-truth quantity.

    True-positive quantity is the count actually matched, i.e. for each feature
    ``min(predicted_qty, ground_truth_qty)`` summed. Over-prediction does not
    inflate the numerator. Empty ground truth returns 1.0 iff no prediction.
    """
    pred = _clean_counts(predicted, normalize=normalize)
    gt = _clean_counts(ground_truth, normalize=normalize)
    gt_total = sum(gt.values())
    if gt_total == 0:
        return 1.0 if not pred else 0.0
    tp = sum(min(pred.get(f, 0), q) for f, q in gt.items())
    return tp / gt_total


def hallucination_rate(predicted, ground_truth, *, normalize=False):
    """HR (Eq. 3): hallucinated quantity / total predicted quantity.

    Hallucinated quantity is the predicted count NOT backed by ground truth:
    ``predicted_total - true_positive_quantity`` (over-counts of a real feature
    plus every count of a feature that does not exist). If nothing is predicted,
    returns 0.0 (no hallucination possible).
    """
    pred = _clean_counts(predicted, normalize=normalize)
    gt = _clean_counts(ground_truth, normalize=normalize)
    pred_total = sum(pred.values())
    if pred_total == 0:
        return 0.0
    tp = sum(min(q, gt.get(f, 0)) for f, q in pred.items())
    return (pred_total - tp) / pred_total


def mean_absolute_error(predicted, ground_truth, *, normalize=False,
                        feature_space=None):
    """MAE (Eq. 4): mean |ground_truth_qty - predicted_qty| over features.

    The set of features evaluated is the union of predicted and ground-truth
    feature names, unless ``feature_space`` (an iterable of names) is given, in
    which case exactly those n features are scored (missing => quantity 0). With
    an empty evaluation set MAE is 0.0.
    """
    pred = _clean_counts(predicted, normalize=normalize)
    gt = _clean_counts(ground_truth, normalize=normalize)
    if feature_space is not None:
        feats = [
            (_normalize(f) if (normalize and _normalize) else f)
            for f in feature_space
        ]
        # de-duplicate while preserving order
        seen = set()
        feats = [f for f in feats if not (f in seen or seen.add(f))]
    else:
        feats = sorted(set(pred) | set(gt))
    if not feats:
        return 0.0
    total = sum(abs(gt.get(f, 0) - pred.get(f, 0)) for f in feats)
    return total / len(feats)


def evaluate_design(predicted, ground_truth, *, normalize=False,
                    feature_space=None):
    """All four metrics for one CAD design as a dict."""
    return {
        "fna": feature_name_accuracy(predicted, ground_truth,
                                     normalize=normalize),
        "fqa": feature_quantity_accuracy(predicted, ground_truth,
                                         normalize=normalize),
        "hr": hallucination_rate(predicted, ground_truth, normalize=normalize),
        "mae": mean_absolute_error(predicted, ground_truth,
                                   normalize=normalize,
                                   feature_space=feature_space),
    }


# --------------------------------------------------------------------------- #
# Dataset-level aggregation (mean over designs)
# --------------------------------------------------------------------------- #
def afr_scorecard(samples, *, normalize=False, feature_space=None):
    """Aggregate the four metrics over a dataset of CAD designs.

    ``samples``: iterable of ``(predicted, ground_truth)`` count-dict pairs.
    Returns per-metric means (``mean_fna``/``mean_fqa``/``mean_hr``/``mean_mae``)
    plus ``n``. Matches the paper's per-model reporting (averaged over designs).
    """
    rows = [
        evaluate_design(p, g, normalize=normalize, feature_space=feature_space)
        for p, g in samples
    ]
    if not rows:
        raise ValueError("no samples")
    n = len(rows)
    return {
        "n": n,
        "mean_fna": sum(r["fna"] for r in rows) / n,
        "mean_fqa": sum(r["fqa"] for r in rows) / n,
        "mean_hr": sum(r["hr"] for r in rows) / n,
        "mean_mae": sum(r["mae"] for r in rows) / n,
        "rows": tuple(rows),
    }
