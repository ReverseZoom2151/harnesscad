"""Empirical-CDF threshold selection for GIFT geometric feedback.

The GIFT paper picks its filtering thresholds "empirically from the cumulative
distribution of inference-time IoU scores" (Section 3.1): tau_low is set where
roughly 10% of generated programs fall below it (degenerate / non-executable),
and tau_valid where roughly 40% fall below it (separating recoverable near-miss
from high-fidelity solutions). tau_match is fixed near 1.0.

This module implements that data-driven choice deterministically from a pool of
observed IoU scores, so the SRS/FDA bands adapt to the model's actual inference
behaviour instead of using hard-coded constants.

Deterministic, stdlib-only.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right


def empirical_cdf(scores):
    """Return sorted unique values and their cumulative fractions (<= v).

    Result is a list of (value, cumulative_fraction) with the fraction being
    the share of scores less than or equal to that value.
    """
    if not scores:
        raise ValueError("scores must be non-empty")
    ordered = sorted(float(s) for s in scores)
    n = len(ordered)
    out = []
    seen = set()
    for v in ordered:
        if v in seen:
            continue
        seen.add(v)
        out.append((v, bisect_right(ordered, v) / n))
    return out


def fraction_below(scores, x):
    """Share of ``scores`` strictly below ``x`` (the empirical CDF just under x)."""
    if not scores:
        raise ValueError("scores must be non-empty")
    ordered = sorted(float(s) for s in scores)
    return bisect_left(ordered, float(x)) / len(ordered)


def quantile(scores, fraction):
    """Value v such that (approximately) ``fraction`` of scores fall below v.

    Uses the lower-nearest-rank convention so the returned value is one of the
    observed scores; fraction is clamped to [0, 1]. This is the threshold that
    places ``fraction`` of the mass below it.
    """
    if not scores:
        raise ValueError("scores must be non-empty")
    if not (0.0 <= fraction <= 1.0):
        raise ValueError("fraction must lie in [0, 1]")
    ordered = sorted(float(s) for s in scores)
    n = len(ordered)
    idx = int(round(fraction * (n - 1)))
    if idx < 0:
        idx = 0
    elif idx > n - 1:
        idx = n - 1
    return ordered[idx]


def select_gift_thresholds(scores, low_fraction=0.10, valid_fraction=0.40,
                           tau_match=0.99):
    """Choose (tau_low, tau_valid, tau_match) from an IoU pool.

    tau_low is the ``low_fraction`` quantile (paper: ~0.10 -> ~0.5), tau_valid
    the ``valid_fraction`` quantile (paper: ~0.40 -> ~0.9). tau_match is fixed.
    The result is monotone-repaired so tau_low <= tau_valid <= tau_match.
    """
    if valid_fraction < low_fraction:
        raise ValueError("valid_fraction must be >= low_fraction")
    if not (0.0 <= tau_match <= 1.0):
        raise ValueError("tau_match must lie in [0, 1]")
    tau_low = quantile(scores, low_fraction)
    tau_valid = quantile(scores, valid_fraction)
    tau_valid = max(tau_valid, tau_low)
    tau_match = max(tau_match, tau_valid)
    return {"tau_low": tau_low, "tau_valid": tau_valid, "tau_match": tau_match}


def band_mass(scores, tau_low, tau_valid, tau_match):
    """Fraction of the pool landing in each GIFT band, given thresholds.

    Returns reject / near_miss(FDA) / valid(SRS) / match shares that sum to 1.
    Useful to check that chosen thresholds reproduce the intended split
    (~10% reject, ~40% below tau_valid, etc.).
    """
    if not (0.0 <= tau_low <= tau_valid <= tau_match <= 1.0):
        raise ValueError("require 0 <= tau_low <= tau_valid <= tau_match <= 1")
    n = len(scores)
    if n == 0:
        raise ValueError("scores must be non-empty")
    reject = fraction_below(scores, tau_low)
    below_valid = fraction_below(scores, tau_valid)
    below_match = fraction_below(scores, tau_match)
    return {
        "reject": reject,
        "near_miss": below_valid - reject,
        "valid": below_match - below_valid,
        "match": 1.0 - below_match,
    }
