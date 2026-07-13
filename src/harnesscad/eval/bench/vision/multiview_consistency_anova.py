"""img2cadsvg_multiview_consistency -- validated multi-view consistency metric.

Img2CAD renders 36 views per object and argues its Structured-Visual-Geometry
conditioning yields **multi-view consistency**: the CAD-generation quality metrics
"remain consistent across various viewpoints" (paper, Sec. V.D).  It validates
this with a statistical test:

    "we conducted an ANOVA test, which showed no significant correlation between
    the viewpoint and the qualitative metrics (p < 0.01).  This shows that our
    method effectively preserves multi-view consistency."

The one-way ANOVA F-statistic and the associated dispersion summaries are
deterministic statistics.  This module implements:

* :func:`one_way_anova` -- the one-way ANOVA F-statistic over grouped samples
  (here: metric samples grouped by viewpoint bucket) with between/within sums of
  squares and degrees of freedom;
* :func:`f_critical` -- a small deterministic table lookup for common
  significance levels, plus :func:`is_consistent` which declares multi-view
  consistency when ``F < F_crit`` (i.e. viewpoint explains no significant
  variance);
* :func:`consistency_score` -- a scale-free spread summary (1 minus the
  coefficient of variation, clamped to ``[0, 1]``) that quantifies how flat a
  metric is across viewpoints.

Pure stdlib, deterministic; no learned components and no external stats library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AnovaResult:
    f_statistic: float
    df_between: int
    df_within: int
    ss_between: float
    ss_within: float
    grand_mean: float


def one_way_anova(groups: list[list[float]]) -> AnovaResult:
    """One-way ANOVA over ``groups`` (one list of metric samples per viewpoint).

    Requires >= 2 groups, each non-empty, and at least one group with >1 sample
    (so within-group variance is defined).  ``F = MS_between / MS_within``.
    """
    groups = [g for g in groups]
    if len(groups) < 2:
        raise ValueError("need at least 2 groups")
    if any(len(g) == 0 for g in groups):
        raise ValueError("every group must be non-empty")
    n_total = sum(len(g) for g in groups)
    k = len(groups)
    if n_total <= k:
        raise ValueError("need more samples than groups for within-variance")
    grand = sum(sum(g) for g in groups) / n_total
    ss_between = 0.0
    ss_within = 0.0
    for g in groups:
        m = sum(g) / len(g)
        ss_between += len(g) * (m - grand) ** 2
        for v in g:
            ss_within += (v - m) ** 2
    df_between = k - 1
    df_within = n_total - k
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f = math.inf if ms_within == 0.0 else ms_between / ms_within
    return AnovaResult(
        f_statistic=f,
        df_between=df_between,
        df_within=df_within,
        ss_between=ss_between,
        ss_within=ss_within,
        grand_mean=grand,
    )


# A small, deterministic table of upper-tail F critical values, F(df1, df2).
# Keyed by (alpha, df1, df2); values from standard F-distribution tables.
_F_TABLE: dict[tuple[float, int, int], float] = {
    (0.01, 1, 10): 10.04,
    (0.01, 2, 10): 7.56,
    (0.01, 3, 10): 6.55,
    (0.01, 2, 30): 5.39,
    (0.01, 3, 30): 4.51,
    (0.05, 1, 10): 4.96,
    (0.05, 2, 10): 4.10,
    (0.05, 3, 10): 3.71,
    (0.05, 2, 30): 3.32,
    (0.05, 3, 30): 2.92,
}


def f_critical(alpha: float, df1: int, df2: int) -> float:
    """Look up an F critical value; falls back to the nearest tabulated df2.

    Raises ``KeyError`` if ``(alpha, df1)`` is not tabulated at all.
    """
    if (alpha, df1, df2) in _F_TABLE:
        return _F_TABLE[(alpha, df1, df2)]
    candidates = [
        (a, d1, d2) for (a, d1, d2) in _F_TABLE if a == alpha and d1 == df1
    ]
    if not candidates:
        raise KeyError(f"no tabulated F for alpha={alpha}, df1={df1}")
    # nearest tabulated df2
    best = min(candidates, key=lambda key: abs(key[2] - df2))
    return _F_TABLE[best]


def is_consistent(result: AnovaResult, alpha: float = 0.01) -> bool:
    """True iff viewpoint explains no significant variance (``F < F_crit``).

    Reproduces the paper's conclusion criterion: "no significant correlation
    between the viewpoint and the qualitative metrics".
    """
    crit = f_critical(alpha, result.df_between, result.df_within)
    return result.f_statistic < crit


def consistency_score(values: list[float]) -> float:
    """Scale-free flatness of a metric across viewpoints, in ``[0, 1]``.

    ``1 - CV`` where ``CV`` is the coefficient of variation (std / |mean|),
    clamped to ``[0, 1]``.  ``1`` = perfectly consistent across viewpoints.
    """
    if len(values) < 2:
        raise ValueError("need at least 2 values")
    mean = sum(values) / len(values)
    if mean == 0.0:
        return 0.0 if any(v != 0.0 for v in values) else 1.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    cv = math.sqrt(var) / abs(mean)
    return max(0.0, min(1.0, 1.0 - cv))
