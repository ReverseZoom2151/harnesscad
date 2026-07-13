"""MUSE judge-vs-human agreement protocol (deterministic, stdlib-only).

Re-implements the deterministic statistics of
``scripts/bench_step3/agreement_three_judges.py`` from the muse-benchmark repo
(Dong et al., "MUSE") without numpy/scipy. MUSE validates its automatic VLM
judge by measuring how well its scores agree with a human panel across THREE
aggregation levels:

  * Item   -- one pair per (case, model, sample, rubric-item): finest grain.
  * Cell   -- one pair per (case, model, sample) using the overall score.
  * System -- one pair per model, averaging cell scores within the model:
              the ranking the leaderboard ultimately reports.

For each level it reports Pearson, Spearman and Kendall (tau-b) correlation, the
mean human / mean judge score, and the signed ``bias`` = mean(judge) - mean(human)
(is the judge systematically generous or harsh?). Item-level correlations, when
there are enough pairs, also get a seeded bootstrap 95% confidence interval.

Everything here is deterministic: correlations are closed-form, and the
bootstrap uses ``random.Random(seed)`` so a given input always yields the same
interval. Correlations of a constant series are reported as NaN (undefined),
matching the repo's ``safe_corr`` guard.

The reusable pieces:
  * ``pearson`` / ``spearman`` / ``kendall_tau_b`` -- stdlib correlations
    (kendall tau-b handles ties, unlike a ranking-only Kendall);
  * ``bootstrap_ci`` -- seeded percentile CI for any correlation of paired data;
  * ``agreement_report`` -- the full Item/Cell/System protocol from paired rows.

No wall clock; randomness is seeded.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

NAN = float("nan")


# --- Rank helper -------------------------------------------------------------

def _average_ranks(values):
    """Fractional ranks with ties averaged (1-based), preserving input order."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank over the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


# --- Correlations ------------------------------------------------------------

def pearson(x, y):
    """Pearson product-moment correlation; NaN if either series is constant."""
    n = len(x)
    if n < 2 or len(y) != n:
        return NAN
    mx = sum(x) / n
    my = sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0.0 or syy <= 0.0:
        return NAN
    return sxy / math.sqrt(sxx * syy)


def spearman(x, y):
    """Spearman rank correlation = Pearson on average ranks (ties handled)."""
    if len(x) < 2 or len(y) != len(x):
        return NAN
    return pearson(_average_ranks(x), _average_ranks(y))


def kendall_tau_b(x, y):
    """Kendall tau-b correlation over paired data (ties corrected).

    tau_b = (C - D) / sqrt((C + D + Tx) * (C + D + Ty)) where C/D are concordant
    / discordant pairs and Tx/Ty are pairs tied only on x / only on y. NaN if the
    denominator is 0 (e.g. a constant series).
    """
    n = len(x)
    if n < 2 or len(y) != n:
        return NAN
    concordant = discordant = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + tx) * (concordant + discordant + ty))
    if denom <= 0.0:
        return NAN
    return (concordant - discordant) / denom


# --- Bootstrap CI ------------------------------------------------------------

def bootstrap_ci(corr_fn, x, y, n_boot=2000, seed=42, lo_pct=2.5, hi_pct=97.5):
    """Seeded percentile bootstrap CI for ``corr_fn`` over paired (x, y).

    Resamples pair indices with replacement ``n_boot`` times using
    ``random.Random(seed)``; discards resamples whose correlation is NaN; returns
    (lo, hi) percentiles, or (NaN, NaN) if too few finite resamples.
    """
    n = len(x)
    if n < 2:
        return (NAN, NAN)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        v = corr_fn([x[i] for i in idx], [y[i] for i in idx])
        if v == v:  # not NaN
            vals.append(v)
    if len(vals) < 2:
        return (NAN, NAN)
    return (_percentile(vals, lo_pct), _percentile(vals, hi_pct))


def _percentile(values, pct):
    """Linear-interpolation percentile (numpy default), deterministic."""
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (pct / 100.0) * (len(s) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return s[lo]
    frac = rank - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


# --- The three-level agreement protocol --------------------------------------

_CORRS = (("pearson", pearson), ("spearman", spearman), ("kendall", kendall_tau_b))


def _level_stats(pairs, *, with_ci, min_ci_n, n_boot, seed):
    """Correlations + bias for a list of (human, judge) pairs."""
    n = len(pairs)
    h = [float(p[0]) for p in pairs]
    l = [float(p[1]) for p in pairs]
    row = {"n": n}
    if n == 0:
        for name, _ in _CORRS:
            row[name] = (NAN, None, None)
        row["mean_human"] = NAN
        row["mean_judge"] = NAN
        row["bias"] = NAN
        return row
    row["mean_human"] = sum(h) / n
    row["mean_judge"] = sum(l) / n
    row["bias"] = row["mean_judge"] - row["mean_human"]
    for name, fn in _CORRS:
        pt = fn(h, l)
        if with_ci and n >= min_ci_n:
            lo, hi = bootstrap_ci(fn, h, l, n_boot=n_boot, seed=seed)
            row[name] = (pt, lo, hi)
        else:
            row[name] = (pt, None, None)
    return row


def agreement_report(item_pairs, cell_pairs, *, min_ci_n=30, n_boot=2000, seed=42):
    """Full Item / Cell / System agreement report for one judge.

    item_pairs : iterable of (human_score, judge_score) at rubric-item grain.
    cell_pairs : iterable of (human_score, judge_score, model_label) at
                 (case, model, sample) grain; model_label drives the System
                 level (per-model means of the cell pairs).

    Returns {"Item": row, "Cell": row, "System": row} where each row has n,
    mean_human, mean_judge, bias, and (point, lo, hi) tuples for pearson,
    spearman and kendall. Only the Item level (n >= min_ci_n) carries CIs, matching
    the repo; lo/hi are None otherwise.
    """
    item_pairs = [(float(a), float(b)) for a, b in item_pairs]
    cell_list = [(float(a), float(b), m) for a, b, m in cell_pairs]

    by_model_h = defaultdict(list)
    by_model_l = defaultdict(list)
    for h, l, m in cell_list:
        by_model_h[m].append(h)
        by_model_l[m].append(l)
    models = sorted(by_model_h)
    system_pairs = [
        (sum(by_model_h[m]) / len(by_model_h[m]), sum(by_model_l[m]) / len(by_model_l[m]))
        for m in models
    ]

    return {
        "Item": _level_stats(item_pairs, with_ci=True, min_ci_n=min_ci_n,
                             n_boot=n_boot, seed=seed),
        "Cell": _level_stats([(h, l) for h, l, _ in cell_list], with_ci=False,
                             min_ci_n=min_ci_n, n_boot=n_boot, seed=seed),
        "System": _level_stats(system_pairs, with_ci=False, min_ci_n=min_ci_n,
                               n_boot=n_boot, seed=seed),
    }
