"""Joint reconstruction + segmentation analysis (joint SDF paper).

The paper reports (Table 4) Pearson correlations between reconstruction metrics
(CD, NC, F1) and segmentation metrics (mIoU, Acc., Consis.) across a set of
shapes, plus per-shape part counts (predicted vs ground truth).  All of this is
a deterministic statistical analysis of *already-computed* per-shape metrics --
no learned model is involved -- so it lives here as a scoring utility.

Provided:
  * ``pearson`` correlation of two aligned metric series,
  * ``correlation_table`` building the recon x seg grid of Table 4,
  * ``part_count_agreement`` (exact match rate and mean absolute error of the
    predicted vs GT part counts),
  * ``joint_score`` a single combined figure of merit (mean of a
    higher-is-better segmentation score and a normalised reconstruction score).
"""

from __future__ import annotations

import math


def pearson(xs, ys):
    """Pearson correlation r of two equal-length numeric series.

    Returns 0.0 when either series has zero variance (undefined correlation),
    which is the conventional deterministic fallback.
    """
    if len(xs) != len(ys):
        raise ValueError("series length mismatch")
    n = len(xs)
    if n == 0:
        raise ValueError("empty series")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0.0 or syy == 0.0:
        return 0.0
    return sxy / math.sqrt(sxx * syy)


def correlation_table(recon, seg):
    """Grid ``{recon_metric: {seg_metric: r}}`` (Table 4).

    ``recon`` and ``seg`` are dicts of ``metric_name -> list-of-per-shape-values``
    aligned by shape index.
    """
    table = {}
    for rname, rvals in recon.items():
        row = {}
        for sname, svals in seg.items():
            row[sname] = pearson(rvals, svals)
        table[rname] = row
    return table


def part_count_agreement(pred_counts, gt_counts):
    """Exact-match rate and mean absolute error of part counts across shapes."""
    if len(pred_counts) != len(gt_counts):
        raise ValueError("count series length mismatch")
    n = len(pred_counts)
    if n == 0:
        raise ValueError("empty series")
    exact = sum(1 for p, g in zip(pred_counts, gt_counts) if p == g) / n
    mae = sum(abs(p - g) for p, g in zip(pred_counts, gt_counts)) / n
    return {"exact_match": exact, "mae": mae}


def joint_score(seg_score, cd, *, cd_scale=1.0):
    """Single figure of merit combining a segmentation and a reconstruction term.

    ``seg_score`` is a higher-is-better value in ``[0, 1]`` (e.g. mIoU).  ``cd`` is
    a lower-is-better Chamfer distance; it is mapped to ``1/(1+cd/cd_scale)`` in
    ``(0, 1]`` so both terms share an orientation, then averaged.
    """
    if cd < 0:
        raise ValueError("chamfer distance must be non-negative")
    if cd_scale <= 0:
        raise ValueError("cd_scale must be positive")
    recon_term = 1.0 / (1.0 + cd / cd_scale)
    return 0.5 * (seg_score + recon_term)


def mean_std(values):
    """Population mean and standard deviation (the paper's mean +/- std)."""
    n = len(values)
    if n == 0:
        raise ValueError("empty series")
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / n
    return {"mean": m, "std": math.sqrt(var)}
