"""Text2CAD-Bench L1-L3 scorecard and leaderboard aggregation.

Deterministic re-encoding of the Text2CAD-Bench main evaluation protocol (Wang
et al., "Text2CAD-Bench", Sections 3.3.3 / 4.1 / 4.2, Tables 1-3). The protocol
jointly reports three complementary metrics per (model, level, prompt-style):

  * CD  -- Chamfer Distance (x10^3, lower better),
  * IR  -- Invalidity Rate (%, lower better),
  * IoU -- Intersection over Union (higher better).

Critical protocol rules implemented here (the genuinely-new benchmark logic):

  1. Best-of-N retry: each sample allows up to N attempts with error feedback;
     the reported result is the *best* attempt, and a sample is invalid only if
     *all* attempts fail (Section 4.1.2).
  2. CD and IoU are averaged over *valid (successfully executed) samples only*;
     invalid samples contribute to IR but not to CD/IoU (Section 4.1.2:
     "For invalid samples, we do not calculate their CD and IoU").
  3. Sample-count weighted averages across levels (Section 4.2).
  4. Degradation analysis: L1 -> L3 CD ratio (paper: 1.3-2.1x).
  5. Survivorship-bias caveat: flag comparisons where IR rises yet CD improves,
     since CD/IoU over a shrunken valid set can be misleadingly optimistic
     (Section 4.4, "Survivorship bias caveat").
  6. Leaderboard ranking with joint (multi-metric) tie handling.

This does not overlap ``bench/muse_scorecard`` (assemblability funnel) or
``bench/engdesign_*``: it is the CD/IR/IoU joint parametric-CAD scorecard with
retry-best and survivorship logic specific to Text2CAD-Bench. Per-sample results
are injected. No wall clock, no randomness.
"""

from __future__ import annotations


def resolve_sample(sample):
    """Reduce a multi-attempt sample to its best (valid) result.

    sample : mapping with key "attempts" -> iterable of attempt dicts, each
        {"valid": bool, "cd": float|None, "iou": float|None}. A single-attempt
        sample may instead pass "valid"/"cd"/"iou" directly.

    Best attempt = the valid attempt with the lowest CD (ties broken by higher
    IoU). Returns {"valid": bool, "cd": float|None, "iou": float|None}. If no
    attempt is valid the sample is invalid (cd/iou None).
    """
    if "attempts" in sample:
        attempts = list(sample["attempts"])
    else:
        attempts = [sample]
    valid = [a for a in attempts if a.get("valid")]
    if not valid:
        return {"valid": False, "cd": None, "iou": None}
    best = min(valid, key=lambda a: (float(a.get("cd", float("inf"))),
                                     -float(a.get("iou", 0.0))))
    return {"valid": True, "cd": float(best["cd"]), "iou": float(best["iou"])}


def cell_scorecard(samples):
    """Aggregate one (model, level, style) cell into CD / IR / IoU.

    samples : iterable of multi-attempt sample records (see ``resolve_sample``).

    Returns a dict:
      n_total, n_valid : sample counts,
      ir               : invalidity rate in percent (invalid / total * 100),
      cd, iou          : means over valid samples only (None if none valid).
    """
    resolved = [resolve_sample(s) for s in samples]
    n_total = len(resolved)
    if n_total == 0:
        raise ValueError("no samples")
    valid = [r for r in resolved if r["valid"]]
    n_valid = len(valid)
    ir = 100.0 * (n_total - n_valid) / n_total
    cd = sum(r["cd"] for r in valid) / n_valid if n_valid else None
    iou = sum(r["iou"] for r in valid) / n_valid if n_valid else None
    return {"n_total": n_total, "n_valid": n_valid, "ir": ir,
            "cd": cd, "iou": iou}


def weighted_average(cells):
    """Sample-count weighted average of CD / IR / IoU across cells.

    cells : iterable of cell scorecards (from ``cell_scorecard``).

    IR is weighted by n_total; CD and IoU are weighted by n_valid (the set they
    are defined over). Returns {ir, cd, iou, n_total, n_valid}; cd/iou are None
    if no cell has a valid sample.
    """
    cells = list(cells)
    tot = sum(c["n_total"] for c in cells)
    if tot == 0:
        raise ValueError("no samples")
    ir = sum(c["ir"] * c["n_total"] for c in cells) / tot
    valid = sum(c["n_valid"] for c in cells)
    if valid:
        cd = sum(c["cd"] * c["n_valid"] for c in cells if c["cd"] is not None) \
            / valid
        iou = sum(c["iou"] * c["n_valid"] for c in cells if c["iou"] is not None) \
            / valid
    else:
        cd = iou = None
    return {"ir": ir, "cd": cd, "iou": iou, "n_total": tot, "n_valid": valid}


def degradation_ratio(cell_low, cell_high):
    """CD degradation ratio between two levels (e.g. L1 -> L3).

    Returns {cd_ratio, ir_delta, iou_delta}. cd_ratio = high.cd / low.cd
    (None if either CD undefined or low.cd == 0).
    """
    lo, hi = cell_low["cd"], cell_high["cd"]
    ratio = (hi / lo) if (lo not in (None, 0) and hi is not None) else None
    return {
        "cd_ratio": ratio,
        "ir_delta": cell_high["ir"] - cell_low["ir"],
        "iou_delta": (None if cell_high["iou"] is None or cell_low["iou"] is None
                      else cell_high["iou"] - cell_low["iou"]),
    }


def survivorship_flag(cell_a, cell_b):
    """Flag a survivorship-bias-suspect comparison (a -> b).

    Returns True when IR *rises* from a to b while CD *improves* (drops) -- the
    exact pattern the paper warns inflates apparent quality on a shrunken valid
    set (Section 4.4). CD undefined on either side -> False.
    """
    if cell_a["cd"] is None or cell_b["cd"] is None:
        return False
    return cell_b["ir"] > cell_a["ir"] and cell_b["cd"] < cell_a["cd"]


def rank_leaderboard(entries, metric="cd"):
    """Rank models by a single metric, returning entries with a 1-based rank.

    entries : iterable of dicts each with "model" and the chosen metric key.
    metric  : "cd" or "ir" (ascending, lower better) or "iou" (descending).
    Entries whose metric is None sort last. Ties share the metric value but get
    distinct ranks in a stable model-name order.

    Returns a list of {model, value, rank} sorted best-first.
    """
    ascending = metric in ("cd", "ir")
    rows = list(entries)

    def key(e):
        v = e.get(metric)
        missing = v is None
        val = float(v) if not missing else 0.0
        # None always last; among present, sort by metric then model name.
        return (missing, val if ascending else -val, str(e.get("model", "")))

    ordered = sorted(rows, key=key)
    return [{"model": e.get("model"), "value": e.get(metric), "rank": i}
            for i, e in enumerate(ordered, 1)]


def prompt_style_comparison(geo_cell, seq_cell):
    """Compare geometric vs sequence prompt cells at one level.

    Returns {cd_better, ir_better, iou_better} each "geo"/"seq"/"tie",
    reflecting the paper's finding that geometric prompts win on L1-L2 while
    sequence prompts can win on L3.
    """
    def cmp_lower(a, b):
        if a is None or b is None:
            return "tie"
        if a < b:
            return "geo"
        if b < a:
            return "seq"
        return "tie"

    def cmp_higher(a, b):
        if a is None or b is None:
            return "tie"
        if a > b:
            return "geo"
        if b > a:
            return "seq"
        return "tie"

    return {
        "cd_better": cmp_lower(geo_cell["cd"], seq_cell["cd"]),
        "ir_better": cmp_lower(geo_cell["ir"], seq_cell["ir"]),
        "iou_better": cmp_higher(geo_cell["iou"], seq_cell["iou"]),
    }
