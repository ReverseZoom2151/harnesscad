"""B-Spline geometric-property metrics and augmentation statistics.

Chen, Shu, Hong, Taber, Li & Klenk, *Learning From Design Procedure To Generate
CAD Programs for Data Augmentation* (NeurIPS 2025 Workshop), Sec. 4.1-4.3.

The paper evaluates whether the augmentation enriches organic (B-Spline)
geometry using a small set of deterministic geometric proxies:

  * per-object **B-Spline ratio** (Eq. 1):
        beta_i = [ (f_bi / f_i) + (e_bi / e_i) ] / 2
    where f_i / f_bi are faces / B-Spline faces and e_i / e_bi are curves /
    B-Spline curves;
  * dataset **geometric properties** (Table 1): avg #STEP-lines, avg #faces,
    avg #curves, % of objects with any B-Spline face, % with any B-Spline curve,
    and mean B-Spline ratio;
  * **diversity of shape complexity** (Fig. 3): the distribution of per-object
    B-Spline ratios across bins -- the paper's method spreads more evenly over
    the [0, 1] range than the DeepCAD-derived baselines (which pile up at 0).

This module computes those metrics from the synthesised programs produced by
``datagen.designproc_program_synthesis`` (or from any object exposing face/curve
counts), plus an augmentation-vs-baseline comparison and a validity/diversity
gate. The learned generation model is external; these are the paper's proxy
metrics only. Pure functions; stdlib only; no randomness, no wall clock.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

from harnesscad.data.datagen.designproc_program_synthesis import program_totals


# ---------------------------------------------------------------------------
# Per-object B-Spline ratio (Eq. 1)
# ---------------------------------------------------------------------------

def bspline_ratio(faces: int, bspline_faces: int,
                  curves: int, bspline_curves: int) -> float:
    """B-Spline ratio beta = [(f_b/f) + (e_b/e)] / 2 (paper Eq. 1).

    A face/curve group with zero elements contributes a zero sub-ratio (rather
    than a division error). Result lies in ``[0, 1]``.
    """
    for name, v in (("faces", faces), ("bspline_faces", bspline_faces),
                    ("curves", curves), ("bspline_curves", bspline_curves)):
        if v < 0:
            raise ValueError("%s must be >= 0" % name)
    if bspline_faces > faces or bspline_curves > curves:
        raise ValueError("B-Spline count cannot exceed total count")
    fr = (bspline_faces / faces) if faces else 0.0
    er = (bspline_curves / curves) if curves else 0.0
    return (fr + er) / 2.0


def program_bspline_ratio(program: Sequence[dict]) -> float:
    """B-Spline ratio of a synthesised program (via its face/curve totals)."""
    t = program_totals(program)
    return bspline_ratio(t["faces"], t["bspline_faces"],
                         t["curves"], t["bspline_curves"])


# ---------------------------------------------------------------------------
# Dataset geometric properties (Table 1)
# ---------------------------------------------------------------------------

def _mean(xs: Sequence[float]) -> float:
    return (sum(xs) / len(xs)) if xs else 0.0


def geometric_properties(programs: Sequence[Sequence[dict]]) -> Dict[str, float]:
    """Compute the Table-1 geometric properties for a set of programs.

    Returns a dict with:
      ``n``, ``avg_lines``, ``avg_faces``, ``avg_curves``,
      ``frac_with_bspline_faces``, ``frac_with_bspline_curves``,
      ``mean_bspline_ratio``.
    """
    if not programs:
        return {
            "n": 0, "avg_lines": 0.0, "avg_faces": 0.0, "avg_curves": 0.0,
            "frac_with_bspline_faces": 0.0, "frac_with_bspline_curves": 0.0,
            "mean_bspline_ratio": 0.0,
        }
    lines, faces, curves, ratios = [], [], [], []
    n_bf, n_bc = 0, 0
    for prog in programs:
        t = program_totals(prog)
        lines.append(t["lines"])
        faces.append(t["faces"])
        curves.append(t["curves"])
        ratios.append(bspline_ratio(t["faces"], t["bspline_faces"],
                                    t["curves"], t["bspline_curves"]))
        if t["bspline_faces"] > 0:
            n_bf += 1
        if t["bspline_curves"] > 0:
            n_bc += 1
    n = len(programs)
    return {
        "n": n,
        "avg_lines": _mean(lines),
        "avg_faces": _mean(faces),
        "avg_curves": _mean(curves),
        "frac_with_bspline_faces": n_bf / n,
        "frac_with_bspline_curves": n_bc / n,
        "mean_bspline_ratio": _mean(ratios),
    }


# ---------------------------------------------------------------------------
# Diversity of shape complexity (Fig. 3): B-Spline-ratio histogram
# ---------------------------------------------------------------------------

def ratio_histogram(ratios: Sequence[float], n_bins: int = 10) -> List[int]:
    """Bin per-object B-Spline ratios into ``n_bins`` equal bins over [0, 1].

    A ratio of exactly 1.0 lands in the last bin. Returns bin counts.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    hist = [0] * n_bins
    for r in ratios:
        if not (0.0 <= r <= 1.0):
            raise ValueError("ratio %r outside [0, 1]" % r)
        idx = min(n_bins - 1, int(r * n_bins))
        hist[idx] += 1
    return hist


def distribution_entropy(hist: Sequence[int]) -> float:
    """Shannon entropy (bits) of a histogram -- a scalar diversity measure.

    An evenly-spread distribution (the paper's goal, Fig. 3) has high entropy;
    a spike at a single bin (the DeepCAD baselines piling up at ratio 0) has
    entropy 0. Normalised is available via :func:`normalized_diversity`.
    """
    total = sum(hist)
    if total == 0:
        return 0.0
    ent = 0.0
    for c in hist:
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


def normalized_diversity(hist: Sequence[int]) -> float:
    """Entropy normalised to ``[0, 1]`` by ``log2(n_bins)`` (1 = perfectly even)."""
    n = len(hist)
    if n <= 1:
        return 0.0
    return distribution_entropy(hist) / math.log2(n)


def diversity_report(programs: Sequence[Sequence[dict]],
                     n_bins: int = 10) -> dict:
    """Full diversity report: ratios, histogram, entropy, normalised diversity."""
    ratios = [program_bspline_ratio(p) for p in programs]
    hist = ratio_histogram(ratios, n_bins)
    return {
        "n": len(programs),
        "ratios": ratios,
        "histogram": hist,
        "entropy_bits": distribution_entropy(hist),
        "normalized_diversity": normalized_diversity(hist),
        "mean_ratio": _mean(ratios),
    }


# ---------------------------------------------------------------------------
# Validity / diversity gate on generated programs
# ---------------------------------------------------------------------------

def program_is_valid(program: Sequence[dict], require_bspline: bool = False,
                     min_faces: int = 1) -> bool:
    """Structural validity stand-in for a synthesised program.

    A program is admitted if it exports, has at least ``min_faces`` faces and
    (optionally) retains B-Spline geometry. The paper's real gate additionally
    compiles to a watertight B-rep with OpenCascade / DTGBrepGen validity --
    that geometric check is external.
    """
    if not program:
        return False
    if not any(c.get("op") == "export" for c in program):
        return False
    t = program_totals(program)
    if t["faces"] < min_faces:
        return False
    if require_bspline and (t["bspline_faces"] + t["bspline_curves"]) == 0:
        return False
    return True


def filter_valid(programs: Sequence[Sequence[dict]],
                 require_bspline: bool = False) -> List[Sequence[dict]]:
    """Keep only programs passing :func:`program_is_valid`."""
    return [p for p in programs if program_is_valid(p, require_bspline)]


# ---------------------------------------------------------------------------
# Augmentation-vs-baseline comparison (Table 1 / Table 2 style)
# ---------------------------------------------------------------------------

def augmentation_gain(baseline: Sequence[Sequence[dict]],
                      augmented: Sequence[Sequence[dict]]) -> dict:
    """Compare augmented programs against a baseline set (paper Tables 1-2).

    Returns the two property blocks plus deltas on the headline metrics
    (mean B-Spline ratio, fraction with B-Spline faces/curves, diversity).
    Positive deltas mean the augmentation enriched organic geometry.
    """
    base = geometric_properties(baseline)
    aug = geometric_properties(augmented)
    base_div = diversity_report(baseline) if baseline else {"normalized_diversity": 0.0}
    aug_div = diversity_report(augmented) if augmented else {"normalized_diversity": 0.0}
    return {
        "baseline": base,
        "augmented": aug,
        "delta_mean_bspline_ratio": aug["mean_bspline_ratio"] - base["mean_bspline_ratio"],
        "delta_frac_bspline_faces": aug["frac_with_bspline_faces"] - base["frac_with_bspline_faces"],
        "delta_frac_bspline_curves": aug["frac_with_bspline_curves"] - base["frac_with_bspline_curves"],
        "delta_normalized_diversity": (
            aug_div["normalized_diversity"] - base_div["normalized_diversity"]),
    }
