"""Deterministic diversity / novelty / coverage metrics for mechanical component feature vectors.

Motivated by "Generative AI and CAD Automation for Diverse and Novel Mechanical
Component Designs Under Data Constraints", which argues that under data
constraints a generator must be judged not only on validity but on whether it
produces DIVERSE and NOVEL components spanning the design space. This module
provides numeric measures for that: mean pairwise distance (diversity index),
nearest-neighbour spread, grid coverage of the design space, and novelty of
candidates relative to a reference (training) set. A categorical style-diversity
pair (Simpson / Shannon) complements the numeric metrics for style labels such
as 'five-spoke', 'multispoke', 'mesh', 'minimalist' discussed in the paper.

Distinction from sibling modules:
  * bench/feasibility_novelty.py operates on HUMAN RATINGS: Spearman /
    Mann-Whitney rank statistics and feasibility-vs-novelty Pareto fronts. It
    does not touch geometric feature vectors.
  * dataengine/distribution_audit.py works on CATEGORICAL op-tag HISTOGRAMS with
    KL / chi-square divergence against a target distribution.
  * THIS module works on NUMERIC component feature vectors (spoke_count,
    symmetry_order, rim_diameter, aspect ratios, ...) using distance-geometry:
    pairwise diversity, nearest-neighbour spread, occupancy coverage, and
    distance-to-reference novelty. No rank stats, no op-tag histograms.

A "component" is a feature vector: a tuple/list of floats. All functions are
deterministic and stdlib-only.
"""
from __future__ import annotations

from math import sqrt, log
from collections import Counter


def euclidean(a, b):
    """Euclidean distance between two feature vectors. ValueError on dim mismatch."""
    if len(a) != len(b):
        raise ValueError("dimension mismatch: %d vs %d" % (len(a), len(b)))
    return sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def normalized(components, bounds=None):
    """Return components scaled per-dimension into [0,1] using bounds.

    bounds: optional list of (min,max) per dimension; if None it is derived from
    the components themselves. Degenerate dimensions (min==max) map to 0.0.
    """
    comps = [list(map(float, c)) for c in components]
    if not comps:
        return []
    d = len(comps[0])
    for c in comps:
        if len(c) != d:
            raise ValueError("inconsistent dimensionality among components")
    if bounds is None:
        bounds = _bounds_from(comps)
    out = []
    for c in comps:
        row = []
        for i in range(d):
            lo, hi = bounds[i]
            span = hi - lo
            row.append(0.0 if span == 0 else (c[i] - lo) / span)
        out.append(row)
    return out


def _bounds_from(components):
    comps = [list(map(float, c)) for c in components]
    d = len(comps[0])
    for c in comps:
        if len(c) != d:
            raise ValueError("inconsistent dimensionality among components")
    lows = [min(c[i] for c in comps) for i in range(d)]
    highs = [max(c[i] for c in comps) for i in range(d)]
    return list(zip(lows, highs))


def pairwise_diversity(components, metric=euclidean):
    """Mean distance over all unordered pairs (standard diversity index).

    Returns 0.0 if fewer than 2 items.
    """
    n = len(components)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += metric(components[i], components[j])
            count += 1
    return total / count


def nearest_neighbor_distances(components, metric=euclidean):
    """For each item, the min distance to any OTHER item (spread / uniqueness).

    Single-item (or empty) sets: returns a list of 0.0 (no other item).
    """
    n = len(components)
    if n < 2:
        return [0.0] * n
    out = []
    for i in range(n):
        best = None
        for j in range(n):
            if i == j:
                continue
            dist = metric(components[i], components[j])
            if best is None or dist < best:
                best = dist
        out.append(best)
    return out


def _cell_index(component, bounds, bins):
    """Map a component to its integer grid cell tuple given per-dim bounds."""
    idx = []
    for value, (lo, hi) in zip(component, bounds):
        span = hi - lo
        if span == 0:
            b = 0
        else:
            b = int((float(value) - lo) / span * bins)
            if b >= bins:
                b = bins - 1
            elif b < 0:
                b = 0
        idx.append(b)
    return tuple(idx)


def _resolve_bounds(ref_or_bounds):
    """Accept explicit per-dim (min,max) bounds, or a reference component set."""
    if not ref_or_bounds:
        raise ValueError("ref_or_bounds must be non-empty")
    first = ref_or_bounds[0]
    # A (min,max) bound entry is a 2-item pair with min<=max; a reference set is
    # a list of feature vectors. Heuristic: treat as bounds if every entry is a
    # length-2 pair AND at least one entry is not a plausible feature vector is
    # ambiguous, so require an explicit shape: bounds entries are 2-tuples of
    # (lo,hi) with lo<=hi for all. Reference vectors may also be length-2, so
    # we prefer the explicit-bounds interpretation only when all pairs satisfy
    # lo<=hi.
    is_bounds = all(
        (hasattr(e, "__len__") and len(e) == 2 and float(e[0]) <= float(e[1]))
        for e in ref_or_bounds
    )
    if is_bounds:
        return [(float(lo), float(hi)) for lo, hi in ref_or_bounds]
    return _bounds_from(ref_or_bounds)


def coverage(components, ref_or_bounds=None, bins=8):
    """Occupancy-spread coverage of the design space.

    Discretizes each dimension into `bins` bins (using explicit per-dimension
    (min,max) bounds, or bounds derived from a reference component set, or from
    the components themselves when ref_or_bounds is None). Returns

        coverage = (# distinct occupied cells) / (# components)

    which lies in (0, 1]. 1.0 means every component lands in its own cell
    (maximal spread); lower values indicate clumping into shared cells.
    Returns 0.0 for an empty set.
    """
    n = len(components)
    if n == 0:
        return 0.0
    bounds = _bounds_from(components) if ref_or_bounds is None else _resolve_bounds(ref_or_bounds)
    occupied = {_cell_index(c, bounds, bins) for c in components}
    return len(occupied) / n


def grid_occupancy(components, bounds, bins=8):
    """Raw occupancy over the full discretized grid.

    Returns (occupied_cells, total_cells) where total_cells = bins ** d for the
    dimensionality d. Raises ValueError if bins ** d exceeds 10_000_000 (the
    grid would be too large to reason about) or if inputs are empty.
    """
    if not components:
        raise ValueError("components must be non-empty")
    resolved = _resolve_bounds(bounds)
    d = len(resolved)
    total_cells = bins ** d
    if total_cells > 10_000_000:
        raise ValueError("grid too large: bins**d = %d > 10_000_000" % total_cells)
    occupied = {_cell_index(c, resolved, bins) for c in components}
    return len(occupied), total_cells


def novelty_vs_reference(candidates, reference, k=1, metric=euclidean):
    """For each candidate, mean distance to its k nearest neighbours in reference.

    Higher = more novel (further from known/training components). Empty
    reference raises ValueError.
    """
    if not reference:
        raise ValueError("reference set must be non-empty")
    kk = min(k, len(reference))
    if kk < 1:
        raise ValueError("k must be >= 1")
    out = []
    for cand in candidates:
        dists = sorted(metric(cand, r) for r in reference)
        out.append(sum(dists[:kk]) / kk)
    return out


def novelty_score(candidate, reference, k=1, metric=euclidean):
    """Single-candidate convenience wrapper around novelty_vs_reference."""
    return novelty_vs_reference([candidate], reference, k=k, metric=metric)[0]


def is_novel(candidate, reference, threshold, k=1, metric=euclidean):
    """True if the candidate's novelty score meets/exceeds threshold."""
    return novelty_score(candidate, reference, k=k, metric=metric) >= threshold


def diversity_report(components, reference=None, bins=8):
    """Deterministic summary dict of diversity/coverage (and novelty if given).

    Keys: n, pairwise_diversity, mean_nn_distance, min_nn_distance, coverage;
    plus mean_novelty and max_novelty when a non-empty reference is provided.
    """
    n = len(components)
    nn = nearest_neighbor_distances(components)
    report = {
        "n": n,
        "pairwise_diversity": pairwise_diversity(components),
        "mean_nn_distance": (sum(nn) / len(nn)) if nn else 0.0,
        "min_nn_distance": min(nn) if nn else 0.0,
        "coverage": coverage(components, bins=bins) if n else 0.0,
    }
    if reference:
        nov = novelty_vs_reference(components, reference) if components else []
        report["mean_novelty"] = (sum(nov) / len(nov)) if nov else 0.0
        report["max_novelty"] = max(nov) if nov else 0.0
    return report


def simpson_diversity(labels):
    """Simpson diversity index (1 - sum p_i^2) over categorical style labels.

    0.0 for a single distinct label (or empty); approaches 1.0 as labels become
    numerous and evenly spread. Higher = more diverse.
    """
    labels = list(labels)
    total = len(labels)
    if total == 0:
        return 0.0
    counts = Counter(labels)
    return 1.0 - sum((c / total) ** 2 for c in counts.values())


def shannon_entropy(labels, base=2):
    """Shannon entropy over categorical style labels.

    0.0 for a single label (or empty); higher for uniform label mixes.
    """
    labels = list(labels)
    total = len(labels)
    if total == 0:
        return 0.0
    counts = Counter(labels)
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * (log(p) / log(base))
    return ent
