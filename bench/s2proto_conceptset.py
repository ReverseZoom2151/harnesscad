"""Concept-set diversity / alignment metrics from a pairwise SIMILARITY matrix.

Motivated by "Sketch2Prototype" (Edwards, Man, Ahmed), Sec. 3.3 and Figures 6-7.
The paper evaluates a set of AI-generated concepts NOT from geometric feature
vectors but from CLIP SIMILARITY scores:

  * "To measure diversity, we compute the average pairwise CLIP score for each
    of the 4 images in each sample. Higher CLIP scores indicate less diverse
    datasets." (Sec. 3.3) -- similarity-based set diversity, where LOWER mean
    pairwise similarity == MORE diverse.
  * It then ranks each sample's diversity by its PERCENTILE in the distribution
    of pairwise scores (5th / 50th / 95th percentiles, Fig. 7).
  * Alignment across modalities is measured by average cross-set CLIP score:
    sketch-vs-text 25.8, image-vs-text 28.1, sketch-vs-image 64.4 (Sec. 3.3,
    Fig. 6).

This module implements those SIMILARITY-driven measures. It is DELIBERATELY
distinct from ``bench/datacon_diversity`` (which takes NUMERIC feature vectors +
Euclidean distance geometry: pairwise distance, grid coverage, nearest-neighbour
novelty) and from ``bench/feasibility_novelty`` (human-rating rank statistics).
Here the input is an already-computed symmetric SIMILARITY matrix (e.g. CLIP
cosine scores) or a cross-set similarity matrix -- no feature vectors, no
distances, no rank tests. The organising idea is "similarity high -> diversity
low", the inverse of the distance world.

All functions are deterministic and standard-library only.
"""
from __future__ import annotations


def _check_square_symmetric(matrix):
    """Validate a square, (approximately) symmetric similarity matrix."""
    n = len(matrix)
    if n == 0:
        raise ValueError("similarity matrix must be non-empty")
    for row in matrix:
        if len(row) != n:
            raise ValueError("similarity matrix must be square")
    for i in range(n):
        for j in range(i + 1, n):
            if abs(float(matrix[i][j]) - float(matrix[j][i])) > 1e-9:
                raise ValueError("similarity matrix must be symmetric at (%d,%d)" % (i, j))
    return n


def mean_pairwise_similarity(matrix):
    """Average off-diagonal (unordered-pair) similarity of a concept set.

    This is the paper's per-sample diversity statistic: higher == LESS diverse.
    Returns 0.0 for a set with fewer than two concepts.
    """
    n = _check_square_symmetric(matrix)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += float(matrix[i][j])
            count += 1
    return total / count


def set_diversity(matrix, max_similarity=1.0):
    """Diversity of a concept set = ``max_similarity - mean pairwise similarity``.

    Converts the "higher similarity == less diverse" statistic into a
    diversity score that increases with spread. With CLIP cosine similarity in
    [0, max_similarity], a fully-identical set scores 0 and a maximally spread
    set approaches ``max_similarity``.
    """
    return float(max_similarity) - mean_pairwise_similarity(matrix)


def effective_concept_count(matrix, threshold):
    """Greedy count of mutually-dissimilar concepts (effective design coverage).

    Walks concepts in index order, keeping a concept only if its similarity to
    every already-kept concept is STRICTLY BELOW ``threshold``. The returned
    count is how many genuinely distinct concepts the set covers -- the paper's
    "inherent design expansion" idea made concrete: a set can nominally have N
    images yet collapse to far fewer distinct concepts when they are near
    duplicates (the 95th-percentile "almost identical geometries" case).
    """
    n = _check_square_symmetric(matrix)
    kept = []
    for i in range(n):
        if all(float(matrix[i][k]) < threshold for k in kept):
            kept.append(i)
    return len(kept)


def cross_modal_alignment(cross):
    """Average similarity of a rectangular cross-set (cross-modal) matrix.

    ``cross[i][j]`` is the similarity between item i of set A and item j of set
    B (e.g. sketches vs generated images). Returns the mean over all pairs --
    the paper's average cross-modal CLIP score (25.8 / 28.1 / 64.4). Raises
    ValueError on an empty or ragged matrix.
    """
    rows = len(cross)
    if rows == 0:
        raise ValueError("cross matrix must be non-empty")
    cols = len(cross[0])
    if cols == 0:
        raise ValueError("cross matrix rows must be non-empty")
    total = 0.0
    count = 0
    for row in cross:
        if len(row) != cols:
            raise ValueError("cross matrix must be rectangular")
        for v in row:
            total += float(v)
            count += 1
    return total / count


def percentile_rank(value, distribution):
    """Fraction of ``distribution`` values that are <= ``value`` (in [0, 1]).

    Locates where one concept set's diversity statistic falls within the
    distribution of statistics across many samples (Fig. 7's 5th/50th/95th
    percentile framing). Raises ValueError on an empty distribution.
    """
    dist = list(distribution)
    if not dist:
        raise ValueError("distribution must be non-empty")
    below_or_equal = sum(1 for x in dist if float(x) <= float(value))
    return below_or_equal / len(dist)


def percentile_value(distribution, percentile):
    """Value at the given percentile (0..100) via nearest-rank on sorted data.

    Deterministic nearest-rank method: index = ceil(p/100 * n) - 1, clamped to
    [0, n-1]. Used to reproduce the paper's 5th/50th/95th-percentile exemplars.
    """
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    data = sorted(float(x) for x in distribution)
    if not data:
        raise ValueError("distribution must be non-empty")
    n = len(data)
    import math
    rank = math.ceil(percentile / 100.0 * n)
    idx = min(max(rank - 1, 0), n - 1)
    return data[idx]


def rank_methods_by_diversity(method_matrices, max_similarity=1.0):
    """Rank named concept-generation methods by set diversity (most first).

    ``method_matrices`` maps a method name (e.g. ``"sketch_alone"``,
    ``"controlnet"``, ``"sketch2prototype"``) to its pairwise similarity matrix.
    Returns a list of ``(name, diversity)`` tuples sorted by descending
    diversity, ties broken by name for determinism -- reproducing the paper's
    claim that the text-intermediary method is more diverse than the
    sketch-alone and ControlNet baselines (Sec. 3.1).
    """
    scored = [
        (name, set_diversity(matrix, max_similarity=max_similarity))
        for name, matrix in method_matrices.items()
    ]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored


def conceptset_report(matrix, threshold=None, max_similarity=1.0):
    """Deterministic summary dict for a single concept set.

    Keys: ``n``, ``mean_pairwise_similarity``, ``set_diversity``; plus
    ``effective_concept_count`` when a ``threshold`` is supplied.
    """
    n = _check_square_symmetric(matrix)
    report = {
        "n": n,
        "mean_pairwise_similarity": mean_pairwise_similarity(matrix),
        "set_diversity": set_diversity(matrix, max_similarity=max_similarity),
    }
    if threshold is not None:
        report["effective_concept_count"] = effective_concept_count(matrix, threshold)
    return report
