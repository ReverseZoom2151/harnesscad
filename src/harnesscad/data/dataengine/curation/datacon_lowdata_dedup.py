"""Low-data training-set construction via scale-invariant de-duplication.

Implements the data-curation protocol from "Generative AI and CAD Automation
for Diverse and Novel Mechanical Component Designs Under Data Constraints"
(Section 4.1). The authors collected 463 rim files, then "Each design was
thoroughly reviewed, and those with only dimensional differences (e.g.,
variations between 17-inch and 20-inch rims) were excluded. After filtering,
218 rim images were selected."

This module models that funnel deterministically:

  1. Preprocessing normalization: a scale-invariant canonical signature. Two
     designs that differ ONLY by a global scale (a 17-inch vs a 20-inch rim,
     which is the same geometry uniformly scaled) map to the SAME normalized
     feature vector and therefore the same signature.
  2. De-duplication: keep one canonical representative per distinct design
     family (first-seen wins), yielding a small, high-quality training set.
  3. Curated-subset selection: if the de-duplicated set is still larger than a
     target size, deterministically down-select using farthest-point (max-min
     distance) greedy sampling to preserve design diversity.

Normalization choice: L2 normalization. Dividing a feature vector by its
Euclidean (L2) norm removes any global scalar multiplier k > 0, because
(k * v) / ||k * v|| == v / ||v||. This is why two rims differing only in
dimension collapse to one representative. The zero vector is guarded and maps
to zeros.

This complements dataengine/ (annotation / consensus / distribution): it is a
scale-invariant dedup + curated-subset selector that did not previously exist.
Stdlib-only, deterministic (all randomness seeded via random.Random).
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple


def scale_normalize(features: Sequence[float]) -> Tuple[float, ...]:
    """Return the L2-normalized feature vector as a tuple.

    Dividing by the L2 norm makes the result invariant to a global positive
    scale factor, so two designs differing only in dimension (e.g. a 17-inch
    vs 20-inch rim) map to the same normalized vector. A zero vector (norm 0)
    is returned unchanged as zeros.
    """
    vec = [float(x) for x in features]
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return tuple(0.0 for _ in vec)
    return tuple(x / norm for x in vec)


def canonical_signature(features: Sequence[float], precision: int = 3) -> Tuple[float, ...]:
    """Scale-normalize then round each component to `precision` decimals.

    The returned tuple is hashable and identical for two scale-only variants,
    so it can be used as a dictionary key for de-duplication.
    """
    normalized = scale_normalize(features)
    return tuple(round(x, precision) for x in normalized)


def is_scale_variant(
    a_features: Sequence[float],
    b_features: Sequence[float],
    precision: int = 3,
) -> bool:
    """True if the two feature vectors share a canonical signature.

    That means they are the same design differing only by a global scale.
    """
    return canonical_signature(a_features, precision) == canonical_signature(
        b_features, precision
    )


def _features_of(record: Dict[str, Any]) -> Sequence[float]:
    return record["features"]


def dedup_by_scale(
    records: Sequence[Dict[str, Any]], precision: int = 3
) -> Dict[str, Any]:
    """Remove scale-only near-duplicates, keeping one per distinct design.

    Deterministic: input order is preserved and the first occurrence of each
    canonical signature is kept. Mirrors the paper's 463 -> 218 exclusion of
    designs that differ only in dimension.

    Returns a dict with keys:
      kept:            list of records, one per distinct signature
      removed:         list of records dropped as scale-variants
      n_in:            number of input records
      n_out:           number of kept records
      reduction_ratio: 1 - n_out / n_in (0.0 for empty input)
    """
    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    seen: Dict[Tuple[float, ...], bool] = {}

    for record in records:
        sig = canonical_signature(_features_of(record), precision)
        if sig in seen:
            removed.append(record)
        else:
            seen[sig] = True
            kept.append(record)

    n_in = len(records)
    n_out = len(kept)
    reduction_ratio = 0.0 if n_in == 0 else 1.0 - (n_out / n_in)

    return {
        "kept": kept,
        "removed": removed,
        "n_in": n_in,
        "n_out": n_out,
        "reduction_ratio": reduction_ratio,
    }


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def select_training_subset(
    records: Sequence[Dict[str, Any]],
    target_size: int,
    seed: int,
    precision: int = 3,
    diversity_key: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Build a curated training subset of at most `target_size` records.

    First de-duplicates by scale. If the de-duplicated set still exceeds
    `target_size`, greedily down-selects using farthest-point (max-min
    distance) sampling over the scale-normalized feature vectors: the first
    representative is chosen with random.Random(seed), then each subsequent
    pick is the record whose minimum distance to the already-selected set is
    largest. Ties break on the earliest (lowest) index for determinism.

    If fewer than or equal to `target_size` distinct designs remain after
    de-duplication, all kept records are returned. Deterministic given `seed`.

    `diversity_key` is accepted for API symmetry; when provided it is a callable
    mapping a record to the vector used for diversity distances (defaults to the
    scale-normalized "features").
    """
    kept = dedup_by_scale(records, precision)["kept"]

    if target_size <= 0:
        return []
    if len(kept) <= target_size:
        return list(kept)

    def vector_of(record: Dict[str, Any]) -> Tuple[float, ...]:
        if diversity_key is not None:
            return tuple(float(x) for x in diversity_key(record))
        return scale_normalize(_features_of(record))

    vectors = [vector_of(r) for r in kept]

    rng = random.Random(seed)
    first_idx = rng.randrange(len(kept))

    selected_idx = [first_idx]
    selected_set = {first_idx}
    # min distance from each candidate to the current selected set
    min_dist = [
        _euclidean(vectors[i], vectors[first_idx]) for i in range(len(kept))
    ]

    while len(selected_idx) < target_size:
        best_i = -1
        best_d = -1.0
        for i in range(len(kept)):
            if i in selected_set:
                continue
            d = min_dist[i]
            if d > best_d:
                best_d = d
                best_i = i
        if best_i == -1:
            break
        selected_idx.append(best_i)
        selected_set.add(best_i)
        for i in range(len(kept)):
            if i in selected_set:
                continue
            d = _euclidean(vectors[i], vectors[best_i])
            if d < min_dist[i]:
                min_dist[i] = d

    return [kept[i] for i in selected_idx]


def construction_report(
    records: Sequence[Dict[str, Any]],
    target_size: int,
    seed: int,
    precision: int = 3,
) -> Dict[str, Any]:
    """Summarize the paper's raw -> deduped -> selected curation funnel.

    Mirrors the 463 -> 218 -> (training subset) pipeline described in
    Section 4.1. Returns a dict with keys:
      n_raw, n_after_dedup, n_selected, reduction_ratio, target_size, seed.
    """
    dedup = dedup_by_scale(records, precision)
    subset = select_training_subset(records, target_size, seed, precision)

    return {
        "n_raw": dedup["n_in"],
        "n_after_dedup": dedup["n_out"],
        "n_selected": len(subset),
        "reduction_ratio": dedup["reduction_ratio"],
        "target_size": target_size,
        "seed": seed,
    }
