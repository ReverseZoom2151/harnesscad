"""Vitruvion dataset curation: sketch filtering, exact token dedup, balanced sharding.

The three deterministic data-pipeline rules Vitruvion (Seff et al., ICLR 2022) applies
before any model sees a sketch -- ``img2cad/pipeline/filter_sequences.py``,
``img2cad/pipeline/uniqueify_tokens.py``, ``img2cad/pipeline/prerender_images.py``.

1. Filtering (``filter_sketch``)
--------------------------------
A sketch is kept only if it has **6 to 16 entities inclusive** (the paper's training
regime) *and* it survives a renderability check:

  * normalisation must succeed and must not return the zero-extent sentinel ``-1``;
  * no zero-radius circle or arc;
  * no zero-length line (start point equal to end point);
  * no arc whose start point coincides with its mid point (a zero-sweep arc).

The last three are exactly the cases that make the arc re-fit (three points through a
circumcentre) or the bbox normalisation degenerate, so the filter is what guarantees the
quantiser never sees a NaN.  The check is run **on the normalised sketch**, so it is
scale-invariant; this module normalises a *copy*, leaving the caller's entities intact.

2. Exact token dedup (``unique_indices``)
-----------------------------------------
SketchGraphs contains many sketches that are distinct as *sequences* but identical once
tokenised (same primitives, same bins).  Vitruvion deduplicates on the **quantised token
stream**, not on the geometry: sequences are bucketed by length (only equal-length streams
can be equal), and within a bucket the first occurrence of each distinct stream is kept;
the surviving indices are returned in ascending order.  This is a *lossless exact* dedup
and is deliberately different from ``bench.skexgen_dedup_hash``, which hashes a token
stream to a digest (a collision-prone approximation, and it dedups per-branch).  Note the
consequence: two sketches that differ only below the quantiser's resolution collapse into
one, so the dedup rate is a function of ``num_bins``.

3. Balanced sharding (``shard_range``)
--------------------------------------
Renders are farmed out to array jobs with a remainder-aware split: with ``n`` items over
``k`` shards, the first ``n mod k`` shards get ``floor(n/k) + 1`` items and the rest get
``floor(n/k)``.  Shard sizes therefore differ by at most one, the ranges are contiguous
and non-overlapping, and they exactly cover ``[0, n)`` -- unlike a naive
``ceil``-based split, which can hand the last shard an empty (or negative) range.

Pure stdlib.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, Hashable, List, Sequence, Tuple

from harnesscad.domain.geometry.vitruvion_sketch_norm import (
    VArc,
    VCircle,
    VLine,
    normalize_sketch,
)

__all__ = [
    "FilterConfig",
    "sketch_is_renderable",
    "filter_sketch",
    "filter_indices",
    "unique_indices",
    "shard_range",
]

TOLERANCE = 1e-9


@dataclass(frozen=True)
class FilterConfig:
    """Vitruvion's ``FilterConfiguration`` defaults."""

    min_entities: int = 6
    max_entities: int = 16


def _close(a: Sequence[float], b: Sequence[float], tol: float = TOLERANCE) -> bool:
    return all(math.isclose(x, y, rel_tol=1e-5, abs_tol=tol) for x, y in zip(a, b))


def sketch_is_renderable(entities: Sequence[object], tol: float = TOLERANCE) -> bool:
    """Whether a sketch normalises and contains no degenerate entity.

    Operates on a deep copy: the caller's entities are never modified.
    """
    work = [copy.deepcopy(e) for e in entities]
    try:
        scale = normalize_sketch(work)
    except (ValueError, ZeroDivisionError):
        return False

    if scale == -1:
        return False

    for entity in work:
        if isinstance(entity, (VArc, VCircle)) and entity.radius == 0:
            return False
        if isinstance(entity, VLine) and _close(entity.start_point, entity.end_point, tol):
            return False
        if isinstance(entity, VArc) and _close(entity.start_point, entity.mid_point, tol):
            return False
    return True


def filter_sketch(entities: Sequence[object], config: FilterConfig = FilterConfig()) -> bool:
    """Whether a sketch passes the entity-count bounds and the renderability check."""
    if len(entities) < config.min_entities or len(entities) > config.max_entities:
        return False
    return sketch_is_renderable(entities)


def filter_indices(
    sketches: Sequence[Sequence[object]], config: FilterConfig = FilterConfig()
) -> List[int]:
    """The indices of the sketches that survive :func:`filter_sketch`, in order."""
    return [i for i, sketch in enumerate(sketches) if filter_sketch(sketch, config)]


def unique_indices(sequences: Sequence[Sequence[Hashable]]) -> List[int]:
    """Indices of the first occurrence of each distinct token stream, ascending.

    Bucketing by length first is what makes this cheap: only equal-length streams can
    compare equal, so no cross-length comparisons are ever made.
    """
    seen_by_length: Dict[int, Dict[Tuple[Hashable, ...], int]] = {}
    keep: List[int] = []

    for index, sequence in enumerate(sequences):
        key = tuple(sequence)
        bucket = seen_by_length.setdefault(len(key), {})
        if key in bucket:
            continue
        bucket[key] = index
        keep.append(index)

    keep.sort()
    return keep


def shard_range(shard_id: int, n_items: int, n_shards: int) -> Tuple[int, int]:
    """The half-open ``[start, end)`` slice of ``n_items`` assigned to ``shard_id``.

    Remainder items go one each to the lowest-numbered shards, so shard sizes differ by
    at most one and the shards exactly tile ``[0, n_items)``.
    """
    if n_shards <= 0:
        raise ValueError("n_shards must be positive")
    if not 0 <= shard_id < n_shards:
        raise ValueError("shard_id out of range")
    if n_items < 0:
        raise ValueError("n_items must be non-negative")

    remainder = n_items % n_shards
    size = n_items // n_shards
    size_plus_one = size + 1

    if shard_id < remainder:
        start = shard_id * size_plus_one
        return (start, start + size_plus_one)

    start = remainder * size_plus_one + (shard_id - remainder) * size
    return (start, start + size)
