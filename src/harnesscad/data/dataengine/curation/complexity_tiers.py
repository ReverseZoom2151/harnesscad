"""Complexity-based stratification of an ICL database (DST Appendix B.1).

The DST paper partitions its 151,940 (NL, code, CAD) triplets into Easy /
Middle / Hard tiers by a composite complexity score built from three
complementary, min-max-normalised metrics (Eq. 26):

  * ``Len(NL_i)``   -- length of the natural-language design specification.
  * ``Geom(CAD_i)`` -- geometric complexity = number of edges + faces.
  * ``Ops(Code_i)`` -- operational complexity = number of valid CadQuery ops.

Each metric is min-max normalised across the corpus, the three normalised
values are summed into ``Complexity_i``, samples are ranked, and split into
three equal-sized tiers. This module reproduces that stratification exactly and
deterministically (stable sort, ties by input order), which is useful for
building difficulty-balanced evaluation splits and ICL databases.

stdlib-only; no wall clock, no RNG.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Sequence, Tuple


class Tier(str, Enum):
    EASY = "easy"
    MIDDLE = "middle"
    HARD = "hard"


@dataclass(frozen=True)
class ComplexitySample:
    """One corpus sample with its three raw complexity dimensions."""

    sample_id: str
    nl_length: float      # Len(NL_i)
    geom: float           # Geom(CAD_i) = edges + faces
    ops: float            # Ops(Code_i) = valid CadQuery operation count


@dataclass(frozen=True)
class ScoredSample:
    sample_id: str
    complexity: float     # summed normalised score (Eq. 26)
    tier: Tier


def _minmax(values: Sequence[float]) -> List[float]:
    """Min-max normalise -- ``(x - min) / (max - min)`` (Eq. 26 N(.)).

    A degenerate range (all equal) maps every value to 0.0, matching the paper's
    "equal contribution" intent without dividing by zero.
    """
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span == 0:
        return [0.0 for _ in values]
    return [(v - lo) / span for v in values]


def complexity_scores(samples: Sequence[ComplexitySample]) -> List[float]:
    """Composite complexity score per sample: sum of the three normalised dims."""
    if not samples:
        return []
    nl = _minmax([s.nl_length for s in samples])
    gm = _minmax([s.geom for s in samples])
    op = _minmax([s.ops for s in samples])
    return [nl[i] + gm[i] + op[i] for i in range(len(samples))]


def _tier_bounds(n: int) -> Tuple[int, int]:
    """Split ``n`` ranked items into three (near-)equal tiers.

    Returns (easy_end, middle_end): indices [0, easy_end) are Easy,
    [easy_end, middle_end) Middle, [middle_end, n) Hard. Remainders go to the
    later tiers so Easy is never larger than Hard.
    """
    base = n // 3
    rem = n % 3
    # Remainder goes to Middle then Hard so Easy is never the largest tier.
    easy_size = base
    middle_size = base + (1 if rem >= 2 else 0)
    easy_end = easy_size
    middle_end = easy_size + middle_size
    return easy_end, middle_end


def partition(
    samples: Sequence[ComplexitySample],
) -> List[ScoredSample]:
    """Rank by composite complexity and assign Easy/Middle/Hard tiers.

    Deterministic: stable sort by (score, original index) so ties resolve by
    input order. Returns one :class:`ScoredSample` per input, in the *input*
    order (not ranked order) so callers can zip back to their records.
    """
    n = len(samples)
    if n == 0:
        return []
    scores = complexity_scores(samples)
    # Rank ascending: lowest complexity == Easy.
    order = sorted(range(n), key=lambda i: (scores[i], i))
    easy_end, middle_end = _tier_bounds(n)
    tier_of: Dict[int, Tier] = {}
    for rank, idx in enumerate(order):
        if rank < easy_end:
            tier_of[idx] = Tier.EASY
        elif rank < middle_end:
            tier_of[idx] = Tier.MIDDLE
        else:
            tier_of[idx] = Tier.HARD
    return [
        ScoredSample(
            sample_id=samples[i].sample_id,
            complexity=scores[i],
            tier=tier_of[i],
        )
        for i in range(n)
    ]


def tier_counts(scored: Sequence[ScoredSample]) -> Dict[Tier, int]:
    counts = {Tier.EASY: 0, Tier.MIDDLE: 0, Tier.HARD: 0}
    for s in scored:
        counts[s.tier] += 1
    return counts
