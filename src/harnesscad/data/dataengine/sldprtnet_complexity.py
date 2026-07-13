"""SldprtNet four-level complexity taxonomy and curriculum ordering.

SldprtNet (Li et al., ICRA 2026, Sec. III.D) categorises each part by its number
of feature operations into four complexity levels, and uses that stratification
for complexity-specific benchmarking and curriculum learning:

  * Level 1 (Simple):   1-5 features    (93,188 samples in the paper)
  * Level 2 (Moderate): 6-10 features   (78,926 samples)
  * Level 3 (Advanced): 11-100 features (69,259 samples)
  * Level 4 (Expert):   101+ features   (1,234 samples)

This module makes that taxonomy deterministic and reusable: classify a feature
count into a level, stratify a collection, order it for curriculum learning
(simple -> expert, ties broken by feature count then id), and compare an observed
level histogram against the paper's reference distribution. It is distinct from
the generic op-frequency auditing in :mod:`dataengine.distribution_audit`: the
boundaries and level labels here are SldprtNet-specific.

Stdlib-only and deterministic (no wall clock, no RNG; stratification and
curriculum order are fully determined by feature count and id).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

#: (level number, label, inclusive lower bound, inclusive upper bound or None).
COMPLEXITY_LEVELS: Tuple[Tuple[int, str, int, object], ...] = (
    (1, "Simple", 1, 5),
    (2, "Moderate", 6, 10),
    (3, "Advanced", 11, 100),
    (4, "Expert", 101, None),
)

#: Reference sample counts reported in the paper (for distribution comparison).
REFERENCE_COUNTS: Dict[int, int] = {1: 93188, 2: 78926, 3: 69259, 4: 1234}

LEVEL_LABELS: Dict[int, str] = {n: label for n, label, _lo, _hi in COMPLEXITY_LEVELS}


def classify_complexity(feature_count: int) -> int:
    """Return the complexity level (1..4) for a given feature count.

    A part must contain at least one feature (SldprtNet retains only parts with
    >= 1 supported feature); ``feature_count < 1`` is rejected.
    """
    if feature_count < 1:
        raise ValueError("feature_count must be >= 1")
    for level, _label, lo, hi in COMPLEXITY_LEVELS:
        if feature_count >= lo and (hi is None or feature_count <= hi):
            return level
    raise ValueError(f"unclassifiable feature_count: {feature_count}")


def level_label(level: int) -> str:
    if level not in LEVEL_LABELS:
        raise ValueError(f"unknown level: {level}")
    return LEVEL_LABELS[level]


@dataclass(frozen=True)
class ComplexityItem:
    """An item to stratify: an id and its feature count."""

    id: str
    feature_count: int

    @property
    def level(self) -> int:
        return classify_complexity(self.feature_count)


def stratify(items: Sequence[ComplexityItem]) -> Dict[int, List[ComplexityItem]]:
    """Bucket items by complexity level (keys 1..4 always present)."""
    buckets: Dict[int, List[ComplexityItem]] = {1: [], 2: [], 3: [], 4: []}
    for it in items:
        buckets[it.level].append(it)
    return buckets


def level_histogram(items: Sequence[ComplexityItem]) -> Dict[int, int]:
    """Count of items per level (keys 1..4 always present)."""
    hist = {1: 0, 2: 0, 3: 0, 4: 0}
    for it in items:
        hist[it.level] += 1
    return hist


def curriculum_order(items: Sequence[ComplexityItem]) -> List[ComplexityItem]:
    """Order items for curriculum learning: simplest first.

    Sort key: (level, feature_count, id) -- deterministic and stable, so a model
    is exposed to simple parts before advancing to expert-level intricacy.
    """
    return sorted(items, key=lambda it: (it.level, it.feature_count, it.id))


def level_proportions(items: Sequence[ComplexityItem]) -> Dict[int, float]:
    """Fraction of items in each level (keys 1..4; empty -> all zeros)."""
    hist = level_histogram(items)
    total = sum(hist.values())
    if total == 0:
        return {lvl: 0.0 for lvl in (1, 2, 3, 4)}
    return {lvl: hist[lvl] / total for lvl in (1, 2, 3, 4)}


def reference_proportions() -> Dict[int, float]:
    """The paper's reference level proportions (normalised REFERENCE_COUNTS)."""
    total = sum(REFERENCE_COUNTS.values())
    return {lvl: REFERENCE_COUNTS[lvl] / total for lvl in (1, 2, 3, 4)}


def distribution_l1(items: Sequence[ComplexityItem]) -> float:
    """L1 distance between observed and reference level proportions (0..2).

    A smaller value means the observed complexity mix better matches the paper's
    balanced-plus-rare-expert distribution.
    """
    obs = level_proportions(items)
    ref = reference_proportions()
    return sum(abs(obs[lvl] - ref[lvl]) for lvl in (1, 2, 3, 4))
