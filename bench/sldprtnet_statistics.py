"""SldprtNet dataset statistics and quality report.

SldprtNet (Li et al., ICRA 2026, Sec. III.D) reports two headline distributions
over the corpus -- the frequency of each of the 13 CAD feature types (Fig. 5)
and the four-level complexity mix (Fig. 4) -- plus, implicitly, the multimodal
completeness that makes it a *multimodal* dataset. This module folds a bag of
:class:`~dataengine.sldprtnet_record.SldprtNetRecord` samples (each carrying an
``encoder_txt`` feature-tree script) into a single :class:`SldprtNetStats`
report combining:

  * per-feature-type frequency (over all 13 types, zeros included);
  * complexity-level histogram + proportions
    (via :mod:`dataengine.sldprtnet_complexity`);
  * multimodal coverage -- per-modality coverage and fully-aligned rate
    (via :mod:`dataengine.sldprtnet_record`);
  * a mean/most-common feature summary.

It composes the other SldprtNet modules rather than reimplementing them, and is
distinct from :mod:`bench.histcad_history_quality` (history-dataset metrics) and
the generic :mod:`dataengine.distribution_audit`. Stdlib-only and deterministic
(sorted keys throughout; no wall clock, no RNG).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from dataengine.sldprtnet_complexity import (
    ComplexityItem,
    level_histogram,
    level_proportions,
)
from dataengine.sldprtnet_record import (
    SldprtNetRecord,
    fully_aligned_rate,
    modality_coverage,
)
from reconstruction.sldprtnet_feature_tree import FEATURE_TYPES, FeatureTree


@dataclass
class SldprtNetStats:
    """Aggregate statistics over a SldprtNet sample collection."""

    num_samples: int = 0
    feature_frequency: Dict[str, int] = field(default_factory=dict)
    complexity_histogram: Dict[int, int] = field(default_factory=dict)
    complexity_proportions: Dict[int, float] = field(default_factory=dict)
    modality_coverage: Dict[str, float] = field(default_factory=dict)
    fully_aligned_rate: float = 0.0
    mean_features_per_part: float = 0.0

    @property
    def most_common_feature(self) -> str:
        """Feature type with the highest frequency (ties -> lexicographically first).

        Returns ``""`` for an empty corpus.
        """
        if not self.feature_frequency or all(
            v == 0 for v in self.feature_frequency.values()
        ):
            return ""
        return max(sorted(self.feature_frequency), key=self.feature_frequency.get)


def _feature_count_for(record: SldprtNetRecord) -> int:
    """Feature count for a record: parse encoder_txt if present, else the field."""
    if record.encoder_txt:
        try:
            return FeatureTree.from_text(record.encoder_txt).num_features
        except ValueError:
            return record.feature_count
    return record.feature_count


def _feature_counts_by_type(record: SldprtNetRecord) -> Dict[str, int]:
    if record.encoder_txt:
        try:
            return FeatureTree.from_text(record.encoder_txt).feature_counts()
        except ValueError:
            return {}
    return {}


def compute_statistics(records: Sequence[SldprtNetRecord]) -> SldprtNetStats:
    """Compute the full SldprtNet statistics report over ``records``."""
    records = list(records)
    stats = SldprtNetStats(num_samples=len(records))

    # Feature-type frequency over all 13 types (zeros included).
    freq: Dict[str, int] = {ft: 0 for ft in FEATURE_TYPES}
    total_features = 0
    items: List[ComplexityItem] = []
    for rec in records:
        fc = _feature_count_for(rec)
        total_features += fc
        if fc >= 1:
            items.append(ComplexityItem(rec.id, fc))
        for ftype, cnt in _feature_counts_by_type(rec).items():
            freq[ftype] = freq.get(ftype, 0) + cnt
    stats.feature_frequency = {ft: freq[ft] for ft in sorted(freq)}

    # Complexity.
    stats.complexity_histogram = level_histogram(items)
    stats.complexity_proportions = level_proportions(items)

    # Multimodal coverage.
    stats.modality_coverage = modality_coverage(records)
    stats.fully_aligned_rate = fully_aligned_rate(records)

    # Mean features/part.
    stats.mean_features_per_part = (
        total_features / len(records) if records else 0.0
    )
    return stats
