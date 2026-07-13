"""SldprtNet data-acquisition cleaning and dedup pipeline.

SldprtNet's automated pipeline (Li et al., ICRA 2026, Sec. III.C) starts from
~680,000 scraped ``.sldprt`` parts and filters down to the ~242,000 retained
samples by the rule: *keep a model only if it contains at least one of the 13
representative feature types*. This module makes that acquisition filter
deterministic and reusable, together with the surrounding cleaning stages a
scalable dataset build needs:

  * drop parts with **no supported feature** (the paper's completeness filter);
  * drop **empty / degenerate** parts (zero features);
  * **deduplicate** parts by a canonical feature-signature so repeated scrapes of
    the same design collapse to one sample (first occurrence wins);
  * report a :class:`CleaningReport` accounting for every dropped part by reason,
    so the retention yield (retained / input) is auditable.

It reuses the 13-type taxonomy from
:mod:`reconstruction.sldprtnet_feature_tree` and is distinct from the generic
dedup in :mod:`dataengine.datacon_lowdata_dedup`: the signature and the
supported-feature filter here are SldprtNet-specific.

Stdlib-only and deterministic: dedup keeps the first occurrence in input order,
and the report counts are a pure function of the input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from harnesscad.domain.reconstruction.sequences.feature_tree import FEATURE_TYPE_SET


@dataclass(frozen=True)
class RawPart:
    """A scraped candidate part prior to cleaning.

    ``feature_types`` is the ordered list of feature-type strings extracted from
    the raw ``.sldprt`` (may include unsupported types or be empty).
    """

    id: str
    feature_types: Tuple[str, ...] = ()

    @property
    def supported_features(self) -> Tuple[str, ...]:
        return tuple(f for f in self.feature_types if f in FEATURE_TYPE_SET)

    @property
    def has_supported_feature(self) -> bool:
        return any(f in FEATURE_TYPE_SET for f in self.feature_types)

    def signature(self) -> Tuple[Tuple[str, int], ...]:
        """Canonical, order-independent multiset of *supported* feature types.

        Two parts with the same supported-feature counts collapse in dedup,
        regardless of the order features were listed.
        """
        counts: Dict[str, int] = {}
        for f in self.supported_features:
            counts[f] = counts.get(f, 0) + 1
        return tuple(sorted(counts.items()))


# Reasons a part is dropped, in the order stages are applied.
REASON_NO_SUPPORTED = "no_supported_feature"
REASON_DUPLICATE = "duplicate"


@dataclass
class CleaningReport:
    """Accounting for a cleaning run."""

    input_count: int = 0
    retained: List[RawPart] = field(default_factory=list)
    dropped: Dict[str, int] = field(
        default_factory=lambda: {REASON_NO_SUPPORTED: 0, REASON_DUPLICATE: 0}
    )

    @property
    def retained_count(self) -> int:
        return len(self.retained)

    @property
    def dropped_count(self) -> int:
        return sum(self.dropped.values())

    @property
    def retention_yield(self) -> float:
        if self.input_count == 0:
            return 0.0
        return self.retained_count / self.input_count


def clean(parts: Sequence[RawPart]) -> CleaningReport:
    """Run the SldprtNet acquisition cleaning + dedup pipeline.

    Stages (in order): drop parts with no supported feature, then deduplicate by
    canonical feature signature (first occurrence wins). A part that is empty has
    no supported features and is therefore dropped by the first stage.
    """
    report = CleaningReport(input_count=len(parts))
    seen_signatures: set = set()
    for part in parts:
        if not part.has_supported_feature:
            report.dropped[REASON_NO_SUPPORTED] += 1
            continue
        sig = part.signature()
        if sig in seen_signatures:
            report.dropped[REASON_DUPLICATE] += 1
            continue
        seen_signatures.add(sig)
        report.retained.append(part)
    return report


def unsupported_feature_ratio(parts: Sequence[RawPart]) -> float:
    """Fraction of all listed feature occurrences that are *unsupported*.

    A quality signal on the raw scrape: high values mean the source repositories
    use many features outside the 13-type taxonomy.
    """
    total = 0
    unsupported = 0
    for part in parts:
        for f in part.feature_types:
            total += 1
            if f not in FEATURE_TYPE_SET:
                unsupported += 1
    if total == 0:
        return 0.0
    return unsupported / total
