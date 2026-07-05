"""Coverage and imbalance audit across CAD dataset provenance dimensions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional


DIMENSIONS = ("source", "geography", "process", "geometry_family")


@dataclass(frozen=True)
class CoverageWarning:
    dimension: str
    value: str
    code: str
    observed: float
    target: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "value": self.value,
            "code": self.code,
            "observed": self.observed,
            "target": self.target,
        }


@dataclass
class BiasAuditReport:
    n_items: int
    distributions: Dict[str, Dict[str, int]] = field(default_factory=dict)
    missing: Dict[str, int] = field(default_factory=dict)
    warnings: List[CoverageWarning] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings

    def to_dict(self) -> dict:
        return {
            "n_items": self.n_items,
            "distributions": {
                key: dict(sorted(values.items()))
                for key, values in sorted(self.distributions.items())
            },
            "missing": dict(sorted(self.missing.items())),
            "warnings": [warning.to_dict() for warning in self.warnings],
            "ok": self.ok,
        }


def _metadata(item: Any) -> Mapping[str, Any]:
    if isinstance(item, Mapping):
        nested = item.get("metadata")
        if isinstance(nested, Mapping):
            return {**item, **nested}
        return item
    meta = getattr(item, "metadata", None)
    if isinstance(meta, Mapping):
        return meta
    return {}


def audit_bias(
    items: Iterable[Any],
    *,
    targets: Optional[Mapping[str, Mapping[str, float]]] = None,
    minimum_share: float = 0.05,
    maximum_missing_share: float = 0.1,
    target_tolerance: float = 0.5,
) -> BiasAuditReport:
    """Audit provenance dimensions and report missing/imbalanced coverage.

    Targets are optional per-dimension probability maps. Without targets the
    audit still flags categories below ``minimum_share`` and excessive missing
    labels. This detects skew; it does not infer social fairness from geometry.
    """
    if not 0 <= minimum_share <= 1:
        raise ValueError("minimum_share must be between 0 and 1")
    if not 0 <= maximum_missing_share <= 1:
        raise ValueError("maximum_missing_share must be between 0 and 1")
    if target_tolerance < 0:
        raise ValueError("target_tolerance must be non-negative")

    rows = list(items)
    counts = {dimension: Counter() for dimension in DIMENSIONS}
    missing = Counter()
    for item in rows:
        meta = _metadata(item)
        for dimension in DIMENSIONS:
            value = meta.get(dimension)
            if value is None or str(value).strip() == "":
                missing[dimension] += 1
            else:
                counts[dimension][str(value).strip().casefold()] += 1

    warnings: List[CoverageWarning] = []
    n = len(rows)
    for dimension in DIMENSIONS:
        known = sum(counts[dimension].values())
        missing_share = (missing[dimension] / n) if n else 0.0
        if n and missing_share > maximum_missing_share:
            warnings.append(CoverageWarning(
                dimension, "(missing)", "missing-metadata", missing_share,
                maximum_missing_share,
            ))
        if known:
            for value, count in sorted(counts[dimension].items()):
                share = count / known
                if share < minimum_share:
                    warnings.append(CoverageWarning(
                        dimension, value, "under-represented", share, minimum_share
                    ))

        target = (targets or {}).get(dimension)
        if target:
            total_weight = sum(max(0.0, float(v)) for v in target.values())
            if total_weight <= 0:
                raise ValueError(f"target weights for {dimension} must be positive")
            for raw_value, weight in sorted(target.items()):
                value = str(raw_value).casefold()
                expected = max(0.0, float(weight)) / total_weight
                observed = (counts[dimension].get(value, 0) / known) if known else 0.0
                lower = expected * (1.0 - target_tolerance)
                upper = expected * (1.0 + target_tolerance)
                if observed < lower:
                    warnings.append(CoverageWarning(
                        dimension, value, "below-target", observed, expected
                    ))
                elif observed > upper:
                    warnings.append(CoverageWarning(
                        dimension, value, "above-target", observed, expected
                    ))

    warnings.sort(key=lambda w: (w.dimension, w.code, w.value))
    return BiasAuditReport(
        n_items=n,
        distributions={
            dimension: dict(sorted(counter.items()))
            for dimension, counter in counts.items()
        },
        missing=dict(missing),
        warnings=warnings,
    )
