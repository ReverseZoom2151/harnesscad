"""Cross-source reconciliation for CAD models, drawings and reference data.

Evidence is joined by a persistent correspondence ID rather than by fragile
array position or geometry order.  The implementation is deterministic,
stdlib-only and operates on already-extracted facts, so it needs no CAD kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Any, Iterable, Mapping, Optional, Sequence


class DiscrepancyKind(str, Enum):
    MISSING_CORRESPONDENCE = "missing_correspondence"
    MISSING_FIELD = "missing_field"
    VALUE_MISMATCH = "value_mismatch"
    NUMERIC_MISMATCH = "numeric_mismatch"
    DUPLICATE_EVIDENCE = "duplicate_evidence"


@dataclass(frozen=True)
class Evidence:
    """Facts extracted from one source for one persistent CAD entity."""

    correspondence_id: str
    source: str
    metrics: Mapping[str, Any] = field(default_factory=dict)
    annotations: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.correspondence_id.strip():
            raise ValueError("correspondence_id must be non-empty")
        if not self.source.strip():
            raise ValueError("source must be non-empty")


@dataclass(frozen=True)
class Discrepancy:
    kind: DiscrepancyKind
    correspondence_id: str
    category: str
    field: str
    sources: tuple[str, ...]
    values: Mapping[str, Any]
    message: str
    relative_delta: Optional[float] = None


@dataclass(frozen=True)
class ReconciliationReport:
    discrepancies: tuple[Discrepancy, ...]
    correspondence_ids: tuple[str, ...]
    sources: tuple[str, ...]
    comparisons: int

    @property
    def ok(self) -> bool:
        return not self.discrepancies

    def by_kind(self, kind: DiscrepancyKind) -> tuple[Discrepancy, ...]:
        return tuple(d for d in self.discrepancies if d.kind is kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "correspondence_ids": list(self.correspondence_ids),
            "sources": list(self.sources),
            "comparisons": self.comparisons,
            "discrepancies": [
                {
                    "kind": d.kind.value,
                    "correspondence_id": d.correspondence_id,
                    "category": d.category,
                    "field": d.field,
                    "sources": list(d.sources),
                    "values": dict(d.values),
                    "message": d.message,
                    "relative_delta": d.relative_delta,
                }
                for d in self.discrepancies
            ],
        }


def reconcile(
    evidence: Iterable[Evidence],
    *,
    required_sources: Sequence[str] = ("model", "drawing", "reference"),
    relative_tolerance: float = 0.01,
    absolute_tolerance: float = 1e-9,
) -> ReconciliationReport:
    """Compare all facts sharing a correspondence ID.

    Numeric metric values use ``max(abs_tol, rel_tol * max(|a|, |b|))``.
    Drawing annotations and reference metadata are compared as canonical
    values. Missing fields are reported only when another source provides that
    field, avoiding false failures for category-specific empty records.
    """

    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("tolerances must be non-negative")
    required = tuple(sorted(set(required_sources)))
    items = sorted(
        evidence,
        key=lambda e: (e.correspondence_id, e.source),
    )
    grouped: dict[str, dict[str, Evidence]] = {}
    discrepancies: list[Discrepancy] = []

    for item in items:
        sources = grouped.setdefault(item.correspondence_id, {})
        if item.source in sources:
            discrepancies.append(
                Discrepancy(
                    DiscrepancyKind.DUPLICATE_EVIDENCE,
                    item.correspondence_id,
                    "identity",
                    item.source,
                    (item.source,),
                    {},
                    f"duplicate evidence for {item.correspondence_id!r} from {item.source!r}",
                )
            )
            continue
        sources[item.source] = item

    comparisons = 0
    for cid in sorted(grouped):
        by_source = grouped[cid]
        missing_sources = sorted(set(required) - set(by_source))
        for source in missing_sources:
            discrepancies.append(
                Discrepancy(
                    DiscrepancyKind.MISSING_CORRESPONDENCE,
                    cid,
                    "identity",
                    "",
                    (source,),
                    {},
                    f"{source!r} has no evidence for correspondence ID {cid!r}",
                )
            )

        for category in ("metrics", "annotations", "metadata"):
            maps = {
                source: getattr(item, category)
                for source, item in sorted(by_source.items())
            }
            fields = sorted({key for values in maps.values() for key in values})
            for field_name in fields:
                present = {source: values[field_name] for source, values in maps.items()
                           if field_name in values}
                for source in sorted(set(maps) - set(present)):
                    discrepancies.append(
                        Discrepancy(
                            DiscrepancyKind.MISSING_FIELD,
                            cid,
                            category,
                            field_name,
                            (source,),
                            dict(present),
                            f"{source!r} is missing {category}.{field_name}",
                        )
                    )
                for left, right in combinations(sorted(present), 2):
                    comparisons += 1
                    a, b = present[left], present[right]
                    mismatch, delta = _mismatch(
                        a, b, category == "metrics",
                        relative_tolerance, absolute_tolerance,
                    )
                    if mismatch:
                        kind = (
                            DiscrepancyKind.NUMERIC_MISMATCH
                            if delta is not None
                            else DiscrepancyKind.VALUE_MISMATCH
                        )
                        discrepancies.append(
                            Discrepancy(
                                kind,
                                cid,
                                category,
                                field_name,
                                (left, right),
                                {left: a, right: b},
                                f"{category}.{field_name} disagrees between "
                                f"{left!r} and {right!r}",
                                delta,
                            )
                        )

    discrepancies.sort(
        key=lambda d: (
            d.correspondence_id, d.kind.value, d.category, d.field, d.sources
        )
    )
    all_sources = tuple(sorted({item.source for item in items} | set(required)))
    return ReconciliationReport(
        tuple(discrepancies), tuple(sorted(grouped)), all_sources, comparisons
    )


def _mismatch(
    left: Any,
    right: Any,
    numeric: bool,
    rel_tol: float,
    abs_tol: float,
) -> tuple[bool, Optional[float]]:
    if numeric and _number(left) and _number(right):
        a, b = float(left), float(right)
        absolute = abs(a - b)
        scale = max(abs(a), abs(b))
        relative = absolute / scale if scale else 0.0
        return absolute > max(abs_tol, rel_tol * scale), relative
    return _canonical(left) != _canonical(right), None


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple((str(k), _canonical(v)) for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_canonical(v) for v in value))
    return value
