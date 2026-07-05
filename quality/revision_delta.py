"""Deterministic quantity and footprint deltas between two CAD revisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from quality.estimate import BOM, BOMLine, PartEstimate, estimate_part


@dataclass(frozen=True)
class ScalarDelta:
    metric: str
    before: Optional[float]
    after: Optional[float]
    absolute: Optional[float]
    percent: Optional[float]
    available: bool

    @property
    def direction(self) -> str:
        if not self.available or self.absolute is None:
            return "unavailable"
        return "increased" if self.absolute > 0 else "decreased" if self.absolute < 0 else "unchanged"


@dataclass(frozen=True)
class BOMLineDelta:
    part: str
    material: str
    before_qty: int
    after_qty: int
    quantity_delta: int
    mass_delta: Optional[float]
    cost_delta: Optional[float]
    carbon_delta: Optional[float]
    energy_delta: Optional[float]

    @property
    def change(self) -> str:
        if self.before_qty == 0:
            return "added"
        if self.after_qty == 0:
            return "removed"
        return "changed" if self.quantity_delta else "unchanged"


@dataclass(frozen=True)
class RevisionDeltaReport:
    metrics: tuple[ScalarDelta, ...]
    bom_lines: tuple[BOMLineDelta, ...]
    available: bool
    note: str

    def metric(self, name: str) -> ScalarDelta:
        return next(item for item in self.metrics if item.metric == name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "note": self.note,
            "metrics": {
                item.metric: {
                    "before": item.before,
                    "after": item.after,
                    "absolute": item.absolute,
                    "percent": item.percent,
                    "available": item.available,
                    "direction": item.direction,
                }
                for item in self.metrics
            },
            "bom_lines": [
                {
                    "part": line.part,
                    "material": line.material,
                    "before_qty": line.before_qty,
                    "after_qty": line.after_qty,
                    "quantity_delta": line.quantity_delta,
                    "mass_delta": line.mass_delta,
                    "cost_delta": line.cost_delta,
                    "carbon_delta": line.carbon_delta,
                    "energy_delta": line.energy_delta,
                    "change": line.change,
                }
                for line in self.bom_lines
            ],
        }


_METRICS = ("volume", "mass", "cost", "carbon", "energy")


def compare_revisions(
    before: Any,
    after: Any,
    *,
    material: str = "aluminium",
) -> RevisionDeltaReport:
    """Compare backends, :class:`PartEstimate`, :class:`BOM`, or metric mappings."""

    left = _snapshot(before, material)
    right = _snapshot(after, material)
    metrics = tuple(_delta(name, left["metrics"].get(name), right["metrics"].get(name))
                    for name in _METRICS)
    lines = _line_deltas(left["lines"], right["lines"])
    available = any(item.available for item in metrics) or bool(lines)
    note = (
        "revision quantities compared"
        if available
        else "comparison unavailable: neither revision exposed measurable quantities"
    )
    return RevisionDeltaReport(metrics, lines, available, note)


def _snapshot(source: Any, material: str) -> dict[str, Any]:
    if isinstance(source, BOM):
        energy = sum(
            (line.estimate.embodied_energy * line.qty)
            for line in source.lines
            if line.estimate is not None and line.estimate.embodied_energy is not None
        )
        energy_known = any(
            line.estimate is not None and line.estimate.embodied_energy is not None
            for line in source.lines
        )
        return {
            "metrics": {
                "mass": source.total_mass,
                "cost": source.total_cost,
                "carbon": source.total_carbon,
                "energy": energy if energy_known else None,
            },
            "lines": {_line_key(line): line for line in source.lines},
        }

    if isinstance(source, Mapping) and _is_revision_mapping(source):
        totals = source.get("totals", source)
        raw_lines = source.get("lines", ())
        return {
            "metrics": {
                name: _number(totals.get(name, totals.get(f"total_{name}")))
                for name in _METRICS
            },
            "lines": {
                _line_key(line): line
                for raw in raw_lines
                if (line := _mapping_line(raw)) is not None
            },
        }

    estimate = source if isinstance(source, PartEstimate) else estimate_part(source, material)
    return {
        "metrics": {
            "volume": _number(estimate.volume),
            "mass": _number(estimate.mass),
            "cost": _number(estimate.total_cost),
            "carbon": _number(estimate.embodied_carbon),
            "energy": _number(estimate.embodied_energy),
        },
        "lines": {},
    }


def _is_revision_mapping(source: Mapping[str, Any]) -> bool:
    aggregate = set(_METRICS) | {f"total_{name}" for name in _METRICS}
    return bool(aggregate.intersection(source) or "totals" in source or "lines" in source)


def _mapping_line(raw: Any) -> Optional[BOMLine]:
    if isinstance(raw, BOMLine):
        return raw
    if not isinstance(raw, Mapping) or "part" not in raw:
        return None
    return BOMLine(
        part=str(raw["part"]),
        qty=int(raw.get("qty", 0)),
        material=str(raw.get("material", "")),
        unit_mass=_number(raw.get("unit_mass")),
        unit_cost=_number(raw.get("unit_cost")),
        unit_carbon=_number(raw.get("unit_carbon")),
    )


def _delta(name: str, before: Optional[float], after: Optional[float]) -> ScalarDelta:
    if before is None or after is None:
        return ScalarDelta(name, before, after, None, None, False)
    absolute = after - before
    percent = None if before == 0 else absolute / abs(before) * 100.0
    return ScalarDelta(name, before, after, absolute, percent, True)


def _line_key(line: BOMLine) -> tuple[str, str]:
    return line.part, line.material


def _line_deltas(
    before: Mapping[tuple[str, str], BOMLine],
    after: Mapping[tuple[str, str], BOMLine],
) -> tuple[BOMLineDelta, ...]:
    out = []
    for part, material in sorted(set(before) | set(after)):
        left = before.get((part, material))
        right = after.get((part, material))
        bq, aq = left.qty if left else 0, right.qty if right else 0
        out.append(
            BOMLineDelta(
                part, material, bq, aq, aq - bq,
                _optional_sub(_extended(right, "mass"), _extended(left, "mass")),
                _optional_sub(_extended(right, "cost"), _extended(left, "cost")),
                _optional_sub(_extended(right, "carbon"), _extended(left, "carbon")),
                _optional_sub(_extended(right, "energy"), _extended(left, "energy")),
            )
        )
    return tuple(out)


def _extended(line: Optional[BOMLine], metric: str) -> Optional[float]:
    if line is None:
        return 0.0
    if metric == "energy":
        value = line.estimate.embodied_energy if line.estimate else None
        return None if value is None else value * line.qty
    value = getattr(line, f"unit_{metric}")
    return None if value is None else value * line.qty


def _optional_sub(after: Optional[float], before: Optional[float]) -> Optional[float]:
    return None if after is None or before is None else after - before


def _number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None
