"""CAD instruction/model complexity metrics used for dataset balancing."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Complexity:
    unit_count: int
    parameter_density: float
    occupancy_entropy: float


def voxel_entropy(occupancy: Iterable[int | bool]) -> float:
    values = tuple(bool(value) for value in occupancy)
    if not values:
        return 0.0
    occupied = sum(values)
    probabilities = (occupied / len(values), (len(values) - occupied) / len(values))
    entropy = -sum(p * log2(p) for p in probabilities if p)
    return entropy


def measure_complexity(
    units: Sequence[Mapping[str, object]], occupancy: Iterable[int | bool]
) -> Complexity:
    parameter_count = sum(
        len(unit.get("parameters", {}))
        for unit in units
        if isinstance(unit.get("parameters", {}), Mapping)
    )
    count = len(units)
    return Complexity(
        count,
        parameter_count / count if count else 0.0,
        voxel_entropy(occupancy),
    )
