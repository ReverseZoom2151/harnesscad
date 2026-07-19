"""Embodied-carbon (CO2e) material accounting for generated designs.

A deterministic post-processing accounting pipeline: identify the materials
present in a design, pair each with a carbon-dioxide-equivalent (CO2e) value
from a materials dictionary, and surface the top contributors so a designer can
iterate toward lower-impact choices.

This module implements that accounting:

*   :data:`DEFAULT_CO2E` -- a small general materials dictionary (kg CO2e per kg).
*   :func:`embodied_carbon` -- CO2e for one material given its mass.
*   :func:`aggregate` -- total embodied carbon across a bill of materials.
*   :func:`top_contributors` -- the top-N materials by CO2e, the ranked list
    shown back to the designer.

Deterministic and stdlib-only. Ties in the ranking break by material name so the
output order is stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

__all__ = [
    "DEFAULT_CO2E",
    "MaterialUse",
    "embodied_carbon",
    "aggregate",
    "top_contributors",
    "carbon_intensity",
]

#: Representative cradle-to-gate CO2e factors, kg CO2e per kg of material.
#: Order-of-magnitude values sufficient for a deterministic ranking demo.
DEFAULT_CO2E: Dict[str, float] = {
    "timber": 0.45,
    "bamboo": 0.30,
    "cork": 0.19,
    "wool": 0.98,
    "cotton": 5.90,
    "glass": 0.85,
    "ceramic": 0.70,
    "concrete": 0.11,
    "brick": 0.24,
    "gypsum": 0.39,
    "steel": 1.46,
    "aluminium": 8.24,
    "copper": 2.80,
    "plastic": 3.10,
    "paint": 2.10,
    "leather": 17.0,
    "marble": 0.13,
    "granite": 0.64,
    "stone": 0.08,
    "vinyl": 2.41,
}


@dataclass(frozen=True)
class MaterialUse:
    """A material and its mass (kg) in a design's bill of materials."""

    material: str
    mass_kg: float

    def __post_init__(self) -> None:
        if self.mass_kg < 0:
            raise ValueError("mass_kg must be non-negative")


def carbon_intensity(material: str, table: Mapping[str, float] = DEFAULT_CO2E) -> float:
    """CO2e factor (kg CO2e / kg) for a material. Raises if unknown."""
    key = material.strip().lower()
    if key not in table:
        raise KeyError(f"unknown material {material!r}")
    return table[key]


def embodied_carbon(
    use: MaterialUse, table: Mapping[str, float] = DEFAULT_CO2E
) -> float:
    """Embodied CO2e (kg) for one material use."""
    return carbon_intensity(use.material, table) * use.mass_kg


def aggregate(
    uses: Sequence[MaterialUse], table: Mapping[str, float] = DEFAULT_CO2E
) -> float:
    """Total embodied CO2e (kg) across a bill of materials.

    Repeated materials are summed. Unknown materials raise ``KeyError``.
    """
    return sum(embodied_carbon(u, table) for u in uses)


def top_contributors(
    uses: Sequence[MaterialUse],
    n: int = 10,
    table: Mapping[str, float] = DEFAULT_CO2E,
) -> List[Tuple[str, float]]:
    """Top-``n`` materials by total embodied CO2e, descending.

    Masses of repeated materials are combined before ranking. Ties break by
    material name (ascending) for a deterministic order.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    totals: Dict[str, float] = {}
    for u in uses:
        key = u.material.strip().lower()
        totals[key] = totals.get(key, 0.0) + embodied_carbon(u, table)
    ranked = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:n]
