"""HarnessCAD parts library — retrieve-and-instantiate standard parts.

Instead of modelling gears, flanges, shafts, bearings and brackets from scratch,
retrieve a parametric :class:`ModelCard` from a functionally-indexed
:class:`PartCatalog` and instantiate it (range-validated) into a CISP op stream.
Cards are admitted only if their ops build (the Voyager gate reused from
``memory.skills``).
"""

from __future__ import annotations

from library.parts import (
    ModelCard,
    flange_ops, bracket_part_ops, spur_gear_blank_ops, shaft_ops, bearing_seat_ops,
    flange_card, bracket_card, spur_gear_blank_card, shaft_card, bearing_seat_card,
    default_cards,
)
from library.catalog import PartCatalog, build_default_catalog

__all__ = [
    "ModelCard",
    "PartCatalog",
    "build_default_catalog",
    "default_cards",
    "flange_ops", "bracket_part_ops", "spur_gear_blank_ops", "shaft_ops",
    "bearing_seat_ops",
    "flange_card", "bracket_card", "spur_gear_blank_card", "shaft_card",
    "bearing_seat_card",
]
