"""Complexity scaling taxonomy for text-to-CAD tasks (Kumar et al.,
"Generative AI for CAD Automation", 2025, Table 2 / sec. 3.2).

The paper grades ten FreeCAD test cases on a 1-10 complexity scale, from a
single fixed-dimension primitive (level 1) up to a fully constrained parametric
frame with reinforcement ribs (level 10).  Sec. 3.2 groups these bands:

    levels 1-3  individual primitives, explicit dimensions, no interdependency
    levels 4-7  boolean ops, parametric constraints, hierarchical dependencies
    levels 8-10 intricate feature dependencies / specialised geometry

This module encodes that reference table and a *deterministic feature-based
scorer* that maps a natural-language design description onto an estimated level
by detecting the design features the paper's bands are defined by.  No LLM: pure
keyword/feature signals, reproducible.

Stdlib only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

# Table 2: the reference ten levels (level -> model type, key features).
REFERENCE_LEVELS: Tuple[Tuple[int, str, str], ...] = (
    (1, "Cube", "Basic shape, fixed dimensions"),
    (2, "Cylinder", "Defined radius and height"),
    (3, "Filleted Cuboid", "Edge fillets, feature modifications"),
    (4, "Boolean Union", "Merging a box and cylinder"),
    (5, "Boolean Subtraction", "Cutting a hole through a solid"),
    (6, "Parametric Plate", "Fully constrained model with drilled holes"),
    (7, "Parametric Hinge", "Multiple segments, constraints"),
    (8, "Gear", "Involute profile, precise tooth count"),
    (9, "Plate with Cutouts", "Complex feature constraints"),
    (10, "Parametric Frame", "Reinforcement ribs, multiple constraints"),
)

# Sec. 3.2 complexity bands.
BANDS: Tuple[Tuple[str, int, int], ...] = (
    ("primitive", 1, 3),
    ("compositional", 4, 7),
    ("specialised", 8, 10),
)


def band_of(level: int) -> str:
    """Return the sec. 3.2 band name for a 1-10 level."""
    if not 1 <= level <= 10:
        raise ValueError("level must be in 1..10")
    for name, lo, hi in BANDS:
        if lo <= level <= hi:
            return name
    raise AssertionError("unreachable")


# Feature signals, each contributing to the estimated level.  Ordered so the
# scorer can report which features fired.
_FEATURE_SIGNALS: Tuple[Tuple[str, int, Tuple[str, ...]], ...] = (
    ("fillet", 1, ("fillet",)),
    ("chamfer", 1, ("chamfer",)),
    ("boolean_union", 3, ("union", "merge", "combine")),
    ("boolean_subtraction", 3, ("subtract", "cut", "hole", "bore", "drill")),
    ("parametric", 2, ("parametric", "fully constrained", "constraint")),
    ("hierarchical", 3, ("hinge", "assembly", "segment", "leaf", "knuckle")),
    ("specialised", 5, ("gear", "involute", "thread", "spline", "rib",
                        "reinforcement")),
    ("multi_feature", 1, ("cutout", "multiple", "each corner", "several")),
)


@dataclass
class ComplexityEstimate:
    """Result of scoring a description."""
    level: int
    band: str
    features: List[str]


def score_description(description: str) -> ComplexityEstimate:
    """Estimate the 1-10 complexity level of a design description.

    Starts at level 1 (a plain primitive) and adds the weight of every feature
    signal detected, clamped to 1..10.  Deterministic and case-insensitive.
    """
    if not description or not description.strip():
        raise ValueError("description must be non-empty")
    low = description.lower()
    level = 1
    fired: List[str] = []
    for name, weight, keys in _FEATURE_SIGNALS:
        if any(k in low for k in keys):
            level += weight
            fired.append(name)
    level = max(1, min(10, level))
    return ComplexityEstimate(level=level, band=band_of(level), features=fired)


def band_histogram(levels: List[int]) -> Dict[str, int]:
    """Count how many levels fall in each band (for dataset summaries)."""
    hist = {name: 0 for name, _, _ in BANDS}
    for lv in levels:
        hist[band_of(lv)] += 1
    return hist
