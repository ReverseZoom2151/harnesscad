"""Bounding-box assembly-placement gates (adjacency + floating-part).

Ported from CADCLAW's ``adjacency.py`` and ``floating.py`` -- two of its
assembly verification gates.  CADCLAW checks an authored CAD assembly for
mis-placed components using nothing but part bounding boxes:

*   **Adjacency gate** (:func:`adjacency_check`) -- every part of type ``source``
    must have a part of type ``target`` within ``max_distance`` (centre-to-centre).
    Catches scattered or wrongly-placed components ("every motor needs a bracket
    within 50 mm").  Mirror of ``AdjacencyCheck``.
*   **Floating-part gate** (:func:`floating_check`) -- every non-exempt part must
    be within ``max_gap`` of at least one *structural* anchor part (bbox-to-bbox
    gap).  Catches parts that pass inventory and interference checks yet are
    disconnected from the assembly (the "idler floating in the centre" bug).
    Mirror of ``FloatingCheck``.

Both gates in the original operate on CadQuery solids via ``.BoundingBox()``.
Here a part is the pure-data :class:`Part` (a label + an axis-aligned bounding
box), so the gates are stdlib-only and deterministic -- the exact geometric
predicate is preserved, only the geometry source is abstracted.  The
:func:`bbox_distance` helper (minimum L2 gap between two AABBs, zero when they
overlap) is the shared distance primitive.

The harness already has ``domain.assembly.interference`` (solid overlap); these
gates are the complementary *too-far-apart* checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

__all__ = [
    "BBox6",
    "Point3",
    "Part",
    "bbox_distance",
    "center_distance",
    "AdjacencyRule",
    "AdjacencyViolation",
    "AdjacencyResult",
    "adjacency_check",
    "FloatingPart",
    "FloatingResult",
    "floating_check",
]

BBox6 = Tuple[float, float, float, float, float, float]
Point3 = Tuple[float, float, float]


@dataclass(frozen=True)
class Part:
    """An assembly part: a label plus an axis-aligned bounding box.

    ``bbox`` is ``(xmin, ymin, zmin, xmax, ymax, zmax)``.
    """

    label: str
    bbox: BBox6

    @property
    def center(self) -> Point3:
        b = self.bbox
        return ((b[0] + b[3]) / 2.0, (b[1] + b[4]) / 2.0, (b[2] + b[5]) / 2.0)


def bbox_distance(a: BBox6, b: BBox6) -> float:
    """Minimum L2 distance between two axis-aligned bounding boxes.

    Zero when the boxes overlap; otherwise the L2 norm of the per-axis gaps.
    Mirror of CADCLAW's ``bbox_distance``.
    """
    dx = max(0.0, a[0] - b[3], b[0] - a[3])
    dy = max(0.0, a[1] - b[4], b[1] - a[4])
    dz = max(0.0, a[2] - b[5], b[2] - a[5])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def center_distance(a: Point3, b: Point3) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


# ---------------------------------------------------------------------------
# Adjacency gate.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdjacencyRule:
    """Every ``source`` part must have a ``target`` part within ``max_distance``."""

    source: str
    target: str
    max_distance: float = 50.0
    source_filter: Optional[Callable[[Part], bool]] = None


@dataclass(frozen=True)
class AdjacencyViolation:
    source_label: str
    source_center: Point3
    nearest_target_label: str
    nearest_distance: float
    max_allowed: float


@dataclass
class AdjacencyResult:
    passed: bool
    violations: List[AdjacencyViolation] = field(default_factory=list)


def adjacency_check(parts: Sequence[Part], rules: Sequence[AdjacencyRule]) -> AdjacencyResult:
    """Run adjacency rules; a violation is a source part with no near target.

    Centre-to-centre distance, mirror of ``AdjacencyCheck.run``.  A missing
    target type yields a violation with distance ``inf``.
    """
    by_label: Dict[str, List[Part]] = {}
    for p in parts:
        by_label.setdefault(p.label, []).append(p)

    violations: List[AdjacencyViolation] = []
    for rule in rules:
        sources = by_label.get(rule.source, [])
        targets = by_label.get(rule.target, [])
        for src in sources:
            if rule.source_filter and not rule.source_filter(src):
                continue
            sc = src.center
            if not targets:
                violations.append(AdjacencyViolation(
                    rule.source, sc, rule.target, math.inf, rule.max_distance))
                continue
            nearest = min(targets, key=lambda t: center_distance(sc, t.center))
            d = center_distance(sc, nearest.center)
            if d > rule.max_distance:
                violations.append(AdjacencyViolation(
                    rule.source, sc, rule.target, d, rule.max_distance))
    return AdjacencyResult(passed=not violations, violations=violations)


# ---------------------------------------------------------------------------
# Floating-part gate.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FloatingPart:
    label: str
    center: Point3
    bbox: BBox6
    nearest_label: Optional[str]
    nearest_center: Optional[Point3]
    nearest_distance_mm: float


@dataclass
class FloatingResult:
    passed: bool
    checked: int
    floating: List[FloatingPart] = field(default_factory=list)


def floating_check(
    parts: Sequence[Part],
    structural_labels: Set[str],
    *,
    max_gap_mm: float = 5.0,
    exempt_labels: Optional[Set[str]] = None,
) -> FloatingResult:
    """Flag non-exempt parts not within ``max_gap_mm`` of any structural anchor.

    Mirror of ``FloatingCheck.run``.  Disabled (``checked=0``, passed) when
    ``structural_labels`` is empty or no structural part is present -- the
    reference treats those as setup conditions, not failures.
    """
    structural_labels = set(structural_labels)
    exempt = set(exempt_labels) if exempt_labels else {"belt"}
    if not structural_labels:
        return FloatingResult(passed=True, checked=0, floating=[])

    structural = [p for p in parts if p.label in structural_labels]
    if not structural:
        return FloatingResult(passed=True, checked=0, floating=[])

    floating: List[FloatingPart] = []
    checked = 0
    for p in parts:
        if p.label in exempt or p.label in structural_labels:
            continue
        checked += 1
        best_dist = math.inf
        best: Optional[Part] = None
        for s in structural:
            d = bbox_distance(p.bbox, s.bbox)
            if d < best_dist:
                best_dist = d
                best = s
                if best_dist == 0.0:
                    break
        if best_dist > max_gap_mm:
            floating.append(FloatingPart(
                label=p.label,
                center=p.center,
                bbox=p.bbox,
                nearest_label=best.label if best else None,
                nearest_center=best.center if best else None,
                nearest_distance_mm=best_dist if best_dist != math.inf else -1.0,
            ))
    return FloatingResult(passed=not floating, checked=checked, floating=floating)
