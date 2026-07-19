"""Rule-based physical-plausibility gate and hard-constraint checker for CAD solids.

A deterministic geometric-constraint checker for text-to-CAD code generation.
Unlike the harness's ``quality/anomaly.py`` -- which turns the same
bounding-box shape ratios into a scale-robust *feature vector* for statistical
outlier scoring -- this module is a
deterministic **acceptance gate**: it applies fixed absolute thresholds to a
solid's measured properties and returns concrete issue/warning strings plus a
pass/fail flag. It also evaluates a solid against an explicit constraint dict
(dimension caps, volume bounds, topology requirements).

All computation is stdlib-only and deterministic. Inputs are plain numbers (the
caller extracts them from whatever kernel it uses), so nothing here depends on a
CAD kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Default physical-plausibility thresholds.
MIN_DIMENSION = 0.1          # mm; below this a dimension is "very small"
MAX_DIMENSION = 10000.0      # mm; above this a dimension is "very large"
EXTREME_ASPECT_HI = 100.0    # side-ratio flagged as extreme when above this
EXTREME_ASPECT_LO = 0.01     # ...or below this
LOW_FILL_RATIO = 0.01        # solidity below this -> sparse geometry warning
# A solid cannot occupy more volume than its own bounding box, so a fill ratio
# above 1.0 is IMPOSSIBLE and means the volume and the bbox disagree. That -- and
# only that -- is an issue. This was 0.95, which flagged every plain plate: a
# solid plate is ~1.0 BECAUSE IT IS SOLID (red-team sweep: 18 of 45
# provably-correct parts). The threshold now sits on the theorem.
HIGH_FILL_RATIO = 1.0 + 1e-6  # solidity above this is geometrically impossible
HIGH_SA_TO_VOL = 1000.0      # surface-area / volume above this -> thin/complex


@dataclass(frozen=True)
class AABB:
    """Axis-aligned bounding box.

    ``width`` spans x, ``depth`` spans y, ``height`` spans z -- the convention
    where height is the vertical (z) extent.
    """

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def depth(self) -> float:
        return self.max_y - self.min_y

    @property
    def height(self) -> float:
        return self.max_z - self.min_z

    @property
    def volume(self) -> float:
        return self.width * self.height * self.depth

    @property
    def center(self) -> tuple:
        return (
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
            (self.min_z + self.max_z) / 2.0,
        )

    def extents(self) -> tuple:
        """(width, depth, height) as a tuple."""
        return (self.width, self.depth, self.height)


def _pairwise_aspect_ratios(bbox: AABB) -> List[float]:
    """The three side-to-side ratios; a zero denominator yields +inf."""
    w, d, h = bbox.width, bbox.depth, bbox.height
    inf = float("inf")
    return [
        w / h if h > 0 else inf,
        w / d if d > 0 else inf,
        h / d if d > 0 else inf,
    ]


def fill_ratio(volume: float, bbox: AABB) -> Optional[float]:
    """Solidity: solid volume / bounding-box volume.

    Returns ``None`` when either volume is non-positive (ratio undefined).
    A cuboid has ratio ~1; a thin frame or sparse lattice has a small ratio.
    """
    bv = bbox.volume
    if volume <= 0 or bv <= 0:
        return None
    return volume / bv


def check_physical_plausibility(
    volume: float,
    surface_area: float,
    bbox: AABB,
    *,
    min_dimension: float = MIN_DIMENSION,
    max_dimension: float = MAX_DIMENSION,
    aspect_hi: float = EXTREME_ASPECT_HI,
    aspect_lo: float = EXTREME_ASPECT_LO,
    low_fill: float = LOW_FILL_RATIO,
    high_fill: float = HIGH_FILL_RATIO,
    high_sa_to_vol: float = HIGH_SA_TO_VOL,
) -> Dict[str, Any]:
    """Deterministic plausibility screen over measured solid properties.

    Returns a dict with:
      * ``plausible`` -- ``True`` when no hard *issue* was raised (warnings are
        non-fatal, matching SpatialHero's semantics);
      * ``issues``    -- fatal problems (e.g. suspiciously high fill ratio);
      * ``warnings``  -- soft flags (tiny/huge dimensions, extreme aspect,
        sparse fill, high surface-area-to-volume);
      * ``metrics``   -- the intermediate numbers used, for inspection.
    """
    issues: List[str] = []
    warnings: List[str] = []

    extents = bbox.extents()
    min_dim = min(extents)
    max_dim = max(extents)

    if min_dim < min_dimension:
        warnings.append(f"Very small dimension detected: {min_dim}mm")
    if max_dim > max_dimension:
        warnings.append(f"Very large dimension detected: {max_dim}mm")

    ratios = _pairwise_aspect_ratios(bbox)
    finite = [r for r in ratios if r != float("inf")]
    extreme = [r for r in ratios if r > aspect_hi or r < aspect_lo]
    if extreme and finite:
        warnings.append(
            f"Extreme aspect ratio detected: {min(finite):.2f} - {max(finite):.2f}"
        )

    fr = fill_ratio(volume, bbox)
    if fr is not None:
        if fr < low_fill:
            warnings.append(
                f"Very low fill ratio: {fr:.2%} - geometry may be too sparse"
            )
        elif fr > high_fill:
            # The bound is 1.0 and it is a THEOREM: a solid cannot occupy more
            # volume than its own bounding box, so fill ratio > 1 means the
            # volume and the bbox disagree -- a measurement or a build is wrong.
            #
            # It used to be 0.95, and that flagged a plain plate. A solid plate
            # has a fill ratio near 1.0 BECAUSE IT IS SOLID; the rule mistook
            # "solid" for "suspicious" and fired on 18 of 45 provably-correct
            # parts in the red-team sweep, hardest on exactly the parts most
            # likely to be right. A rule that fires on correctness is not a
            # verifier. The impossible case is still caught, and only it.
            issues.append(
                f"Fill ratio {fr:.2%} exceeds 1.0: the measured volume "
                f"({volume:g}) is larger than its own bounding box "
                f"({bbox.volume:g}), which no solid can be. The volume "
                f"measurement or the bbox is wrong."
            )

    sa_to_vol = None
    if volume > 0:
        sa_to_vol = surface_area / volume
        if sa_to_vol > high_sa_to_vol:
            warnings.append(
                "High surface area to volume ratio - "
                "may indicate thin or complex geometry"
            )

    return {
        "plausible": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "metrics": {
            "min_dimension": min_dim,
            "max_dimension": max_dim,
            "aspect_ratios": ratios,
            "fill_ratio": fr,
            "sa_to_vol": sa_to_vol,
        },
    }


def check_constraints(
    bbox: AABB,
    volume: float,
    constraints: Dict[str, Any],
    *,
    num_faces: Optional[int] = None,
    is_closed: Optional[bool] = None,
) -> Dict[str, bool]:
    """Evaluate a solid against an explicit constraint dict.

    Recognised keys (all optional): ``max_width``, ``max_height``,
    ``max_depth``, ``min_volume``, ``max_volume``, ``must_be_closed``,
    ``min_faces``. Only keys present in ``constraints`` (and whose backing
    measurement is available) appear in the result, each mapped to a pass bool.
    """
    results: Dict[str, bool] = {}

    if "max_width" in constraints:
        results["max_width"] = bbox.width <= constraints["max_width"]
    if "max_height" in constraints:
        results["max_height"] = bbox.height <= constraints["max_height"]
    if "max_depth" in constraints:
        results["max_depth"] = bbox.depth <= constraints["max_depth"]
    if "min_volume" in constraints:
        results["min_volume"] = volume >= constraints["min_volume"]
    if "max_volume" in constraints:
        results["max_volume"] = volume <= constraints["max_volume"]
    if "must_be_closed" in constraints and is_closed is not None:
        results["must_be_closed"] = bool(is_closed) == bool(constraints["must_be_closed"])
    if "min_faces" in constraints and num_faces is not None:
        results["min_faces"] = num_faces >= constraints["min_faces"]

    return results


def all_constraints_pass(results: Dict[str, bool]) -> bool:
    """True when every evaluated constraint passed (vacuously true if empty)."""
    return all(results.values())
