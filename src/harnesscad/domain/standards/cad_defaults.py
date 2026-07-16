"""First-pass parametric modeling defaults, ported from text-to-cad's CAD skill.

Source: earthtojake/text-to-cad (resources/cad_repos/text-to-cad-main),
``skills/cad/SKILL.md`` ("Default assumptions") and
``skills/cad/references/build123d-modeling.md`` / ``positioning.md``. That
skill pack encodes the working defaults a CAD agent should assume when the
user does not specify them -- explicitly framed there as "first-pass modeling
defaults, not manufacturability, tolerance, or certification claims". The
same framing applies here.

What is ported (values verbatim from the source skill):

* metric **normal clearance hole** diameters -- M3/M4/M5 -> 3.4/4.5/5.5 mm;
* **small plastic enclosure wall**: 2.0-3.0 mm when unspecified;
* **cosmetic fillet**: 1.0-3.0 mm "when safe for local geometry";
* per-part-type **origin conventions** (plates at footprint center,
  axisymmetric parts on the rotational axis, adapters on the mounting datum);
* the **feature-order heuristic**: base solid, major additions, subtractive
  features, shell, through-wall holes, then fillets and chamfers last,
  because "fillets are the most failure-prone operation and every boolean
  invalidates selectors";
* the **boolean overshoot** rule: extend cutting tools roughly 1 mm past the
  faces they enter and exit, because coincident tool/target faces are a
  classic kernel failure.

These pair with the harness's existing standards data
(:mod:`harnesscad.domain.standards.thread_database` for thread geometry,
``heatsert_bores`` for insert bores); this module is the agent-facing
*assumption* layer those tables do not cover.

Stdlib-only, deterministic, absolute imports.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional, Tuple

__all__ = [
    "CLEARANCE_HOLE_NORMAL_MM",
    "ENCLOSURE_WALL_RANGE_MM",
    "COSMETIC_FILLET_RANGE_MM",
    "BOOLEAN_OVERSHOOT_MM",
    "FEATURE_ORDER",
    "ORIGIN_CONVENTIONS",
    "clearance_hole_diameter",
    "clearance_hole_radius",
    "default_wall_thickness",
    "default_fillet_radius",
    "origin_convention",
    "feature_order",
    "main",
]

# skills/cad/SKILL.md: "M3/M4/M5 normal clearance holes: 3.4/4.5/5.5 mm
# unless another standard is requested."
CLEARANCE_HOLE_NORMAL_MM: Dict[str, float] = {
    "M3": 3.4,
    "M4": 4.5,
    "M5": 5.5,
}

# "Small plastic enclosure wall: 2.0-3.0 mm when unspecified."
ENCLOSURE_WALL_RANGE_MM: Tuple[float, float] = (2.0, 3.0)

# "Cosmetic fillet: 1.0-3.0 mm when safe for local geometry."
COSMETIC_FILLET_RANGE_MM: Tuple[float, float] = (1.0, 3.0)

# build123d-modeling.md, "Overshoot boolean tools": "for through-cuts, go
# roughly 1 mm beyond both faces."
BOOLEAN_OVERSHOOT_MM: float = 1.0

# build123d-modeling.md, "Order operations so fragile steps come last and
# failures localize."
FEATURE_ORDER: Tuple[str, ...] = (
    "base_solid",
    "major_additions",
    "subtractive_features",
    "shell",
    "through_wall_holes",
    "fillets_and_chamfers",
)

# positioning.md, "Good defaults" for part-local origins.
ORIGIN_CONVENTIONS: Dict[str, str] = {
    "symmetric": "origin at body center",
    "plate": "origin at footprint center; thickness along Z",
    "enclosure": ("origin at footprint center; base/lid mating surfaces "
                  "controlled by Z parameters"),
    "axisymmetric": "origin on rotational axis",
    "adapter": ("origin on the primary mounting datum or center of the "
                "bolt pattern"),
}


def clearance_hole_diameter(size: str) -> float:
    """Normal-fit clearance hole diameter (mm) for a metric screw size.

    Only the sizes the source skill specifies are answered; asking for any
    other size raises rather than inventing a value (soundness over
    completeness -- extend the table from a cited standard, do not
    extrapolate here).
    """
    key = size.strip().upper()
    if key not in CLEARANCE_HOLE_NORMAL_MM:
        known = ", ".join(sorted(CLEARANCE_HOLE_NORMAL_MM))
        raise KeyError(
            f"no ported clearance-hole value for {size!r} (known: {known}); "
            f"extend CLEARANCE_HOLE_NORMAL_MM from a cited standard instead "
            f"of extrapolating")
    return CLEARANCE_HOLE_NORMAL_MM[key]


def clearance_hole_radius(size: str) -> float:
    return clearance_hole_diameter(size) / 2.0


def default_wall_thickness() -> float:
    """Midpoint of the source range, for a single first-pass number."""
    lo, hi = ENCLOSURE_WALL_RANGE_MM
    return (lo + hi) / 2.0


def default_fillet_radius(smallest_local_extent_mm: Optional[float] = None) -> float:
    """A cosmetic fillet radius that is 'safe for local geometry'.

    Without context, the low end of the source range (1.0 mm). With a known
    smallest local extent, the radius is additionally capped below half of
    it -- the same infeasibility bound the harness preflight enforces
    (a fillet radius at or above half the smallest extent consumes the part).
    """
    lo, hi = COSMETIC_FILLET_RANGE_MM
    radius = lo
    if smallest_local_extent_mm is not None:
        if smallest_local_extent_mm <= 0:
            raise ValueError("smallest_local_extent_mm must be positive")
        cap = smallest_local_extent_mm / 2.0
        if cap <= lo:
            radius = cap * 0.5  # stay strictly inside the feasible bound
        else:
            radius = min(hi, cap * 0.5) if cap * 0.5 >= lo else lo
    return radius


def origin_convention(part_type: str) -> str:
    """The part-local origin convention for a part type ('symmetric',
    'plate', 'enclosure', 'axisymmetric', 'adapter')."""
    key = part_type.strip().lower()
    if key not in ORIGIN_CONVENTIONS:
        known = ", ".join(sorted(ORIGIN_CONVENTIONS))
        raise KeyError(f"unknown part type {part_type!r} (known: {known})")
    return ORIGIN_CONVENTIONS[key]


def feature_order() -> List[str]:
    """The fragile-steps-last modeling order, base solid first."""
    return list(FEATURE_ORDER)


def _selfcheck() -> int:
    failures: List[str] = []

    def check(cond: bool, message: str) -> None:
        if not cond:
            failures.append(message)

    check(clearance_hole_diameter("m3") == 3.4, "M3 normal clearance is 3.4")
    check(clearance_hole_diameter("M4") == 4.5, "M4 normal clearance is 4.5")
    check(clearance_hole_radius("M5") == 2.75, "M5 clearance radius")
    try:
        clearance_hole_diameter("M6")
        check(False, "unknown size must raise, not extrapolate")
    except KeyError:
        pass
    check(default_wall_thickness() == 2.5, "wall default is the range midpoint")
    check(default_fillet_radius() == 1.0, "context-free fillet is the low end")
    check(default_fillet_radius(20.0) <= 3.0, "fillet capped at range top")
    check(default_fillet_radius(2.0) < 1.0,
          "fillet stays strictly inside the feasibility bound on thin stock")
    try:
        default_fillet_radius(0.0)
        check(False, "non-positive extent must raise")
    except ValueError:
        pass
    check(feature_order()[0] == "base_solid"
          and feature_order()[-1] == "fillets_and_chamfers",
          "fragile steps come last")
    check("footprint center" in origin_convention("plate"),
          "plate origin convention ported")
    try:
        origin_convention("spaceship")
        check(False, "unknown part type must raise")
    except KeyError:
        pass

    for message in failures:
        print("selfcheck FAIL: " + message)
    print("selfcheck: %s" % ("PASS" if not failures else "FAIL"))
    return 0 if not failures else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cad_defaults",
        description="first-pass CAD modeling defaults ported from "
                    "earthtojake/text-to-cad")
    parser.add_argument("--selfcheck", action="store_true",
                        help="run the built-in self-test and exit")
    args = parser.parse_args(argv)
    if args.selfcheck:
        return _selfcheck()
    print("clearance holes (normal fit, mm):")
    for size in sorted(CLEARANCE_HOLE_NORMAL_MM):
        print(f"  {size}: {CLEARANCE_HOLE_NORMAL_MM[size]}")
    print(f"enclosure wall: {ENCLOSURE_WALL_RANGE_MM} mm")
    print(f"cosmetic fillet: {COSMETIC_FILLET_RANGE_MM} mm")
    print(f"boolean overshoot: {BOOLEAN_OVERSHOOT_MM} mm")
    print("feature order: " + " -> ".join(FEATURE_ORDER))
    return 0


if __name__ == "__main__":
    sys.exit(main())
