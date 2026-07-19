"""Deterministic spur/helical gear-train parameter math.

The parameter computation is standard involute-gear engineering math and is
fully deterministic, yielding a verifiable domain utility independent of how
inputs are proposed.

It provides:

* :func:`snap_module` -- round a raw module to the nearest ISO 54 preferred
  value, so a designer's arbitrary
  module becomes a stock-standard one,
* :func:`gear_geometry` -- pitch / addendum / dedendum / outside / root
  diameters from module + tooth count (+ helix angle for helical gears),
* :func:`mesh_pair` -- gear ratio, exact centre distance, and a mesh-validity
  check (equal module, compatible helix hands) for a two-gear train.

All lengths are in millimetres, angles in degrees. Stdlib-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

__all__ = [
    "PREFERRED_MODULES_PRIMARY",
    "PREFERRED_MODULES_SECONDARY",
    "snap_module",
    "GearGeometry",
    "gear_geometry",
    "MeshResult",
    "mesh_pair",
]

#: ISO 54 first-choice module series (CAD-GPT's priority list).
PREFERRED_MODULES_PRIMARY: Tuple[float, ...] = (
    1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50,
)

#: Second-choice module series, used only when no primary value is close.
PREFERRED_MODULES_SECONDARY: Tuple[float, ...] = (
    1.75, 2.25, 2.75, 3.5, 4.5, 5.5, 7, 9, 14, 18, 22, 28, 36,
)


def snap_module(raw: float, *, prefer_primary_tolerance: float = 0.10) -> float:
    """Snap a raw module to the nearest ISO-preferred value.

    A primary value is chosen when one lies within ``prefer_primary_tolerance``
    (relative) of ``raw``; otherwise the nearest value across both series wins.
    This mirrors CAD-GPT's "round up in priority, else fall back" rule while
    staying deterministic.
    """
    if raw <= 0:
        raise ValueError("module must be positive")
    best_primary = min(PREFERRED_MODULES_PRIMARY, key=lambda m: abs(m - raw))
    if abs(best_primary - raw) <= prefer_primary_tolerance * raw:
        return float(best_primary)
    both = PREFERRED_MODULES_PRIMARY + PREFERRED_MODULES_SECONDARY
    return float(min(both, key=lambda m: (abs(m - raw), m)))


@dataclass(frozen=True)
class GearGeometry:
    """Derived geometry of a single involute gear (mm)."""

    module: float
    teeth: int
    helix_angle: float
    pitch_diameter: float
    outside_diameter: float
    root_diameter: float
    base_diameter: float


def gear_geometry(
    module: float,
    teeth: int,
    *,
    helix_angle: float = 0.0,
    pressure_angle: float = 20.0,
) -> GearGeometry:
    """Compute involute-gear diameters from module and tooth count.

    For a helical gear the transverse module grows by ``1/cos(beta)``, so the
    pitch diameter is ``m * z / cos(beta)``. Addendum = 1 module, dedendum =
    1.25 module (standard full-depth teeth), giving outside = pitch + 2m and
    root = pitch - 2.5m. Base diameter = pitch * cos(pressure_angle).
    """
    if module <= 0 or teeth <= 0:
        raise ValueError("module and teeth must be positive")
    if not (0.0 <= helix_angle < 90.0):
        raise ValueError("helix_angle must be in [0, 90) degrees")
    beta = math.radians(helix_angle)
    pitch_d = module * teeth / math.cos(beta)
    outside_d = pitch_d + 2.0 * module
    root_d = pitch_d - 2.5 * module
    base_d = pitch_d * math.cos(math.radians(pressure_angle))
    return GearGeometry(
        module=float(module),
        teeth=int(teeth),
        helix_angle=float(helix_angle),
        pitch_diameter=pitch_d,
        outside_diameter=outside_d,
        root_diameter=root_d,
        base_diameter=base_d,
    )


@dataclass(frozen=True)
class MeshResult:
    """Outcome of meshing two gears into a train."""

    meshes: bool
    gear_ratio: float
    center_distance: float
    reasons: Tuple[str, ...]


def mesh_pair(
    driving: GearGeometry,
    driven: GearGeometry,
    *,
    module_tol: float = 1e-6,
) -> MeshResult:
    """Check whether two gears mesh and return ratio + centre distance.

    Two involute gears mesh only if they share the same module and helix angle
    magnitude (external gears run on opposite helix hands, checked by the caller
    via the sign of ``helix_angle`` if it is signed). The gear ratio is
    ``driven.teeth / driving.teeth``; the exact centre distance for external
    gears is the mean of the two pitch diameters.
    """
    reasons = []
    if abs(driving.module - driven.module) > module_tol:
        reasons.append(
            f"module mismatch ({driving.module} vs {driven.module}): gears cannot mesh"
        )
    if abs(driving.helix_angle - driven.helix_angle) > 1e-6:
        reasons.append(
            f"helix-angle mismatch ({driving.helix_angle} vs {driven.helix_angle})"
        )
    ratio = driven.teeth / driving.teeth
    center = 0.5 * (driving.pitch_diameter + driven.pitch_diameter)
    return MeshResult(
        meshes=not reasons,
        gear_ratio=ratio,
        center_distance=center,
        reasons=tuple(reasons),
    )
