"""Involute spur-gear geometry (deterministic, stdlib-only).

Ported from the OpenSCAD parametric-gear library bundled with CAD-GPT
(``paper_code/common/*/gears.scad``, R. Huttary). CAD-GPT itself is an LLM
that emits OpenSCAD ``gear(m=..., z=..., h=..., w=...)`` calls; the *geometry*
those calls expand to is a fully deterministic rack-generated involute profile.
The harness previously carried only a toothless gear *blank* (a pitch-diameter
cylinder, see ``library/parts.py::spur_gear_blank_ops``) which explicitly notes
"no involute teeth"; this module supplies the missing tooth geometry.

Everything here is closed-form and angle units are DEGREES (matching OpenSCAD's
trig convention, where ``tan(w)`` takes ``w`` in degrees).

Definitions (module ``m``, teeth ``z``, profile shift ``x``, pressure angle ``w``):

    pitch radius   r_wk = m*z/2 + x
    addendum       a    = m                (tooth height above pitch line)
    dedendum       b    = m                (before clearance)
    tip radius     r_kk = r_wk + m
    root radius    r_fk = r_wk - m
    base radius    r_b  = r_wk_nominal * cos(w)      (nominal = m*z/2)
    circular pitch p    = pi*m

The clearance factor mirrors the OpenSCAD ``gear_info`` module: a positive
clearance shrinks the tip and grows the root (external splines), a negative one
does the reverse (internal splines / rings).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

__all__ = [
    "pitch_radius",
    "pitch_diameter",
    "base_radius",
    "tip_radius",
    "root_radius",
    "circular_pitch",
    "involute_point",
    "rack_profile",
    "GearGeometry",
    "gear_geometry",
]


def pitch_radius(module: float, teeth: int, profile_shift: float = 0.0) -> float:
    """Working (pitch) radius ``r_wk = m*z/2 + x``."""
    return module * teeth / 2.0 + profile_shift


def pitch_diameter(module: float, teeth: int) -> float:
    """Nominal pitch diameter ``d = m*z``."""
    return module * teeth


def base_radius(module: float, teeth: int, pressure_angle: float = 20.0) -> float:
    """Base-circle radius ``r_b = (m*z/2) * cos(w)`` (nominal, degrees)."""
    return (module * teeth / 2.0) * math.cos(math.radians(pressure_angle))


def tip_radius(module: float, teeth: int, profile_shift: float = 0.0,
               clearance: float = 0.0) -> float:
    """Tip (addendum) radius ``r_kk = r_wk + m*(1 - clearance/2)``."""
    r_wk = pitch_radius(module, teeth, profile_shift)
    return r_wk + module * (1.0 - clearance / 2.0)


def root_radius(module: float, teeth: int, profile_shift: float = 0.0,
                clearance: float = 0.0) -> float:
    """Root (dedendum) radius ``r_fk = r_wk - m*(1 + clearance/2)``."""
    r_wk = pitch_radius(module, teeth, profile_shift)
    return r_wk - module * (1.0 + clearance / 2.0)


def circular_pitch(module: float) -> float:
    """Circular pitch ``p = pi*m`` (arc length between adjacent teeth)."""
    return math.pi * module


def involute_point(base_r: float, roll_angle_deg: float) -> Tuple[float, float]:
    """A point on the involute of a circle of radius ``base_r``.

    ``roll_angle_deg`` is the angle (degrees) through which the generating line
    has unwound. The classic parametrisation::

        x = r_b * (cos(t) + t*sin(t))
        y = r_b * (sin(t) - t*cos(t))

    with ``t`` in radians. At ``t=0`` the point sits on the base circle at
    ``(r_b, 0)``.
    """
    t = math.radians(roll_angle_deg)
    x = base_r * (math.cos(t) + t * math.sin(t))
    y = base_r * (math.sin(t) - t * math.cos(t))
    return (x, y)


def rack_profile(module: float = 2.0, teeth: int = 10, profile_shift: float = 0.0,
                 pressure_angle: float = 20.0,
                 clearance: float = 0.0) -> List[Tuple[float, float]]:
    """The rack-cutter polygon that generates an involute gear.

    Faithful port of the OpenSCAD ``rack()`` function. In the OpenSCAD library a
    gear tooth face is cut by sweeping this rack around a blank; the rack polygon
    itself is a deterministic list of ``(x, y)`` vertices (scaled by ``module``).

    ``pressure_angle`` (``w``) is in degrees. Returns the vertex list in the same
    order OpenSCAD emits it (leading anchor point, four points per tooth index
    ``i`` running ``-1..z`` inclusive, then two closing points).
    """
    m = float(module)
    z = int(teeth)
    x = float(profile_shift)
    w = float(pressure_angle)

    dx = 2.0 * math.tan(math.radians(w))
    c = clearance / m if m != 0 else 0.0
    o = dx / 2.0 - math.pi / 4.0
    r = z / 2.0 + x + 1.0
    X = [c, math.pi / 2.0 - dx - c, math.pi / 2.0 - c, math.pi - dx + c]
    Y = [r - c, r - c, r - 2.0 - c, r - 2.0 - c]

    pts: List[Tuple[float, float]] = [(-math.pi + o, r + 5.0)]
    for i in range(-1, z + 1):          # OpenSCAD [-1:z] is inclusive of z
        for j in range(4):
            pts.append((o + i * math.pi + X[j], Y[j]))
    pts.append((o + math.pi * (z + 1) + c, r - c))
    pts.append((o + math.pi * (z + 1) + c, r + 5.0))

    return [(m * px, m * py) for px, py in pts]


@dataclass(frozen=True)
class GearGeometry:
    """Derived geometry for one involute spur gear (mirrors ``gear_info``)."""

    module: float
    teeth: int
    profile_shift: float
    pressure_angle: float
    clearance: float
    pitch_radius: float
    pitch_diameter: float
    base_radius: float
    tip_radius: float
    root_radius: float
    tip_diameter: float
    root_diameter: float
    circular_pitch: float

    def to_dict(self) -> dict:
        return {
            "module": self.module,
            "teeth": self.teeth,
            "profile_shift": self.profile_shift,
            "pressure_angle": self.pressure_angle,
            "clearance": self.clearance,
            "pitch_radius": self.pitch_radius,
            "pitch_diameter": self.pitch_diameter,
            "base_radius": self.base_radius,
            "tip_radius": self.tip_radius,
            "root_radius": self.root_radius,
            "tip_diameter": self.tip_diameter,
            "root_diameter": self.root_diameter,
            "circular_pitch": self.circular_pitch,
        }


def gear_geometry(module: float, teeth: int, profile_shift: float = 0.0,
                  pressure_angle: float = 20.0,
                  clearance: float = 0.0) -> GearGeometry:
    """Compute the full derived geometry bundle for one spur gear."""
    if module <= 0:
        raise ValueError("module must be positive")
    if teeth < 1:
        raise ValueError("teeth must be >= 1")
    r_wk = pitch_radius(module, teeth, profile_shift)
    r_kk = tip_radius(module, teeth, profile_shift, clearance)
    r_fk = root_radius(module, teeth, profile_shift, clearance)
    return GearGeometry(
        module=float(module),
        teeth=int(teeth),
        profile_shift=float(profile_shift),
        pressure_angle=float(pressure_angle),
        clearance=float(clearance),
        pitch_radius=r_wk,
        pitch_diameter=pitch_diameter(module, teeth),
        base_radius=base_radius(module, teeth, pressure_angle),
        tip_radius=r_kk,
        root_radius=r_fk,
        tip_diameter=2.0 * r_kk,
        root_diameter=2.0 * r_fk,
        circular_pitch=circular_pitch(module),
    )
