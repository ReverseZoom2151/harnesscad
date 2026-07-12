"""Heat-set (threaded) insert bore schedule keyed to screw designation.

Source rule: the ``heatserts`` plugin of the CadQuery community plugin
collection.  A heat-set insert is a knurled brass sleeve melted into a
thermoplastic part so that a machine screw can be driven into it.  The
plugin encodes a *schedule*: for each metric screw designation (M3, M4,
M5, M6) it stores the insert bore diameter, the bore depth, and the bolt
diameter, and it builds the hole as the union of

* the insert bore -- a cylinder of ``diam`` x ``depth`` down from the face,
* an optional bolt-clearance bore -- a cylinder of ``1.2 * bolt_diam``
  running ``bolt_clear`` down from the *face* (so it only has an effect
  when ``bolt_clear > depth``), and
* an optional lead-in chamfer -- a truncated cone from ``diam/2 + setback``
  at the face down to ``diam/2`` over ``chamfer_depth``; a scalar chamfer
  means setback == depth, i.e. a 45 degree lead-in.

This module reimplements the schedule and the resulting axial profile with
stdlib arithmetic, and adds the derived quantities a planner needs: melt
displacement volume, minimum boss diameter (the wall of plastic that must
surround the insert), and validation of the bore against a wall thickness.

Relation to the rest of the harness: ``geometry/cqcontrib_hole_features.py``
models plain / counterbored / countersunk holes given explicit dimensions;
this module is the *standards* layer that turns a screw designation into
those dimensions for the heat-set-insert case, in the same spirit as the
other tables in ``standards/``.  ``geometry/solidpy_screw_thread.py`` cuts
real threads -- a heat-set insert is the alternative to a cut thread in a
printed part, so the two are complementary, not duplicates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Bolt-clearance bore is 20 percent oversize on the nominal screw diameter.
BOLT_CLEARANCE_FACTOR = 1.2

# Recommended minimum plastic wall around an insert, as a multiple of the
# insert bore diameter (a boss of 2 x bore diameter is the usual guidance).
MIN_BOSS_FACTOR = 2.0


class HeatsertError(ValueError):
    """Raised for an unknown designation or an impossible bore."""


@dataclass(frozen=True)
class InsertDims:
    """Bore dimensions for one heat-set insert size."""

    designation: str
    bore_diameter: float
    bore_depth: float
    bolt_diameter: float

    @property
    def bolt_clearance_diameter(self) -> float:
        return self.bolt_diameter * BOLT_CLEARANCE_FACTOR

    @property
    def min_boss_diameter(self) -> float:
        return self.bore_diameter * MIN_BOSS_FACTOR


_SCHEDULE: Dict[str, InsertDims] = {
    "M3": InsertDims("M3", 4.0, 5.8, 3.0),
    "M4": InsertDims("M4", 5.6, 8.1, 4.0),
    "M5": InsertDims("M5", 6.4, 9.5, 5.0),
    "M6": InsertDims("M6", 8.0, 12.7, 6.0),
}


def designations() -> List[str]:
    """Supported screw designations, ordered by bolt diameter."""
    return sorted(_SCHEDULE, key=lambda k: _SCHEDULE[k].bolt_diameter)


def insert_dims(designation: str) -> InsertDims:
    """Bore schedule entry for a screw designation such as ``"M4"``."""
    key = designation.strip().upper()
    try:
        return _SCHEDULE[key]
    except KeyError:
        raise HeatsertError(
            "unknown heat-set designation %r (known: %s)"
            % (designation, ", ".join(designations())))


def with_extra_size(designation: str, bore_diameter: float, bore_depth: float,
                    bolt_diameter: float) -> InsertDims:
    """Build an ad-hoc schedule entry (the plugin allows user-added sizes).

    Returns a new :class:`InsertDims`; the built-in table is not mutated, so
    the module stays deterministic across calls.
    """
    if min(bore_diameter, bore_depth, bolt_diameter) <= 0.0:
        raise HeatsertError("insert dimensions must be positive")
    if bolt_diameter >= bore_diameter:
        raise HeatsertError("bolt diameter must be smaller than the bore")
    return InsertDims(designation.strip().upper(), float(bore_diameter),
                      float(bore_depth), float(bolt_diameter))


@dataclass(frozen=True)
class BoreSection:
    """One axial slice of the bore, measured downward from the face.

    ``z_start`` / ``z_end`` are depths below the face (both non-negative,
    ``z_end > z_start``).  ``d_start`` / ``d_end`` are the diameters at those
    depths; a cone has ``d_start != d_end``.
    """

    z_start: float
    z_end: float
    d_start: float
    d_end: float

    @property
    def length(self) -> float:
        return self.z_end - self.z_start

    @property
    def volume(self) -> float:
        r1 = self.d_start / 2.0
        r2 = self.d_end / 2.0
        return math.pi * self.length * (r1 * r1 + r1 * r2 + r2 * r2) / 3.0


def _normalise_chamfer(chamfer) -> Optional[Tuple[float, float]]:
    if chamfer is None:
        return None
    if isinstance(chamfer, (int, float)):
        c = float(chamfer)
        vals = (c, c)
    else:
        setback, depth = chamfer
        vals = (float(setback), float(depth))
    if vals[0] <= 0.0 or vals[1] <= 0.0:
        raise HeatsertError("chamfer setback and depth must be positive")
    return vals


def heatsert_bore(designation: str,
                  bolt_clear: float = 0.0,
                  chamfer=None,
                  dims: Optional[InsertDims] = None) -> List[BoreSection]:
    """Axial bore profile for a heat-set insert hole.

    Sections run from the face downward and are contiguous.  ``bolt_clear``
    is measured from the face, matching the plugin, so a value at or below
    the insert depth contributes nothing.  ``chamfer`` is either a scalar
    (45 degree lead-in) or a ``(setback, depth)`` pair.
    """
    d = dims if dims is not None else insert_dims(designation)
    if bolt_clear < 0.0:
        raise HeatsertError("bolt_clear must not be negative")
    ch = _normalise_chamfer(chamfer)
    if ch is not None and ch[1] >= d.bore_depth:
        raise HeatsertError("chamfer depth must be shallower than the bore")

    sections: List[BoreSection] = []
    if ch is not None:
        setback, depth = ch
        sections.append(BoreSection(
            0.0, depth, d.bore_diameter + 2.0 * setback, d.bore_diameter))
        sections.append(BoreSection(
            depth, d.bore_depth, d.bore_diameter, d.bore_diameter))
    else:
        sections.append(BoreSection(
            0.0, d.bore_depth, d.bore_diameter, d.bore_diameter))

    if bolt_clear > d.bore_depth:
        sections.append(BoreSection(
            d.bore_depth, bolt_clear,
            d.bolt_clearance_diameter, d.bolt_clearance_diameter))
    return sections


def bore_depth(sections: List[BoreSection]) -> float:
    """Total depth of a bore profile below the face."""
    return max(s.z_end for s in sections)


def bore_volume(sections: List[BoreSection]) -> float:
    """Material removed by the bore."""
    return sum(s.volume for s in sections)


def chamfer_angle(chamfer) -> float:
    """Lead-in half-angle from the bore axis, in degrees."""
    ch = _normalise_chamfer(chamfer)
    if ch is None:
        raise HeatsertError("no chamfer given")
    setback, depth = ch
    return math.degrees(math.atan2(setback, depth))


def melt_displacement(designation: str,
                      insert_length: Optional[float] = None,
                      dims: Optional[InsertDims] = None) -> float:
    """Plastic volume displaced when the insert is melted in.

    The insert occupies its own outside envelope; the bore already removed a
    cylinder of ``bore_diameter``.  Real inserts are slightly larger than the
    bore (the knurls bite into the wall), so this returns the bore cylinder
    volume as the nominal displacement over ``insert_length`` (defaulting to
    the full bore depth) -- the quantity a planner uses to size the boss.
    """
    d = dims if dims is not None else insert_dims(designation)
    length = d.bore_depth if insert_length is None else float(insert_length)
    if length <= 0.0:
        raise HeatsertError("insert length must be positive")
    r = d.bore_diameter / 2.0
    return math.pi * r * r * length


def fits_in_wall(designation: str,
                 wall_thickness: float,
                 bolt_clear: float = 0.0,
                 chamfer=None,
                 dims: Optional[InsertDims] = None) -> bool:
    """True when the bore (including any bolt clearance) stays inside a wall."""
    sections = heatsert_bore(designation, bolt_clear, chamfer, dims)
    return bore_depth(sections) < float(wall_thickness)


def boss_ok(designation: str,
            boss_diameter: float,
            dims: Optional[InsertDims] = None) -> bool:
    """True when a boss is wide enough to carry the insert without splitting."""
    d = dims if dims is not None else insert_dims(designation)
    return float(boss_diameter) >= d.min_boss_diameter


def select_for_bolt(bolt_diameter: float) -> InsertDims:
    """Smallest scheduled insert that accepts a bolt of the given diameter."""
    for name in designations():
        d = _SCHEDULE[name]
        if d.bolt_diameter >= float(bolt_diameter) - 1e-9:
            return d
    raise HeatsertError(
        "no scheduled insert for a bolt of diameter %g" % bolt_diameter)
