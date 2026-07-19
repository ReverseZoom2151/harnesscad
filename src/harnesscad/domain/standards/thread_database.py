"""Standard screw-thread / fastener dimension database.

Thread lookup tables implemented from published thread standards.  A named
lookup for the four common thread standards, each entry carrying the nominal
major radius, pitch, taper and hex head flat-to-flat distance in consistent
millimetres:

* **ISO metric** coarse & fine (``M3x0.5`` ... ``M64x6``);
* **UTS** Unified coarse (UNC) and fine (UNF), specified in inch/TPI;
* **NPT** National Pipe Thread (tapered, 1/32 taper);

plus derived hex-head geometry (:func:`hex_radius`, :func:`hex_height`) so a
fastener's wrench flats and head height can be computed from its name alone.

This is the *standards data* layer that pairs with the *profile geometry* in
:mod:`geometry.sdfx_thread_profile`: look a fastener up here to get its pitch and
major radius, then generate its tooth section there.  It is distinct from
:mod:`standards.cqplug_heatsert_schedule` (heat-set insert schedule) and the
helical mesh sweep in :mod:`geometry.solidpy_screw_thread`.

All dimensions are normalised to millimetres.  Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional

__all__ = [
    "ThreadParameters",
    "thread_lookup",
    "thread_names",
    "hex_radius",
    "hex_height",
    "MM_PER_INCH",
]

MM_PER_INCH = 25.4
_COS30 = math.cos(math.radians(30.0))


class ThreadParameters(NamedTuple):
    """Resolved dimensions for a named thread, in millimetres."""

    name: str
    radius: float        # nominal major radius (mm)
    pitch: float         # axial thread-to-thread distance (mm)
    taper: float         # taper half-angle (radians); 0 for straight threads
    hex_flat2flat: float  # hex head wrench flat-to-flat distance (mm)
    units: str           # original specification units ("mm" or "inch")


# --- builders in native units, normalised to mm on insertion ---

_DB: Dict[str, ThreadParameters] = {}


def _iso_add(name: str, diameter: float, pitch: float, ftof: float) -> None:
    """ISO metric thread, specified in mm."""
    _DB[name] = ThreadParameters(
        name=name,
        radius=0.5 * diameter,
        pitch=pitch,
        taper=0.0,
        hex_flat2flat=ftof,
        units="mm",
    )


def _uts_add(name: str, diameter_in: float, tpi: float, ftof_in: float) -> None:
    """Unified thread standard, specified in inch / threads-per-inch."""
    _DB[name] = ThreadParameters(
        name=name,
        radius=0.5 * diameter_in * MM_PER_INCH,
        pitch=(1.0 / tpi) * MM_PER_INCH,
        taper=0.0,
        hex_flat2flat=ftof_in * MM_PER_INCH,
        units="inch",
    )


def _npt_add(name: str, diameter_in: float, tpi: float, ftof_mm: float) -> None:
    """National pipe thread (tapered 1/32), diameter/tpi in inch, ftof in mm."""
    _DB[name] = ThreadParameters(
        name=name,
        radius=0.5 * diameter_in * MM_PER_INCH,
        pitch=(1.0 / tpi) * MM_PER_INCH,
        taper=math.atan(1.0 / 32.0),
        hex_flat2flat=ftof_mm,
        units="inch",
    )


def _build() -> None:
    # UTS Coarse (UNC)
    _uts_add("unc_1/4", 1.0 / 4.0, 20, 7.0 / 16.0)
    _uts_add("unc_5/16", 5.0 / 16.0, 18, 1.0 / 2.0)
    _uts_add("unc_3/8", 3.0 / 8.0, 16, 9.0 / 16.0)
    _uts_add("unc_7/16", 7.0 / 16.0, 14, 5.0 / 8.0)
    _uts_add("unc_1/2", 1.0 / 2.0, 13, 3.0 / 4.0)
    _uts_add("unc_9/16", 9.0 / 16.0, 12, 13.0 / 16.0)
    _uts_add("unc_5/8", 5.0 / 8.0, 11, 15.0 / 16.0)
    _uts_add("unc_3/4", 3.0 / 4.0, 10, 9.0 / 8.0)
    _uts_add("unc_7/8", 7.0 / 8.0, 9, 21.0 / 16.0)
    _uts_add("unc_1", 1.0, 8, 3.0 / 2.0)

    # UTS Fine (UNF)
    _uts_add("unf_1/4", 1.0 / 4.0, 28, 7.0 / 16.0)
    _uts_add("unf_5/16", 5.0 / 16.0, 24, 1.0 / 2.0)
    _uts_add("unf_3/8", 3.0 / 8.0, 24, 9.0 / 16.0)
    _uts_add("unf_7/16", 7.0 / 16.0, 20, 5.0 / 8.0)
    _uts_add("unf_1/2", 1.0 / 2.0, 20, 3.0 / 4.0)
    _uts_add("unf_9/16", 9.0 / 16.0, 18, 13.0 / 16.0)
    _uts_add("unf_5/8", 5.0 / 8.0, 18, 15.0 / 16.0)
    _uts_add("unf_3/4", 3.0 / 4.0, 16, 9.0 / 8.0)
    _uts_add("unf_7/8", 7.0 / 8.0, 14, 21.0 / 16.0)
    _uts_add("unf_1", 1.0, 12, 3.0 / 2.0)

    # National Pipe Thread (ftof from ASME B16.11 plug, mm)
    _npt_add("npt_1/8", 0.405, 27, 11.2)
    _npt_add("npt_1/4", 0.540, 18, 15.7)
    _npt_add("npt_3/8", 0.675, 18, 17.5)
    _npt_add("npt_1/2", 0.840, 14, 22.4)
    _npt_add("npt_3/4", 1.050, 14, 26.9)
    _npt_add("npt_1", 1.315, 11.5, 35.1)
    _npt_add("npt_1_1/4", 1.660, 11.5, 44.5)
    _npt_add("npt_1_1/2", 1.900, 11.5, 50.8)
    _npt_add("npt_2", 2.375, 11.5, 63.5)
    _npt_add("npt_2_1/2", 2.875, 8, 76.2)
    _npt_add("npt_3", 3.500, 8, 88.9)
    _npt_add("npt_4", 4.500, 8, 117.3)

    # ISO Coarse
    _iso_add("M1x0.25", 1, 0.25, 1.75)
    _iso_add("M1.2x0.25", 1.2, 0.25, 2.0)
    _iso_add("M1.6x0.35", 1.6, 0.35, 3.2)
    _iso_add("M2x0.4", 2, 0.4, 4)
    _iso_add("M2.5x0.45", 2.5, 0.45, 5)
    _iso_add("M3x0.5", 3, 0.5, 6)
    _iso_add("M4x0.7", 4, 0.7, 7)
    _iso_add("M5x0.8", 5, 0.8, 8)
    _iso_add("M6x1", 6, 1, 10)
    _iso_add("M8x1.25", 8, 1.25, 13)
    _iso_add("M10x1.5", 10, 1.5, 17)
    _iso_add("M12x1.75", 12, 1.75, 19)
    _iso_add("M16x2", 16, 2, 24)
    _iso_add("M20x2.5", 20, 2.5, 30)
    _iso_add("M24x3", 24, 3, 36)
    _iso_add("M30x3.5", 30, 3.5, 46)
    _iso_add("M36x4", 36, 4, 55)
    _iso_add("M42x4.5", 42, 4.5, 65)
    _iso_add("M48x5", 48, 5, 75)
    _iso_add("M56x5.5", 56, 5.5, 85)
    _iso_add("M64x6", 64, 6, 95)

    # ISO Fine
    _iso_add("M1x0.2", 1, 0.2, 1.75)
    _iso_add("M1.2x0.2", 1.2, 0.2, 2.0)
    _iso_add("M1.6x0.2", 1.6, 0.2, 3.2)
    _iso_add("M2x0.25", 2, 0.25, 4)
    _iso_add("M2.5x0.35", 2.5, 0.35, 5)
    _iso_add("M3x0.35", 3, 0.35, 6)
    _iso_add("M4x0.5", 4, 0.5, 7)
    _iso_add("M5x0.5", 5, 0.5, 8)
    _iso_add("M6x0.75", 6, 0.75, 10)
    _iso_add("M8x1", 8, 1, 13)
    _iso_add("M10x1.25", 10, 1.25, 17)
    _iso_add("M12x1.5", 12, 1.5, 19)
    _iso_add("M16x1.5", 16, 1.5, 24)
    _iso_add("M20x2", 20, 2, 30)
    _iso_add("M24x2", 24, 2, 36)
    _iso_add("M30x2", 30, 2, 46)
    _iso_add("M36x3", 36, 3, 55)
    _iso_add("M42x3", 42, 3, 65)
    _iso_add("M48x3", 48, 3, 75)
    _iso_add("M56x4", 56, 4, 85)
    _iso_add("M64x4", 64, 4, 95)


_build()


def thread_lookup(name: str) -> ThreadParameters:
    """Look up a thread's dimensions by name, or raise ``KeyError``."""
    try:
        return _DB[name]
    except KeyError:
        raise KeyError('thread "%s" not found' % name)


def thread_names(prefix: Optional[str] = None) -> List[str]:
    """Sorted list of known thread names, optionally filtered by prefix."""
    names = sorted(_DB.keys())
    if prefix is not None:
        names = [n for n in names if n.startswith(prefix)]
    return names


def hex_radius(t: ThreadParameters) -> float:
    """Hex-head corner radius (circumradius) from the flat-to-flat distance."""
    return t.hex_flat2flat / (2.0 * _COS30)


def hex_height(t: ThreadParameters) -> float:
    """Empirical hex-head height (sdfx heuristic: 5/6 of the corner radius)."""
    return 2.0 * hex_radius(t) * (5.0 / 12.0)
