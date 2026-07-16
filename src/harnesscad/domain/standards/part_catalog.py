"""Standard-parts catalog (anvilate).

Port of the standard-component dimension tables and resolver pattern from
anvilate (MIT License, Copyright (c) 2026 Clay Good).  Source material:

* ``src/anvilate/standards/data/metric_clearance.yaml`` -- ISO 273 metric
  clearance holes, M2 ... M30, at the close / normal / coarse fits;
* ``src/anvilate/standards/data/cap_screws.yaml`` -- ISO 4762 (DIN 912)
  socket-head cap screw head geometry;
* ``src/anvilate/standards/data/hex_bolts.yaml`` -- ISO 4014 / ISO 4017
  hexagon-head bolt and screw head geometry;
* ``src/anvilate/standards/data/hex_nuts.yaml`` -- ISO 4032 style-1 hex nuts;
* ``src/anvilate/standards/data/washers.yaml`` -- ISO 7089 plain washers
  (normal series, 200 HV);
* ``src/anvilate/standards/data/dowel_pins.yaml`` -- ISO 2338 parallel pins;
* ``src/anvilate/standards/data/bearings.yaml`` -- ISO 15 deep-groove ball
  bearing boundary dimensions (miniature, 68, 60, 62, 63 series);
* ``src/anvilate/standards/data/extrusions.yaml`` -- T-slot aluminum extrusion
  profiles, 20/30/40/45 mm series (Bosch Rexroth / Misumi HFS convention);
* ``src/anvilate/standards/data/nema_frames.yaml`` -- NEMA ICS 16 stepper
  frame mounting geometry for NEMA 17 / 23 / 34;
* ``src/anvilate/standards/resolver.py`` -- the ``StandardsResolver`` lookup
  pattern (:class:`PartCatalog` here mirrors its has-component /
  known-components behaviour across every table at once).

The YAML tables are embedded here as Python dicts (converted offline; no yaml
dependency).  Each dataset's provenance block (name / version / source /
license / retrieved) is preserved verbatim in :data:`PROVENANCE`, keyed by
dataset name.  The dimension values themselves are CC0-1.0 facts; the source
standards are not redistributed.

anvilate's ``metric_thread.yaml`` is deliberately NOT ported: thread pitch and
major-diameter lookups are already covered by
:mod:`harnesscad.domain.standards.thread_database` (the sdfx-derived thread
table), and duplicating them here would create two sources of truth.  Use that
module for thread parameters; use this one for clearance holes, head/nut/washer
geometry, pins, bearings, extrusions, and motor frames.

Lookup conventions.  Every category accepts the bare user-facing designation
("M5", "608", "2020", "NEMA17") and, where the source table keys carry a
standard prefix ("ISO4762-M5", "EXT-2020"), the prefixed form as well.  The
catalog-wide :func:`resolve` mirrors anvilate's resolver: it answers "which
standard part is this reference?" across all categories at once.

All dimensions are millimetres.  Pure stdlib, deterministic.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

__all__ = [
    "PROVENANCE",
    "ClearanceHole",
    "CapScrew",
    "HexBolt",
    "HexNut",
    "Washer",
    "DowelPin",
    "Bearing",
    "ExtrusionProfile",
    "NemaDimension",
    "NemaFrame",
    "clearance_hole",
    "cap_screw",
    "hex_bolt",
    "hex_nut",
    "washer",
    "dowel_pin",
    "bearing",
    "extrusion",
    "nema_frame",
    "clearance_sizes",
    "cap_screw_sizes",
    "hex_bolt_sizes",
    "hex_nut_sizes",
    "washer_sizes",
    "dowel_pin_sizes",
    "bearing_designations",
    "extrusion_designations",
    "nema_frame_designations",
    "resolve",
    "known_designations",
    "CLEARANCE_FITS",
    "DOWEL_TOLERANCE_CLASS",
    "main",
]

# ---------------------------------------------------------------------------
# Provenance: each source yaml's dataset block, preserved verbatim.
# ---------------------------------------------------------------------------

PROVENANCE: Dict[str, Dict[str, str]] = {
    "metric_clearance": {
        "name": "anvilate-metric-clearance-seed",
        "version": "0.1.0",
        "source": "ISO 273 metric clearance holes",
        "license": "CC0-1.0 (dimension values only; source standard not redistributed)",
        "retrieved": "2026-07-08",
    },
    "cap_screws": {
        "name": "anvilate-cap-screws-seed",
        "version": "0.1.0",
        "source": "ISO 4762 (DIN 912) socket-head cap screw head dimensions",
        "license": "CC0-1.0 (dimension values only; source standard not redistributed)",
        "retrieved": "2026-07-08",
    },
    "hex_bolts": {
        "name": "anvilate-hex-bolts-seed",
        "version": "0.1.0",
        "source": "ISO 4014 / ISO 4017 hexagon-head bolt and screw head dimensions",
        "license": "CC0-1.0 (dimension values only; source standard not redistributed)",
        "retrieved": "2026-07-08",
    },
    "hex_nuts": {
        "name": "anvilate-hex-nuts-seed",
        "version": "0.1.0",
        "source": "ISO 4032 style-1 hexagon nut dimensions",
        "license": "CC0-1.0 (dimension values only; source standard not redistributed)",
        "retrieved": "2026-07-08",
    },
    "washers": {
        "name": "anvilate-washers-seed",
        "version": "0.1.0",
        "source": "ISO 7089 plain washer dimensions (normal series, 200 HV)",
        "license": "CC0-1.0 (dimension values only; source standard not redistributed)",
        "retrieved": "2026-07-08",
    },
    "dowel_pins": {
        "name": "anvilate-dowel-pins-seed",
        "version": "0.1.0",
        "source": "ISO 2338 parallel-pin dimensions",
        "license": "CC0-1.0 (dimension values only; source standard not redistributed)",
        "retrieved": "2026-07-08",
    },
    "bearings": {
        "name": "anvilate-bearings-seed",
        "version": "0.1.0",
        "source": "ISO 15 deep-groove ball bearing boundary dimensions",
        "license": "CC0-1.0 (dimension values only; source standard not redistributed)",
        "retrieved": "2026-07-08",
    },
    "extrusions": {
        "name": "anvilate-extrusions-seed",
        "version": "0.1.0",
        "source": "T-slot profile geometry (Bosch Rexroth / Misumi HFS common metric convention)",
        "license": "CC0-1.0 (dimension values only; vendor catalogs not redistributed)",
        "retrieved": "2026-07-08",
    },
    "nema_frames": {
        "name": "anvilate-nema-frames-seed",
        "version": "0.1.0",
        "source": "NEMA ICS 16 stepper frame mounting dimensions",
        "license": "CC0-1.0 (dimension values only; source standard not redistributed)",
        "retrieved": "2026-07-08",
    },
}

# ---------------------------------------------------------------------------
# Typed records.
# ---------------------------------------------------------------------------


class ClearanceHole(NamedTuple):
    """ISO 273 clearance-hole diameter for one metric size at one fit (mm)."""

    size: str        # nominal thread size, e.g. "M5"
    fit: str         # "close", "normal", or "coarse"
    diameter: float  # clearance-hole diameter (mm)


class CapScrew(NamedTuple):
    """ISO 4762 (DIN 912) socket-head cap screw head geometry (mm).

    Length and shank are order-specific and omitted (anvilate's "mount, not
    body" rule); pitch and tap/clearance holes come from the thread and
    clearance tables.
    """

    designation: str     # table key, e.g. "ISO4762-M5"
    size: str            # nominal thread size, e.g. "M5"
    head_diameter: float  # dk (mm)
    head_height: float    # k (mm)
    socket: float         # hex key width across flats s (mm)


class HexBolt(NamedTuple):
    """ISO 4014 / ISO 4017 hexagon-head bolt/screw head geometry (mm)."""

    designation: str          # table key, e.g. "ISO4014-M5"
    size: str                 # nominal thread size, e.g. "M5"
    width_across_flats: float  # s (mm) -- wrench size / pocket to clear
    head_height: float         # k (mm)


class HexNut(NamedTuple):
    """ISO 4032 style-1 hexagon nut dimensions (mm).

    Note: ISO 4032 widths across flats differ from the older DIN 934 at M10
    (16 vs 17) and M12 (18 vs 19); these are the ISO 4032 values.
    """

    designation: str           # table key, e.g. "ISO4032-M5"
    size: str                  # nominal thread size, e.g. "M5"
    width_across_flats: float  # s (mm)
    height: float              # m (mm)


class Washer(NamedTuple):
    """ISO 7089 plain washer, normal series, 200 HV (formerly DIN 125 A) (mm)."""

    designation: str      # table key, e.g. "ISO7089-M5"
    size: str             # nominal thread size served, e.g. "M5"
    inner_diameter: float  # d1 (mm)
    outer_diameter: float  # d2 (mm)
    thickness: float       # h (mm)


class DowelPin(NamedTuple):
    """ISO 2338 parallel (cylindrical) dowel pin (mm).

    The pin seats in a reamed hole sized from ``nominal_diameter`` and the
    tolerance class through the ISO 286 tables; only the pin's own standardized
    dimensions live here, not the mating fit.
    """

    designation: str        # table key, e.g. "ISO2338-6"
    nominal_diameter: float  # d (mm)
    chamfer: float           # c (mm)
    length_min: float        # smallest stocked commercial length (mm)
    length_max: float        # largest stocked commercial length (mm)
    tolerance_class: str     # "m6" standard grade (h8 is the alternative)


class Bearing(NamedTuple):
    """ISO 15 deep-groove ball bearing boundary dimensions (mm).

    Load ratings and internal geometry are manufacturer-specific and omitted.
    """

    designation: str       # e.g. "608", "6204"
    bore: float            # d (mm)
    outer_diameter: float  # D (mm)
    width: float           # B (mm)


class ExtrusionProfile(NamedTuple):
    """T-slot aluminum extrusion profile (mm).

    Vendor variance warning (from the source table): T-slot cross-sections are
    NOT covered by a single ISO/DIN standard.  The slot widths here follow the
    Bosch Rexroth / Misumi HFS convention (20->6, 30->8, 40->10, 45->10 mm);
    other vendors differ.  Only the module width is truly vendor-independent.
    """

    designation: str      # table key, e.g. "EXT-2020"
    name: str             # human name, e.g. "T-slot 20x20 (6 mm slot)"
    profile_width: float  # square module width (mm) -- the modular grid
    slot_width: float     # T-slot mouth a T-nut / bolt head engages (mm)


class NemaDimension(NamedTuple):
    """One cited NEMA mounting dimension: a magnitude with its condition note."""

    magnitude: float  # value in `unit`
    unit: str         # always "mm" in this table
    condition: str    # the source table's citation condition


class NemaFrame(NamedTuple):
    """NEMA ICS 16 stepper frame mounting geometry.

    Only the standardized mounting interface is recorded: the square bolt
    pattern, the pilot locating boss, the nominal faceplate, and the mounting
    screw a bracket drills a clearance hole for.  Body length and shaft are
    manufacturer-specific and omitted.  Coverage: NEMA 17, 23, 34; NEMA
    8/11/14/42 are not recorded, so a reference to them surfaces as a lookup
    failure rather than a guess.
    """

    designation: str              # e.g. "NEMA17"
    name: str                     # e.g. "NEMA 17 stepper frame"
    faceplate_width: NemaDimension
    bolt_spacing: NemaDimension   # square mounting-hole pattern pitch
    pilot_diameter: NemaDimension  # raised pilot boss diameter
    mounting_hole: NemaDimension   # nominal mounting-screw size (M3/M5)


# ---------------------------------------------------------------------------
# Embedded tables (converted offline from the anvilate yaml files).
# ---------------------------------------------------------------------------

CLEARANCE_FITS: Tuple[str, str, str] = ("close", "normal", "coarse")

# size: {fit: clearance diameter in mm}
_CLEARANCE: Dict[str, Dict[str, float]] = {
    "M2": {"close": 2.2, "normal": 2.4, "coarse": 2.6},
    "M2.5": {"close": 2.7, "normal": 2.9, "coarse": 3.1},
    "M3": {"close": 3.2, "normal": 3.4, "coarse": 3.6},
    "M4": {"close": 4.3, "normal": 4.5, "coarse": 4.8},
    "M5": {"close": 5.3, "normal": 5.5, "coarse": 5.8},
    "M6": {"close": 6.4, "normal": 6.6, "coarse": 7.0},
    "M8": {"close": 8.4, "normal": 9.0, "coarse": 10.0},
    "M10": {"close": 10.5, "normal": 11.0, "coarse": 12.0},
    "M12": {"close": 13.0, "normal": 13.5, "coarse": 14.5},
    "M14": {"close": 15.0, "normal": 15.5, "coarse": 16.5},
    "M16": {"close": 17.0, "normal": 17.5, "coarse": 18.5},
    "M20": {"close": 21.0, "normal": 22.0, "coarse": 24.0},
    "M22": {"close": 23.0, "normal": 24.0, "coarse": 26.0},
    "M24": {"close": 25.0, "normal": 26.0, "coarse": 28.0},
    "M27": {"close": 28.0, "normal": 30.0, "coarse": 32.0},
    "M30": {"close": 31.0, "normal": 33.0, "coarse": 35.0},
}

# thread designation: head_diameter (dk), head_height (k), socket (s) in mm.
_CAP_SCREWS: Dict[str, Dict[str, float]] = {
    "ISO4762-M2": {"head_diameter": 3.8, "head_height": 2.0, "socket": 1.5},
    "ISO4762-M2.5": {"head_diameter": 4.5, "head_height": 2.5, "socket": 2.0},
    "ISO4762-M3": {"head_diameter": 5.5, "head_height": 3.0, "socket": 2.5},
    "ISO4762-M4": {"head_diameter": 7.0, "head_height": 4.0, "socket": 3.0},
    "ISO4762-M5": {"head_diameter": 8.5, "head_height": 5.0, "socket": 4.0},
    "ISO4762-M6": {"head_diameter": 10.0, "head_height": 6.0, "socket": 5.0},
    "ISO4762-M8": {"head_diameter": 13.0, "head_height": 8.0, "socket": 6.0},
    "ISO4762-M10": {"head_diameter": 16.0, "head_height": 10.0, "socket": 8.0},
    "ISO4762-M12": {"head_diameter": 18.0, "head_height": 12.0, "socket": 10.0},
    "ISO4762-M16": {"head_diameter": 24.0, "head_height": 16.0, "socket": 14.0},
    "ISO4762-M20": {"head_diameter": 30.0, "head_height": 20.0, "socket": 17.0},
}

# nominal thread size: width_across_flats (s), head_height (k) in mm.
_HEX_BOLTS: Dict[str, Dict[str, float]] = {
    "ISO4014-M3": {"width_across_flats": 5.5, "head_height": 2.0},
    "ISO4014-M4": {"width_across_flats": 7.0, "head_height": 2.8},
    "ISO4014-M5": {"width_across_flats": 8.0, "head_height": 3.5},
    "ISO4014-M6": {"width_across_flats": 10.0, "head_height": 4.0},
    "ISO4014-M8": {"width_across_flats": 13.0, "head_height": 5.3},
    "ISO4014-M10": {"width_across_flats": 16.0, "head_height": 6.4},
    "ISO4014-M12": {"width_across_flats": 18.0, "head_height": 7.5},
    "ISO4014-M16": {"width_across_flats": 24.0, "head_height": 10.0},
    "ISO4014-M20": {"width_across_flats": 30.0, "head_height": 12.5},
}

# nominal thread size: width_across_flats (s), height (m) in mm.
_HEX_NUTS: Dict[str, Dict[str, float]] = {
    "ISO4032-M3": {"width_across_flats": 5.5, "height": 2.4},
    "ISO4032-M4": {"width_across_flats": 7.0, "height": 3.2},
    "ISO4032-M5": {"width_across_flats": 8.0, "height": 4.7},
    "ISO4032-M6": {"width_across_flats": 10.0, "height": 5.2},
    "ISO4032-M8": {"width_across_flats": 13.0, "height": 6.8},
    "ISO4032-M10": {"width_across_flats": 16.0, "height": 8.4},
    "ISO4032-M12": {"width_across_flats": 18.0, "height": 10.8},
    "ISO4032-M16": {"width_across_flats": 24.0, "height": 14.8},
    "ISO4032-M20": {"width_across_flats": 30.0, "height": 18.0},
}

# nominal thread size: inner_diameter (d1), outer_diameter (d2), thickness (h) mm.
_WASHERS: Dict[str, Dict[str, float]] = {
    "ISO7089-M3": {"inner_diameter": 3.2, "outer_diameter": 7.0, "thickness": 0.5},
    "ISO7089-M4": {"inner_diameter": 4.3, "outer_diameter": 9.0, "thickness": 0.8},
    "ISO7089-M5": {"inner_diameter": 5.3, "outer_diameter": 10.0, "thickness": 1.0},
    "ISO7089-M6": {"inner_diameter": 6.4, "outer_diameter": 12.0, "thickness": 1.6},
    "ISO7089-M8": {"inner_diameter": 8.4, "outer_diameter": 16.0, "thickness": 1.6},
    "ISO7089-M10": {"inner_diameter": 10.5, "outer_diameter": 20.0, "thickness": 2.0},
    "ISO7089-M12": {"inner_diameter": 13.0, "outer_diameter": 24.0, "thickness": 2.5},
    "ISO7089-M14": {"inner_diameter": 15.0, "outer_diameter": 28.0, "thickness": 2.5},
    "ISO7089-M16": {"inner_diameter": 17.0, "outer_diameter": 30.0, "thickness": 3.0},
    "ISO7089-M20": {"inner_diameter": 21.0, "outer_diameter": 37.0, "thickness": 3.0},
}

# The standard tolerance class m6 applies to d unless a design selects h8.
DOWEL_TOLERANCE_CLASS = "m6"

# nominal_diameter (d), chamfer (c), length_min / length_max (l) in mm.
_DOWEL_PINS: Dict[str, Dict[str, float]] = {
    "ISO2338-1": {"nominal_diameter": 1.0, "chamfer": 0.20, "length_min": 4.0, "length_max": 10.0},
    "ISO2338-1.5": {"nominal_diameter": 1.5, "chamfer": 0.30, "length_min": 4.0, "length_max": 16.0},
    "ISO2338-2": {"nominal_diameter": 2.0, "chamfer": 0.35, "length_min": 6.0, "length_max": 20.0},
    "ISO2338-2.5": {"nominal_diameter": 2.5, "chamfer": 0.40, "length_min": 6.0, "length_max": 24.0},
    "ISO2338-3": {"nominal_diameter": 3.0, "chamfer": 0.50, "length_min": 8.0, "length_max": 30.0},
    "ISO2338-4": {"nominal_diameter": 4.0, "chamfer": 0.63, "length_min": 8.0, "length_max": 40.0},
    "ISO2338-5": {"nominal_diameter": 5.0, "chamfer": 0.80, "length_min": 10.0, "length_max": 50.0},
    "ISO2338-6": {"nominal_diameter": 6.0, "chamfer": 1.2, "length_min": 12.0, "length_max": 60.0},
    "ISO2338-8": {"nominal_diameter": 8.0, "chamfer": 1.6, "length_min": 14.0, "length_max": 80.0},
    "ISO2338-10": {"nominal_diameter": 10.0, "chamfer": 2.0, "length_min": 18.0, "length_max": 95.0},
    "ISO2338-12": {"nominal_diameter": 12.0, "chamfer": 2.5, "length_min": 22.0, "length_max": 140.0},
    "ISO2338-16": {"nominal_diameter": 16.0, "chamfer": 3.0, "length_min": 26.0, "length_max": 180.0},
    "ISO2338-20": {"nominal_diameter": 20.0, "chamfer": 3.5, "length_min": 35.0, "length_max": 200.0},
}

# designation: {bore (d), outer_diameter (D), width (B)} in mm.
_BEARINGS: Dict[str, Dict[str, float]] = {
    # Miniature / instrument sizes
    "623": {"bore": 3.0, "outer_diameter": 10.0, "width": 4.0},
    "625": {"bore": 5.0, "outer_diameter": 16.0, "width": 5.0},
    "626": {"bore": 6.0, "outer_diameter": 19.0, "width": 6.0},
    "608": {"bore": 8.0, "outer_diameter": 22.0, "width": 7.0},
    # 68-series (thin section)
    "6800": {"bore": 10.0, "outer_diameter": 19.0, "width": 5.0},
    "6801": {"bore": 12.0, "outer_diameter": 21.0, "width": 5.0},
    "6802": {"bore": 15.0, "outer_diameter": 24.0, "width": 5.0},
    "6803": {"bore": 17.0, "outer_diameter": 26.0, "width": 5.0},
    "6804": {"bore": 20.0, "outer_diameter": 32.0, "width": 7.0},
    "6805": {"bore": 25.0, "outer_diameter": 37.0, "width": 7.0},
    "6806": {"bore": 30.0, "outer_diameter": 42.0, "width": 7.0},
    "6807": {"bore": 35.0, "outer_diameter": 47.0, "width": 7.0},
    "6808": {"bore": 40.0, "outer_diameter": 52.0, "width": 7.0},
    "6809": {"bore": 45.0, "outer_diameter": 58.0, "width": 7.0},
    "6810": {"bore": 50.0, "outer_diameter": 65.0, "width": 7.0},
    # 60-series (extra light)
    "6000": {"bore": 10.0, "outer_diameter": 26.0, "width": 8.0},
    "6001": {"bore": 12.0, "outer_diameter": 28.0, "width": 8.0},
    "6002": {"bore": 15.0, "outer_diameter": 32.0, "width": 9.0},
    "6003": {"bore": 17.0, "outer_diameter": 35.0, "width": 10.0},
    "6004": {"bore": 20.0, "outer_diameter": 42.0, "width": 12.0},
    "6005": {"bore": 25.0, "outer_diameter": 47.0, "width": 12.0},
    "6006": {"bore": 30.0, "outer_diameter": 55.0, "width": 13.0},
    "6007": {"bore": 35.0, "outer_diameter": 62.0, "width": 14.0},
    "6008": {"bore": 40.0, "outer_diameter": 68.0, "width": 15.0},
    "6009": {"bore": 45.0, "outer_diameter": 75.0, "width": 16.0},
    "6010": {"bore": 50.0, "outer_diameter": 80.0, "width": 16.0},
    # 62-series (light)
    "6200": {"bore": 10.0, "outer_diameter": 30.0, "width": 9.0},
    "6201": {"bore": 12.0, "outer_diameter": 32.0, "width": 10.0},
    "6202": {"bore": 15.0, "outer_diameter": 35.0, "width": 11.0},
    "6203": {"bore": 17.0, "outer_diameter": 40.0, "width": 12.0},
    "6204": {"bore": 20.0, "outer_diameter": 47.0, "width": 14.0},
    "6205": {"bore": 25.0, "outer_diameter": 52.0, "width": 15.0},
    "6206": {"bore": 30.0, "outer_diameter": 62.0, "width": 16.0},
    "6207": {"bore": 35.0, "outer_diameter": 72.0, "width": 17.0},
    "6208": {"bore": 40.0, "outer_diameter": 80.0, "width": 18.0},
    "6209": {"bore": 45.0, "outer_diameter": 85.0, "width": 19.0},
    "6210": {"bore": 50.0, "outer_diameter": 90.0, "width": 20.0},
    # 63-series (medium)
    "6300": {"bore": 10.0, "outer_diameter": 35.0, "width": 11.0},
    "6301": {"bore": 12.0, "outer_diameter": 37.0, "width": 12.0},
    "6302": {"bore": 15.0, "outer_diameter": 42.0, "width": 13.0},
    "6303": {"bore": 17.0, "outer_diameter": 47.0, "width": 14.0},
    "6304": {"bore": 20.0, "outer_diameter": 52.0, "width": 15.0},
    "6305": {"bore": 25.0, "outer_diameter": 62.0, "width": 17.0},
    "6306": {"bore": 30.0, "outer_diameter": 72.0, "width": 19.0},
    "6307": {"bore": 35.0, "outer_diameter": 80.0, "width": 21.0},
    "6308": {"bore": 40.0, "outer_diameter": 90.0, "width": 23.0},
    "6309": {"bore": 45.0, "outer_diameter": 100.0, "width": 25.0},
    "6310": {"bore": 50.0, "outer_diameter": 110.0, "width": 27.0},
}

# designation: {name, profile_width (square module), slot_width} in mm.
_EXTRUSIONS: Dict[str, Dict[str, object]] = {
    "EXT-2020": {"name": "T-slot 20x20 (6 mm slot)", "profile_width": 20.0, "slot_width": 6.0},
    "EXT-3030": {"name": "T-slot 30x30 (8 mm slot)", "profile_width": 30.0, "slot_width": 8.0},
    "EXT-4040": {"name": "T-slot 40x40 (10 mm slot)", "profile_width": 40.0, "slot_width": 10.0},
    "EXT-4545": {"name": "T-slot 45x45 (10 mm slot)", "profile_width": 45.0, "slot_width": 10.0},
}

# frame: per-dimension {magnitude (mm), condition} pairs, as in the source table.
_NEMA_FRAMES: Dict[str, Dict[str, object]] = {
    "NEMA17": {
        "name": "NEMA 17 stepper frame",
        "faceplate_width": (42.3, "nominal faceplate, varies by manufacturer"),
        "bolt_spacing": (31.0, "square mounting-hole pattern pitch"),
        "pilot_diameter": (22.0, "raised pilot boss diameter"),
        "mounting_hole": (3.0, "M3 mounting screw"),
    },
    "NEMA23": {
        "name": "NEMA 23 stepper frame",
        "faceplate_width": (56.4, "nominal faceplate, varies by manufacturer"),
        "bolt_spacing": (47.14, "square mounting-hole pattern pitch (1.856 in)"),
        "pilot_diameter": (38.1, "raised pilot boss diameter (1.500 in)"),
        "mounting_hole": (5.0, "M5 mounting screw"),
    },
    "NEMA34": {
        "name": "NEMA 34 stepper frame",
        "faceplate_width": (86.0, "nominal faceplate, varies by manufacturer"),
        "bolt_spacing": (69.6, "square mounting-hole pattern pitch (2.74 in)"),
        "pilot_diameter": (73.025, "raised pilot boss diameter (2.875 in)"),
        "mounting_hole": (5.0, "M5 mounting screw"),
    },
}


# ---------------------------------------------------------------------------
# Designation normalisation helpers.
# ---------------------------------------------------------------------------


def _norm(designation: str) -> str:
    return designation.strip()


def _prefixed(designation: str, prefix: str) -> str:
    """Return the table key for a bare or already-prefixed designation.

    ``_prefixed("M5", "ISO4762")`` -> ``"ISO4762-M5"``;
    ``_prefixed("ISO4762-M5", "ISO4762")`` -> ``"ISO4762-M5"``.
    """
    d = _norm(designation)
    upper = d.upper()
    if upper.startswith(prefix.upper() + "-"):
        return prefix + "-" + d[len(prefix) + 1:].upper()
    return prefix + "-" + upper


def _metric_key(designation: str) -> str:
    """Normalise a bare metric size: "m5" -> "M5"."""
    return _norm(designation).upper()


# ---------------------------------------------------------------------------
# Per-category typed accessors.
# ---------------------------------------------------------------------------


def clearance_hole(size: str, fit: str = "normal") -> ClearanceHole:
    """ISO 273 clearance hole for a metric size ("M5") at a fit.

    ``fit`` is one of :data:`CLEARANCE_FITS` (close / normal / coarse); "loose"
    is accepted as an alias for the coarse fit.  Raises ``KeyError`` on an
    unknown size or fit.
    """
    key = _metric_key(size)
    fit_key = fit.strip().lower()
    if fit_key == "loose":
        fit_key = "coarse"
    try:
        row = _CLEARANCE[key]
    except KeyError:
        raise KeyError('clearance size "%s" not found' % size)
    try:
        dia = row[fit_key]
    except KeyError:
        raise KeyError('clearance fit "%s" not one of %s' % (fit, list(CLEARANCE_FITS)))
    return ClearanceHole(size=key, fit=fit_key, diameter=dia)


def cap_screw(designation: str) -> CapScrew:
    """ISO 4762 socket-head cap screw by "M5" or "ISO4762-M5"."""
    key = _prefixed(designation, "ISO4762")
    try:
        row = _CAP_SCREWS[key]
    except KeyError:
        raise KeyError('cap screw "%s" not found' % designation)
    return CapScrew(
        designation=key,
        size=key.split("-", 1)[1],
        head_diameter=row["head_diameter"],
        head_height=row["head_height"],
        socket=row["socket"],
    )


def hex_bolt(designation: str) -> HexBolt:
    """ISO 4014 / ISO 4017 hex bolt head by "M5" or "ISO4014-M5"."""
    key = _prefixed(designation, "ISO4014")
    try:
        row = _HEX_BOLTS[key]
    except KeyError:
        raise KeyError('hex bolt "%s" not found' % designation)
    return HexBolt(
        designation=key,
        size=key.split("-", 1)[1],
        width_across_flats=row["width_across_flats"],
        head_height=row["head_height"],
    )


def hex_nut(designation: str) -> HexNut:
    """ISO 4032 style-1 hex nut by "M5" or "ISO4032-M5"."""
    key = _prefixed(designation, "ISO4032")
    try:
        row = _HEX_NUTS[key]
    except KeyError:
        raise KeyError('hex nut "%s" not found' % designation)
    return HexNut(
        designation=key,
        size=key.split("-", 1)[1],
        width_across_flats=row["width_across_flats"],
        height=row["height"],
    )


def washer(designation: str) -> Washer:
    """ISO 7089 plain washer by the thread it serves: "M5" or "ISO7089-M5"."""
    key = _prefixed(designation, "ISO7089")
    try:
        row = _WASHERS[key]
    except KeyError:
        raise KeyError('washer "%s" not found' % designation)
    return Washer(
        designation=key,
        size=key.split("-", 1)[1],
        inner_diameter=row["inner_diameter"],
        outer_diameter=row["outer_diameter"],
        thickness=row["thickness"],
    )


def dowel_pin(designation: str) -> DowelPin:
    """ISO 2338 parallel pin by nominal diameter: "6", 6, or "ISO2338-6"."""
    d = _norm(str(designation))
    # Accept "6", "6.0", or the prefixed key.
    if d.upper().startswith("ISO2338-"):
        d = d.split("-", 1)[1]
    # Normalise numeric spellings: "6.0" -> "6", "1.5" stays "1.5".
    try:
        num = float(d)
        d = ("%g" % num)
    except ValueError:
        pass
    key = "ISO2338-" + d
    try:
        row = _DOWEL_PINS[key]
    except KeyError:
        raise KeyError('dowel pin "%s" not found' % designation)
    return DowelPin(
        designation=key,
        nominal_diameter=row["nominal_diameter"],
        chamfer=row["chamfer"],
        length_min=row["length_min"],
        length_max=row["length_max"],
        tolerance_class=DOWEL_TOLERANCE_CLASS,
    )


def bearing(designation: str) -> Bearing:
    """ISO 15 deep-groove ball bearing by designation ("608", "6204")."""
    key = _norm(str(designation))
    try:
        row = _BEARINGS[key]
    except KeyError:
        raise KeyError('bearing "%s" not found' % designation)
    return Bearing(
        designation=key,
        bore=row["bore"],
        outer_diameter=row["outer_diameter"],
        width=row["width"],
    )


def extrusion(designation: str) -> ExtrusionProfile:
    """T-slot extrusion profile by series ("2020") or table key ("EXT-2020")."""
    d = _norm(designation).upper()
    key = d if d.startswith("EXT-") else "EXT-" + d
    try:
        row = _EXTRUSIONS[key]
    except KeyError:
        raise KeyError('extrusion profile "%s" not found' % designation)
    return ExtrusionProfile(
        designation=key,
        name=str(row["name"]),
        profile_width=float(row["profile_width"]),  # type: ignore[arg-type]
        slot_width=float(row["slot_width"]),  # type: ignore[arg-type]
    )


def nema_frame(designation: str) -> NemaFrame:
    """NEMA stepper frame by "NEMA17", "nema 17", or bare "17"."""
    d = _norm(str(designation)).upper().replace(" ", "")
    key = d if d.startswith("NEMA") else "NEMA" + d
    try:
        row = _NEMA_FRAMES[key]
    except KeyError:
        raise KeyError('NEMA frame "%s" not found (recorded: %s)'
                       % (designation, ", ".join(sorted(_NEMA_FRAMES))))

    def dim(field: str) -> NemaDimension:
        magnitude, condition = row[field]  # type: ignore[misc]
        return NemaDimension(magnitude=float(magnitude), unit="mm", condition=str(condition))

    return NemaFrame(
        designation=key,
        name=str(row["name"]),
        faceplate_width=dim("faceplate_width"),
        bolt_spacing=dim("bolt_spacing"),
        pilot_diameter=dim("pilot_diameter"),
        mounting_hole=dim("mounting_hole"),
    )


# ---------------------------------------------------------------------------
# Per-category listing functions (sorted, deterministic).
# ---------------------------------------------------------------------------


def _metric_sort_key(size: str) -> float:
    return float(size.lstrip("Mm"))


def clearance_sizes() -> List[str]:
    """Metric sizes with clearance-hole rows, in ascending size order."""
    return sorted(_CLEARANCE, key=_metric_sort_key)


def cap_screw_sizes() -> List[str]:
    """Cap-screw table keys (ISO4762-*), in ascending thread size order."""
    return sorted(_CAP_SCREWS, key=lambda k: _metric_sort_key(k.split("-", 1)[1]))


def hex_bolt_sizes() -> List[str]:
    """Hex-bolt table keys (ISO4014-*), in ascending thread size order."""
    return sorted(_HEX_BOLTS, key=lambda k: _metric_sort_key(k.split("-", 1)[1]))


def hex_nut_sizes() -> List[str]:
    """Hex-nut table keys (ISO4032-*), in ascending thread size order."""
    return sorted(_HEX_NUTS, key=lambda k: _metric_sort_key(k.split("-", 1)[1]))


def washer_sizes() -> List[str]:
    """Washer table keys (ISO7089-*), in ascending thread size order."""
    return sorted(_WASHERS, key=lambda k: _metric_sort_key(k.split("-", 1)[1]))


def dowel_pin_sizes() -> List[str]:
    """Dowel-pin table keys (ISO2338-*), in ascending diameter order."""
    return sorted(_DOWEL_PINS, key=lambda k: float(k.split("-", 1)[1]))


def bearing_designations() -> List[str]:
    """Bearing designations, sorted lexicographically."""
    return sorted(_BEARINGS)


def extrusion_designations() -> List[str]:
    """Extrusion table keys (EXT-*), sorted."""
    return sorted(_EXTRUSIONS)


def nema_frame_designations() -> List[str]:
    """Recorded NEMA frame designations, sorted."""
    return sorted(_NEMA_FRAMES)


# ---------------------------------------------------------------------------
# Catalog-wide resolution (anvilate resolver.py pattern).
# ---------------------------------------------------------------------------

# (category name, accessor) in a fixed resolution order.  Clearance sizes are
# not components (they are hole dimensions for a size), so like anvilate's
# resolver the component walk covers the part tables only.
_CATEGORIES: Sequence[Tuple[str, object]] = (
    ("nema_frame", nema_frame),
    ("bearing", bearing),
    ("dowel_pin", dowel_pin),
    ("cap_screw", cap_screw),
    ("washer", washer),
    ("hex_nut", hex_nut),
    ("hex_bolt", hex_bolt),
    ("extrusion", extrusion),
)


def resolve(designation: str) -> Tuple[str, NamedTuple]:
    """Resolve a designation across every part table at once.

    Mirrors anvilate ``StandardsResolver.has_component``: the tables are tried
    in a fixed order (NEMA frames, bearings, dowel pins, cap screws, washers,
    hex nuts, hex bolts, extrusions) and the first hit wins.  Prefixed keys
    ("ISO4032-M5") are unambiguous; a bare metric size ("M5") resolves to the
    first fastener table that carries it (cap screw before washer/nut/bolt),
    so pass the prefixed form when the category matters.  Returns
    ``(category, record)``; raises ``KeyError`` when no table knows the
    designation.
    """
    for category, accessor in _CATEGORIES:
        try:
            return category, accessor(designation)  # type: ignore[operator]
        except KeyError:
            continue
    raise KeyError('designation "%s" not found in any part table' % designation)


def known_designations() -> List[str]:
    """Every resolvable table key, across all part tables, sorted.

    Mirrors anvilate ``StandardsResolver.known_components``.
    """
    return sorted(
        set(_CAP_SCREWS)
        | set(_HEX_BOLTS)
        | set(_HEX_NUTS)
        | set(_WASHERS)
        | set(_DOWEL_PINS)
        | set(_BEARINGS)
        | set(_EXTRUSIONS)
        | set(_NEMA_FRAMES)
    )


# ---------------------------------------------------------------------------
# Self-check.
# ---------------------------------------------------------------------------


def _selfcheck() -> None:
    # Clearance holes (ISO 273).
    assert clearance_hole("M5").diameter == 5.5
    assert clearance_hole("M5", "close").diameter == 5.3
    assert clearance_hole("M5", "loose").diameter == 5.8  # alias for coarse
    assert clearance_hole("m3", "coarse").diameter == 3.6
    assert clearance_hole("M30", "coarse").diameter == 35.0
    assert len(_CLEARANCE) == 16

    # Cap screws (ISO 4762).
    s = cap_screw("M5")
    assert s.designation == "ISO4762-M5"
    assert s.head_diameter == 8.5 and s.head_height == 5.0 and s.socket == 4.0
    assert cap_screw("ISO4762-M2.5").socket == 2.0
    assert len(_CAP_SCREWS) == 11

    # Hex bolts (ISO 4014) and nuts (ISO 4032).
    assert hex_bolt("M10").width_across_flats == 16.0
    assert hex_bolt("M8").head_height == 5.3
    assert hex_nut("M10").width_across_flats == 16.0  # ISO 4032, not DIN 934's 17
    assert hex_nut("M5").height == 4.7
    assert len(_HEX_BOLTS) == 9 and len(_HEX_NUTS) == 9

    # Washers (ISO 7089).
    w = washer("M5")
    assert w.inner_diameter == 5.3 and w.outer_diameter == 10.0 and w.thickness == 1.0
    assert len(_WASHERS) == 10

    # Dowel pins (ISO 2338).
    p = dowel_pin("6")
    assert p.nominal_diameter == 6.0 and p.chamfer == 1.2
    assert p.length_min == 12.0 and p.length_max == 60.0
    assert p.tolerance_class == "m6"
    assert dowel_pin(1.5).designation == "ISO2338-1.5"
    assert len(_DOWEL_PINS) == 13

    # Bearings (ISO 15).
    b = bearing("608")
    assert b.bore == 8.0 and b.outer_diameter == 22.0 and b.width == 7.0
    assert bearing("6204").outer_diameter == 47.0
    assert len(_BEARINGS) == 48

    # Extrusions (T-slot).
    e = extrusion("2020")
    assert e.profile_width == 20.0 and e.slot_width == 6.0
    assert extrusion("EXT-4545").slot_width == 10.0
    assert len(_EXTRUSIONS) == 4

    # NEMA frames (NEMA ICS 16).
    f = nema_frame("NEMA17")
    assert f.bolt_spacing.magnitude == 31.0 and f.bolt_spacing.unit == "mm"
    assert f.pilot_diameter.magnitude == 22.0
    assert f.mounting_hole.magnitude == 3.0
    assert nema_frame("23").bolt_spacing.magnitude == 47.14
    assert nema_frame("nema 34").pilot_diameter.magnitude == 73.025
    assert len(_NEMA_FRAMES) == 3

    # Catalog-wide resolution.
    cat, rec = resolve("608")
    assert cat == "bearing" and rec.bore == 8.0  # type: ignore[union-attr]
    cat, rec = resolve("NEMA17")
    assert cat == "nema_frame"
    cat, rec = resolve("ISO4032-M5")
    assert cat == "hex_nut"
    cat, rec = resolve("EXT-3030")
    assert cat == "extrusion"
    try:
        resolve("NEMA42")
    except KeyError:
        pass
    else:
        raise AssertionError("NEMA42 must surface as a coverage gap")

    known = known_designations()
    assert "ISO4762-M5" in known and "608" in known and "NEMA17" in known
    assert known == sorted(known)
    assert len(known) == 11 + 9 + 9 + 10 + 13 + 48 + 4 + 3

    # Provenance preserved per dataset.
    assert len(PROVENANCE) == 9
    for meta in PROVENANCE.values():
        for field in ("name", "version", "source", "license", "retrieved"):
            assert meta[field]
    assert PROVENANCE["bearings"]["source"].startswith("ISO 15")
    assert PROVENANCE["nema_frames"]["source"].startswith("NEMA ICS 16")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="part_catalog",
        description="Standard-parts catalog (anvilate port): clearance holes, "
                    "fasteners, pins, bearings, extrusions, NEMA frames.",
    )
    parser.add_argument(
        "--selfcheck", action="store_true",
        help="assert known table values and exit 0",
    )
    parser.add_argument(
        "designation", nargs="?",
        help="optional designation to resolve (e.g. M5, 608, EXT-2020, NEMA17)",
    )
    args = parser.parse_args(argv)

    if args.selfcheck:
        _selfcheck()
        print("part_catalog selfcheck: OK "
              "(%d designations across %d datasets)"
              % (len(known_designations()), len(PROVENANCE)))
        return 0

    if args.designation:
        try:
            category, record = resolve(args.designation)
        except KeyError as exc:
            print(str(exc))
            return 1
        print("%s: %s" % (category, record))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
