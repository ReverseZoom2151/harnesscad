"""Wheel-rim ISO specification code parser and derived geometry.

This module implements a wheel-rim specification-code parser and the derived
rim geometry.

Automotive wheel rims are described by compact ISO-style specification codes such
as::

    "17 4H PCD 114.3 7J ET34 C/B:73"

which decode as:

    * ``17``          -- nominal rim-diameter code (inches),
    * ``4H``          -- bolt count (4 holes),
    * ``PCD 114.3``   -- pitch-circle diameter of the bolt holes (mm),
    * ``7J``          -- rim width (7 inches) with flange profile ``J``,
    * ``ET34``        -- wheel offset / "Einpresstiefe" (mm),
    * ``C/B:73``      -- centre-bore diameter (mm).

Cross-reference tables reproduced here:

    * Table 1 -- nominal rim-diameter code -> specified rim
      diameter ``D`` (mm).  Note the specified diameter is measured at the bead
      seat, so it is larger than the nominal code times 25.4.
    * Table 2 -- flange type -> flange height ``G`` (mm).  The
      "3.00 B" flange family uses ``G = 14.5 mm`` while the "J" family
      (nominal widths 14-21) uses ``G = 17.5 mm``.

Derived-geometry equations implemented here:

    * External circle radius::

          Rcs = D / 2 + G

      For ``D = 436.6 mm`` (code 17) and ``G = 17.5 mm`` this gives
      ``Rcs ~= 235.8 mm``.

    * Bolt-circle outer diameter::

          Do = PCD + 4 * Rs

      where ``Rs`` is the bolt-hole radius.

    * Centre-bore inner diameter::

          Di = CB

    * Transform ratio between the specified and the
      actual external circle radii::

          rho = Rcs / Rca

All functions are deterministic and depend only on the Python standard library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Table 1: nominal rim-diameter code -> specified diameter D
# (mm), measured at the bead seat.
# ---------------------------------------------------------------------------
SPECIFIED_DIAMETER_MM = {
    10: 253.2,
    12: 304.0,
    13: 329.4,
    14: 354.8,
    15: 380.2,
    16: 405.6,
    17: 436.6,
    18: 462.0,
    19: 487.4,
    20: 512.8,
    21: 538.2,
    22: 563.6,
    23: 589.0,
    24: 614.4,
    25: 639.8,
    26: 665.2,
    28: 716.0,
    30: 766.8,
}


# ---------------------------------------------------------------------------
# Table 2: flange type -> flange height G (mm).
#   * "3.00 B" family        -> G = 14.5 mm
#   * "J" family (14-21)     -> G = 17.5 mm  (J, JJ, JX variants)
# ---------------------------------------------------------------------------
FLANGE_HEIGHT_MM = {
    "B": 14.5,
    "J": 17.5,
    "JJ": 17.5,
    "JX": 17.5,
    "JK": 17.5,
}


def specified_diameter(code: int) -> float:
    """Return the specified rim diameter D (mm) for a nominal diameter code.

    Implements the Table 1 cross-reference of the nominal rim-diameter
    paper.  Raises ``ValueError`` for a code not present in the table.
    """
    try:
        return SPECIFIED_DIAMETER_MM[int(code)]
    except (KeyError, TypeError, ValueError):
        raise ValueError("unknown rim-diameter code: {0!r}".format(code))


def flange_height(flange: str) -> float:
    """Return the flange height G (mm) for a flange type string.

    Implements the Table 2 cross-reference of the flange
    paper.  Supports ``'B'`` (14.5 mm) and the ``'J'`` family variants
    (``'J'``, ``'JJ'``, ``'JX'``, ``'JK'`` -> 17.5 mm).  Raises ``ValueError``
    for an unknown flange type.
    """
    if flange is None:
        raise ValueError("flange type is required")
    key = str(flange).strip().upper()
    try:
        return FLANGE_HEIGHT_MM[key]
    except KeyError:
        raise ValueError("unknown flange type: {0!r}".format(flange))


@dataclass
class RimSpec:
    """Parsed fields of a wheel-rim specification code."""

    diameter_code: int
    specified_diameter: Optional[float] = None
    bolt_count: Optional[int] = None
    pcd: Optional[float] = None
    width: Optional[float] = None
    flange: Optional[str] = None
    et: Optional[float] = None
    center_bore: Optional[float] = None


# Regex fragments used by the parser.
_RE_BOLT = re.compile(r"\b(\d+)\s*H\b", re.IGNORECASE)
_RE_PCD = re.compile(r"\bPCD\s*[:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_RE_ET = re.compile(r"\bET\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_RE_CB = re.compile(r"\bC\s*/\s*B\s*[:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
# Width + flange, e.g. "7J", "6 1/2-JJ", "8.5JJ".  The width may be an integer,
# a decimal, or a "n 1/2" fraction.  Flange is one of the known flange keys.
_RE_WIDTH_FLANGE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:[- ]?(\d+)\s*/\s*(\d+))?\s*[- ]?(JJ|JX|JK|J|B)\b",
    re.IGNORECASE,
)
_RE_INT = re.compile(r"\b(\d+)\b")


def _parse_width_flange(code: str):
    """Return ``(width, flange)`` or ``(None, None)`` from a spec code."""
    match = _RE_WIDTH_FLANGE.search(code)
    if match is None:
        return None, None
    whole = float(match.group(1))
    num, den = match.group(2), match.group(3)
    if num is not None and den is not None and float(den) != 0.0:
        whole += float(num) / float(den)
    flange = match.group(4).upper()
    return whole, flange


def parse_rim_spec(code: str) -> RimSpec:
    """Parse a wheel-rim specification code into a :class:`RimSpec`.

    the specification-code grammar.  The first integer token is required and
    becomes ``diameter_code``; all other tokens are optional and default to
    ``None`` when absent.  Parsing is token-order independent (regex based).

    Example::

        >>> spec = parse_rim_spec("17 4H PCD 114.3 7J ET34 C/B:73")
        >>> spec.diameter_code, spec.pcd, spec.flange
        (17, 114.3, 'J')
    """
    if code is None or not str(code).strip():
        raise ValueError("empty rim specification code")
    text = str(code).strip()

    # Width + flange first, so we can strip it before hunting the diameter code
    # (otherwise "7" from "7J" could be mistaken for the diameter code).
    width, flange = _parse_width_flange(text)
    residual = text
    if width is not None:
        residual = _RE_WIDTH_FLANGE.sub(" ", residual, count=1)

    int_match = _RE_INT.search(residual)
    if int_match is None:
        raise ValueError("no diameter code found in: {0!r}".format(code))
    diameter_code = int(int_match.group(1))

    try:
        spec_d = specified_diameter(diameter_code)
    except ValueError:
        spec_d = None

    bolt_match = _RE_BOLT.search(text)
    bolt_count = int(bolt_match.group(1)) if bolt_match else None

    pcd_match = _RE_PCD.search(text)
    pcd = float(pcd_match.group(1)) if pcd_match else None

    et_match = _RE_ET.search(text)
    et = float(et_match.group(1)) if et_match else None

    cb_match = _RE_CB.search(text)
    center_bore = float(cb_match.group(1)) if cb_match else None

    return RimSpec(
        diameter_code=diameter_code,
        specified_diameter=spec_d,
        bolt_count=bolt_count,
        pcd=pcd,
        width=width,
        flange=flange,
        et=et,
        center_bore=center_bore,
    )


# ---------------------------------------------------------------------------
# Derived geometry (Sections 3.3 and 3.5).
# ---------------------------------------------------------------------------
def external_circle_radius(D: float, G: float) -> float:
    """External circle radius Rcs = D/2 + G.

    For ``D = 436.6 mm`` and ``G = 17.5 mm`` this returns ``235.8 mm``.
    """
    return D / 2.0 + G


def bolt_circle_outer_diameter(pcd: float, bolt_radius: float) -> float:
    """Bolt-circle outer diameter Do = PCD + 4*Rs."""
    return pcd + 4.0 * bolt_radius


def center_bore_inner_diameter(cb: float) -> float:
    """Centre-bore inner diameter Di = CB."""
    return cb


def transform_ratio(rcs: float, rca: float) -> float:
    """Transform ratio rho = Rcs / Rca.

    Raises ``ValueError`` when ``rca <= 0``.
    """
    if rca <= 0:
        raise ValueError("actual external circle radius must be positive")
    return rcs / rca


def spec_summary(spec: RimSpec) -> dict:
    """Return a dict of the parsed fields plus derived external circle radius.

    ``external_circle_radius`` is included only when both the specified diameter
    ``D`` and a resolvable flange height ``G`` are available.
    """
    summary = {
        "diameter_code": spec.diameter_code,
        "specified_diameter": spec.specified_diameter,
        "bolt_count": spec.bolt_count,
        "pcd": spec.pcd,
        "width": spec.width,
        "flange": spec.flange,
        "et": spec.et,
        "center_bore": spec.center_bore,
    }
    if spec.specified_diameter is not None and spec.flange is not None:
        try:
            g = flange_height(spec.flange)
        except ValueError:
            g = None
        if g is not None:
            summary["external_circle_radius"] = external_circle_radius(
                spec.specified_diameter, g
            )
    return summary
