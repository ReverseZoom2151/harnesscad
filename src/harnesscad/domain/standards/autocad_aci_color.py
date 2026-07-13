"""autocad_aci_color -- AutoCAD Color Index (ACI) names, RGB, nearest match.

``AutoCAD.py`` carries a ``Color`` enum mapping human names to ACI integers
(RED=1, YELLOW=2, GREEN=3, CYAN=4, BLUE=5, MAGENTA=6, WHITE=7, GRAY=8,
ORANGE=30, PURPLE=40, BROWN=41) plus the special ``ByBlock`` / ``ByLayer``
pseudo-colours every AutoCAD document uses. The name<->index dictionary and the
RGB values of the seven standard ACI colours are a fixed, host-independent
lookup, and mapping an arbitrary RGB triple to the nearest standard ACI is a
small deterministic routine.

This module provides:

  * :data:`NAME_TO_ACI` / :func:`name_to_aci` / :func:`aci_to_name`;
  * :data:`ACI_RGB` -- exact RGB for the seven standard colours (1-7), which are
    universally defined, plus the pure-white terminator 255;
  * :func:`aci_to_rgb` and :func:`nearest_aci` (Euclidean match over the entries
    with a defined RGB).

Only entries with an unambiguous, standard RGB are given colour values; the
chromatic mid-range of the 256-entry palette is host/skin dependent and is
deliberately not fabricated. Stdlib-only, deterministic.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

RGB = Tuple[int, int, int]

# Special pseudo-colours.
BYBLOCK = 0
BYLAYER = 256
BYENTITY = 257

# Name -> ACI index (from the AutoCAD.py Color enum).
NAME_TO_ACI: Dict[str, int] = {
    "RED": 1,
    "YELLOW": 2,
    "GREEN": 3,
    "CYAN": 4,
    "BLUE": 5,
    "MAGENTA": 6,
    "WHITE": 7,
    "GRAY": 8,
    "ORANGE": 30,
    "PURPLE": 40,
    "BROWN": 41,
}

_ACI_TO_NAME: Dict[int, str] = {v: k for k, v in NAME_TO_ACI.items()}

# Unambiguous standard RGB values (the seven primary ACI colours + white cap).
ACI_RGB: Dict[int, RGB] = {
    1: (255, 0, 0),
    2: (255, 255, 0),
    3: (0, 255, 0),
    4: (0, 255, 255),
    5: (0, 0, 255),
    6: (255, 0, 255),
    7: (255, 255, 255),
    255: (255, 255, 255),
}


def name_to_aci(name: str) -> int:
    """Return the ACI index for a colour name (case-insensitive)."""
    key = name.upper()
    if key not in NAME_TO_ACI:
        raise ValueError(f"unknown colour name '{name}'")
    return NAME_TO_ACI[key]


def aci_to_name(index: int) -> Optional[str]:
    """Return the colour name for an ACI index, or None if unnamed."""
    return _ACI_TO_NAME.get(index)


def is_special(index: int) -> bool:
    """True for the ByBlock / ByLayer / ByEntity pseudo-colours."""
    return index in (BYBLOCK, BYLAYER, BYENTITY)


def validate_aci(index: int) -> int:
    """Return ``index`` if it is a valid ACI value, else raise ValueError.

    Valid values are 0..256 (0 ByBlock, 1..255 true colours, 256 ByLayer) plus
    the 257 ByEntity extension.
    """
    if not isinstance(index, int):
        raise ValueError("ACI index must be an int")
    if index < 0 or index > 257:
        raise ValueError(f"ACI index {index} out of range 0..257")
    return index


def aci_to_rgb(index: int) -> Optional[RGB]:
    """Return the standard RGB for ``index`` or None if not defined here."""
    return ACI_RGB.get(index)


def nearest_aci(rgb: RGB) -> int:
    """Return the standard ACI index whose RGB is closest to ``rgb``.

    Uses squared Euclidean distance over the entries in :data:`ACI_RGB`. Ties
    resolve to the lower ACI index for determinism. White is offered as index 7
    (not the 255 alias).
    """
    r, g, b = rgb
    best_idx = 7
    best_d = None
    for idx in sorted(ACI_RGB):
        if idx == 255:  # skip the white alias; 7 already covers white
            continue
        cr, cg, cb = ACI_RGB[idx]
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if best_d is None or d < best_d:
            best_d = d
            best_idx = idx
    return best_idx
