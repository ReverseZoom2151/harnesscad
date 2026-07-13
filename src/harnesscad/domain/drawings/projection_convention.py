"""t2cadtd_projection_convention — first/third-angle layout + view sufficiency.

Text2CAD (Yavartanoo et al., "Text to 3D CAD Generation via Technical Drawings")
notes that technical drawings "follow standards like ANSI Y14.5 and ISO 8015 ...
using orthogonal projections (ISO 128, ASME Y14.3)". Those standards define two
mutually exclusive ways to *arrange* the front/top/side views on a sheet:

  * **Third-angle** (ASME Y14.3, common in the US): the top view is placed ABOVE
    the front view and the right-side view to the RIGHT of the front view — each
    view sits on the side of the object it looks from.
  * **First-angle** (ISO 128, common in Europe): the top view is placed BELOW the
    front view and the right-side view to the LEFT — the object is "pushed through"
    onto the far plane.

The two are visually identical per view but differ in *placement*; a sheet that
mixes them is invalid, and reading a sheet with the wrong assumed convention
mislabels views. None of the existing modules model this:
:mod:`drawings.cad2program_canvas_layout` hard-codes one third-angle arrangement,
:mod:`drawings.creft_view_consistency` only checks shared extents. This module
adds:

  * :func:`view_placements` — the (column, row) grid cell of each view for a
    given convention.
  * :func:`infer_convention` — given labelled view placements on a sheet, decide
    whether the layout is first-angle, third-angle, or inconsistent.
  * :func:`convert_layout` — reflect a placement dict between the two conventions.
  * :func:`dimensions_covered` / :func:`views_sufficient` — the paper's core
    requirement that the orthographic views "provide sufficient information to
    reconstruct 3D CAD models": which of the width/height/depth extents the given
    view set pins down, and whether all three are covered.

Pure stdlib, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

FRONT = "front"
TOP = "top"
SIDE = "side"

THIRD_ANGLE = "third_angle"
FIRST_ANGLE = "first_angle"

# Grid cells (column, row) with front at the origin. +col = rightward,
# +row = upward.  The front view anchors both conventions at (0, 0).
_THIRD_ANGLE_PLACEMENT: Dict[str, Tuple[int, int]] = {
    FRONT: (0, 0),
    TOP: (0, 1),    # top ABOVE front
    SIDE: (1, 0),   # right-side view to the RIGHT
}
_FIRST_ANGLE_PLACEMENT: Dict[str, Tuple[int, int]] = {
    FRONT: (0, 0),
    TOP: (0, -1),   # top BELOW front
    SIDE: (-1, 0),  # right-side view to the LEFT
}

# Which of the three overall extents each view measures (front spans width X and
# height Z, top spans width X and depth Y, side spans depth Y and height Z).
_VIEW_DIMENSIONS: Dict[str, Tuple[str, str]] = {
    FRONT: ("width", "height"),
    TOP: ("width", "depth"),
    SIDE: ("depth", "height"),
}

ALL_DIMENSIONS: Tuple[str, str, str] = ("width", "height", "depth")


def view_placements(convention: str = THIRD_ANGLE) -> Dict[str, Tuple[int, int]]:
    """Grid ``(column, row)`` of each of the three views for a convention."""
    if convention == THIRD_ANGLE:
        return dict(_THIRD_ANGLE_PLACEMENT)
    if convention == FIRST_ANGLE:
        return dict(_FIRST_ANGLE_PLACEMENT)
    raise ValueError("unknown convention %r" % (convention,))


def convert_layout(placement: Dict[str, Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
    """Reflect a placement dict to the other convention (about the front view).

    First- and third-angle placements are point reflections of each other through
    the anchoring front view, so negating each non-front cell converts between
    them. The front view (at the origin) is preserved.
    """
    out: Dict[str, Tuple[int, int]] = {}
    for name, (col, row) in placement.items():
        if name == FRONT:
            out[name] = (col, row)
        else:
            out[name] = (-col, -row)
    return out


def infer_convention(placement: Dict[str, Tuple[int, int]]) -> str:
    """Classify a labelled sheet layout as first/third angle, or inconsistent.

    ``placement`` maps view names to ``(column, row)`` cells. Only the *signs* of
    the top and side offsets relative to the front matter, so arbitrary spacing
    is tolerated. Returns :data:`THIRD_ANGLE`, :data:`FIRST_ANGLE`, or
    ``"inconsistent"`` when the two views disagree or a required view is missing.
    """
    if FRONT not in placement or TOP not in placement or SIDE not in placement:
        return "inconsistent"
    fc = placement[FRONT]
    top_row = _sign(placement[TOP][1] - fc[1])
    side_col = _sign(placement[SIDE][0] - fc[0])
    # Third angle: top above (row > 0), side right (col > 0).
    if top_row > 0 and side_col > 0:
        return THIRD_ANGLE
    # First angle: top below (row < 0), side left (col < 0).
    if top_row < 0 and side_col < 0:
        return FIRST_ANGLE
    return "inconsistent"


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


@dataclass(frozen=True)
class SufficiencyResult:
    sufficient: bool
    covered: Tuple[str, ...]
    missing: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, object]:
        return {"sufficient": self.sufficient,
                "covered": list(self.covered),
                "missing": list(self.missing)}


def dimensions_covered(view_names: Sequence[str]) -> Set[str]:
    """The overall extents (width/height/depth) pinned down by the given views."""
    covered: Set[str] = set()
    for name in view_names:
        if name in _VIEW_DIMENSIONS:
            covered.update(_VIEW_DIMENSIONS[name])
    return covered


def views_sufficient(view_names: Sequence[str]) -> SufficiencyResult:
    """Are the given views enough to reconstruct all three overall dimensions?

    A prismatic solid needs all of width (X), height (Z) and depth (Y) determined.
    Any single orthographic view fixes only two of the three, so at least two
    distinct views are required; :func:`dimensions_covered` reports exactly which.
    """
    covered = dimensions_covered(view_names)
    missing = tuple(d for d in ALL_DIMENSIONS if d not in covered)
    ordered_covered = tuple(d for d in ALL_DIMENSIONS if d in covered)
    return SufficiencyResult(sufficient=not missing,
                             covered=ordered_covered,
                             missing=missing)
