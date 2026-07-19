"""Flat-pack decomposition of a box cabinet into cut-ready 2D panels.

The recurring cabinet example turns a
3D box-cabinet (exterior height/width/depth plus a plywood thickness)
into flat panels for laser/CNC cutting.  Wall thickness is the classic
failure mode here: side boards made too thin, a back board that comes out
too short, and board thickness confused with cabinet depth.  A finite
laser bed also forces oversized panels (for example the back board wider
than the 12-inch bed) to be split into pieces.

This module implements the deterministic geometry: butt-joint panel
decomposition that correctly accounts for wall thickness, material-area
accounting, bed-fit testing in either orientation, and equal-strip
splitting of oversized panels.

Butt-joint convention (documented and used by :func:`decompose_cabinet`):

  * 2 side panels: each ``width = D``, ``height = H`` (the full sides).
  * top and bottom panels: each ``width = W - 2*t`` (they fit BETWEEN the
    two sides), ``height = D``.
  * back panel: ``width = W - 2*t``, ``height = H - 2*t`` (it fits inside,
    behind the shelves, between the top/bottom and the two sides).  The
    alternative full-cover back (``W`` by ``H``) is available via the
    ``back_full_cover`` flag but the inner-fit convention is the default.
  * each shelf: ``width = W - 2*t``, ``height = D`` (fits between the
    sides).  ``num_shelves`` such shelves are produced.

All panels carry the plywood thickness ``t``.  All dimensions are plain
numbers in a single consistent unit (the tests use inches).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Panel:
    """A single flat 2D piece to be cut from stock of the given thickness.

    Attributes:
        name: Human-readable identifier (for example ``"side_left"``).
        width: Width of the flat piece (2D, in the cutting plane).
        height: Height of the flat piece (2D, in the cutting plane).
        thickness: Material thickness of the stock the piece is cut from.
        holes: Optional list of ``(x, y, diameter)`` hole features.
    """

    name: str
    width: float
    height: float
    thickness: float
    holes: List[Tuple[float, float, float]] = field(default_factory=list)

    def area(self) -> float:
        """Return the flat area (``width * height``) of this panel."""
        return self.width * self.height


def decompose_cabinet(
    exterior_height: float,
    exterior_width: float,
    exterior_depth: float,
    thickness: float,
    num_shelves: int = 1,
    back_full_cover: bool = False,
) -> List[Panel]:
    """Decompose a box cabinet into flat butt-joint panels.

    Args:
        exterior_height: Exterior height ``H`` of the assembled cabinet.
        exterior_width: Exterior width ``W`` of the assembled cabinet.
        exterior_depth: Exterior depth ``D`` of the assembled cabinet.
        thickness: Plywood thickness ``t`` of every panel.
        num_shelves: Number of interior shelves (each fits between the
            sides).  Defaults to 1.
        back_full_cover: If True the back panel covers the full ``W`` by
            ``H`` rectangle instead of the default inner-fit rectangle.

    Returns:
        A list of :class:`Panel` objects: two sides, a top, a bottom, a
        back, then ``num_shelves`` shelves.

    Raises:
        ValueError: If any dimension or the thickness is not positive, if
            ``num_shelves`` is negative, or if the wall thickness leaves a
            non-positive inner dimension (``2*t >= W`` or ``2*t >= H``).
    """
    H = exterior_height
    W = exterior_width
    D = exterior_depth
    t = thickness

    if H <= 0 or W <= 0 or D <= 0:
        raise ValueError("exterior dimensions must be positive")
    if t <= 0:
        raise ValueError("thickness must be positive")
    if num_shelves < 0:
        raise ValueError("num_shelves must be non-negative")

    inner_w = W - 2 * t
    inner_h = H - 2 * t
    if inner_w <= 0:
        raise ValueError(
            "2*thickness >= exterior_width: no positive inner width "
            "(walls would overlap)"
        )
    if inner_h <= 0:
        raise ValueError(
            "2*thickness >= exterior_height: no positive inner height "
            "(top/bottom would overlap)"
        )

    panels: List[Panel] = [
        Panel("side_left", D, H, t),
        Panel("side_right", D, H, t),
        Panel("top", inner_w, D, t),
        Panel("bottom", inner_w, D, t),
    ]

    if back_full_cover:
        panels.append(Panel("back", W, H, t))
    else:
        panels.append(Panel("back", inner_w, inner_h, t))

    for i in range(num_shelves):
        panels.append(Panel("shelf_{0}".format(i + 1), inner_w, D, t))

    return panels


def total_material_area(panels: List[Panel]) -> float:
    """Return the summed flat area (``width * height``) of all panels."""
    return sum(p.area() for p in panels)


def fits_on_bed(panel: Panel, bed_w: float, bed_h: float) -> bool:
    """Return True if ``panel`` fits the bed in either orientation.

    The panel fits if it fits as-is (``width <= bed_w`` and
    ``height <= bed_h``) or rotated 90 degrees (``width <= bed_h`` and
    ``height <= bed_w``).
    """
    as_is = panel.width <= bed_w and panel.height <= bed_h
    rotated = panel.width <= bed_h and panel.height <= bed_w
    return as_is or rotated


def split_panel_to_fit(
    panel: Panel,
    bed_w: float,
    bed_h: float,
    kerf: float = 0.0,
) -> List[Panel]:
    """Split an oversized panel into the fewest equal strips that fit.

    If ``panel`` already fits the bed (in either orientation) it is
    returned unchanged as ``[panel]``.  Otherwise the panel is divided
    along its longer dimension into ``ceil(long / bed_long)`` equal
    strips -- for example splitting an oversized back board into
    halves.  The strips are named ``panel.name + "_1"``, ``"_2"``, ...

    Args:
        panel: The panel to split.
        bed_w: Bed width.
        bed_h: Bed height.
        kerf: Join/kerf allowance.  Kept simple: the oversized dimension
            is divided equally and strips are not shrunk, so ``kerf`` only
            affects the sufficiency check (each strip plus one kerf must
            still fit).  Defaults to 0.0.

    Returns:
        A list of :class:`Panel` strips, each of which fits the bed.

    Raises:
        ValueError: If the panel's shorter dimension alone cannot fit the
            bed's larger dimension (splitting the long side cannot help),
            or if ``bed_w``/``bed_h`` are not positive.
    """
    if bed_w <= 0 or bed_h <= 0:
        raise ValueError("bed dimensions must be positive")

    if fits_on_bed(panel, bed_w, bed_h):
        return [panel]

    bed_long = max(bed_w, bed_h)
    bed_short = min(bed_w, bed_h)

    long_dim = max(panel.width, panel.height)
    short_dim = min(panel.width, panel.height)
    split_along_width = panel.width >= panel.height

    # The dimension we are NOT splitting must fit the bed on its own; if
    # it exceeds even the larger bed dimension, splitting the long side
    # cannot rescue it.
    if short_dim > bed_long + 1e-9:
        raise ValueError(
            "panel short dimension {0} exceeds largest bed dimension {1}; "
            "cannot fit even by splitting the long side".format(
                short_dim, bed_long
            )
        )

    # Choose how many strips: divide the long dimension so each strip fits
    # the bed's long dimension (the short dimension is placed along the
    # bed's short dimension, already verified to fit somewhere).  Using the
    # larger bed dimension for the long side gives the minimum strip count.
    n = max(2, int(math.ceil((long_dim + kerf) / bed_long - 1e-9)))

    # Grow the strip count until each resulting strip genuinely fits.
    while True:
        strip_long = long_dim / n
        if split_along_width:
            probe = Panel(panel.name, strip_long, panel.height, panel.thickness)
        else:
            probe = Panel(panel.name, panel.width, strip_long, panel.thickness)
        if fits_on_bed(probe, bed_w, bed_h):
            break
        n += 1
        if n > 10000:  # pragma: no cover - defensive guard
            raise ValueError("unable to split panel to fit the bed")

    strips: List[Panel] = []
    for i in range(n):
        if split_along_width:
            strip = Panel(
                "{0}_{1}".format(panel.name, i + 1),
                long_dim / n,
                panel.height,
                panel.thickness,
            )
        else:
            strip = Panel(
                "{0}_{1}".format(panel.name, i + 1),
                panel.width,
                long_dim / n,
                panel.thickness,
            )
        strips.append(strip)

    # Silence unused-variable warnings while keeping the names meaningful.
    _ = (bed_short, short_dim)
    return strips


def nest_report(
    panels: List[Panel],
    bed_w: float,
    bed_h: float,
    kerf: float = 0.0,
) -> dict:
    """Report bed-fit status and the fully split panel list.

    Args:
        panels: Panels to check against the bed.
        bed_w: Bed width.
        bed_h: Bed height.
        kerf: Join/kerf allowance forwarded to :func:`split_panel_to_fit`.

    Returns:
        A dict with keys:
          * ``"fit"``: names of panels that fit as-is.
          * ``"needs_split"``: names of panels that had to be split.
          * ``"split_panels"``: the full list of :class:`Panel` after
            splitting every oversized panel.
          * ``"total_area"``: total flat area of ``split_panels``.
    """
    fit: List[str] = []
    needs_split: List[str] = []
    split_panels: List[Panel] = []

    for panel in panels:
        if fits_on_bed(panel, bed_w, bed_h):
            fit.append(panel.name)
            split_panels.append(panel)
        else:
            needs_split.append(panel.name)
            split_panels.extend(
                split_panel_to_fit(panel, bed_w, bed_h, kerf=kerf)
            )

    return {
        "fit": fit,
        "needs_split": needs_split,
        "split_panels": split_panels,
        "total_area": total_material_area(split_panels),
    }
