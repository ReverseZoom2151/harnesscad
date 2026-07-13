"""cad2program_canvas_layout — arrange orthographic views on a fixed canvas.

To feed a 2D drawing to an image encoder, CAD2PROGRAM (Wang et al., AAAI 2025,
App. A.1 / Fig. 8) renders an "engineering drawing" by *arranging the three
axis-aligned orthographic views on a fixed-size canvas*: the top, front and side
views are placed at the top-left, bottom-left and bottom-right of the canvas
respectively, and the canvas is a fixed resolution (512x512 in the paper's
PlankAssembly (ViT) experiment).

The placement itself — projecting a 3D box-composed solid to three views, scaling
them to fit inside the canvas quadrants, and computing the pixel rectangles — is
deterministic geometry with no learned component.  This module implements it.

Layout (third-angle convention, matching :mod:`drawings.creft_projection`)::

    +------------------+------------------+
    | TOP  (X h, Y v)  |                  |
    +------------------+------------------+
    | FRONT (X h, Z v) | SIDE  (Y h, Z v) |
    +------------------+------------------+

Each view is uniformly scaled (same factor) to preserve relative proportions, so
shared extents stay comparable across quadrants — the property the model exploits
to align dimensions between views.

Pure stdlib, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from harnesscad.domain.drawings.view_lifting import FRONT, SIDE, TOP
from harnesscad.domain.reconstruction.translate.shape_program import Bbox, ShapeProgram

# Which (axis_min, axis_max) of a box map to (h, v) of each view, plus the
# view's placement quadrant on the canvas.
_VIEW_AXES: Dict[str, Tuple[str, str]] = {
    TOP: ("x", "y"),
    FRONT: ("x", "z"),
    SIDE: ("y", "z"),
}
_QUADRANT: Dict[str, str] = {
    TOP: "top_left",
    FRONT: "bottom_left",
    SIDE: "bottom_right",
}


@dataclass(frozen=True)
class PixelRect:
    """An integer pixel rectangle on the canvas (left, top, width, height)."""

    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


@dataclass(frozen=True)
class CanvasLayout:
    """Placed rectangles for one drawing on a fixed canvas."""

    width: int
    height: int
    scale: float
    rects: Dict[str, List[PixelRect]]   # view name -> per-primitive rects


def _box_interval(box: Bbox, axis: str) -> Tuple[float, float]:
    lo = {"x": box.position_x - box.scale_x / 2.0,
          "y": box.position_y - box.scale_y / 2.0,
          "z": box.position_z - box.scale_z / 2.0}[axis]
    size = {"x": box.scale_x, "y": box.scale_y, "z": box.scale_z}[axis]
    return (lo, lo + size)


def _program_span(program: ShapeProgram) -> Dict[str, Tuple[float, float]]:
    lo = {"x": float("inf"), "y": float("inf"), "z": float("inf")}
    hi = {"x": float("-inf"), "y": float("-inf"), "z": float("-inf")}
    for inst in program.instances:
        for axis in ("x", "y", "z"):
            a, b = _box_interval(inst.bbox, axis)
            lo[axis] = min(lo[axis], a)
            hi[axis] = max(hi[axis], b)
    return {axis: (lo[axis], hi[axis]) for axis in ("x", "y", "z")}


def layout_program(program: ShapeProgram, canvas: int = 512,
                   margin: int = 8, gap: int = 8) -> CanvasLayout:
    """Project a box-composed program to three views placed on a square canvas.

    A single uniform ``scale`` maps model units to pixels so that all three views
    fit within their quadrants; every primitive contributes one rectangle per
    view.  The Y axis of each quadrant is flipped so larger model coordinates map
    to smaller pixel rows (image convention, origin at top-left).
    """
    if not program.instances:
        raise ValueError("cannot lay out an empty program")
    span = _program_span(program)
    ext = {axis: max(0.0, span[axis][1] - span[axis][0]) for axis in span}

    # Quadrant pixel size (a square split with margins and a central gap).
    quad = (canvas - 2 * margin - gap) // 2
    if quad <= 0:
        raise ValueError("canvas too small for margins/gap")

    # Horizontal content: left column spans X, right column spans Y.
    # Vertical content: top row spans Y, bottom row spans Z.
    horiz = max(ext["x"], ext["y"], 1e-9)
    vert = max(ext["y"], ext["z"], 1e-9)
    scale = quad / max(horiz, vert)

    quad_origin = {
        "top_left": (margin, margin),
        "bottom_left": (margin, margin + quad + gap),
        "bottom_right": (margin + quad + gap, margin + quad + gap),
    }

    rects: Dict[str, List[PixelRect]] = {FRONT: [], TOP: [], SIDE: []}
    for view, (ha, va) in _VIEW_AXES.items():
        qx, qy = quad_origin[_QUADRANT[view]]
        h_lo = span[ha][0]
        v_hi = span[va][1]
        for inst in program.instances:
            a0, a1 = _box_interval(inst.bbox, ha)
            b0, b1 = _box_interval(inst.bbox, va)
            left = qx + int(round((a0 - h_lo) * scale))
            width = max(1, int(round((a1 - a0) * scale)))
            # Flip vertical: larger model coord -> smaller pixel row.
            top = qy + int(round((v_hi - b1) * scale))
            height = max(1, int(round((b1 - b0) * scale)))
            rects[view].append(PixelRect(left, top, width, height))
    return CanvasLayout(canvas, canvas, scale, rects)
