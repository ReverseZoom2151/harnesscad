"""Drawing-space <-> canvas-space viewport transforms (from the ``arcs`` CAD system).

``arcs/src/window/utils.rs`` builds the affine matrix that maps *drawing space*
(y-up, real units, unbounded) onto *canvas space* (y-down, pixels, window-sized)
from just two viewport parameters -- the drawing-space point the window is
centred on and the number of pixels per drawing unit:

    drawing_units_per_pixel = 1 / pixels_per_drawing_unit
    x_basis = ( 1,  0) * dupp
    y_basis = ( 0, -1) * dupp                      <- the y-flip
    origin  = centre + (-w/2, +h/2) * dupp

    canvas -> drawing = | x_basis.x  x_basis.y  0 |
                        | y_basis.x  y_basis.y  0 |
                        | origin.x   origin.y   1 |

and the drawing -> canvas direction is its inverse. ``arcs/src/components/
dimension.rs`` complements it with the ``Dimension`` enum: a length is either
pinned in *pixels* (stroke widths, handles -- constant on screen at any zoom) or
expressed in *drawing units* (real geometry -- scales with the zoom).

The harness had no world<->screen viewport model: ``vision.cvcad_pixel_calibration``
is a camera/pixel-scale estimator and ``drawings.cad2program_canvas_layout`` lays
out sheets; neither gives an invertible, zoomable, pannable viewport. This module
adds it, plus zoom/pan, visible-bounds and zoom-to-fit, which the Rust source
leaves to the application.

Affines are 6-tuples ``(a, b, c, d, e, f)`` acting as
``x' = a*x + c*y + e``, ``y' = b*x + d*y + f`` (the same row-major order euclid
uses). Pure standard library, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

Point = Tuple[float, float]
Size = Tuple[float, float]
Affine = Tuple[float, float, float, float, float, float]
BBox = Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)

IDENTITY: Affine = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

PIXELS = "pixels"
DRAWING_UNITS = "drawing_units"

__all__ = [
    "DRAWING_UNITS",
    "IDENTITY",
    "PIXELS",
    "Dimension",
    "Viewport",
    "apply_affine",
    "compose_affine",
    "invert_affine",
    "to_canvas_coordinates",
    "to_drawing_coordinates",
    "transform_to_canvas_space",
    "transform_to_drawing_space",
    "visible_bounds",
    "zoom_to_fit",
]


def apply_affine(transform: Affine, point: Point) -> Point:
    a, b, c, d, e, f = transform
    x, y = point
    return (a * x + c * y + e, b * x + d * y + f)


def compose_affine(first: Affine, then: Affine) -> Affine:
    """Affine applying ``first`` and then ``then``."""
    a1, b1, c1, d1, e1, f1 = first
    a2, b2, c2, d2, e2, f2 = then
    return (
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
        e1 * a2 + f1 * c2 + e2,
        e1 * b2 + f1 * d2 + f2,
    )


def invert_affine(transform: Affine) -> Affine:
    a, b, c, d, e, f = transform
    determinant = a * d - b * c
    if determinant == 0.0:
        raise ValueError("affine is not invertible")
    ia = d / determinant
    ib = -b / determinant
    ic = -c / determinant
    id_ = a / determinant
    ie = -(e * ia + f * ic)
    if_ = -(e * ib + f * id_)
    return (ia, ib, ic, id_, ie, if_)


@dataclass(frozen=True)
class Dimension:
    """A length that is either pinned in pixels or measured in drawing units."""

    kind: str
    value: float

    @staticmethod
    def pixels(value: float) -> "Dimension":
        return Dimension(PIXELS, float(value))

    @staticmethod
    def drawing_units(value: float) -> "Dimension":
        return Dimension(DRAWING_UNITS, float(value))

    def in_pixels(self, pixels_per_drawing_unit: float) -> float:
        if self.kind == PIXELS:
            return self.value
        if self.kind == DRAWING_UNITS:
            return self.value * pixels_per_drawing_unit
        raise ValueError("unknown dimension kind: " + str(self.kind))

    def in_drawing_units(self, pixels_per_drawing_unit: float) -> float:
        if pixels_per_drawing_unit <= 0.0:
            raise ValueError("pixels_per_drawing_unit must be positive")
        if self.kind == DRAWING_UNITS:
            return self.value
        if self.kind == PIXELS:
            return self.value / pixels_per_drawing_unit
        raise ValueError("unknown dimension kind: " + str(self.kind))


@dataclass(frozen=True)
class Viewport:
    """The window onto drawing space."""

    centre: Point
    pixels_per_drawing_unit: float

    def __post_init__(self) -> None:
        if not self.pixels_per_drawing_unit > 0.0:
            raise ValueError("pixels_per_drawing_unit must be positive")

    def zoomed(self, scale_factor: float) -> "Viewport":
        """Zoom in (``scale_factor > 1``) or out, keeping the centre fixed.

        Follows ``arcs``: a positive factor divides the pixels-per-unit scale,
        so ``zoomed(2)`` doubles the visible drawing area.
        """
        if scale_factor == 0.0:
            raise ValueError("scale_factor must be non-zero")
        return Viewport(
            self.centre, self.pixels_per_drawing_unit / scale_factor
        )

    def translated(self, dx: float, dy: float) -> "Viewport":
        """Pan by a displacement given in *drawing* units."""
        return Viewport(
            (self.centre[0] + dx, self.centre[1] + dy),
            self.pixels_per_drawing_unit,
        )


def transform_to_drawing_space(viewport: Viewport, window: Size) -> Affine:
    """Canvas (pixel, y-down) -> drawing (unit, y-up) affine."""
    width, height = window
    dupp = 1.0 / viewport.pixels_per_drawing_unit

    x_basis = (dupp, 0.0)
    y_basis = (0.0, -dupp)
    origin = (
        viewport.centre[0] - width / 2.0 * dupp,
        viewport.centre[1] + height / 2.0 * dupp,
    )
    return (
        x_basis[0],
        x_basis[1],
        y_basis[0],
        y_basis[1],
        origin[0],
        origin[1],
    )


def transform_to_canvas_space(viewport: Viewport, window: Size) -> Affine:
    """Drawing (unit, y-up) -> canvas (pixel, y-down) affine."""
    return invert_affine(transform_to_drawing_space(viewport, window))


def to_canvas_coordinates(
    point: Point, viewport: Viewport, window: Size
) -> Point:
    return apply_affine(transform_to_canvas_space(viewport, window), point)


def to_drawing_coordinates(
    point: Point, viewport: Viewport, window: Size
) -> Point:
    return apply_affine(transform_to_drawing_space(viewport, window), point)


def visible_bounds(viewport: Viewport, window: Size) -> BBox:
    """The drawing-space region currently visible in the window."""
    width, height = window
    bottom_left = to_drawing_coordinates((0.0, height), viewport, window)
    top_right = to_drawing_coordinates((width, 0.0), viewport, window)
    return (bottom_left[0], bottom_left[1], top_right[0], top_right[1])


def zoom_to_fit(bounds: BBox, window: Size, margin: float = 0.0) -> Viewport:
    """Viewport framing ``bounds`` inside ``window``, keeping the aspect ratio.

    ``margin`` is a fraction of the window reserved as padding on every side
    (``0.05`` -> 5 percent). Degenerate (zero-extent) bounds fall back to a
    scale of 1 pixel per drawing unit.
    """
    min_x, min_y, max_x, max_y = bounds
    if max_x < min_x or max_y < min_y:
        raise ValueError("bounds are inverted")
    if not 0.0 <= margin < 0.5:
        raise ValueError("margin must be in [0, 0.5)")

    width, height = window
    if width <= 0.0 or height <= 0.0:
        raise ValueError("window must have a positive size")

    usable_w = width * (1.0 - 2.0 * margin)
    usable_h = height * (1.0 - 2.0 * margin)

    span_x = max_x - min_x
    span_y = max_y - min_y

    scales = []
    if span_x > 0.0:
        scales.append(usable_w / span_x)
    if span_y > 0.0:
        scales.append(usable_h / span_y)
    scale = min(scales) if scales else 1.0

    centre = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
    return Viewport(centre, scale)
