"""cvcad_pixel_calibration — pixel-to-metric calibration from a reference object.

Deterministic core of the "vision-based measurement" pipeline in Bhandari &
Manandhar, *Integrating Computer Vision and CAD for Precise Dimension Extraction
and 3D Solid Model Regeneration* (Machines 2023, 11, 1083), Sec. 6 / Fig. 9.

The paper measures an object in an image by first calibrating with a reference
object of *known* physical size (a circular red sticker of known diameter). The
calibration ratio is

    pixels_per_metric = reference_width_in_pixels / known_reference_width      (I)

and any pixel coordinate is mapped to a real-world coordinate with the paper's
relation

    (x, y) = ( (phi / delta) * p(x) , (phi / delta) * p(y) )                    (II)

where ``phi`` is the reference dimension, ``delta`` the reference pixel width and
``p`` a contour pixel coordinate. Note ``phi / delta == 1 / pixels_per_metric``,
i.e. millimetres per pixel.

Everything here is closed-form geometry over supplied pixel data. The learned /
external parts of the paper (GrabCut segmentation, Canny edge detection, OCR of
the reference size) are NOT implemented — this module consumes their outputs
(pixel widths, contour points) and turns them into calibrated measurements.

Stdlib-only, deterministic, no wall clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

Point = Tuple[float, float]


@dataclass(frozen=True)
class Calibration:
    """A pixel<->metric scale derived from one reference object.

    ``pixels_per_metric`` is pixels per unit length (e.g. px/mm); its reciprocal
    ``mm_per_pixel`` is the paper's ``phi / delta`` factor.
    """

    pixels_per_metric: float
    unit: str = "mm"

    @property
    def mm_per_pixel(self) -> float:
        return 1.0 / self.pixels_per_metric

    def to_dict(self) -> dict:
        return {
            "pixels_per_metric": self.pixels_per_metric,
            "mm_per_pixel": self.mm_per_pixel,
            "unit": self.unit,
        }


def calibrate_from_reference(reference_pixel_width: float,
                             known_width: float,
                             unit: str = "mm") -> Calibration:
    """Eq. (I): pixels_per_metric = object width in pixels / known width."""
    if reference_pixel_width <= 0.0:
        raise ValueError("reference_pixel_width must be positive")
    if known_width <= 0.0:
        raise ValueError("known_width must be positive")
    return Calibration(reference_pixel_width / float(known_width), unit)


def pixels_to_metric(cal: Calibration, pixels: float) -> float:
    """Convert a pixel length to a metric length."""
    return pixels * cal.mm_per_pixel


def metric_to_pixels(cal: Calibration, length: float) -> float:
    """Inverse of :func:`pixels_to_metric`."""
    return length * cal.pixels_per_metric


def point_to_metric(cal: Calibration, p: Point) -> Point:
    """Eq. (II): map a pixel coordinate to a real-world coordinate."""
    f = cal.mm_per_pixel
    return (p[0] * f, p[1] * f)


def contour_to_metric(cal: Calibration, points: Sequence[Point]) -> List[Point]:
    """Map a whole contour (list of pixel points) to metric coordinates."""
    return [point_to_metric(cal, p) for p in points]


def euclidean_pixels(a: Point, b: Point) -> float:
    """Pixel distance between two points."""
    return math.hypot(b[0] - a[0], b[1] - a[1])


def distance_metric(cal: Calibration, a: Point, b: Point) -> float:
    """Real-world distance between two pixel points."""
    return pixels_to_metric(cal, euclidean_pixels(a, b))


def bounding_box(points: Sequence[Point]) -> Tuple[float, float, float, float]:
    """Axis-aligned pixel bounding box (min_x, min_y, max_x, max_y)."""
    if not points:
        raise ValueError("points must be non-empty")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def bounding_box_pixel_size(points: Sequence[Point]) -> Tuple[float, float]:
    """Return (width_px, height_px) of the bounding box of ``points``."""
    min_x, min_y, max_x, max_y = bounding_box(points)
    return (max_x - min_x, max_y - min_y)


def measure_object_size(cal: Calibration,
                        object_points: Sequence[Point]) -> Tuple[float, float]:
    """Measure an object's (width, height) in metric units from its contour."""
    w_px, h_px = bounding_box_pixel_size(object_points)
    return (pixels_to_metric(cal, w_px), pixels_to_metric(cal, h_px))


def calibrate_from_reference_contour(reference_points: Sequence[Point],
                                     known_width: float,
                                     unit: str = "mm") -> Calibration:
    """Calibrate directly from the reference object's contour points.

    Uses the reference bounding-box *width* as ``reference_pixel_width``. This
    mirrors the paper: the smaller contour (the red circular sticker) is the
    reference, and its known diameter fixes the scale.
    """
    w_px, _ = bounding_box_pixel_size(reference_points)
    return calibrate_from_reference(w_px, known_width, unit)
