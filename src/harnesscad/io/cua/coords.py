"""coords — decode a model's coordinate output, and refuse to guess its space.

This is the F5 layer: the model emits a point in some IMAGE space, and it must
land as an exact pixel in the frame's image space before :mod:`frames` maps it to
the screen. Every reference repo loses coordinates HERE, three different ways, and
each way is reproduced-as-a-test below so the fix is pinned:

**Aspect-distorting downscale (computer-use-agent).** Its ``ResolutionScaler``
resizes with ``fit: "fill"`` (non-uniform) and then, tellingly, computes a SINGLE
``scaleFactor = sqrt((sw/w)*(sh/h))`` — the geometric mean of two DIFFERENT
per-axis scales — and divides both x and y by it in ``scaleToOriginal``. When the
source is not 16:10 the x-scale and y-scale differ, so one scalar cannot invert
both axes: the click drifts, worse the further from centre.
:func:`detect_nonuniform_scale` flags the setup; :class:`Downscale` keeps the two
axis scales SEPARATE and inverts each correctly.

**bbox strings (os_computer_use grounding).** OS-Atlas returns
``<|box_start|>x1, y1, x2, y2<|box_end|>`` and the midpoint is taken; ShowUI
returns a normalised ``[u, v]`` multiplied by image size. :func:`parse_point`
handles the tag, the 2- and 4-number forms, and normalised vs pixel — but only
when TOLD which it is.

**Magnitude guessing (TuriX, called out in frames.py).** Deciding "0-1 vs 0-1000"
from how big the number is is a bug. Here :class:`CoordSpace` is DECLARED by the
caller and an out-of-range value RAISES; the magnitude is never a hint.

Stdlib only (``re``). Produces integer image pixels ready for
:meth:`frames.Frame.to_screen`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class CoordError(ValueError):
    """A coordinate could not be decoded, or was outside its declared space."""


class CoordSpace(Enum):
    """The DECLARED space a model's numbers live in. Never inferred."""

    PIXELS = "pixels"          # already image pixels (0..w, 0..h)
    UNIT = "unit"              # normalised 0..1 (ShowUI)
    THOUSAND = "thousand"      # normalised 0..1000 (Qwen-VL / many bbox models)


_BOX_TAG = re.compile(r"<\|box_start\|>(.*?)<\|box_end\|>", re.S)
_NUMBER = re.compile(r"-?\d+\.\d+|-?\d+")


def _numbers(text: str) -> List[float]:
    match = _BOX_TAG.search(text)
    inner = match.group(1) if match else text
    return [float(n) for n in _NUMBER.findall(inner)]


def parse_numbers(text: str) -> List[float]:
    """The raw numbers a model emitted, unwrapping a ``<|box_start|>...`` tag if
    present. Empty list if there are none (caller decides if that is an error)."""
    return _numbers(text)


def midpoint(numbers: List[float]) -> Tuple[float, float]:
    """A point from a 2-number ``(x, y)`` or a 4-number ``(x1,y1,x2,y2)`` bbox.

    The 4-number case returns the bbox CENTRE, matching the grounding repos. Any
    other count is an error, not a best-effort guess."""
    if len(numbers) == 2:
        return numbers[0], numbers[1]
    if len(numbers) >= 4:
        return (numbers[0] + numbers[2]) / 2.0, (numbers[1] + numbers[3]) / 2.0
    raise CoordError("need 2 or 4 numbers to form a point, got %d: %r"
                     % (len(numbers), numbers))


def denormalize(u: float, v: float, space: CoordSpace,
                img_w: int, img_h: int) -> Tuple[float, float]:
    """Map a point from its declared space into image pixels, RAISING on range.

    The whole anti-magnitude-guessing rule lives here: for :attr:`CoordSpace.UNIT`
    a value must be in ``[0, 1]``; for :attr:`CoordSpace.THOUSAND` in ``[0, 1000]``;
    for :attr:`CoordSpace.PIXELS` within the image. Out of range is a decode
    error, never a hint that the caller meant a different space.
    """
    if img_w <= 0 or img_h <= 0:
        raise CoordError("degenerate image size %dx%d" % (img_w, img_h))
    if space is CoordSpace.UNIT:
        _require_range(u, v, 0.0, 1.0, space)
        return u * img_w, v * img_h
    if space is CoordSpace.THOUSAND:
        _require_range(u, v, 0.0, 1000.0, space)
        return (u / 1000.0) * img_w, (v / 1000.0) * img_h
    # PIXELS
    _require_range(u, v, 0.0, float(max(img_w, img_h)), space)
    if not (0.0 <= u <= img_w and 0.0 <= v <= img_h):
        raise CoordError("pixel point (%s, %s) is outside the %dx%d image"
                         % (u, v, img_w, img_h))
    return u, v


def _require_range(u: float, v: float, lo: float, hi: float,
                   space: CoordSpace) -> None:
    for label, val in (("x", u), ("y", v)):
        if not (lo <= val <= hi):
            raise CoordError(
                "%s=%s outside declared space %s [%s, %s]; the magnitude is NOT "
                "used to reinterpret the space" % (label, val, space.value, lo, hi))


def parse_point(text: str, space: CoordSpace, img_w: int, img_h: int
                ) -> Tuple[int, int]:
    """Full decode: model text -> integer image pixel, in one call.

    Unwraps a bbox tag, forms the point (2-number or 4-number midpoint), then
    denormalises from the DECLARED ``space``. Rounds to the nearest pixel last, so
    rounding never changes which side of a range bound a value falls on.
    """
    px, py = midpoint(_numbers(text))
    fx, fy = denormalize(px, py, space, img_w, img_h)
    return int(round(fx)), int(round(fy))


# --- the downscale, done honestly -------------------------------------------
@dataclass(frozen=True)
class Downscale:
    """A resize that keeps its TWO axis scales, so a coordinate round-trips.

    ``src`` is the screen/source size and ``dst`` the model-image size. Unlike the
    single-scalar ``scaleFactor`` in the reference repo, ``sx`` and ``sy`` are kept
    apart and inverted independently — the only mapping that is correct when the
    resize is non-uniform.
    """

    src_w: int
    src_h: int
    dst_w: int
    dst_h: int

    def __post_init__(self) -> None:
        if min(self.src_w, self.src_h, self.dst_w, self.dst_h) <= 0:
            raise CoordError("degenerate downscale %r" % (self,))

    @property
    def sx(self) -> float:
        return self.dst_w / float(self.src_w)

    @property
    def sy(self) -> float:
        return self.dst_h / float(self.src_h)

    @property
    def uniform(self) -> bool:
        return abs(self.sx - self.sy) < 1e-9

    def to_image(self, x: float, y: float) -> Tuple[float, float]:
        return x * self.sx, y * self.sy

    def to_source(self, x: float, y: float) -> Tuple[float, float]:
        return x / self.sx, y / self.sy


def detect_nonuniform_scale(src_w: int, src_h: int, dst_w: int, dst_h: int
                            ) -> Optional[str]:
    """A warning string if this resize distorts aspect (a circle -> ellipse), else
    None. This is the ``fit: "fill"`` bug: the caller should letterbox instead (see
    :meth:`frames.Frame.letterbox`) or at least keep both axis scales."""
    ds = Downscale(src_w, src_h, dst_w, dst_h)
    if ds.uniform:
        return None
    return ("non-uniform downscale: x-scale %.4f != y-scale %.4f; a single "
            "scaleFactor cannot invert both axes and a click drifts off-centre. "
            "Letterbox (uniform scale + pad) instead." % (ds.sx, ds.sy))


def geometric_mean_error(src_w: int, src_h: int, dst_w: int, dst_h: int,
                         x: float, y: float) -> Tuple[float, float]:
    """The pixel error the reference repo's ``sqrt(sx*sy)`` mapping introduces at
    ``(x, y)`` in source pixels: (correct - buggy) per axis. Zero only when the
    scale is uniform. Exposed so a test can PROVE the bug is real, not asserted."""
    import math
    ds = Downscale(src_w, src_h, dst_w, dst_h)
    g = math.sqrt(ds.sx * ds.sy)
    # The model SEES the fill-resized point (x*sx, y*sy). The buggy code then
    # inverts with the single factor /g instead of the true per-axis /sx, /sy.
    # That mismatch is the drift:
    recovered_x, recovered_y = (x * ds.sx) / g, (y * ds.sy) / g
    return (x - recovered_x), (y - recovered_y)
