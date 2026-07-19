"""Coordinate normalisation + 6-bit quantisation for parametric sketch primitives.

Every primitive parameter is quantized into a 6-bit integer, and the boolean flags
are encoded as 1 (true) or 0 (false) -- i.e. every coordinate is mapped onto one of
``2**6 = 64`` discrete levels. An accuracy protocol then allows a coordinate error
within a threshold ``eta = 1`` "out of 64 levels" to count as correct, and a
*quantisation error* metric reports the Chamfer gap introduced purely by
integerising the floats.

This module implements the deterministic pieces:

  * :func:`normalize_sketch` -- fit an axis-aligned bounding box over all primitive
    control points and radii and map coordinates into ``[0, 1]`` (returning the box
    so the mapping is invertible);
  * :func:`quantize` / :func:`dequantize` -- the ``[0, 1] <-> {0..levels-1}`` 6-bit
    lattice with round-half-up-at-.5 nearest-level rounding;
  * :func:`quantize_primitive` / :func:`dequantize_primitive` -- quantise the *padded
    7-slot* parameter row of a primitive (padding zeros stay zero, radius scaled by
    the box extent), preserving type and flag;
  * :func:`quantization_error` -- the mean absolute coordinate error and the max
    error introduced by a round-trip, in normalised units.

Pure stdlib. Reuses the primitive representation in
:mod:`reconstruction.ppa_primitive`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from harnesscad.domain.reconstruction.sketch import primitives as pp

# 6-bit quantisation -> 64 levels (quantized into 6-bit integers).
DEFAULT_BITS = 6


def levels(bits: int = DEFAULT_BITS) -> int:
    """Number of discrete levels for a ``bits``-bit quantiser (``2**bits``)."""
    if bits <= 0:
        raise ValueError("bits must be positive")
    return 1 << bits


def quantize(value: float, bits: int = DEFAULT_BITS) -> int:
    """Map ``value in [0, 1]`` to an integer level in ``{0 .. 2**bits - 1}``.

    Values outside ``[0, 1]`` are clamped. Rounding is nearest-level, ties up.
    """
    n = levels(bits)
    lvl = math.floor(value * (n - 1) + 0.5)
    if lvl < 0:
        return 0
    if lvl > n - 1:
        return n - 1
    return lvl


def dequantize(level: int, bits: int = DEFAULT_BITS) -> float:
    """Inverse of :func:`quantize`: level -> the ``[0, 1]`` cell centre."""
    n = levels(bits)
    return level / (n - 1)


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box ``[xmin, xmax] x [ymin, ymax]`` used for normalisation."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def extent(self) -> float:
        """Uniform scale = the larger side (keeps aspect ratio; avoids zero div)."""
        return max(self.xmax - self.xmin, self.ymax - self.ymin, 1e-12)

    def to_unit(self, x: float, y: float) -> tuple[float, float]:
        e = self.extent
        return ((x - self.xmin) / e, (y - self.ymin) / e)

    def from_unit(self, u: float, v: float) -> tuple[float, float]:
        e = self.extent
        return (u * e + self.xmin, v * e + self.ymin)


def sketch_bbox(sketch: pp.Sketch) -> BoundingBox:
    """Bounding box over all primitive control points (radii expand circle bounds)."""
    xs: list[float] = []
    ys: list[float] = []
    for prim in sketch:
        for (x, y) in prim.control_points():
            xs.append(x)
            ys.append(y)
        if prim.ptype == pp.CIRCLE:
            cx, cy = prim.control_points()[0]
            r = prim.radius
            xs.extend((cx - r, cx + r))
            ys.extend((cy - r, cy + r))
    if not xs:
        return BoundingBox(0.0, 0.0, 1.0, 1.0)
    return BoundingBox(min(xs), min(ys), max(xs), max(ys))


def normalize_sketch(sketch: pp.Sketch, box: BoundingBox | None = None):
    """Map a sketch's coordinates into ``[0, 1]`` using a uniform-scale box.

    Returns ``(normalized_sketch, box)``; pass the same ``box`` back to
    :func:`denormalize_sketch` to invert. Coordinates keep aspect ratio (uniform
    scale by the larger side).
    """
    if box is None:
        box = sketch_bbox(sketch)
    e = box.extent
    out = []
    for prim in sketch:
        p = prim.params
        u = list(p)
        # slots 0..5 are (x1,y1,x2,y2,x3,y3); slot 6 is radius (circle only).
        for i in range(0, 6, 2):
            if p[i] == 0.0 and p[i + 1] == 0.0 and i >= 2 * (prim.meaningful // 2):
                continue  # padding pair -- leave as 0
            nx, ny = box.to_unit(p[i], p[i + 1])
            u[i], u[i + 1] = nx, ny
        if prim.ptype == pp.CIRCLE:
            u[6] = p[6] / e  # radius scales by extent only (no offset)
        out.append(pp.Primitive(prim.ptype, prim.flag, tuple(u)))
    return pp.Sketch(out), box


def denormalize_sketch(sketch: pp.Sketch, box: BoundingBox) -> pp.Sketch:
    """Invert :func:`normalize_sketch` given the same ``box``."""
    e = box.extent
    out = []
    for prim in sketch:
        p = prim.params
        u = list(p)
        for i in range(0, 6, 2):
            if p[i] == 0.0 and p[i + 1] == 0.0 and i >= 2 * (prim.meaningful // 2):
                continue
            x, y = box.from_unit(p[i], p[i + 1])
            u[i], u[i + 1] = x, y
        if prim.ptype == pp.CIRCLE:
            u[6] = p[6] * e
        out.append(pp.Primitive(prim.ptype, prim.flag, tuple(u)))
    return pp.Sketch(out)


def quantize_primitive(prim: pp.Primitive, bits: int = DEFAULT_BITS) -> tuple[int, ...]:
    """Quantise a primitive's 7-slot *normalised* row to integer levels.

    The primitive's params are assumed to already lie in ``[0, 1]`` (see
    :func:`normalize_sketch`). Padding zeros quantise to level 0.
    """
    return tuple(quantize(v, bits) for v in prim.params)


def dequantize_primitive(prim_type: str, flag: bool, qparams, bits: int = DEFAULT_BITS
                         ) -> pp.Primitive:
    """Rebuild a primitive from integer levels (inverse of :func:`quantize_primitive`)."""
    return pp.Primitive(prim_type, bool(flag),
                        tuple(dequantize(q, bits) for q in qparams))


def roundtrip_primitive(prim: pp.Primitive, bits: int = DEFAULT_BITS) -> pp.Primitive:
    """Quantise then dequantise a normalised primitive (the value it collapses to)."""
    return dequantize_primitive(prim.ptype, prim.flag,
                                quantize_primitive(prim, bits), bits)


def quantization_error(sketch: pp.Sketch, bits: int = DEFAULT_BITS) -> dict:
    """Mean and max absolute coordinate error from a 6-bit round-trip.

    ``sketch`` must already be normalised to ``[0, 1]``. Only meaningful (non-padding)
    coordinate slots contribute. Errors are in normalised units; the maximum possible
    error is half a level = ``0.5 / (2**bits - 1)``.
    """
    errs: list[float] = []
    for prim in sketch:
        rt = roundtrip_primitive(prim, bits)
        m = prim.meaningful
        # meaningful coordinate slots: first `m` of (x1,y1,x2,y2,x3,y3) for
        # line/arc/point, plus the radius slot for circles.
        if prim.ptype == pp.CIRCLE:
            idxs = (0, 1, 6)
        else:
            idxs = tuple(range(m))
        for i in idxs:
            errs.append(abs(prim.params[i] - rt.params[i]))
    if not errs:
        return {"mean": 0.0, "max": 0.0, "count": 0}
    return {"mean": sum(errs) / len(errs), "max": max(errs), "count": len(errs)}
