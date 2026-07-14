"""Prove a render is a picture of something. Stdlib zlib, no PIL, no numpy.

A renderer that writes a 1600x1000 slab of background is not a failure the eye
catches in a contact sheet -- it is a failure the eye *forgives*. So every PNG
this package ships is decoded back off disk and measured:

    silhouette  -- fraction of pixels that differ from the background colour
                   (the corner). Too low: the part missed the frame or the
                   camera is inside it. Too high: the part fills the frame or
                   the buffer is a solid colour.
    variance    -- luminance variance over the whole image. A flat fill (blank,
                   black, white) has ~0; a shaded solid has hundreds.
    shades      -- how many distinct luminance levels the silhouette contains.
                   A flat-coloured blob (no shading, no depth) scores ~1.
    not_black   -- the image is not (near-)uniformly dark.

An image that fails any check is never published: it is fixed or dropped.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

__all__ = [
    "PngError",
    "PngImage",
    "PngStats",
    "load_png",
    "png_stats",
    "validate_png",
    "MIN_SILHOUETTE",
    "MAX_SILHOUETTE",
    "MIN_VARIANCE",
    "MIN_SHADES",
]

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

#: Thresholds. A hero render of a part on a background sits comfortably inside
#: these; a blank/flat/overflowing frame does not.
MIN_SILHOUETTE = 0.02
MAX_SILHOUETTE = 0.95
MIN_VARIANCE = 25.0
MIN_SHADES = 8


class PngError(ValueError):
    """The file is not a PNG we can decode (or is not a PNG at all)."""


@dataclass
class PngImage:
    width: int
    height: int
    channels: int
    #: Row-major, `channels` samples per pixel, 8 bits each.
    pixels: List[int]

    def rgb(self, i: int) -> Tuple[int, int, int]:
        o = i * self.channels
        if self.channels >= 3:
            return (self.pixels[o], self.pixels[o + 1], self.pixels[o + 2])
        g = self.pixels[o]
        return (g, g, g)

    @property
    def pixel_count(self) -> int:
        return self.width * self.height


@dataclass
class PngStats:
    width: int
    height: int
    silhouette: float
    variance: float
    shades: int
    mean_luma: float
    distinct_colours: int
    ok: bool
    failures: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "silhouette": round(self.silhouette, 4),
            "variance": round(self.variance, 2),
            "shades": self.shades,
            "mean_luma": round(self.mean_luma, 2),
            "distinct_colours": self.distinct_colours,
            "ok": self.ok,
            "failures": self.failures,
        }


# --- decode ----------------------------------------------------------------
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def load_png(path: str) -> PngImage:
    """Decode a non-interlaced 8-bit PNG with stdlib zlib only."""
    with open(path, "rb") as fh:
        data = fh.read()
    if not data.startswith(PNG_MAGIC):
        raise PngError(f"{path}: not a PNG (bad magic)")

    pos = len(PNG_MAGIC)
    idat = bytearray()
    width = height = depth = colour = interlace = -1
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length  # length + tag + body + crc
        if tag == b"IHDR":
            width, height, depth, colour, _comp, _filt, interlace = struct.unpack(
                ">IIBBBBB", body)
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break
    if width <= 0 or height <= 0:
        raise PngError(f"{path}: no IHDR / zero-sized image")
    if depth != 8:
        raise PngError(f"{path}: only 8-bit PNGs are decoded (got depth {depth})")
    if interlace:
        raise PngError(f"{path}: interlaced PNGs are not decoded")
    if colour not in _CHANNELS or colour == 3:
        raise PngError(f"{path}: unsupported colour type {colour}")
    if not idat:
        raise PngError(f"{path}: no IDAT data")

    channels = _CHANNELS[colour]
    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    expected = (stride + 1) * height
    if len(raw) < expected:
        raise PngError(f"{path}: truncated image data "
                       f"({len(raw)} bytes, expected {expected})")

    out = bytearray(stride * height)
    prev = bytearray(stride)
    src = 0
    for y in range(height):
        ftype = raw[src]
        src += 1
        line = bytearray(raw[src:src + stride])
        src += stride
        if ftype == 1:  # Sub
            for x in range(channels, stride):
                line[x] = (line[x] + line[x - channels]) & 0xFF
        elif ftype == 2:  # Up
            for x in range(stride):
                line[x] = (line[x] + prev[x]) & 0xFF
        elif ftype == 3:  # Average
            for x in range(stride):
                a = line[x - channels] if x >= channels else 0
                line[x] = (line[x] + ((a + prev[x]) >> 1)) & 0xFF
        elif ftype == 4:  # Paeth
            for x in range(stride):
                a = line[x - channels] if x >= channels else 0
                c = prev[x - channels] if x >= channels else 0
                line[x] = (line[x] + _paeth(a, prev[x], c)) & 0xFF
        elif ftype != 0:
            raise PngError(f"{path}: unknown filter type {ftype} on row {y}")
        out[y * stride:(y + 1) * stride] = line
        prev = line

    return PngImage(width=width, height=height, channels=channels,
                    pixels=list(out))


# --- measure ---------------------------------------------------------------
def _luma(rgb: Tuple[int, int, int]) -> float:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def png_stats(path: str, bg_tolerance: int = 12) -> PngStats:
    """Decode `path` and measure it against the publish thresholds."""
    img = load_png(path)
    n = img.pixel_count
    bg = img.rgb(0)  # top-left corner: the renderer's background
    fg = 0
    lumas: List[float] = []
    fg_shades = set()
    colours = set()
    total = 0.0
    for i in range(n):
        c = img.rgb(i)
        colours.add(c)
        l = _luma(c)
        lumas.append(l)
        total += l
        if (abs(c[0] - bg[0]) > bg_tolerance or abs(c[1] - bg[1]) > bg_tolerance
                or abs(c[2] - bg[2]) > bg_tolerance):
            fg += 1
            fg_shades.add(int(l) >> 2)  # 4-level buckets: real shading, not dither

    mean = total / n if n else 0.0
    var = sum((l - mean) ** 2 for l in lumas) / n if n else 0.0
    silhouette = fg / n if n else 0.0

    failures: List[str] = []
    if silhouette < MIN_SILHOUETTE:
        failures.append("blank: silhouette %.3f < %.3f" % (silhouette, MIN_SILHOUETTE))
    if silhouette > MAX_SILHOUETTE:
        failures.append("overflowing/flat: silhouette %.3f > %.3f"
                        % (silhouette, MAX_SILHOUETTE))
    if var < MIN_VARIANCE:
        failures.append("flat fill: luminance variance %.1f < %.1f" % (var, MIN_VARIANCE))
    if len(fg_shades) < MIN_SHADES:
        failures.append("unshaded: %d distinct shades < %d" % (len(fg_shades), MIN_SHADES))
    if mean < 6.0:
        failures.append("black image: mean luminance %.1f" % mean)

    return PngStats(
        width=img.width, height=img.height, silhouette=silhouette, variance=var,
        shades=len(fg_shades), mean_luma=mean, distinct_colours=len(colours),
        ok=not failures, failures=failures,
    )


def validate_png(path: str) -> Dict[str, object]:
    """png_stats as a plain dict; a decode failure is reported, never raised."""
    try:
        return png_stats(path).to_dict()
    except (PngError, OSError) as exc:
        return {"ok": False, "failures": [str(exc)], "width": 0, "height": 0}
