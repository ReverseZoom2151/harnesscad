"""Continuous B-Rep tokenization for surface/edge token sequences.

This module converts a B-Rep into two sequences of continuous-valued tokens
before autoregressive generation:

  * a **surface token** = the 6 axis-aligned bounding-box coordinates enclosing
    the surface (its topology characteristics) concatenated with a per-surface
    hidden feature vector (from a surface VAE);
  * an **edge token** = the edge bounding box concatenated with its two adjacent
    vertices (start point and end point). The vertex information is integrated
    into the edge tokens, so no separate vertex tokens are generated.

After tokenization the tokens are ordered per family in ascending order of the
3D coordinate values x1, y1, z1, x2, y2, z2 within the bounding boxes, yielding
the ordered sequences ``S`` and ``E``.

The learned surface/edge VAE encoders are external; the deterministic packing,
ordering and quantization are implemented here. The hidden feature slots are
carried opaquely so the layout matches the token even when the VAE output is
supplied from elsewhere.
"""

from __future__ import annotations

Point = tuple[float, float, float]
Box = tuple[float, float, float, float, float, float]


def bounding_box(points: tuple[Point, ...]) -> Box:
    """Axis-aligned box ``(x1, y1, z1, x2, y2, z2)`` enclosing ``points``."""
    if not points:
        raise ValueError("bounding_box needs at least one point")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


def edge_token(start: Point, end: Point,
               box: Box | None = None,
               features: tuple[float, ...] = ()) -> tuple[float, ...]:
    """Pack one edge token: ``bbox(6) + start(3) + end(3) + features``.

    The bounding box defaults to the box spanned by the two endpoints when not
    supplied (curved edges pass their own box). No separate vertex token is
    emitted -- the two adjacent vertices live inside the edge token.
    """
    if box is None:
        box = bounding_box((start, end))
    if len(box) != 6:
        raise ValueError("edge box must have 6 coordinates")
    return (tuple(float(v) for v in box)
            + tuple(float(v) for v in start)
            + tuple(float(v) for v in end)
            + tuple(float(v) for v in features))


def surface_token(box: Box, features: tuple[float, ...] = ()) -> tuple[float, ...]:
    """Pack one surface token: ``bbox(6) + features``."""
    if len(box) != 6:
        raise ValueError("surface box must have 6 coordinates")
    return tuple(float(v) for v in box) + tuple(float(v) for v in features)


def _order_key(token: tuple[float, ...]) -> tuple[float, ...]:
    # Sort by (x1, y1, z1, x2, y2, z2), i.e. the leading box coordinates,
    # with the full token as a deterministic tie-break.
    return tuple(token[:6]) + tuple(token)


def order_tokens(tokens: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    """Order a token family ascending by leading bounding-box coordinates."""
    return tuple(sorted(tokens, key=_order_key))


def quantize(value: float, bits: int, lo: float = 0.0, hi: float = 1.0) -> int:
    """Uniformly quantize ``value`` in ``[lo, hi]`` to a ``bits``-bit level."""
    if bits <= 0:
        raise ValueError("bits must be positive")
    if hi <= lo:
        raise ValueError("hi must exceed lo")
    levels = (1 << bits) - 1
    t = (value - lo) / (hi - lo)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return int(round(t * levels))


def dequantize(level: int, bits: int, lo: float = 0.0, hi: float = 1.0) -> float:
    levels = (1 << bits) - 1
    return lo + (level / levels) * (hi - lo)


def quantize_token(token: tuple[float, ...], bits: int,
                   lo: float = 0.0, hi: float = 1.0) -> tuple[int, ...]:
    """Quantize every component of a token (the 4-bit validity check)."""
    return tuple(quantize(v, bits, lo, hi) for v in token)
