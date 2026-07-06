"""Solid-Profile-Loop (S-P-L) hierarchical tree for HNC-CAD (Xu et al., ICML 2023).

"Hierarchical Neural Coding for Controllable CAD Model Generation" represents the
high-level design concept of a sketch-and-extrude CAD model as a *three-level tree*
(paper Sec. 3, Fig. 3): (S)olid -> (P)rofile -> (L)oop. The distinctive idea -- and
what sets it apart from a flat command list or the raw curve-at-every-level tree of
:mod:`reconstruction.geofusion_hierarchy` -- is that the two upper levels are
**bounding-box abstractions** of the arrangement of the level below, not the raw
geometry:

* **Loop (L)** -- the leaf. A series of x-y coordinates separated by ``<SEP>`` tokens
  (paper Eq. 1). The *curve type is identified by the number of points* (Willis et
  al. 2021a): a line = 2 points (start, end), an arc = 3 points (start, mid, end), a
  circle = 4 equally-spaced points. Curves are sorted so the initial curve is the one
  with the smallest starting-point coordinate, followed by its connected curve in
  counter-clockwise order.

* **Profile (P)** -- above the leaf. A series of 2D bounding-box parameters
  ``(x, y, w, h)`` of the loops within the sketch plane (paper Eq. 2), where
  ``(x, y)`` is the bottom-left corner and ``(w, h)`` the size. Ordered by sorting the
  bottom-left corners of all boxes in ascending order.

* **Solid (S)** -- the root. A series of 3D bounding-box parameters
  ``(x, y, z, w, h, d)`` capturing the arrangement of the extruded profiles (paper
  Eq. 3), ordered by sorting the bottom-left corners ascending.

Numeric fields are 6-bit quantized (paper Sec. 4): a coordinate is one of 64 levels
and ``<SEP>`` occupies one extra dimension, giving a 65-D one-hot token.

This module is the fully deterministic, network-agnostic core of that representation:
the typed tree, the curve-type-by-point-count rule, loop-curve canonical ordering,
bottom-left bounding-box abstraction with the ascending sort, and 6-bit / 65-D token
encoding. The learned VQ-VAE and transformers are out of scope.

Pure stdlib, no wall-clock, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- 6-bit quantization / 65-D token ---------------------------------------
QUANT_BITS = 6
QUANT_LEVELS = 1 << QUANT_BITS   # 64 coordinate levels
SEP_TOKEN = QUANT_LEVELS         # index 64 -> the <SEP> separator dimension
TOKEN_DIM = QUANT_LEVELS + 1     # 65-D one-hot (paper Sec. 4)


def quantize6(value: float) -> int:
    """6-bit uniform quantization of ``value in [0, 1]`` into ``[0, 63]``.

    Values outside the unit interval are clamped, matching the paper's fixed 6-bit
    coordinate tokenization.
    """
    if value <= 0.0:
        return 0
    if value >= 1.0:
        return QUANT_LEVELS - 1
    return int(round(value * (QUANT_LEVELS - 1)))


def dequantize6(code: int) -> float:
    """Inverse of :func:`quantize6`: integer code in ``[0, 63]`` -> ``[0, 1]``."""
    if not 0 <= code < QUANT_LEVELS:
        raise ValueError(f"code {code} outside 6-bit range [0, {QUANT_LEVELS - 1}]")
    return code / (QUANT_LEVELS - 1)


def token_onehot(index: int) -> tuple[int, ...]:
    """65-D one-hot for a coordinate code (0..63) or the ``<SEP>`` token (64)."""
    if not 0 <= index <= SEP_TOKEN:
        raise ValueError(f"token index {index} outside [0, {SEP_TOKEN}]")
    vec = [0] * TOKEN_DIM
    vec[index] = 1
    return tuple(vec)


# --- curve-type identification (by point count) ----------------------------
LINE = "line"
ARC = "arc"
CIRCLE = "circle"

_TYPE_BY_NPOINTS = {2: LINE, 3: ARC, 4: CIRCLE}
_NPOINTS_BY_TYPE = {LINE: 2, ARC: 3, CIRCLE: 4}


def curve_type(n_points: int) -> str:
    """Identify a curve's type from its number of points (Willis et al. 2021a).

    line = 2 (start, end); arc = 3 (start, mid, end); circle = 4 (equally spaced).
    """
    try:
        return _TYPE_BY_NPOINTS[n_points]
    except KeyError:
        raise ValueError(f"no curve type has {n_points} points (expected 2, 3 or 4)")


# --- typed tree -------------------------------------------------------------
Point = tuple[float, float]


@dataclass(frozen=True)
class Curve:
    """A loop primitive as a tuple of (x, y) points; its type is the point count.

    2 points = line, 3 = arc, 4 = circle. Coordinates are floats in ``[0, 1]``
    (quantized on demand by :func:`loop_token_indices`).
    """

    points: tuple[Point, ...]

    def __post_init__(self):
        if len(self.points) not in _TYPE_BY_NPOINTS:
            raise ValueError(f"curve needs 2, 3 or 4 points, got {len(self.points)}")

    @property
    def type(self) -> str:
        return curve_type(len(self.points))

    @property
    def start(self) -> Point:
        return self.points[0]

    @property
    def end(self) -> Point:
        return self.points[-1]


@dataclass(frozen=True)
class Loop:
    """A closed path of connected curves (Eq. 1)."""

    curves: tuple[Curve, ...]


@dataclass(frozen=True)
class ProfileBBox:
    """One profile node: the 2D bounding boxes of its loops (Eq. 2).

    ``boxes`` is a tuple of ``(x, y, w, h)`` with ``(x, y)`` the bottom-left corner.
    """

    boxes: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class SolidNode:
    """The solid root: the 3D bounding boxes of its extruded profiles (Eq. 3).

    ``boxes`` is a tuple of ``(x, y, z, w, h, d)`` with ``(x, y, z)`` the bottom-left
    corner and ``(w, h, d)`` the dimension.
    """

    boxes: tuple[tuple[float, float, float, float, float, float], ...]


@dataclass(frozen=True)
class SPLTree:
    """A full Solid-Profile-Loop tree: one solid, its profiles, and their loops.

    ``profiles[i]`` corresponds to solid box ``i``; ``loops[i]`` is the tuple of
    loops that make up profile ``i``.
    """

    solid: SolidNode
    profiles: tuple[ProfileBBox, ...]
    loops: tuple[tuple[Loop, ...], ...]


# --- loop-curve canonical ordering -----------------------------------------
def _min_point_index(curves: tuple[Curve, ...]) -> int:
    """Index of the curve whose starting point is smallest (lexicographic x, y)."""
    best = 0
    for i in range(1, len(curves)):
        if curves[i].start < curves[best].start:
            best = i
    return best


def sort_loop_curves(loop: Loop) -> Loop:
    """Reorder a loop's connected curves to begin at the smallest starting point.

    Following the paper: the initial curve is the one with the smallest starting-
    point coordinate, and the sequence then follows the connected (counter-clockwise)
    chain. A closed chain is simply rotated so it starts at that curve; a single
    curve (e.g. a circle) is returned unchanged.
    """
    curves = loop.curves
    if len(curves) <= 1:
        return loop
    k = _min_point_index(curves)
    rotated = curves[k:] + curves[:k]
    return Loop(rotated)


def loop_token_indices(loop: Loop) -> tuple[int, ...]:
    """Serialize a loop to 6-bit token indices with ``<SEP>`` between curves (Eq. 1).

    Each point contributes its quantized x then y; a ``<SEP>`` (index 64) separates
    consecutive curves. The loop is canonicalized first via :func:`sort_loop_curves`.
    """
    canon = sort_loop_curves(loop)
    out: list[int] = []
    for i, c in enumerate(canon.curves):
        if i > 0:
            out.append(SEP_TOKEN)
        for (x, y) in c.points:
            out.append(quantize6(x))
            out.append(quantize6(y))
    return tuple(out)


# --- bounding-box abstraction ----------------------------------------------
def loop_bbox(loop: Loop) -> tuple[float, float, float, float]:
    """2D ``(x, y, w, h)`` bounding box of a loop (bottom-left corner + size)."""
    xs = [p[0] for c in loop.curves for p in c.points]
    ys = [p[1] for c in loop.curves for p in c.points]
    if not xs:
        raise ValueError("empty loop has no bounding box")
    x0, y0 = min(xs), min(ys)
    return (x0, y0, max(xs) - x0, max(ys) - y0)


def profile_from_loops(loops: tuple[Loop, ...]) -> ProfileBBox:
    """Build a profile node from its loops (Eq. 2): the 2D boxes sorted ascending.

    The boxes are sorted by their bottom-left corner ``(x, y)`` in ascending order,
    matching the paper's canonical profile ordering.
    """
    boxes = sorted(loop_bbox(lp) for lp in loops)
    return ProfileBBox(tuple(boxes))


def solid_from_profiles(
    profiles: tuple[ProfileBBox, ...],
    extrusions: tuple[tuple[float, float], ...],
) -> SolidNode:
    """Build the solid node (Eq. 3) from profiles and their ``(z, depth)`` extrusions.

    Each profile's overall 2D extent (the union of its loop boxes) is combined with
    the extrusion base ``z`` and ``depth`` to form a 3D box ``(x, y, z, w, h, d)``.
    Boxes are sorted by their bottom-left corner ``(x, y, z)`` ascending.
    """
    if len(profiles) != len(extrusions):
        raise ValueError("one (z, depth) extrusion required per profile")
    boxes: list[tuple[float, float, float, float, float, float]] = []
    for prof, (z, depth) in zip(profiles, extrusions):
        if not prof.boxes:
            raise ValueError("profile has no boxes")
        x0 = min(b[0] for b in prof.boxes)
        y0 = min(b[1] for b in prof.boxes)
        x1 = max(b[0] + b[2] for b in prof.boxes)
        y1 = max(b[1] + b[3] for b in prof.boxes)
        boxes.append((x0, y0, z, x1 - x0, y1 - y0, depth))
    boxes.sort()
    return SolidNode(tuple(boxes))
