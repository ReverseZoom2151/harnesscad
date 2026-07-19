"""cad2program_view_lifting — lift matched orthographic views to 3D boxes.

A 3D parametric model can be reconstructed from the orthographic views of a 2D
CAD drawing.  A learned VLM reads the raster drawing; but once each view has been
reduced to a set of axis-aligned rectangles (one per prismatic component),
*lifting* those rectangles back to a 3D axis-aligned box is a classical,
deterministic geometry operation from the "recover 3D solids from orthographic
projections" line of work.

This module implements that lifting for box-composed (prismatic) solids, which is
distinct from :mod:`drawings.creft_view_consistency` (which only *checks* whether
three views agree) — here we actually *construct* the 3D box(es).

Conventions match :mod:`drawings.creft_projection` (Z-up, third-angle)::

    Front view — X-Z plane:  h = X, v = Z   (looking along -Y)
    Top view   — X-Y plane:  h = X, v = Y   (looking along -Z)
    Side view  — Y-Z plane:  h = Y, v = Z   (looking along -X)

A single prismatic solid projects to one rectangle per view; the three rectangles
share extents pairwise (Front/Top share X, Front/Side share Z, Top/Side share Y).
Given the three rectangles, the 3D box is uniquely determined.  We also provide
the degenerate two-view *extrusion* case: a profile rectangle in one view plus a
depth (thickness) rectangle in an orthogonal view gives a prism.

Pure stdlib, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from harnesscad.domain.reconstruction.translate.shape_program import Bbox

FRONT = "front"
TOP = "top"
SIDE = "side"
VIEW_NAMES: Tuple[str, ...] = (FRONT, TOP, SIDE)

# Which 3D axis each (view, 2D-axis) pair measures.
_AXIS_OF: Dict[Tuple[str, str], str] = {
    (FRONT, "h"): "x", (FRONT, "v"): "z",
    (TOP, "h"): "x", (TOP, "v"): "y",
    (SIDE, "h"): "y", (SIDE, "v"): "z",
}


@dataclass(frozen=True)
class Rect:
    """An axis-aligned rectangle in a view: (h0, v0) low corner, (hw, vh) size."""

    h0: float
    v0: float
    hw: float
    vh: float

    @property
    def h1(self) -> float:
        return self.h0 + self.hw

    @property
    def v1(self) -> float:
        return self.v0 + self.vh


def parse_rect(spec: Sequence[float]) -> Rect:
    """Parse a 4-number ``(h0, v0, hw, vh)`` rectangle spec."""
    h0, v0, hw, vh = spec
    if hw < 0 or vh < 0:
        raise ValueError("rectangle extents must be non-negative")
    return Rect(float(h0), float(v0), float(hw), float(vh))


def parse_view(rect_specs: Sequence[Sequence[float]]) -> List[Rect]:
    """Parse a whole view: a list of rectangle specs -> list of :class:`Rect`."""
    return [parse_rect(s) for s in rect_specs]


# --------------------------------------------------------------------------- #
# Three-view lifting
# --------------------------------------------------------------------------- #

def _axis_interval(rect: Rect, view: str, axis2d: str) -> Tuple[float, float]:
    if axis2d == "h":
        return (rect.h0, rect.h1)
    return (rect.v0, rect.v1)


def lift_three_views(front: Rect, top: Rect, side: Rect,
                     tol: float = 1e-6) -> Bbox:
    """Lift one rectangle per view into a single 3D axis-aligned box.

    The three views must agree on the shared extents (Front/Top share the X
    extent, Front/Side share Z, Top/Side share Y) to within ``tol``; otherwise a
    :class:`ValueError` is raised (the projections are inconsistent).
    """
    x_front = _axis_interval(front, FRONT, "h")
    x_top = _axis_interval(top, TOP, "h")
    z_front = _axis_interval(front, FRONT, "v")
    z_side = _axis_interval(side, SIDE, "v")
    y_top = _axis_interval(top, TOP, "v")
    y_side = _axis_interval(side, SIDE, "h")

    def agree(a: Tuple[float, float], b: Tuple[float, float], name: str) -> Tuple[float, float]:
        if abs(a[0] - b[0]) > tol or abs(a[1] - b[1]) > tol:
            raise ValueError(f"inconsistent {name} extent: {a} vs {b}")
        return a

    x0, x1 = agree(x_front, x_top, "X")
    z0, z1 = agree(z_front, z_side, "Z")
    y0, y1 = agree(y_top, y_side, "Y")

    sx, sy, sz = x1 - x0, y1 - y0, z1 - z0
    return Bbox((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0,
                sx, sy, sz, 0)


def check_three_view_consistency(front: Rect, top: Rect, side: Rect,
                                 tol: float = 1e-6) -> bool:
    """True iff the three view rectangles share their pairwise extents."""
    try:
        lift_three_views(front, top, side, tol)
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
# Two-view extrusion lifting
# --------------------------------------------------------------------------- #

def extrude_profile(profile: Rect, profile_view: str,
                    depth: float, tol: float = 1e-6) -> Bbox:
    """Extrude a 2D profile rectangle by ``depth`` along its missing axis.

    ``profile_view`` names the plane the profile lives in.  The extrusion axis is
    the one axis *not* spanned by that view (Front -> depth along Y, Top -> depth
    along Z, Side -> depth along X).  The profile is centered on the missing axis
    at the origin of that axis, matching a symmetric prism about 0; callers that
    need an offset can translate the resulting box.
    """
    if depth < -tol:
        raise ValueError("depth must be non-negative")
    if profile_view == FRONT:      # spans X (h) and Z (v); extrude along Y
        sx, sz = profile.hw, profile.vh
        cx = profile.h0 + sx / 2.0
        cz = profile.v0 + sz / 2.0
        return Bbox(cx, depth / 2.0, cz, sx, depth, sz, 0)
    if profile_view == TOP:        # spans X (h) and Y (v); extrude along Z
        sx, sy = profile.hw, profile.vh
        cx = profile.h0 + sx / 2.0
        cy = profile.v0 + sy / 2.0
        return Bbox(cx, cy, depth / 2.0, sx, sy, depth, 0)
    if profile_view == SIDE:       # spans Y (h) and Z (v); extrude along X
        sy, sz = profile.hw, profile.vh
        cy = profile.h0 + sy / 2.0
        cz = profile.v0 + sz / 2.0
        return Bbox(depth / 2.0, cy, cz, depth, sy, sz, 0)
    raise ValueError(f"unknown profile view {profile_view!r}")


# --------------------------------------------------------------------------- #
# Multi-component lifting
# --------------------------------------------------------------------------- #

def lift_matched_components(front: Sequence[Rect], top: Sequence[Rect],
                            side: Sequence[Rect],
                            correspondence: Sequence[Tuple[int, int, int]],
                            tol: float = 1e-6) -> List[Bbox]:
    """Lift several prismatic components given an explicit view correspondence.

    ``correspondence`` is a list of ``(front_idx, top_idx, side_idx)`` triples
    telling which rectangle in each view belongs to the same 3D component.  The
    matching itself (which rectangle is which) is the hard, image-understanding
    problem the VLM solves; here we deterministically construct the boxes once the
    correspondence is known.
    """
    boxes: List[Bbox] = []
    for fi, ti, si in correspondence:
        boxes.append(lift_three_views(front[fi], top[ti], side[si], tol))
    return boxes
