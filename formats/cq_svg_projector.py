"""Deterministic orthographic-projection SVG exporter for wireframe geometry.

CadQuery's ``occ_impl/exporters/svg.py`` renders a shape to SVG by (1) running
OCCT hidden-line removal (``HLRBRep_Algo``) to split edges into visible/hidden
sets, (2) projecting them along a view direction, (3) discretising each edge to
a polyline, and (4) fitting the 2D bounding box into the canvas and emitting SVG
``<path>`` elements.  Step 1 needs the OCCT kernel, but steps 2-4 -- the
orthographic projection, the bounding-box-fit transform, and the SVG path/document
emission -- are pure deterministic geometry, reproduced here.

The harness has AMF/DXF codecs but no SVG projector (the ``svg`` hits in
``drawings/`` are SketchGraphs point datasets, not an exporter).  This module
adds the projection + emission layer:

* :func:`project_point` -- orthographic projection of a 3D point onto a camera
  plane whose out-of-screen axis is the view direction (deterministic
  right/up basis).
* :func:`path_data` -- an ``M``/``L`` SVG path string for a projected polyline.
* :func:`fit_transform` -- the exact ``unitScale`` / translate computation from
  ``getSVG`` (both the fit-to-canvas and the fit-to-width/height branches).
* :func:`get_svg` -- assembles the full SVG document, honouring visible vs
  hidden edge styling, margins, stroke width/colour, and ``showHidden``.

Callers supply edges already discretised to 3D polylines (and, optionally, a
visible/hidden split from an external HLR pass); the numerics here are exact up
to IEEE-754 rounding.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "Vec3",
    "Vec2",
    "camera_basis",
    "project_point",
    "project_polyline",
    "path_data",
    "bounding_box_2d",
    "fit_transform",
    "get_svg",
]

Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]

_SVG_TEMPLATE = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <g transform="scale({unitScale}, -{unitScale}) translate({xTranslate},{yTranslate})" stroke-width="{strokeWidth}" fill="none">
    <g stroke="rgb({hiddenColor})" fill="none" stroke-dasharray="{strokeWidth},{strokeWidth}">
{hiddenContent}    </g>
    <g stroke="rgb({strokeColor})" fill="none">
{visibleContent}    </g>
  </g>
</svg>
"""

_PATH_TEMPLATE = '      <path d="{d}" />\n'


# --------------------------------------------------------------------------
# vector helpers
# --------------------------------------------------------------------------

def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(_dot(a, a))


def _normalize(a: Sequence[float]) -> Vec3:
    n = _norm(a)
    if n == 0.0:
        raise ValueError("cannot normalize a null vector")
    return (a[0] / n, a[1] / n, a[2] / n)


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------

def camera_basis(view_dir: Sequence[float]) -> Tuple[Vec3, Vec3, Vec3]:
    """Return a right-handed ``(right, up, out)`` basis for a view direction.

    ``out`` is the normalized view direction (out of the screen toward the
    viewer); ``right`` and ``up`` span the projection plane.  The choice is
    deterministic: ``up`` prefers world +Z, falling back to +Y when the view is
    (anti)parallel to +Z.
    """
    out = _normalize(view_dir)
    up_hint: Vec3 = (0.0, 0.0, 1.0)
    if abs(_dot(out, up_hint)) > 0.999:
        up_hint = (0.0, 1.0, 0.0)
    right = _normalize(_cross(up_hint, out))
    up = _cross(out, right)  # already unit (out, right orthonormal)
    return right, up, out


def project_point(point: Sequence[float], view_dir: Sequence[float]) -> Vec2:
    """Orthographically project a 3D point onto the view plane -> ``(u, v)``."""
    right, up, _ = camera_basis(view_dir)
    return (_dot(point, right), _dot(point, up))


def project_polyline(
    polyline: Sequence[Sequence[float]], view_dir: Sequence[float]
) -> List[Vec2]:
    """Project every vertex of a 3D polyline to 2D."""
    right, up, _ = camera_basis(view_dir)
    return [(_dot(p, right), _dot(p, up)) for p in polyline]


def _fmt(x: float) -> str:
    # deterministic, compact numeric formatting
    return repr(round(x, 6) + 0.0)


def path_data(polyline2d: Sequence[Vec2]) -> str:
    """Build an SVG path ``d`` string (``M x,y L x,y ...``) from a 2D polyline."""
    if not polyline2d:
        return ""
    it = iter(polyline2d)
    x, y = next(it)
    parts = [f"M{_fmt(x)},{_fmt(y)} "]
    for x, y in it:
        parts.append(f"L{_fmt(x)},{_fmt(y)} ")
    return "".join(parts)


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

def bounding_box_2d(
    polylines: Sequence[Sequence[Vec2]],
) -> Tuple[float, float, float, float]:
    """Return ``(xmin, ymin, xmax, ymax)`` over all 2D points."""
    xs: List[float] = []
    ys: List[float] = []
    for pl in polylines:
        for x, y in pl:
            xs.append(x)
            ys.append(y)
    if not xs:
        raise ValueError("no points to bound")
    return (min(xs), min(ys), max(xs), max(ys))


def fit_transform(
    bbox: Tuple[float, float, float, float],
    width: Optional[float],
    height: Optional[float],
    marginLeft: float = 200.0,
    marginTop: float = 20.0,
) -> Tuple[float, float, float, float, float]:
    """Reproduce ``getSVG``'s canvas fit.

    Returns ``(width, height, unitScale, xTranslate, yTranslate)``.  When both
    width and height are given, the drawing is scaled to fill 75% of the canvas
    (the reference ``bb_scale``); when one is ``None`` it is derived from the
    aspect ratio and the drawing is scaled to the available width.
    """
    xmin, ymin, xmax, ymax = bbox
    xlen = xmax - xmin
    ylen = ymax - ymin
    if xlen == 0.0 or ylen == 0.0:
        raise ValueError("degenerate 2D bounding box")

    if width is None or height is None:
        if width is None:
            height = float(height)
            width = (height - 2.0 * marginTop) * (xlen / ylen) + 2.0 * marginLeft
        else:
            width = float(width)
            height = (width - 2.0 * marginLeft) * (ylen / xlen) + 2.0 * marginTop
        unitScale = (width - 2.0 * marginLeft) / xlen
    else:
        width = float(width)
        height = float(height)
        bb_scale = 0.75
        unitScale = min(width / xlen * bb_scale, height / ylen * bb_scale)

    xTranslate = (0.0 - xmin) + marginLeft / unitScale
    yTranslate = (0.0 - ymax) - marginTop / unitScale
    return width, height, unitScale, xTranslate, yTranslate


# --------------------------------------------------------------------------
# document assembly
# --------------------------------------------------------------------------

def get_svg(
    visible_edges: Sequence[Sequence[Sequence[float]]],
    hidden_edges: Optional[Sequence[Sequence[Sequence[float]]]] = None,
    opts: Optional[dict] = None,
) -> str:
    """Project 3D wireframe edges and emit a complete SVG document.

    ``visible_edges`` / ``hidden_edges`` are sequences of 3D polylines (each a
    sequence of ``(x, y, z)``).  ``opts`` mirrors the reference option keys:
    ``width``, ``height``, ``marginLeft``, ``marginTop``, ``projectionDir``,
    ``strokeWidth`` (-1 = auto = ``1/unitScale``), ``strokeColor``,
    ``hiddenColor``, ``showHidden``.
    """
    d = {
        "width": 800,
        "height": 240,
        "marginLeft": 200,
        "marginTop": 20,
        "projectionDir": (-1.75, 1.1, 5.0),
        "strokeWidth": -1.0,
        "strokeColor": (0, 0, 0),
        "hiddenColor": (160, 160, 160),
        "showHidden": True,
    }
    if opts:
        d.update(opts)

    hidden_edges = hidden_edges or []
    view = d["projectionDir"]

    vis2d = [project_polyline(e, view) for e in visible_edges]
    hid2d = [project_polyline(e, view) for e in hidden_edges]

    all2d = vis2d + hid2d
    if not all2d:
        raise ValueError("no edges to export")

    bbox = bounding_box_2d(all2d)
    width, height, unitScale, xTranslate, yTranslate = fit_transform(
        bbox,
        d["width"],
        d["height"],
        float(d["marginLeft"]),
        float(d["marginTop"]),
    )

    strokeWidth = float(d["strokeWidth"])
    if strokeWidth == -1.0:
        strokeWidth = 1.0 / unitScale

    visibleContent = "".join(
        _PATH_TEMPLATE.format(d=path_data(pl)) for pl in vis2d if pl
    )
    hiddenContent = ""
    if d["showHidden"]:
        hiddenContent = "".join(
            _PATH_TEMPLATE.format(d=path_data(pl)) for pl in hid2d if pl
        )

    return _SVG_TEMPLATE.format(
        width=_fmt(width),
        height=_fmt(height),
        unitScale=_fmt(unitScale),
        strokeWidth=_fmt(strokeWidth),
        strokeColor=",".join(str(int(c)) for c in d["strokeColor"]),
        hiddenColor=",".join(str(int(c)) for c in d["hiddenColor"]),
        xTranslate=_fmt(xTranslate),
        yTranslate=_fmt(yTranslate),
        visibleContent=visibleContent,
        hiddenContent=hiddenContent,
    )
