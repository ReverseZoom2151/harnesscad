"""cadmcp_drawing_commands -- host-independent 2D CAD drawing-command geometry.

Transferred from the ``cad_controller.py`` of CAD-MCP (a Model-Context-Protocol
server that drives AutoCAD / GstarCAD / ZWCAD over a Windows COM Automation
API). The COM host is out of scope, but every ``draw_*`` method in that
controller performs a small, exact, host-independent *geometry* computation
before it hands numbers to ``ModelSpace.Add*``:

  * **line**       -- two endpoints, promoted to 3D;
  * **circle**     -- centre + radius (radius must be > 0);
  * **arc**        -- centre + radius + start/end angles converted deg -> rad
                      (the COM ``AddArc`` convention takes radians);
  * **ellipse**    -- the ``AddEllipse`` convention: a *major-axis endpoint
                      vector* ``(a*cos t, a*sin t, 0)`` from the rotation angle
                      plus a *radius ratio* ``minor / major``;
  * **rectangle**  -- two diagonal corners expanded into a closed 5-vertex
                      polyline ``[c, (x2,y1), (x2,y2), (x1,y2), c]``;
  * **polyline**   -- points promoted to 3D, closed only when there are > 2 of
                      them (the controller's ``Closed`` guard);
  * **text / hatch** -- anchor promotion; a hatch needs >= 3 boundary points;
  * **lineweight** -- snapped to the fixed AutoCAD standard-lineweight set
                      (values outside it fall back to 0).

This module recomputes exactly those numbers and returns each drawing command as
a plain :class:`DrawingEntity` (kind + computed geometry + layer/color/
lineweight), with no CAD host. It is the deterministic counterpart to the 3D
CISP op catalog in ``surfaces/mcp`` -- that surface consumes structured 3D ops;
this one is a 2D draughting-command vocabulary. Stdlib-only, deterministic, no
wall clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float, float]

# AutoCAD's fixed set of valid lineweight codes (hundredths of a millimetre).
VALID_LINEWEIGHTS: Tuple[int, ...] = (
    0, 5, 9, 13, 15, 18, 20, 25, 30, 35, 40, 50, 53, 60, 70, 80, 90, 100,
    106, 120, 140, 158, 200, 211,
)


class DrawingCommandError(ValueError):
    """Raised when a drawing command's parameters are geometrically invalid."""


def ensure_3d(point: Sequence[float]) -> Point:
    """Promote a 2- or 3-tuple to an exact ``(x, y, z)`` triple (z defaults 0)."""
    seq = tuple(float(v) for v in point)
    if len(seq) == 2:
        return (seq[0], seq[1], 0.0)
    if len(seq) == 3:
        return (seq[0], seq[1], seq[2])
    raise DrawingCommandError(f"point needs 2 or 3 coordinates, got {len(seq)}")


def validate_lineweight(lineweight: Optional[float]) -> Optional[int]:
    """Return a valid lineweight code, snapping any out-of-set value to 0.

    Mirrors ``CADController.validate_lineweight``: ``None`` stays ``None`` (use
    the layer default); a value already in :data:`VALID_LINEWEIGHTS` is kept;
    anything else falls back to 0.
    """
    if lineweight is None:
        return None
    code = int(lineweight)
    return code if code in VALID_LINEWEIGHTS else 0


@dataclass(frozen=True)
class DrawingEntity:
    """One resolved 2D drawing command: kind + computed geometry + style."""

    kind: str
    geometry: Dict[str, object]
    layer: Optional[str] = None
    color: Optional[int] = None
    lineweight: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "geometry": self.geometry,
            "layer": self.layer,
            "color": self.color,
            "lineweight": self.lineweight,
        }


def _style(layer, color, lineweight) -> dict:
    return {
        "layer": layer,
        "color": None if color is None else int(color),
        "lineweight": validate_lineweight(lineweight),
    }


def line(start: Sequence[float], end: Sequence[float], *, layer=None,
         color=None, lineweight=None) -> DrawingEntity:
    """A straight segment between two endpoints (promoted to 3D)."""
    return DrawingEntity(
        "line", {"start": ensure_3d(start), "end": ensure_3d(end)},
        **_style(layer, color, lineweight))


def circle(center: Sequence[float], radius: float, *, layer=None, color=None,
           lineweight=None) -> DrawingEntity:
    """A circle; ``radius`` must be strictly positive."""
    r = float(radius)
    if r <= 0:
        raise DrawingCommandError(f"circle radius must be > 0, got {r}")
    return DrawingEntity(
        "circle", {"center": ensure_3d(center), "radius": r},
        **_style(layer, color, lineweight))


def arc(center: Sequence[float], radius: float, start_angle: float,
        end_angle: float, *, layer=None, color=None,
        lineweight=None) -> DrawingEntity:
    """An arc; input angles are degrees, stored also as ``AddArc`` radians."""
    r = float(radius)
    if r <= 0:
        raise DrawingCommandError(f"arc radius must be > 0, got {r}")
    sa, ea = float(start_angle), float(end_angle)
    geom = {
        "center": ensure_3d(center),
        "radius": r,
        "start_angle_deg": sa,
        "end_angle_deg": ea,
        "start_angle_rad": math.radians(sa),
        "end_angle_rad": math.radians(ea),
    }
    return DrawingEntity("arc", geom, **_style(layer, color, lineweight))


def ellipse(center: Sequence[float], major_axis: float, minor_axis: float,
            rotation: float = 0.0, *, layer=None, color=None,
            lineweight=None) -> DrawingEntity:
    """An ellipse in the ``AddEllipse`` convention.

    Emits the *major-axis endpoint vector* ``(a*cos t, a*sin t, 0)`` (relative to
    the centre) plus the *radius ratio* ``minor / major`` that the COM API takes.
    """
    a, b = float(major_axis), float(minor_axis)
    if a <= 0 or b <= 0:
        raise DrawingCommandError(f"ellipse axes must be > 0, got {a}, {b}")
    t = math.radians(float(rotation))
    geom = {
        "center": ensure_3d(center),
        "major_axis": a,
        "minor_axis": b,
        "rotation_deg": float(rotation),
        "major_axis_vector": (a * math.cos(t), a * math.sin(t), 0.0),
        "ratio": b / a,
    }
    return DrawingEntity("ellipse", geom, **_style(layer, color, lineweight))


def polyline(points: Sequence[Sequence[float]], closed: bool = False, *,
             layer=None, color=None, lineweight=None) -> DrawingEntity:
    """A polyline of >= 2 points; ``closed`` is honoured only when > 2 points."""
    pts = [ensure_3d(p) for p in points]
    if len(pts) < 2:
        raise DrawingCommandError("polyline needs at least 2 points")
    effective_closed = bool(closed) and len(pts) > 2
    return DrawingEntity(
        "polyline", {"points": pts, "closed": effective_closed},
        **_style(layer, color, lineweight))


def rectangle(corner1: Sequence[float], corner2: Sequence[float], *, layer=None,
              color=None, lineweight=None) -> DrawingEntity:
    """An axis-aligned rectangle from two diagonal corners.

    Expanded into the controller's closed 5-vertex polyline
    ``[c1, (x2,y1), (x2,y2), (x1,y2), c1]`` (z taken from ``corner1``).
    """
    x1, y1, z1 = ensure_3d(corner1)
    x2, y2, _z2 = ensure_3d(corner2)
    points = [
        (x1, y1, z1),
        (x2, y1, z1),
        (x2, y2, z1),
        (x1, y2, z1),
        (x1, y1, z1),
    ]
    return DrawingEntity(
        "rectangle",
        {"corner1": (x1, y1, z1), "corner2": (x2, y2, z1),
         "points": points, "closed": True},
        **_style(layer, color, lineweight))


def text(position: Sequence[float], content: str, height: float = 2.5,
         rotation: float = 0.0, *, layer=None, color=None) -> DrawingEntity:
    """A single-line text label; ``height`` must be positive."""
    h = float(height)
    if h <= 0:
        raise DrawingCommandError(f"text height must be > 0, got {h}")
    geom = {
        "position": ensure_3d(position),
        "text": str(content),
        "height": h,
        "rotation_deg": float(rotation),
        "rotation_rad": math.radians(float(rotation)),
    }
    return DrawingEntity("text", geom, layer=layer,
                         color=None if color is None else int(color))


def hatch(points: Sequence[Sequence[float]], pattern_name: str = "SOLID",
          scale: float = 1.0, *, layer=None, color=None) -> DrawingEntity:
    """A boundary-fill hatch; needs >= 3 boundary points (a closed loop)."""
    pts = [ensure_3d(p) for p in points]
    if len(pts) < 3:
        raise DrawingCommandError("hatch needs at least 3 boundary points")
    s = float(scale)
    if s <= 0:
        raise DrawingCommandError(f"hatch scale must be > 0, got {s}")
    geom = {
        "points": pts,
        "pattern_name": str(pattern_name).upper(),
        "scale": s,
        "closed": True,
    }
    return DrawingEntity("hatch", geom, layer=layer,
                         color=None if color is None else int(color))


def _entity_points(entity: DrawingEntity) -> List[Point]:
    """All coordinate points an entity spans (for extent computation)."""
    g = entity.geometry
    if "points" in g:
        return list(g["points"])  # type: ignore[arg-type]
    if entity.kind == "line":
        return [g["start"], g["end"]]  # type: ignore[list-item]
    if entity.kind in ("circle", "arc"):
        cx, cy, cz = g["center"]  # type: ignore[misc]
        r = float(g["radius"])  # type: ignore[arg-type]
        return [(cx - r, cy - r, cz), (cx + r, cy + r, cz)]
    if entity.kind == "ellipse":
        cx, cy, cz = g["center"]  # type: ignore[misc]
        a = float(g["major_axis"])  # type: ignore[arg-type]
        return [(cx - a, cy - a, cz), (cx + a, cy + a, cz)]
    if entity.kind == "text":
        return [g["position"]]  # type: ignore[list-item]
    return []


def extents(entities: Sequence[DrawingEntity]) -> Optional[Tuple[Point, Point]]:
    """Axis-aligned ``(min, max)`` bounding box over entities (zoom-extents).

    Returns ``None`` when no entity contributes a point. Circles/arcs/ellipses
    contribute their bounding square/rectangle.
    """
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for e in entities:
        for (x, y, z) in _entity_points(e):
            xs.append(x)
            ys.append(y)
            zs.append(z)
    if not xs:
        return None
    return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))


@dataclass
class DrawingBuilder:
    """Accumulates drawing commands into an ordered, serialisable list."""

    entities: List[DrawingEntity] = field(default_factory=list)

    def add(self, entity: DrawingEntity) -> DrawingEntity:
        self.entities.append(entity)
        return entity

    def extents(self) -> Optional[Tuple[Point, Point]]:
        return extents(self.entities)

    def to_list(self) -> List[dict]:
        return [e.to_dict() for e in self.entities]
