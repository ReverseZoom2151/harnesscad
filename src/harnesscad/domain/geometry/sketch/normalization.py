"""Vitruvion sketch normalisation: exact analytic bounding boxes + centre/rescale.

Vitruvion (Seff et al., ICLR 2022 -- ``img2cad/data_utils.py``) preprocesses every
SketchGraphs sketch before tokenising it, and the exact numerics of that step decide
what the quantiser sees.  Two pieces of it are non-obvious and are reimplemented here
faithfully.

1. Exact arc bounding box (``entity_bbox``)
-------------------------------------------
The harness already has a *sampled* bounding box (``drawings.sgraphs2_entity_render.
bounding_box`` walks polyline samples).  Vitruvion instead computes the bbox
**analytically**, which matters because the bbox drives the normalisation scale and a
sampled bbox is always slightly small (it never touches the true extremum of an arc
between two samples), so the two pipelines quantise to *different bins*.

The analytic rule for an arc: start from the full circle box
``[c - r, c + r]``, then *shrink each of the four sides back to the endpoints* unless
the arc actually sweeps through the axis point on that side.  Vitruvion decides that
with a quadrant-crossing test: the start and end points are assigned quadrants 1..4
around the centre, the traversed quadrant list is built (wrapping 4 -> 1 when the end
quadrant precedes the start quadrant), a clockwise arc is handled by swapping the
endpoints, and the extremum on a side is kept only when the ordered quadrant pair that
straddles that side (4->1 for +x, 1->2 for +y, 2->3 for -x, 3->4 for -y) appears in
order in the traversed list.  The degenerate case -- start and end in the *same*
quadrant -- is disambiguated by comparing the x-coordinates (in quadrants 1/2 the arc
is "long" when x_start <= x_end; in 3/4 when x_start >= x_end).

2. Centre + rescale (``normalize_sketch``)
------------------------------------------
Centre the *bbox* (not the centroid) on the origin, then divide every positional and
scale parameter by ``max(width, height)`` so the long axis of the bbox is exactly 1
and every parameter lands in ``[-0.5, 0.5]`` -- which is precisely the quantiser's
domain (see ``reconstruction.vitruvion_primitive_tokens``).  Note the scale parameters
rescaled per type: radius for arcs/circles, and ``startParam``/``endParam`` for lines
(the line anchor is its midpoint and the direction is unit, so the parameters are
half-lengths and must scale with the sketch).  Arc ``startParam``/``endParam`` are
*angles* and are deliberately not rescaled.  A zero-extent sketch returns the sentinel
scale ``-1`` (Vitruvion's "drop this sketch" signal).

3. Parameterisation (``parameterize_entity`` / ``entity_from_params``)
----------------------------------------------------------------------
The continuous parameter vector that is tokenised is *not* the storage form:

  * ``Arc``   -> 6 values: start, mid, end points; a clockwise arc has its start and
    end swapped first so the vector always reads counter-clockwise.
  * ``Circle``-> 3 values: centre, radius.
  * ``Line``  -> 4 values: start, end points.
  * ``Point`` -> 2 values.

The number of values therefore *uniquely identifies the type* on the way back, and an
arc is reconstructed from its three points by circumcentre (a determinant below
``1e-10`` means the points are collinear and the arc is rejected -- returns ``None``).

Pure stdlib.  Entities are mutable dataclasses with SketchGraphs field names, so the
existing evaluation helpers in ``drawings.sgraphs2_entity_render`` apply unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from harnesscad.domain.drawings.entity_render import (
    arc_endpoints,
    arc_midpoint,
    line_endpoints,
)

Vec2 = Tuple[float, float]
BBox = Tuple[Vec2, Vec2]

__all__ = [
    "VPoint",
    "VLine",
    "VCircle",
    "VArc",
    "NUM_PARAMS",
    "POS_PARAMS",
    "SCALE_PARAMS",
    "entity_bbox",
    "sketch_bbox",
    "center_sketch",
    "rescale_sketch",
    "normalize_sketch",
    "parameterize_entity",
    "entity_from_params",
]


# ---------------------------------------------------------------------------
# Mutable entities (SketchGraphs field names)
# ---------------------------------------------------------------------------
@dataclass
class VPoint:
    """A sketch point."""

    x: float = 0.0
    y: float = 0.0
    entity_id: Optional[str] = None
    is_construction: bool = False


@dataclass
class VLine:
    """A line: anchor ``(pntX, pntY)`` + unit direction + parameter interval."""

    pntX: float = 0.0
    pntY: float = 0.0
    dirX: float = 1.0
    dirY: float = 0.0
    startParam: float = -0.5
    endParam: float = 0.5
    entity_id: Optional[str] = None
    is_construction: bool = False

    @property
    def start_point(self) -> Vec2:
        return line_endpoints(self)[0]

    @property
    def end_point(self) -> Vec2:
        return line_endpoints(self)[1]


@dataclass
class VCircle:
    """A full circle."""

    xCenter: float = 0.0
    yCenter: float = 0.0
    xDir: float = 1.0
    yDir: float = 0.0
    radius: float = 1.0
    clockwise: bool = False
    entity_id: Optional[str] = None
    is_construction: bool = False

    @property
    def center_point(self) -> Vec2:
        return (self.xCenter, self.yCenter)


@dataclass
class VArc:
    """An arc: a circle plus an angular interval taken with a handedness."""

    xCenter: float = 0.0
    yCenter: float = 0.0
    xDir: float = 1.0
    yDir: float = 0.0
    radius: float = 1.0
    clockwise: bool = False
    startParam: float = -0.5
    endParam: float = 0.5
    entity_id: Optional[str] = None
    is_construction: bool = False

    @property
    def center_point(self) -> Vec2:
        return (self.xCenter, self.yCenter)

    @property
    def start_point(self) -> Vec2:
        return arc_endpoints(self)[0]

    @property
    def end_point(self) -> Vec2:
        return arc_endpoints(self)[1]

    @property
    def mid_point(self) -> Vec2:
        return arc_midpoint(self)


# Positional parameters, per type (translated when centring).
POS_PARAMS = {
    VArc: ("xCenter", "yCenter"),
    VCircle: ("xCenter", "yCenter"),
    VLine: ("pntX", "pntY"),
    VPoint: ("x", "y"),
}

# Scale parameters, per type (rescaled along with the positions).
SCALE_PARAMS = {
    VArc: ("radius",),
    VCircle: ("radius",),
    VLine: ("startParam", "endParam"),
    VPoint: (),
}

# Length of the continuous parameter vector, per type.
NUM_PARAMS = {
    VArc: 6,
    VCircle: 3,
    VLine: 4,
    VPoint: 2,
}


# ---------------------------------------------------------------------------
# Bounding boxes
# ---------------------------------------------------------------------------
def _max_circle_bbox(entity) -> BBox:
    return (
        (entity.xCenter - entity.radius, entity.yCenter - entity.radius),
        (entity.xCenter + entity.radius, entity.yCenter + entity.radius),
    )


def _relative_quadrant(point: Vec2, center: Vec2) -> int:
    """Quadrant (1..4, counter-clockwise from +x/+y) of ``point`` about ``center``."""
    x = point[0] - center[0]
    y = point[1] - center[1]
    if x >= 0:
        return 1 if y >= 0 else 4
    return 2 if y >= 0 else 3


def _traversed_quadrants(
    start_quadrant: int, end_quadrant: int, x_start: float, x_end: float
) -> List[int]:
    """The quadrants an arc passes through, in traversal order."""
    if start_quadrant < end_quadrant:
        return list(range(start_quadrant, end_quadrant + 1))
    if start_quadrant > end_quadrant:
        return list(range(start_quadrant, 5)) + list(range(1, end_quadrant + 1))

    # Start and end share a quadrant: either a tiny arc or a nearly-full turn.
    start_ahead = False
    if start_quadrant in (1, 2) and x_start <= x_end:
        start_ahead = True
    if start_quadrant in (3, 4) and x_start >= x_end:
        start_ahead = True
    if start_ahead:
        return list(range(start_quadrant, 5)) + list(range(1, end_quadrant + 1))
    return [start_quadrant]


def _arc_bbox(arc: VArc) -> BBox:
    x_start, y_start = arc.start_point
    x_end, y_end = arc.end_point
    (x0, y0), (x1, y1) = _max_circle_bbox(arc)

    center = arc.center_point
    start_quadrant = _relative_quadrant(arc.start_point, center)
    end_quadrant = _relative_quadrant(arc.end_point, center)

    if arc.clockwise:
        start_quadrant, end_quadrant = end_quadrant, start_quadrant
        x_start, x_end = x_end, x_start
        y_start, y_end = y_end, y_start

    quadrants = _traversed_quadrants(start_quadrant, end_quadrant, x_start, x_end)

    def has_ordered_quadrants(q1: int, q2: int) -> bool:
        if q1 not in quadrants or q2 not in quadrants:
            return False
        return quadrants.index(q2) > quadrants.index(q1)

    # Each side of the full-circle box survives only if the arc sweeps its axis point.
    if not has_ordered_quadrants(4, 1):
        x1 = max(x_start, x_end)
    if not has_ordered_quadrants(1, 2):
        y1 = max(y_start, y_end)
    if not has_ordered_quadrants(2, 3):
        x0 = min(x_start, x_end)
    if not has_ordered_quadrants(3, 4):
        y0 = min(y_start, y_end)

    return ((x0, y0), (x1, y1))


def entity_bbox(entity) -> Optional[BBox]:
    """The exact axis-aligned bbox ``((x0, y0), (x1, y1))`` of one entity.

    Returns ``None`` for entity types Vitruvion does not model (its own behaviour:
    unsupported entities simply do not contribute to the sketch bbox).
    """
    if isinstance(entity, VArc):
        return _arc_bbox(entity)
    if isinstance(entity, VCircle):
        return _max_circle_bbox(entity)
    if isinstance(entity, VLine):
        (sx, sy), (ex, ey) = line_endpoints(entity)
        return ((min(sx, ex), min(sy, ey)), (max(sx, ex), max(sy, ey)))
    if isinstance(entity, VPoint):
        return ((entity.x, entity.y), (entity.x, entity.y))
    return None


def sketch_bbox(entities: Sequence[object]) -> BBox:
    """The bbox of a whole sketch; an empty sketch bounds to the origin."""
    boxes = [b for b in (entity_bbox(e) for e in entities) if b is not None]
    if not boxes:
        return ((0.0, 0.0), (0.0, 0.0))
    x0 = min(b[0][0] for b in boxes)
    y0 = min(b[0][1] for b in boxes)
    x1 = max(b[1][0] for b in boxes)
    y1 = max(b[1][1] for b in boxes)
    return ((x0, y0), (x1, y1))


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def center_sketch(entities: Sequence[object]) -> None:
    """Translate the sketch in place so its bbox centre sits on the origin."""
    (x0, y0), (x1, y1) = sketch_bbox(entities)
    x_offset = (x0 + x1) / 2.0
    y_offset = (y0 + y1) / 2.0

    for entity in entities:
        pos_params = POS_PARAMS.get(type(entity))
        if pos_params is None:
            continue
        for name in pos_params:
            offset = x_offset if "x" in name.lower() else y_offset
            setattr(entity, name, getattr(entity, name) - offset)


def rescale_sketch(entities: Sequence[object]) -> float:
    """Scale the (already centred) sketch in place so its long axis is 1.

    Returns the scale factor that was divided out, or ``-1`` for a zero-extent
    sketch (Vitruvion's signal to discard the sketch).  Raises ``ValueError`` if the
    sketch is not centred.
    """
    (x0, y0), (x1, y1) = sketch_bbox(entities)
    if not math.isclose(x0, -x1, abs_tol=1e-9) or not math.isclose(y0, -y1, abs_tol=1e-9):
        raise ValueError("sketch must be centered before rescaling")

    factor = max(x1 - x0, y1 - y0)
    if factor == 0:
        return -1.0

    for entity in entities:
        pos_params = POS_PARAMS.get(type(entity))
        if pos_params is None:
            continue
        for name in pos_params + SCALE_PARAMS[type(entity)]:
            setattr(entity, name, getattr(entity, name) / factor)
    return factor


def normalize_sketch(entities: Sequence[object]) -> float:
    """Centre then rescale; returns the scale factor (or ``-1``)."""
    center_sketch(entities)
    return rescale_sketch(entities)


# ---------------------------------------------------------------------------
# Parameterisation
# ---------------------------------------------------------------------------
def parameterize_entity(entity) -> Optional[List[float]]:
    """The continuous parameter vector that Vitruvion tokenises, or ``None``."""
    if isinstance(entity, VArc):
        start, end = arc_endpoints(entity)
        if entity.clockwise:
            start, end = end, start
        mid = arc_midpoint(entity)
        return [start[0], start[1], mid[0], mid[1], end[0], end[1]]
    if isinstance(entity, VCircle):
        return [entity.xCenter, entity.yCenter, entity.radius]
    if isinstance(entity, VLine):
        start, end = line_endpoints(entity)
        return [start[0], start[1], end[0], end[1]]
    if isinstance(entity, VPoint):
        return [entity.x, entity.y]
    return None


def _arc_from_params(params: Sequence[float], entity_id: Optional[str]) -> Optional[VArc]:
    """Rebuild an arc from start/mid/end via the circumcentre of the three points."""
    b = (params[0], params[1])
    c = (params[2], params[3])
    d = (params[4], params[5])

    temp = c[0] ** 2 + c[1] ** 2
    bc = (b[0] ** 2 + b[1] ** 2 - temp) / 2.0
    cd = (temp - d[0] ** 2 - d[1] ** 2) / 2.0
    det = (b[0] - c[0]) * (c[1] - d[1]) - (c[0] - d[0]) * (b[1] - c[1])

    if abs(det) < 1.0e-10:
        # Collinear (or degenerate) triple: no circle through the points.
        return None

    cx = (bc * (c[1] - d[1]) - cd * (b[1] - c[1])) / det
    cy = ((b[0] - c[0]) * cd - (c[0] - d[0]) * bc) / det
    radius = math.sqrt((cx - b[0]) ** 2 + (cy - b[1]) ** 2)

    # SketchGraphs `Arc.from_info`: reference direction is +x, the params are the
    # absolute angles of the endpoints, and the arc is stored counter-clockwise.
    start_param = math.atan2(b[1] - cy, b[0] - cx)
    end_param = math.atan2(d[1] - cy, d[0] - cx)

    return VArc(
        xCenter=cx,
        yCenter=cy,
        xDir=1.0,
        yDir=0.0,
        radius=radius,
        clockwise=False,
        startParam=start_param,
        endParam=end_param,
        entity_id=entity_id,
    )


def _line_from_params(params: Sequence[float], entity_id: Optional[str]) -> VLine:
    """SketchGraphs ``Line.from_info``: midpoint anchor, unit dir, half-length params."""
    sx, sy, ex, ey = params
    vx, vy = ex - sx, ey - sy
    length = math.sqrt(vx * vx + vy * vy)
    if length == 0:
        dir_x, dir_y = 1.0, 0.0
    else:
        dir_x, dir_y = vx / length, vy / length
    return VLine(
        pntX=(sx + ex) / 2.0,
        pntY=(sy + ey) / 2.0,
        dirX=dir_x,
        dirY=dir_y,
        startParam=-length / 2.0,
        endParam=length / 2.0,
        entity_id=entity_id,
    )


def entity_from_params(params: Sequence[float], entity_id: Optional[str] = None):
    """Rebuild an entity; the parameter count alone determines the type.

    Returns ``None`` only for a degenerate arc.  Raises ``ValueError`` when the count
    matches no entity type.
    """
    count = len(params)
    if count == NUM_PARAMS[VArc]:
        return _arc_from_params(params, entity_id)
    if count == NUM_PARAMS[VCircle]:
        return VCircle(
            xCenter=params[0], yCenter=params[1], radius=params[2], entity_id=entity_id
        )
    if count == NUM_PARAMS[VLine]:
        return _line_from_params(params, entity_id)
    if count == NUM_PARAMS[VPoint]:
        return VPoint(x=params[0], y=params[1], entity_id=entity_id)
    raise ValueError("Unsupported number of parameters: {}".format(count))
