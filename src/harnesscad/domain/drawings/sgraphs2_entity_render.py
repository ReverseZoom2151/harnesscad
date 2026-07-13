"""Deterministic geometry evaluation and polyline rendering of Onshape sketch entities.

SketchGraphs stores sketch entities in an *implicit* parameterisation (see
``formats/sgraphs2_onshape_json.py``): a line is an anchor point plus a direction
plus a parameter interval; an arc is a circle plus an angular interval taken with
a handedness.  Nothing downstream -- plotting, bounding boxes, chamfer/overlap
checks, rasterisation, image supervision -- can consume that directly.  The
reference implementation resolves it inside a matplotlib plotting module
(``sketchgraphs/data/_plotting.py``), which fuses three separable things: the
parameter-to-point evaluation, the sampling of a curve into a polyline, and the
matplotlib drawing calls.

This module extracts the first two -- the deterministic, backend-free half -- so
the geometry is usable without a plotting library.  The output is a
:class:`RenderedEntity` carrying an explicit polyline (plus the ``construction``
flag, which the reference implementation maps to a dashed linestyle).

Evaluation rules (faithful to the reference)
--------------------------------------------
* **Line**: ``point(t) = (pntX, pntY) + t * (dirX, dirY)`` with
  ``t in [startParam, endParam]``.  Endpoints therefore depend on the direction's
  magnitude; Onshape stores a unit direction, so the parameters are arc lengths.
* **Arc**: the reference direction ``(xDir, yDir)`` defines a base angle
  ``phi = atan2(yDir, xDir)``.  A parameter ``t`` maps to the angle
  ``phi + t`` when counter-clockwise and ``phi - t`` when ``clockwise`` -- i.e.
  ``clockwise`` negates the *parameter*, not the base angle.  The point is then
  ``centre + radius * (cos, sin)`` of that angle.
* **Arc midpoint**: the parameters are first wrapped into ``[0, 2*pi)``; if the
  wrapped start exceeds the wrapped end, the end is lifted by ``2*pi`` so the
  interval runs forwards; the midpoint is the mean of the two.  This is what
  makes the midpoint land on the drawn side of a wrap-around arc.
* **Circle**: sampled counter-clockwise from the reference direction over a full
  turn; the polyline is closed (first point repeated).

Sampling is uniform in the parameter and fully deterministic: ``segments``
samples give ``segments + 1`` points on an open curve.

Public API
----------
``line_point`` / ``arc_point``            -- evaluate one parameter.
``line_endpoints`` / ``arc_endpoints`` / ``arc_midpoint`` / ``circle_center``.
``sample_entity(e, segments)``            -- entity -> list of ``(x, y)``.
``render_entity`` / ``render_sketch``     -- entity/sketch -> RenderedEntity(s).
``bounding_box(rendered)``                -- ``(min_x, min_y, max_x, max_y)``.
``normalize_scene(rendered)``             -- fit to the unit box, aspect preserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from harnesscad.io.formats.sgraphs2_onshape_json import (
    Arc,
    Circle,
    Entity,
    EntityType,
    Line,
    Point,
    Sketch,
)

__all__ = [
    "RenderedEntity",
    "line_point",
    "line_endpoints",
    "arc_point",
    "arc_endpoints",
    "arc_midpoint",
    "circle_center",
    "sample_entity",
    "render_entity",
    "render_sketch",
    "bounding_box",
    "normalize_scene",
    "TAU",
]

TAU = 2.0 * math.pi

Vec2 = Tuple[float, float]


@dataclass(frozen=True)
class RenderedEntity:
    """An entity resolved to explicit geometry.

    ``polyline`` is the sampled point list (a single point for a ``Point``
    entity).  ``closed`` marks a circle.  ``construction`` mirrors
    ``isConstruction`` -- the flag the reference plotter turns into a dashed
    line style.
    """

    entity_id: str
    kind: EntityType
    polyline: Tuple[Vec2, ...]
    closed: bool = False
    construction: bool = False


# ---------------------------------------------------------------------------
# Parameter evaluation
# ---------------------------------------------------------------------------
def line_point(line: Line, t: float) -> Vec2:
    """The point on ``line`` at parameter ``t`` (``pnt + t * dir``)."""
    return (line.pntX + t * line.dirX, line.pntY + t * line.dirY)


def line_endpoints(line: Line) -> Tuple[Vec2, Vec2]:
    """``(start, end)`` of a line, from its parameter interval."""
    return line_point(line, line.startParam), line_point(line, line.endParam)


def _circle_point(
    x_center: float,
    y_center: float,
    x_dir: float,
    y_dir: float,
    radius: float,
    clockwise: bool,
    t: float,
) -> Vec2:
    base = math.atan2(y_dir, x_dir)
    offset = -t if clockwise else t
    angle = base + offset
    return (x_center + math.cos(angle) * radius, y_center + math.sin(angle) * radius)


def arc_point(arc: Arc, t: float) -> Vec2:
    """The point on ``arc`` at angular parameter ``t``.

    ``clockwise`` negates the parameter, so an arc and its mirror share a base
    angle and differ only in traversal sense.
    """
    return _circle_point(
        arc.xCenter, arc.yCenter, arc.xDir, arc.yDir, arc.radius, arc.clockwise, t
    )


def arc_endpoints(arc: Arc) -> Tuple[Vec2, Vec2]:
    """``(start, end)`` points of an arc."""
    return arc_point(arc, arc.startParam), arc_point(arc, arc.endParam)


def arc_midpoint(arc: Arc) -> Vec2:
    """The point halfway along the *drawn* span of an arc.

    The parameters are wrapped into ``[0, 2*pi)`` and the end is lifted by a full
    turn when the interval would otherwise run backwards, so the midpoint lands
    on the arc that is actually drawn rather than on its complement.
    """
    start_param = arc.startParam % TAU
    end_param = arc.endParam % TAU
    if start_param > end_param:
        end_param += TAU
    return arc_point(arc, (start_param + end_param) / 2.0)


def circle_center(circle: Circle) -> Vec2:
    """The centre of a circle (or arc)."""
    return (circle.xCenter, circle.yCenter)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def _linspace(lo: float, hi: float, count: int) -> List[float]:
    if count < 2:
        raise ValueError("need at least 2 samples")
    step = (hi - lo) / (count - 1)
    return [lo + step * i for i in range(count - 1)] + [hi]


def sample_entity(entity: Entity, segments: int = 32) -> List[Vec2]:
    """Sample ``entity`` into a polyline of ``(x, y)`` points.

    A point yields one sample; a line yields two (it is straight, so extra
    samples carry no information); an arc yields ``segments + 1`` samples over
    its parameter interval; a circle yields ``segments + 1`` samples over a full
    turn, with the first point repeated last so the polyline is closed.

    ``ValueError`` for entities whose geometry is not modelled (spline, ellipse,
    unknown).
    """
    if segments < 1:
        raise ValueError("segments must be >= 1")

    if isinstance(entity, Point):
        return [(entity.x, entity.y)]

    if isinstance(entity, Line):
        start, end = line_endpoints(entity)
        return [start, end]

    if isinstance(entity, Arc):
        return [
            arc_point(entity, t)
            for t in _linspace(entity.startParam, entity.endParam, segments + 1)
        ]

    if isinstance(entity, Circle):
        points = [
            _circle_point(
                entity.xCenter,
                entity.yCenter,
                entity.xDir,
                entity.yDir,
                entity.radius,
                entity.clockwise,
                t,
            )
            for t in _linspace(0.0, TAU, segments + 1)
        ]
        # Close exactly, rather than relying on cos/sin of TAU round-tripping.
        points[-1] = points[0]
        return points

    raise ValueError(f"entity type {entity.type!r} has no sampled geometry")


def render_entity(entity: Entity, segments: int = 32) -> RenderedEntity:
    """Resolve one entity into explicit geometry."""
    return RenderedEntity(
        entity_id=entity.entityId,
        kind=entity.type,
        polyline=tuple(sample_entity(entity, segments)),
        closed=isinstance(entity, Circle),
        construction=bool(entity.isConstruction),
    )


def render_sketch(
    sketch: Sketch, segments: int = 32, skip_unsupported: bool = True
) -> List[RenderedEntity]:
    """Resolve every entity of a sketch, in construction order.

    Entities whose geometry is not modelled are skipped by default -- matching
    the reference plotter, which silently ignores entity types it has no drawing
    routine for.
    """
    out: List[RenderedEntity] = []
    for entity in sketch.entities.values():
        try:
            out.append(render_entity(entity, segments))
        except ValueError:
            if not skip_unsupported:
                raise
    return out


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------
def bounding_box(rendered: Iterable[RenderedEntity]) -> Tuple[float, float, float, float]:
    """Axis-aligned bounds ``(min_x, min_y, max_x, max_y)`` of rendered geometry.

    Raises ``ValueError`` on an empty scene.
    """
    xs: List[float] = []
    ys: List[float] = []
    for item in rendered:
        for x, y in item.polyline:
            xs.append(x)
            ys.append(y)
    if not xs:
        raise ValueError("cannot bound an empty scene")
    return (min(xs), min(ys), max(xs), max(ys))


def normalize_scene(
    rendered: Sequence[RenderedEntity], margin: float = 0.0
) -> List[RenderedEntity]:
    """Fit a rendered scene into the unit box ``[0, 1]^2``, preserving aspect.

    A single uniform scale is applied (so shapes are not distorted) and the
    result is centred.  A degenerate scene -- all points coincident, or a scene
    with zero extent in both axes -- maps to the centre of the box.  ``margin``
    shrinks the usable box on every side.

    This is the deterministic normalisation a rasteriser or an image-supervised
    model needs, and it is scale- and translation-invariant, so two sketches that
    differ only by a similarity transform normalise identically.
    """
    if not 0.0 <= margin < 0.5:
        raise ValueError("margin must be in [0, 0.5)")
    if not rendered:
        return []

    min_x, min_y, max_x, max_y = bounding_box(rendered)
    span = max(max_x - min_x, max_y - min_y)
    usable = 1.0 - 2.0 * margin

    if span == 0.0:
        scale = 0.0
    else:
        scale = usable / span

    # Centre the scaled content within the usable box.
    off_x = margin + (usable - (max_x - min_x) * scale) / 2.0
    off_y = margin + (usable - (max_y - min_y) * scale) / 2.0

    out: List[RenderedEntity] = []
    for item in rendered:
        polyline = tuple(
            ((x - min_x) * scale + off_x, (y - min_y) * scale + off_y)
            for x, y in item.polyline
        )
        out.append(
            RenderedEntity(
                entity_id=item.entity_id,
                kind=item.kind,
                polyline=polyline,
                closed=item.closed,
                construction=item.construction,
            )
        )
    return out
