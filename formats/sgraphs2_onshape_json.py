"""Onshape FeatureScript sketch-entity JSON schema, parser and serialiser.

The SketchGraphs paper (165) describes the *taxonomy* of Onshape sketch primitives
and constraints; the harness already models that abstractly
(``reconstruction/sketchgraphs_taxonomy.py``, ``sketchgraphs_graph.py``,
``sketchgraphs_sequence.py``).  What the paper does not give -- and what the
reference implementation encodes in ``sketchgraphs/data/_entity.py`` -- is the
*concrete wire format*: the exact nested ``{type, typeName, message}`` JSON that
Onshape's FeatureScript emits, the numeric type codes, the per-entity parameter
layout, and the rules for recovering an entity type from an untyped blob.

This module reimplements that layer in the standard library.

Wire format
-----------
Every entity is a dict with three keys::

    {"type": <int>, "typeName": <str>, "message": {...}}

The outer ``typeName`` distinguishes the *container* kind:

===================== ==== ==============================================
``typeName``          type contains
===================== ==== ==============================================
BTMSketchPoint        158  a point (``x``/``y`` live directly in message)
BTMSketchCurve          4  an unbounded curve (circle)
BTMSketchCurveSegment 155  a bounded curve (line, arc) -- carries
                           ``startParam``/``endParam`` in the message
===================== ==== ==============================================

and a nested ``message.geometry`` blob carries the curve's intrinsic
parameters, again as ``{type, typeName, message}``:

======================= ==== ==================================================
``geometry.typeName``   type parameters
======================= ==== ==================================================
BTCurveGeometryLine      117 dirX, dirY, pntX, pntY
BTCurveGeometryCircle    115 xCenter, yCenter, xDir, yDir, radius, clockwise
======================= ==== ==================================================

Note the load-bearing subtlety that the entity type is *not* a field: a circle
and an arc share the identical ``BTCurveGeometryCircle`` geometry, and are told
apart only by the container (``BTMSketchCurve`` -> circle,
``BTMSketchCurveSegment`` -> arc, i.e. a circle that has been trimmed by the
``startParam``/``endParam`` on the segment).  :func:`inspect_entity_type`
reproduces that dispatch.

Parameter layout
----------------
Each entity class publishes ``float_ids`` / ``bool_ids`` -- the ordered names of
its continuous and boolean parameters.  This is the layout a generative model
quantises and predicts over, so it is part of the format, not an implementation
detail.  :func:`parameter_layout` and :meth:`Entity.parameters` expose it.

Subnodes
--------
Curves own implicit sub-entities addressable by constraints: a line has
``<id>.start`` / ``<id>.end``, a circle has ``<id>.center``, an arc has all
three.  The suffix convention *is* the format -- constraints reference subnodes
by these derived string ids.

Public API
----------
``EntityType`` / ``SubnodeType``  -- the integer taxonomies.
``Point`` / ``Line`` / ``Circle`` / ``Arc`` / ``GenericEntity`` -- entity records.
``inspect_entity_type(d)``        -- entity type of a raw JSON dict.
``parse_entity(d)`` / ``parse_sketch(d)``  -- JSON -> records.
``entity_to_dict(e)`` / ``sketch_to_dict(s)`` -- records -> JSON (round-trips).
``parameter_layout(t)``           -- (float_ids, bool_ids) for an entity type.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Mapping, Tuple

__all__ = [
    "EntityType",
    "SubnodeType",
    "Entity",
    "Point",
    "Line",
    "Circle",
    "Arc",
    "GenericEntity",
    "Sketch",
    "inspect_entity_type",
    "parse_entity",
    "parse_sketch",
    "entity_to_dict",
    "sketch_to_dict",
    "parameter_layout",
    "subnode_ids",
    "TYPE_CODES",
]


class EntityType(enum.IntEnum):
    """Integer taxonomy of sketch entities (SketchGraphs ordering)."""

    Point = 0
    Line = 1
    Circle = 2
    Ellipse = 3
    Spline = 4
    Conic = 5
    Arc = 6
    External = 7
    Stop = 8
    Unknown = 9


class SubnodeType(enum.IntEnum):
    """Integer taxonomy of implicit sub-entities."""

    SN_Start = 101
    SN_End = 102
    SN_Center = 103


# Numeric ``type`` codes that accompany each ``typeName`` on the wire.
TYPE_CODES: Dict[str, int] = {
    "BTMSketchPoint": 158,
    "BTMSketchCurve": 4,
    "BTMSketchCurveSegment": 155,
    "BTCurveGeometryLine": 117,
    "BTCurveGeometryCircle": 115,
}

_GEOMETRY_KEYWORDS: Tuple[Tuple[str, EntityType], ...] = (
    ("Line", EntityType.Line),
    ("Circle", EntityType.Circle),
    ("Ellipse", EntityType.Ellipse),
    ("Spline", EntityType.Spline),
    ("Conic", EntityType.Conic),
)


def inspect_entity_type(entity_dict: Mapping[str, Any]) -> EntityType:
    """Determine the entity type of a raw Onshape entity dict.

    The dispatch is structural, because the JSON carries no explicit entity-type
    field:

    * ``BTMSketchPoint`` container -> ``Point``;
    * no ``geometry`` in the message (e.g. a text entity) -> ``Unknown``;
    * otherwise the first geometry keyword matched in ``geometry.typeName``
      selects a candidate, then the container refines it:
      circle geometry inside a ``BTMSketchCurveSegment`` is an ``Arc``, and an
      elliptical segment (elliptical arc) is not modelled -> ``Unknown``.
    * a spline is only recognised when its geometry ``type`` code is 117
      (an interpolated spline is not modelled -> ``Unknown``).
    """
    if entity_dict.get("typeName") == "BTMSketchPoint":
        return EntityType.Point

    message = entity_dict.get("message") or {}
    geometry = message.get("geometry")
    if not geometry:
        return EntityType.Unknown

    geom_type_name = geometry.get("typeName", "")

    for keyword, candidate in _GEOMETRY_KEYWORDS:
        if keyword in geom_type_name:
            break
    else:
        return EntityType.Unknown

    is_segment = entity_dict.get("typeName") == "BTMSketchCurveSegment"

    if keyword == "Circle":
        return EntityType.Arc if is_segment else EntityType.Circle
    if keyword == "Ellipse":
        # An elliptical arc is out of scope for the taxonomy.
        return EntityType.Unknown if is_segment else EntityType.Ellipse
    if keyword == "Spline":
        return EntityType.Spline if geometry.get("type") == 117 else EntityType.Unknown

    return candidate


def _common(entity_dict: Mapping[str, Any]) -> Tuple[str, bool]:
    message = entity_dict["message"]
    return message["entityId"], bool(message["isConstruction"])


@dataclass
class Entity:
    """Base record: an entity is an id plus a construction flag."""

    entityId: str
    isConstruction: bool = False

    #: Ordered names of the continuous parameters.
    float_ids: ClassVar[Tuple[str, ...]] = ()
    #: Ordered names of the boolean parameters.
    bool_ids: ClassVar[Tuple[str, ...]] = ("isConstruction",)

    @property
    def type(self) -> EntityType:  # pragma: no cover - overridden
        raise NotImplementedError

    @staticmethod
    def subnode_types() -> Tuple[SubnodeType, ...]:
        return ()

    def subnode_ids(self) -> Tuple[str, ...]:
        """Derived ids of this entity's implicit sub-entities."""
        suffixes = {
            SubnodeType.SN_Start: ".start",
            SubnodeType.SN_End: ".end",
            SubnodeType.SN_Center: ".center",
        }
        return tuple(self.entityId + suffixes[t] for t in self.subnode_types())

    def parameters(self) -> Dict[str, Any]:
        """This entity's parameters in declared layout order."""
        out: Dict[str, Any] = {}
        for name in self.float_ids:
            out[name] = float(getattr(self, name))
        for name in self.bool_ids:
            out[name] = bool(getattr(self, name))
        return out

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass
class Point(Entity):
    """A free point: ``x``/``y`` live directly in the message."""

    x: float = 0.0
    y: float = 0.0

    float_ids: ClassVar[Tuple[str, ...]] = ("x", "y")
    bool_ids: ClassVar[Tuple[str, ...]] = ("isConstruction",)

    @property
    def type(self) -> EntityType:
        return EntityType.Point

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": TYPE_CODES["BTMSketchPoint"],
            "typeName": "BTMSketchPoint",
            "message": {
                "entityId": self.entityId,
                "isConstruction": self.isConstruction,
                "x": self.x,
                "y": self.y,
            },
        }

    @staticmethod
    def from_dict(entity_dict: Mapping[str, Any]) -> "Point":
        if entity_dict.get("typeName") != "BTMSketchPoint":
            raise ValueError("not a BTMSketchPoint")
        message = entity_dict["message"]
        entity_id, is_construction = _common(entity_dict)
        return Point(entity_id, is_construction, float(message["x"]), float(message["y"]))


@dataclass
class Line(Entity):
    """A bounded line, stored as point + unit direction + two parameters.

    Onshape does *not* store the endpoints: it stores an anchor ``(pntX, pntY)``
    on the infinite line, a direction ``(dirX, dirY)``, and the parameter range
    ``[startParam, endParam]`` along that direction.  Endpoints are recovered as
    ``pnt + param * dir`` (see ``drawings/sgraphs2_entity_render.py``).
    """

    pntX: float = 0.0
    pntY: float = 0.0
    dirX: float = 1.0
    dirY: float = 0.0
    startParam: float = -0.5
    endParam: float = 0.5

    float_ids: ClassVar[Tuple[str, ...]] = ("dirX", "dirY", "pntX", "pntY", "startParam", "endParam")
    bool_ids: ClassVar[Tuple[str, ...]] = ("isConstruction",)

    @property
    def type(self) -> EntityType:
        return EntityType.Line

    @staticmethod
    def subnode_types() -> Tuple[SubnodeType, ...]:
        return (SubnodeType.SN_Start, SubnodeType.SN_End)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": TYPE_CODES["BTMSketchCurveSegment"],
            "typeName": "BTMSketchCurveSegment",
            "message": {
                "entityId": self.entityId,
                "startPointId": self.entityId + ".start",
                "endPointId": self.entityId + ".end",
                "isConstruction": self.isConstruction,
                "startParam": self.startParam,
                "endParam": self.endParam,
                "geometry": {
                    "type": TYPE_CODES["BTCurveGeometryLine"],
                    "typeName": "BTCurveGeometryLine",
                    "message": {
                        "dirX": self.dirX,
                        "dirY": self.dirY,
                        "pntX": self.pntX,
                        "pntY": self.pntY,
                    },
                },
            },
        }

    @staticmethod
    def from_dict(entity_dict: Mapping[str, Any]) -> "Line":
        message = entity_dict["message"]
        geometry = message["geometry"]["message"]
        entity_id, is_construction = _common(entity_dict)
        return Line(
            entity_id,
            is_construction,
            float(geometry["pntX"]),
            float(geometry["pntY"]),
            float(geometry["dirX"]),
            float(geometry["dirY"]),
            float(message["startParam"]),
            float(message["endParam"]),
        )


@dataclass
class Circle(Entity):
    """A full circle: centre, reference direction, radius, orientation."""

    xCenter: float = 0.0
    yCenter: float = 0.0
    xDir: float = 1.0
    yDir: float = 0.0
    radius: float = 1.0
    clockwise: bool = False

    float_ids: ClassVar[Tuple[str, ...]] = ("xCenter", "yCenter", "xDir", "yDir", "radius")
    bool_ids: ClassVar[Tuple[str, ...]] = ("isConstruction", "clockwise")

    @property
    def type(self) -> EntityType:
        return EntityType.Circle

    @staticmethod
    def subnode_types() -> Tuple[SubnodeType, ...]:
        return (SubnodeType.SN_Center,)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": TYPE_CODES["BTMSketchCurve"],
            "typeName": "BTMSketchCurve",
            "message": {
                "entityId": self.entityId,
                "centerId": self.entityId + ".center",
                "isConstruction": self.isConstruction,
                "geometry": {
                    "type": TYPE_CODES["BTCurveGeometryCircle"],
                    "typeName": "BTCurveGeometryCircle",
                    "message": {
                        "xCenter": self.xCenter,
                        "yCenter": self.yCenter,
                        "xDir": self.xDir,
                        "yDir": self.yDir,
                        "radius": self.radius,
                        "clockwise": self.clockwise,
                    },
                },
            },
        }

    @staticmethod
    def from_dict(entity_dict: Mapping[str, Any]) -> "Circle":
        message = entity_dict["message"]
        geometry = message["geometry"]["message"]
        entity_id, is_construction = _common(entity_dict)
        return Circle(
            entity_id,
            is_construction,
            float(geometry["xCenter"]),
            float(geometry["yCenter"]),
            float(geometry["xDir"]),
            float(geometry["yDir"]),
            float(geometry["radius"]),
            bool(geometry["clockwise"]),
        )


@dataclass
class Arc(Entity):
    """A trimmed circle: circle geometry plus an angular parameter range.

    ``startParam``/``endParam`` are angular offsets (radians) measured from the
    reference direction ``(xDir, yDir)``, and are applied with the sign given by
    ``clockwise``.
    """

    xCenter: float = 0.0
    yCenter: float = 0.0
    xDir: float = 1.0
    yDir: float = 0.0
    radius: float = 1.0
    clockwise: bool = False
    startParam: float = -0.5
    endParam: float = 0.5

    float_ids: ClassVar[Tuple[str, ...]] = (
        "xCenter",
        "yCenter",
        "xDir",
        "yDir",
        "radius",
        "startParam",
        "endParam",
    )
    bool_ids: ClassVar[Tuple[str, ...]] = ("isConstruction", "clockwise")

    @property
    def type(self) -> EntityType:
        return EntityType.Arc

    @staticmethod
    def subnode_types() -> Tuple[SubnodeType, ...]:
        return (SubnodeType.SN_Center, SubnodeType.SN_Start, SubnodeType.SN_End)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": TYPE_CODES["BTMSketchCurveSegment"],
            "typeName": "BTMSketchCurveSegment",
            "message": {
                "entityId": self.entityId,
                "centerId": self.entityId + ".center",
                "startPointId": self.entityId + ".start",
                "endPointId": self.entityId + ".end",
                "isConstruction": self.isConstruction,
                "startParam": self.startParam,
                "endParam": self.endParam,
                "geometry": {
                    "type": TYPE_CODES["BTCurveGeometryCircle"],
                    "typeName": "BTCurveGeometryCircle",
                    "message": {
                        "xCenter": self.xCenter,
                        "yCenter": self.yCenter,
                        "xDir": self.xDir,
                        "yDir": self.yDir,
                        "radius": self.radius,
                        "clockwise": self.clockwise,
                    },
                },
            },
        }

    @staticmethod
    def from_dict(entity_dict: Mapping[str, Any]) -> "Arc":
        message = entity_dict["message"]
        geometry = message["geometry"]["message"]
        entity_id, is_construction = _common(entity_dict)
        return Arc(
            entity_id,
            is_construction,
            float(geometry["xCenter"]),
            float(geometry["yCenter"]),
            float(geometry["xDir"]),
            float(geometry["yDir"]),
            float(geometry["radius"]),
            bool(geometry["clockwise"]),
            float(message["startParam"]),
            float(message["endParam"]),
        )


@dataclass
class GenericEntity(Entity):
    """An entity type the taxonomy does not model; carried through verbatim."""

    data: Dict[str, Any] = field(default_factory=dict)
    entity_type: EntityType = EntityType.Unknown

    float_ids: ClassVar[Tuple[str, ...]] = ()
    bool_ids: ClassVar[Tuple[str, ...]] = ("isConstruction",)

    @property
    def type(self) -> EntityType:
        return self.entity_type

    def to_dict(self) -> Dict[str, Any]:
        return self.data

    @staticmethod
    def from_dict(entity_dict: Mapping[str, Any]) -> "GenericEntity":
        message = entity_dict["message"]
        return GenericEntity(
            message["entityId"],
            bool(message.get("isConstruction", False)),
            data=dict(entity_dict),
            entity_type=inspect_entity_type(entity_dict),
        )


_PARSERS = {
    EntityType.Point: Point.from_dict,
    EntityType.Line: Line.from_dict,
    EntityType.Circle: Circle.from_dict,
    EntityType.Arc: Arc.from_dict,
}


def parse_entity(entity_dict: Mapping[str, Any]) -> Entity:
    """Parse one raw Onshape entity dict into a record.

    Types outside the modelled taxonomy become a :class:`GenericEntity`, which
    round-trips its source dict unchanged rather than failing the whole sketch.
    """
    entity_type = inspect_entity_type(entity_dict)
    parser = _PARSERS.get(entity_type)
    if parser is None:
        return GenericEntity.from_dict(entity_dict)
    return parser(entity_dict)


def entity_to_dict(entity: Entity) -> Dict[str, Any]:
    """Serialise a record back to the Onshape JSON representation."""
    return entity.to_dict()


@dataclass
class Sketch:
    """A parsed sketch: entities by id, plus the untouched constraint blobs."""

    entities: Dict[str, Entity] = field(default_factory=dict)
    constraints: List[Dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entities)

    def of_type(self, entity_type: EntityType) -> List[Entity]:
        return [e for e in self.entities.values() if e.type is entity_type]


def parse_sketch(sketch_dict: Mapping[str, Any]) -> Sketch:
    """Parse a FeatureScript sketch dict (``{"entities": [...], "constraints": [...]}``).

    Entity insertion order is preserved -- it is the construction order the
    sequence representation depends on.
    """
    entities: Dict[str, Entity] = {}
    for entity_dict in sketch_dict.get("entities", []):
        entity = parse_entity(entity_dict)
        entities[entity.entityId] = entity
    constraints = [dict(c) for c in sketch_dict.get("constraints", [])]
    return Sketch(entities, constraints)


def sketch_to_dict(sketch: Sketch) -> Dict[str, Any]:
    """Serialise a sketch back to its FeatureScript JSON representation."""
    return {
        "entities": [e.to_dict() for e in sketch.entities.values()],
        "constraints": [dict(c) for c in sketch.constraints],
    }


_LAYOUTS: Dict[EntityType, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    EntityType.Point: (Point.float_ids, Point.bool_ids),
    EntityType.Line: (Line.float_ids, Line.bool_ids),
    EntityType.Circle: (Circle.float_ids, Circle.bool_ids),
    EntityType.Arc: (Arc.float_ids, Arc.bool_ids),
}


def parameter_layout(entity_type: EntityType) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """``(float_ids, bool_ids)`` -- the parameter layout of an entity type.

    This is the vector layout a generative model predicts over.  Raises
    ``KeyError`` for types with no fixed layout (spline, ellipse, unknown).
    """
    if entity_type not in _LAYOUTS:
        raise KeyError(f"no fixed parameter layout for {entity_type!r}")
    return _LAYOUTS[entity_type]


def subnode_ids(entity: Entity) -> Tuple[str, ...]:
    """The derived ids of an entity's implicit sub-entities."""
    return entity.subnode_ids()
