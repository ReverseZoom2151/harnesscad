"""Synthetic B-Rep topology for analytic primitives (OpenCAD ``core/topology.py``).

OpenCAD's kernel runs with either an OCCT backend or a pure *analytic* backend.
The analytic backend still has to answer topology queries -- "give me the faces of
this box, with their normals, centroids, areas and semantic tags" -- so it
*synthesises* a canonical subshape map from the shape kind plus its bounding box,
and auto-tags each face by matching its normal against the six axis directions
(``top``/``+Z``, ``bottom``/``-Z``, ``front``/``+Y``, ...).  Those tags are what
make a kernel-free selector query such as "the top face" or "the vertical edges"
work at all, and they survive into the persistence layer as ``shape:face:index``
references.

This module reimplements that synthesis deterministically and completes it: the
upstream ``synthetic_edges`` is a stub that gives every edge the shape centroid;
here box edges are generated with real endpoints, midpoints, directions, lengths
and axis tags, cylinders get their two circular rims and a seam, and the tag
vocabulary is shared with the face generator.  The result is a
:class:`TopologyMap` whose refs are directly consumable by the harness's existing
selector algebra (:mod:`geometry.cascade_entity_selector`) and by face-identity
tracking (:mod:`geometry.opencad_face_fingerprint`).

Deterministic: pure arithmetic, fixed emission order, stable IDs; no clock, no
randomness.

Public API
----------
``BoundingBox``, ``SubshapeRef``, ``TopologyMap``
``direction_tags(normal)``, ``build_topology(shape_id, kind, bbox)``
``synthetic_box_faces``/``synthetic_box_edges``/``synthetic_cylinder_faces``/...
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "BoundingBox",
    "SubshapeRef",
    "TopologyMap",
    "DIRECTION_TAGS",
    "COS_THRESHOLD",
    "direction_tags",
    "synthetic_box_faces",
    "synthetic_box_edges",
    "synthetic_cylinder_faces",
    "synthetic_cylinder_edges",
    "synthetic_sphere_faces",
    "build_topology",
]

Vec3 = Tuple[float, float, float]

DIRECTION_TAGS: Tuple[Tuple[Vec3, Tuple[str, ...]], ...] = (
    ((0.0, 0.0, 1.0), ("top", "+Z")),
    ((0.0, 0.0, -1.0), ("bottom", "-Z")),
    ((0.0, 1.0, 0.0), ("front", "+Y")),
    ((0.0, -1.0, 0.0), ("back", "-Y")),
    ((1.0, 0.0, 0.0), ("right", "+X")),
    ((-1.0, 0.0, 0.0), ("left", "-X")),
)

COS_THRESHOLD = 0.95  # ~18 degrees


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length(v: Vec3) -> float:
    return math.sqrt(_dot(v, v))


def _normalise(v: Vec3) -> Vec3:
    n = _length(v)
    if n < 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def direction_tags(normal: Optional[Vec3], *, undirected: bool = False) -> List[str]:
    """Semantic tags for a face normal (or an edge direction when *undirected*)."""
    if normal is None:
        return []
    unit = _normalise(normal)
    if unit == (0.0, 0.0, 0.0):
        return []
    for direction, labels in DIRECTION_TAGS:
        similarity = _dot(unit, direction)
        if undirected:
            similarity = abs(similarity)
        if similarity >= COS_THRESHOLD:
            return list(labels)
    return []


@dataclass(frozen=True)
class BoundingBox:
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    @property
    def size(self) -> Vec3:
        return (
            self.max_x - self.min_x,
            self.max_y - self.min_y,
            self.max_z - self.min_z,
        )

    @property
    def centre(self) -> Vec3:
        return (
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
            (self.min_z + self.max_z) / 2.0,
        )


@dataclass(frozen=True)
class SubshapeRef:
    id: str
    kind: str  # "face" | "edge"
    index: int
    centroid: Vec3
    normal: Optional[Vec3] = None
    direction: Optional[Vec3] = None
    area: Optional[float] = None
    length: Optional[float] = None
    surface: str = "unknown"  # planar | cylindrical | spherical | line | circle
    tags: Tuple[str, ...] = ()


@dataclass
class TopologyMap:
    shape_id: str
    kind: str
    faces: List[SubshapeRef] = field(default_factory=list)
    edges: List[SubshapeRef] = field(default_factory=list)

    def face(self, index: int) -> SubshapeRef:
        return self.faces[index]

    def by_tag(self, tag: str) -> List[SubshapeRef]:
        return [r for r in self.faces + self.edges if tag in r.tags]

    def ids(self) -> List[str]:
        return [r.id for r in self.faces] + [r.id for r in self.edges]


# ── faces ───────────────────────────────────────────────────────────


def synthetic_box_faces(shape_id: str, bbox: BoundingBox) -> List[SubshapeRef]:
    """The six canonical faces of an axis-aligned box, in +Z,-Z,+Y,-Y,+X,-X order."""
    cx, cy, cz = bbox.centre
    dx, dy, dz = bbox.size
    defs: Sequence[Tuple[Vec3, Vec3, float]] = (
        ((cx, cy, bbox.max_z), (0.0, 0.0, 1.0), dx * dy),
        ((cx, cy, bbox.min_z), (0.0, 0.0, -1.0), dx * dy),
        ((cx, bbox.max_y, cz), (0.0, 1.0, 0.0), dx * dz),
        ((cx, bbox.min_y, cz), (0.0, -1.0, 0.0), dx * dz),
        ((bbox.max_x, cy, cz), (1.0, 0.0, 0.0), dy * dz),
        ((bbox.min_x, cy, cz), (-1.0, 0.0, 0.0), dy * dz),
    )
    return [
        SubshapeRef(
            id="%s:face:%d" % (shape_id, index),
            kind="face",
            index=index,
            centroid=centroid,
            normal=normal,
            area=area,
            surface="planar",
            tags=tuple(direction_tags(normal)),
        )
        for index, (centroid, normal, area) in enumerate(defs)
    ]


def synthetic_cylinder_faces(shape_id: str, bbox: BoundingBox) -> List[SubshapeRef]:
    """Top cap, bottom cap and the lateral surface of a Z-aligned cylinder."""
    cx, cy, cz = bbox.centre
    dx, _, dz = bbox.size
    radius = dx / 2.0
    cap_area = math.pi * radius * radius
    lateral_area = 2.0 * math.pi * radius * dz
    return [
        SubshapeRef(
            id="%s:face:0" % shape_id,
            kind="face",
            index=0,
            centroid=(cx, cy, bbox.max_z),
            normal=(0.0, 0.0, 1.0),
            area=cap_area,
            surface="planar",
            tags=("top", "+Z", "cap"),
        ),
        SubshapeRef(
            id="%s:face:1" % shape_id,
            kind="face",
            index=1,
            centroid=(cx, cy, bbox.min_z),
            normal=(0.0, 0.0, -1.0),
            area=cap_area,
            surface="planar",
            tags=("bottom", "-Z", "cap"),
        ),
        SubshapeRef(
            id="%s:face:2" % shape_id,
            kind="face",
            index=2,
            centroid=(cx, cy, cz),
            normal=None,
            area=lateral_area,
            surface="cylindrical",
            tags=("lateral",),
        ),
    ]


def synthetic_sphere_faces(shape_id: str, bbox: BoundingBox) -> List[SubshapeRef]:
    cx, cy, cz = bbox.centre
    radius = bbox.size[0] / 2.0
    return [
        SubshapeRef(
            id="%s:face:0" % shape_id,
            kind="face",
            index=0,
            centroid=(cx, cy, cz),
            normal=None,
            area=4.0 * math.pi * radius * radius,
            surface="spherical",
            tags=("spherical",),
        )
    ]


# ── edges ───────────────────────────────────────────────────────────

_BOX_CORNERS = (
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
)

_BOX_EDGE_PAIRS = (
    (0, 1), (1, 2), (2, 3), (3, 0),      # bottom loop
    (4, 5), (5, 6), (6, 7), (7, 4),      # top loop
    (0, 4), (1, 5), (2, 6), (3, 7),      # vertical pillars
)


def _corner(bbox: BoundingBox, code: Tuple[int, int, int]) -> Vec3:
    return (
        bbox.max_x if code[0] else bbox.min_x,
        bbox.max_y if code[1] else bbox.min_y,
        bbox.max_z if code[2] else bbox.min_z,
    )


def synthetic_box_edges(shape_id: str, bbox: BoundingBox) -> List[SubshapeRef]:
    """The 12 edges of an axis-aligned box with real endpoints, lengths and tags."""
    refs: List[SubshapeRef] = []
    for index, (a_code, b_code) in enumerate(_BOX_EDGE_PAIRS):
        a = _corner(bbox, _BOX_CORNERS[a_code])
        b = _corner(bbox, _BOX_CORNERS[b_code])
        vector = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        length = _length(vector)
        midpoint = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0)
        tags = list(direction_tags(vector, undirected=True))
        if "+Z" in tags or "-Z" in tags:
            tags.append("vertical")
        elif tags:
            tags.append("horizontal")
        refs.append(
            SubshapeRef(
                id="%s:edge:%d" % (shape_id, index),
                kind="edge",
                index=index,
                centroid=midpoint,
                direction=_normalise(vector),
                length=length,
                surface="line",
                tags=tuple(tags),
            )
        )
    return refs


def synthetic_cylinder_edges(shape_id: str, bbox: BoundingBox) -> List[SubshapeRef]:
    """Top rim, bottom rim and the lateral seam of a Z-aligned cylinder."""
    cx, cy, cz = bbox.centre
    radius = bbox.size[0] / 2.0
    height = bbox.size[2]
    circumference = 2.0 * math.pi * radius
    return [
        SubshapeRef(
            id="%s:edge:0" % shape_id,
            kind="edge",
            index=0,
            centroid=(cx, cy, bbox.max_z),
            direction=None,
            length=circumference,
            surface="circle",
            tags=("top", "+Z", "rim"),
        ),
        SubshapeRef(
            id="%s:edge:1" % shape_id,
            kind="edge",
            index=1,
            centroid=(cx, cy, bbox.min_z),
            direction=None,
            length=circumference,
            surface="circle",
            tags=("bottom", "-Z", "rim"),
        ),
        SubshapeRef(
            id="%s:edge:2" % shape_id,
            kind="edge",
            index=2,
            centroid=(cx + radius, cy, cz),
            direction=(0.0, 0.0, 1.0),
            length=height,
            surface="line",
            tags=("seam", "vertical", "+Z"),
        ),
    ]


# ── dispatch ────────────────────────────────────────────────────────


def build_topology(shape_id: str, kind: str, bbox: BoundingBox) -> TopologyMap:
    """Synthesise the subshape map for a primitive (box / cylinder / sphere)."""
    normalised = kind.lower()
    if normalised == "cylinder":
        faces = synthetic_cylinder_faces(shape_id, bbox)
        edges = synthetic_cylinder_edges(shape_id, bbox)
    elif normalised == "sphere":
        faces = synthetic_sphere_faces(shape_id, bbox)
        edges = []
    else:  # box and anything box-like
        faces = synthetic_box_faces(shape_id, bbox)
        edges = synthetic_box_edges(shape_id, bbox)
    return TopologyMap(shape_id=shape_id, kind=normalised, faces=faces, edges=edges)


def topology_summary(topology: TopologyMap) -> Dict[str, float]:
    """Aggregate metrics (face/edge counts, total area, total edge length)."""
    return {
        "face_count": float(len(topology.faces)),
        "edge_count": float(len(topology.edges)),
        "total_area": float(sum(f.area or 0.0 for f in topology.faces)),
        "total_edge_length": float(sum(e.length or 0.0 for e in topology.edges)),
    }
