"""Structured CAD model schema for CADTESTS (Mallis et al., "Text-to-CAD
Evaluation with CADTESTS", Sec. 3-4).

The paper views Text-to-CAD as code synthesis and evaluates a generated CAD
model with executable *property tests* (CADTESTS) that query the Boundary
Representation (B-rep) of the model through the CadQuery API: topology counts,
bounding-box dimensions, face areas, volumes, center-of-mass, geometric-type
presence and spatial relationships (Sec. 4, supplementary Sec. A). Formally a
B-rep is ``m = (F, E, V)`` -- sets of faces, edges and vertices -- and a
CADTEST is a boolean predicate ``T_i : M -> {0, 1}`` over that structure.

This module provides the DETERMINISTIC, stdlib-only structured model that the
CADTESTS assertion primitives (``bench/cadtests_assertions``) query. It stands in
for the CadQuery B-rep inspection surface without the heavy geometry kernel, so
generation is external and the model is *injected*.

A key requirement of the pipeline is *pose and scale invariance*: the passing set
augments the reference with similarity transforms ``a = t . r . s`` (uniform
scale, a 90-degree rotation, and a translation, Sec. 5). Well-formed CADTESTS
must be invariant to these. This module therefore also exposes
:meth:`CADModel.transformed`, applying such a similarity transform so that a
test-runner can build the passing set and check invariance.

No wall clock, no randomness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

# Recognised B-rep surface types and edge (curve) types. The set mirrors the
# "Geometric Types" category of the paper (planar/cylindrical faces, circular/
# straight edges, etc., supplementary Sec. A).
FACE_TYPES = ("plane", "cylinder", "cone", "sphere", "torus")
EDGE_TYPES = ("line", "circle", "arc", "ellipse", "spline")

_AXES = {"x": 0, "y": 1, "z": 2}


def _axis_index(axis):
    """Normalise an axis given as 0/1/2 or 'x'/'y'/'z' into an index."""
    if isinstance(axis, str):
        key = axis.strip().lower()
        if key not in _AXES:
            raise ValueError("unknown axis: %r" % (axis,))
        return _AXES[key]
    idx = int(axis)
    if idx not in (0, 1, 2):
        raise ValueError("axis index must be 0, 1 or 2, got %r" % (axis,))
    return idx


@dataclass(frozen=True)
class Face:
    """A B-rep face: its surface type and (positive) area."""
    type: str
    area: float

    def __post_init__(self):
        if self.type not in FACE_TYPES:
            raise ValueError("unknown face type: %r" % (self.type,))
        if float(self.area) < 0.0:
            raise ValueError("face area must be non-negative, got %r"
                             % (self.area,))

    def scaled(self, s):
        """Area scales with the square of a uniform linear scale factor."""
        return Face(self.type, float(self.area) * (float(s) ** 2))


@dataclass(frozen=True)
class Edge:
    """A B-rep edge: its curve type and length."""
    type: str
    length: float = 0.0

    def __post_init__(self):
        if self.type not in EDGE_TYPES:
            raise ValueError("unknown edge type: %r" % (self.type,))
        if float(self.length) < 0.0:
            raise ValueError("edge length must be non-negative, got %r"
                             % (self.length,))

    def scaled(self, s):
        return Edge(self.type, float(self.length) * float(s))


@dataclass(frozen=True)
class CADModel:
    """A structured, queryable B-rep model.

    Fields:
      solids       -- number of disconnected solid bodies.
      shells       -- number of connected shells.
      faces        -- tuple of :class:`Face`.
      edges        -- tuple of :class:`Edge`.
      vertices     -- number of vertices.
      bbox_min     -- (x, y, z) lower corner of the axis-aligned bounding box.
      bbox_size    -- (dx, dy, dz) extents of the bounding box (>= 0).
      volume       -- solid volume.
      center_of_mass -- (x, y, z) centroid.

    All geometric queries used by CADTESTS are exposed as methods.
    """
    faces: Tuple[Face, ...]
    edges: Tuple[Edge, ...]
    vertices: int
    bbox_min: Tuple[float, float, float]
    bbox_size: Tuple[float, float, float]
    volume: float
    center_of_mass: Tuple[float, float, float]
    solids: int = 1
    shells: int = 1

    def __post_init__(self):
        if len(self.bbox_min) != 3 or len(self.bbox_size) != 3:
            raise ValueError("bbox_min and bbox_size must be 3-tuples")
        if len(self.center_of_mass) != 3:
            raise ValueError("center_of_mass must be a 3-tuple")
        if any(d < 0.0 for d in self.bbox_size):
            raise ValueError("bbox_size components must be non-negative")
        if self.solids < 0 or self.shells < 0 or self.vertices < 0:
            raise ValueError("counts must be non-negative")

    # -- topology counts ---------------------------------------------------
    def num_solids(self):
        return int(self.solids)

    def num_shells(self):
        return int(self.shells)

    def num_faces(self):
        return len(self.faces)

    def num_edges(self):
        return len(self.edges)

    def num_vertices(self):
        return int(self.vertices)

    def count_faces_of_type(self, face_type):
        if face_type not in FACE_TYPES:
            raise ValueError("unknown face type: %r" % (face_type,))
        return sum(1 for f in self.faces if f.type == face_type)

    def count_edges_of_type(self, edge_type):
        if edge_type not in EDGE_TYPES:
            raise ValueError("unknown edge type: %r" % (edge_type,))
        return sum(1 for e in self.edges if e.type == edge_type)

    def has_face_type(self, face_type):
        return self.count_faces_of_type(face_type) > 0

    def has_edge_type(self, edge_type):
        return self.count_edges_of_type(edge_type) > 0

    # -- geometry / dimensions --------------------------------------------
    def dimension(self, axis):
        """Bounding-box extent along an axis (0/1/2 or 'x'/'y'/'z')."""
        return float(self.bbox_size[_axis_index(axis)])

    def largest_axis(self):
        """Index of the axis with the greatest extent (ties -> lowest index)."""
        return max(range(3), key=lambda i: self.bbox_size[i])

    def smallest_axis(self):
        return min(range(3), key=lambda i: self.bbox_size[i])

    def aspect_ratio(self, axis_a, axis_b):
        """Ratio of extent along ``axis_a`` to extent along ``axis_b``."""
        db = self.dimension(axis_b)
        if db == 0.0:
            raise ZeroDivisionError("zero extent along axis %r" % (axis_b,))
        return self.dimension(axis_a) / db

    def bbox_center(self):
        return tuple(self.bbox_min[i] + self.bbox_size[i] / 2.0
                     for i in range(3))

    def face_area(self, index):
        return float(self.faces[index].area)

    def total_face_area(self):
        return sum(f.area for f in self.faces)

    # -- volumetric / mass -------------------------------------------------
    def get_volume(self):
        return float(self.volume)

    def get_center_of_mass(self):
        return tuple(float(c) for c in self.center_of_mass)

    def bbox_volume(self):
        dx, dy, dz = self.bbox_size
        return float(dx) * float(dy) * float(dz)

    def fill_factor(self):
        """Volume as a fraction of the bounding-box volume (shape factor)."""
        bv = self.bbox_volume()
        if bv == 0.0:
            raise ZeroDivisionError("zero bounding-box volume")
        return self.get_volume() / bv

    # -- solid / shell validity -------------------------------------------
    def is_valid_solid(self):
        """A single, non-degenerate, positive-volume solid (Sec. A validity)."""
        return (self.solids == 1 and self.shells >= 1
                and self.volume > 0.0
                and all(d > 0.0 for d in self.bbox_size))

    # -- similarity transform (pose/scale augmentation, Sec. 5) -----------
    def transformed(self, *, scale=1.0, rotate_axis=None, translate=(0.0, 0.0, 0.0)):
        """Return a new model under a similarity transform ``a = t . r . s``.

        ``scale`` is a uniform positive linear factor; ``rotate_axis`` is one of
        None/'x'/'y'/'z' for a 90-degree rotation about that axis; ``translate``
        is a 3-vector added after scaling and rotation. Geometric *types* are
        preserved (a plane stays a plane), so well-formed CADTESTS are invariant
        to this transform -- exactly the property the passing-set augmentation
        of Sec. 5 is designed to enforce.
        """
        s = float(scale)
        if s <= 0.0:
            raise ValueError("scale must be positive, got %r" % (scale,))

        # Scale: linear dims * s, areas * s^2, volume * s^3.
        faces = tuple(f.scaled(s) for f in self.faces)
        edges = tuple(e.scaled(s) for e in self.edges)
        size = [d * s for d in self.bbox_size]
        com = [c * s for c in self.center_of_mass]
        bmin = [c * s for c in self.bbox_min]
        volume = self.volume * (s ** 3)

        # 90-degree rotation: permute the two axes orthogonal to the rotation
        # axis (types and counts are rotation-invariant).
        if rotate_axis is not None:
            i = _axis_index(rotate_axis)
            a, b = [k for k in range(3) if k != i]
            for vec in (size, com, bmin):
                vec[a], vec[b] = vec[b], vec[a]

        # Translation.
        t = tuple(float(x) for x in translate)
        if len(t) != 3:
            raise ValueError("translate must be a 3-vector")
        com = [com[k] + t[k] for k in range(3)]
        bmin = [bmin[k] + t[k] for k in range(3)]

        return CADModel(
            faces=faces,
            edges=edges,
            vertices=self.vertices,
            bbox_min=tuple(bmin),
            bbox_size=tuple(size),
            volume=volume,
            center_of_mass=tuple(com),
            solids=self.solids,
            shells=self.shells,
        )


def similarity_augmentations(model, *, scales=(1.0,), rotations=(None,),
                             translations=((0.0, 0.0, 0.0),),
                             include_scale=True):
    """Build a passing set of similarity-transformed variants of ``model``.

    Mirrors the augmentation of Sec. 5: small scaling, translation and
    90-degree rotations. When ``include_scale`` is False (the *detailed*
    partition, where prompts fix exact dimensions), uniform scaling is omitted.
    Returns a tuple beginning with the reference ``model`` itself.
    """
    variants = [model]
    use_scales = scales if include_scale else (1.0,)
    for s in use_scales:
        for r in rotations:
            for t in translations:
                if s == 1.0 and r is None and t == (0.0, 0.0, 0.0):
                    continue
                variants.append(model.transformed(scale=s, rotate_axis=r,
                                                   translate=t))
    return tuple(variants)
