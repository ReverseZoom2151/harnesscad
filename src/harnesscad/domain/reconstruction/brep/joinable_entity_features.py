"""Deterministic B-Rep entity features for joint prediction (JoinABLe).

Every B-Rep face and edge of a body becomes a node whose feature vector is
derived *deterministically* from the entity's geometry -- no learning involved.
The feature vector JoinABLe feeds its graph network per entity is:

* ``entity_types``  -- one-hot over the 16 surface/curve types (plane, cylinder,
  cone, sphere, torus, elliptical cylinder/cone, NURBS surface; line, arc,
  circle, ellipse, elliptical arc, infinite line, NURBS curve, degenerate);
* ``area``          -- face area (0 for edges);
* ``length``        -- edge length (0 for faces);
* ``face_reversed`` / ``edge_reversed`` / ``reversed`` -- orientation flags;
* ``convexity``     -- one-hot over None/Convex/Concave/Smooth/Non-manifold/
  Degenerate (faces always carry ``None``);
* ``dihedral_angle`` -- edge dihedral angle in radians (0 for faces);
* ``radius``        -- radius of circles and cylinders, ``-1`` when undefined.

Two bodies are compared in a *common* frame: the point extents of both bodies
are pooled and a single scale factor brings them into the unit box, so that the
features of body one and body two are commensurate.  Also provided is the
entity-label vocabulary (Non-joint / Joint / Ambiguous / JointEquivalent /
AmbiguousEquivalent / Hole / HoleEquivalent) and the candidate-pair enumeration
plus its label matrix, which is the input the joint metrics consume.
Stdlib only.
"""

__all__ = [
    "ENTITY_TYPES",
    "ENTITY_TYPE_INDEX",
    "SURFACE_TYPES",
    "CURVE_TYPES",
    "CONVEXITY_TYPES",
    "CONVEXITY_INDEX",
    "LABEL_TYPES",
    "LABEL_INDEX",
    "POSITIVE_LABELS",
    "EntityFeatureError",
    "one_hot",
    "entity_type_name",
    "entity_type_index",
    "is_face",
    "entity_area",
    "entity_length",
    "entity_size",
    "entity_reversed_flags",
    "entity_convexity",
    "entity_dihedral_angle",
    "entity_radius",
    "entity_feature_vector",
    "entity_feature_names",
    "bounding_box",
    "common_scale",
    "scale_features",
    "label_index",
    "is_positive_label",
    "candidate_pairs",
    "candidate_label_matrix",
]

#: Surface types, then curve types -- the one-hot order used by JoinABLe.
SURFACE_TYPES = (
    "PlaneSurfaceType",
    "CylinderSurfaceType",
    "ConeSurfaceType",
    "SphereSurfaceType",
    "TorusSurfaceType",
    "EllipticalCylinderSurfaceType",
    "EllipticalConeSurfaceType",
    "NurbsSurfaceType",
)
CURVE_TYPES = (
    "Line3DCurveType",
    "Arc3DCurveType",
    "Circle3DCurveType",
    "Ellipse3DCurveType",
    "EllipticalArc3DCurveType",
    "InfiniteLine3DCurveType",
    "NurbsCurve3DCurveType",
    "Degenerate3DCurveType",
)
ENTITY_TYPES = SURFACE_TYPES + CURVE_TYPES
ENTITY_TYPE_INDEX = {name: i for i, name in enumerate(ENTITY_TYPES)}

CONVEXITY_TYPES = ("None", "Convex", "Concave", "Smooth", "Non-manifold",
                   "Degenerate")
CONVEXITY_INDEX = {name: i for i, name in enumerate(CONVEXITY_TYPES)}

LABEL_TYPES = ("Non-joint", "Joint", "Ambiguous", "JointEquivalent",
               "AmbiguousEquivalent", "Hole", "HoleEquivalent")
LABEL_INDEX = {name: i for i, name in enumerate(LABEL_TYPES)}

#: Labels that count as a ground-truth joint entity for top-k evaluation.
POSITIVE_LABELS = ("Joint", "JointEquivalent")

#: Radius reported when the entity has none (planes, lines, ...).
UNDEFINED_RADIUS = -1.0


class EntityFeatureError(ValueError):
    """Raised when an entity cannot be turned into a feature vector."""


def one_hot(index, size):
    """A ``size``-long one-hot list with 1.0 at ``index``."""
    if not 0 <= index < size:
        raise EntityFeatureError(f"one-hot index {index} out of range {size}")
    vector = [0.0] * size
    vector[index] = 1.0
    return vector


def is_face(entity):
    """True when the entity is a B-Rep face."""
    return "surface_type" in entity


def entity_type_name(entity):
    """The surface or curve type of the entity, degenerate edges included."""
    if "surface_type" in entity:
        name = entity["surface_type"]
    elif entity.get("is_degenerate"):
        name = "Degenerate3DCurveType"
    elif "curve_type" in entity:
        name = entity["curve_type"]
    else:
        raise EntityFeatureError("entity has neither surface_type nor "
                                 "curve_type")
    if name not in ENTITY_TYPE_INDEX:
        raise EntityFeatureError(f"unknown entity type: {name!r}")
    return name


def entity_type_index(entity):
    """Index of the entity's type in :data:`ENTITY_TYPES`."""
    return ENTITY_TYPE_INDEX[entity_type_name(entity)]


def entity_area(entity):
    """Face area; 0.0 for edges or when the area is missing."""
    if is_face(entity):
        return float(entity.get("area", 0.0))
    return 0.0


def entity_length(entity):
    """Edge length; 0.0 for faces or when the length is missing."""
    if is_face(entity):
        return 0.0
    return float(entity.get("length", 0.0))


def entity_size(entity):
    """The entity's area (face) or length (edge) -- its natural size measure."""
    return entity_area(entity) if is_face(entity) else entity_length(entity)


def entity_reversed_flags(entity):
    """``(face_reversed, edge_reversed, reversed)`` orientation flags."""
    reversed_flag = 1 if entity.get("reversed") else 0
    if is_face(entity):
        return reversed_flag, 0, reversed_flag
    return 0, reversed_flag, reversed_flag


def entity_convexity(entity):
    """Convexity name of the entity; faces are always ``"None"``."""
    if is_face(entity):
        return "None"
    if entity.get("is_degenerate"):
        return "Degenerate"
    name = entity.get("convexity", "None")
    if name not in CONVEXITY_INDEX:
        raise EntityFeatureError(f"unknown convexity: {name!r}")
    return name


def entity_dihedral_angle(entity):
    """Edge dihedral angle in radians; 0.0 for faces or when missing."""
    if is_face(entity):
        return 0.0
    return float(entity.get("dihedral_angle", 0.0))


def entity_radius(entity):
    """Radius of circles / cylinders; ``-1.0`` when the entity has none."""
    if "radius" not in entity:
        return UNDEFINED_RADIUS
    return float(entity["radius"])


def entity_feature_names():
    """Names of the components of :func:`entity_feature_vector`, in order."""
    names = [f"entity_type::{name}" for name in ENTITY_TYPES]
    names += ["is_face", "area", "length", "face_reversed", "edge_reversed",
              "reversed"]
    names += [f"convexity::{name}" for name in CONVEXITY_TYPES]
    names += ["dihedral_angle", "radius"]
    return names


def entity_feature_vector(entity):
    """The full deterministic feature vector for one B-Rep entity."""
    vector = one_hot(entity_type_index(entity), len(ENTITY_TYPES))
    face_reversed, edge_reversed, reversed_flag = entity_reversed_flags(entity)
    vector.append(1.0 if is_face(entity) else 0.0)
    vector.append(entity_area(entity))
    vector.append(entity_length(entity))
    vector.append(float(face_reversed))
    vector.append(float(edge_reversed))
    vector.append(float(reversed_flag))
    vector.extend(one_hot(CONVEXITY_INDEX[entity_convexity(entity)],
                          len(CONVEXITY_TYPES)))
    vector.append(entity_dihedral_angle(entity))
    vector.append(entity_radius(entity))
    return vector


# --------------------------------------------------------------------------- #
# Common scaling of a pair of bodies
# --------------------------------------------------------------------------- #
def bounding_box(points):
    """Axis-aligned ``(min_xyz, max_xyz)`` of a sequence of 3D points."""
    points = [tuple(float(c) for c in p) for p in points]
    if not points:
        raise EntityFeatureError("no points to bound")
    mins = tuple(min(p[i] for p in points) for i in range(3))
    maxs = tuple(max(p[i] for p in points) for i in range(3))
    return mins, maxs


def common_scale(points1, points2, epsilon=1e-6):
    """Single scale factor bringing both bodies inside the unit box.

    Follows JoinABLe: the factor is ``0.999999 / max(|coordinate|)`` over the
    pooled bounding boxes of the two bodies, so both are scaled identically and
    relative placement is preserved.
    """
    box1 = bounding_box(points1)
    box2 = bounding_box(points2)
    extent = max(abs(c) for box in (box1, box2) for corner in box
                 for c in corner)
    if extent < epsilon:
        return 1.0
    return (1.0 / extent) * 0.999999


def scale_features(scale, points=None, areas=None, lengths=None):
    """Apply a common scale to points and to the area / length features.

    Areas and lengths are linear measures in JoinABLe's feature vector and are
    scaled by the same factor as the coordinates.
    """
    scale = float(scale)
    scaled_points = None
    if points is not None:
        scaled_points = [tuple(float(c) * scale for c in p) for p in points]
    scaled_areas = None
    if areas is not None:
        scaled_areas = [float(a) * scale for a in areas]
    scaled_lengths = None
    if lengths is not None:
        scaled_lengths = [float(v) * scale for v in lengths]
    return scaled_points, scaled_areas, scaled_lengths


# --------------------------------------------------------------------------- #
# Candidate entity pairs and their labels
# --------------------------------------------------------------------------- #
def label_index(name):
    """Index of an entity label in :data:`LABEL_TYPES`."""
    if name not in LABEL_INDEX:
        raise EntityFeatureError(f"unknown label: {name!r}")
    return LABEL_INDEX[name]


def is_positive_label(name):
    """True for labels that count as a ground-truth joint entity."""
    return name in POSITIVE_LABELS


def candidate_pairs(entities1, entities2):
    """Every ``(i, j)`` candidate entity pair, in row-major order."""
    return [(i, j) for i in range(len(entities1))
            for j in range(len(entities2))]


def candidate_label_matrix(entities1, entities2, joint_pairs):
    """0/1 label matrix over candidate pairs given the ground-truth pairs.

    ``joint_pairs`` is the set of ``(index_in_body1, index_in_body2)`` pairs
    that form the joint, including its equivalents.  The result is a list of
    rows suitable for :mod:`bench.joinable_joint_metrics`.
    """
    truth = set()
    for pair in joint_pairs:
        i, j = int(pair[0]), int(pair[1])
        if not 0 <= i < len(entities1) or not 0 <= j < len(entities2):
            raise EntityFeatureError(f"joint pair {(i, j)} out of range")
        truth.add((i, j))
    return [[1 if (i, j) in truth else 0 for j in range(len(entities2))]
            for i in range(len(entities1))]
