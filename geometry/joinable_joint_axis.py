"""Joint-axis geometry for B-Rep entities (JoinABLe, CVPR 2022).

A *joint axis* is the infinite line ``(origin, direction)`` that a B-Rep entity
(a face or an edge) contributes to a joint.  Two entities -- one from each body
-- define a joint when their axis lines can be made colinear.  This module
derives that axis line deterministically from the entity's parametric
description, and provides the colinearity / distance predicates needed to test
whether a predicted joint axis matches a ground-truth one.

Entities are plain dicts, as in the Fusion 360 Gallery joint dataset:

* a face carries ``surface_type`` plus the parameters of that surface
  (``origin``/``axis`` for cylinders, cones, tori; ``centroid``/``normal`` for
  planes);
* an edge carries ``curve_type`` plus its parameters (``start_point`` /
  ``end_point`` for lines, ``center`` / ``normal`` for arcs and circles).

Points and vectors may be given either as ``{"x":..,"y":..,"z":..}`` dicts or
as any 3-sequence.  Stdlib only.
"""

import math

__all__ = [
    "SURFACE_AXIS_SOURCES",
    "CURVE_AXIS_SOURCES",
    "AxisLineError",
    "as_vec3",
    "normalize",
    "vec_sub",
    "vec_add",
    "vec_scale",
    "dot",
    "cross",
    "norm",
    "angle_between",
    "direction_between",
    "dist_point_to_line",
    "closest_point_on_line",
    "find_axis_line",
    "find_axis_line_from_face",
    "find_axis_line_from_edge",
    "axis_lines_colinear",
    "joint_axis_error",
]

# Surface type -> (origin-key, direction-key).  A plane's axis passes through
# its centroid along its normal; every quadric of revolution uses its own
# origin and axis of revolution.
SURFACE_AXIS_SOURCES = {
    "PlaneSurfaceType": ("centroid", "normal"),
    "CylinderSurfaceType": ("origin", "axis"),
    "EllipticalCylinderSurfaceType": ("origin", "axis"),
    "ConeSurfaceType": ("origin", "axis"),
    "EllipticalConeSurfaceType": ("origin", "axis"),
    "TorusSurfaceType": ("origin", "axis"),
    # A sphere has no distinguished axis; JoinABLe uses +Z through the origin.
    "SphereSurfaceType": ("origin", None),
}

# Curve type -> (origin-key, direction-key).  ``None`` for the direction of a
# line means "derive it from start_point -> end_point".
CURVE_AXIS_SOURCES = {
    "Line3DCurveType": ("start_point", None),
    "Arc3DCurveType": ("center", "normal"),
    "Circle3DCurveType": ("center", "normal"),
    "Ellipse3DCurveType": ("center", "normal"),
    "EllipticalArc3DCurveType": ("center", "normal"),
}

_DEFAULT_SPHERE_AXIS = (0.0, 0.0, 1.0)


class AxisLineError(ValueError):
    """Raised when no joint axis can be derived from an entity."""


# --------------------------------------------------------------------------- #
# Small vector helpers (stdlib only)
# --------------------------------------------------------------------------- #
def as_vec3(value, name=None):
    """Coerce a point/vector to a ``(x, y, z)`` float tuple.

    ``value`` may be a dict with ``x``/``y``/``z`` keys, a dict with
    ``<name>_x`` style keys when ``name`` is given, or any 3-sequence.
    """
    if isinstance(value, dict):
        if name is not None and f"{name}_x" in value:
            keys = (f"{name}_x", f"{name}_y", f"{name}_z")
        else:
            keys = ("x", "y", "z")
        try:
            return (float(value[keys[0]]), float(value[keys[1]]),
                    float(value[keys[2]]))
        except KeyError as exc:  # pragma: no cover - defensive
            raise AxisLineError(f"missing component {exc} in {value!r}")
    seq = list(value)
    if len(seq) != 3:
        raise AxisLineError("points and vectors must have 3 components")
    return (float(seq[0]), float(seq[1]), float(seq[2]))


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def norm(v):
    return math.sqrt(dot(v, v))


def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec_scale(v, s):
    return (v[0] * s, v[1] * s, v[2] * s)


def normalize(v):
    """Unit vector; a zero vector is returned unchanged (as in JoinABLe)."""
    length = norm(v)
    if length == 0.0:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def direction_between(p1, p2):
    """Unit direction from ``p1`` to ``p2``; zero vector for coincident points."""
    return normalize(vec_sub(p2, p1))


def angle_between(v1, v2):
    """Angle in radians between two vectors, clamped against round-off."""
    denom = norm(v1) * norm(v2)
    if denom == 0.0:
        return 0.0
    c = dot(v1, v2) / denom
    c = max(-1.0, min(1.0, c))
    return math.acos(c)


def closest_point_on_line(point, line_start, line_direction):
    """Foot of the perpendicular from ``point`` onto the infinite line."""
    d = line_direction
    dd = dot(d, d)
    if dd == 0.0:
        return tuple(line_start)
    t = dot(vec_sub(point, line_start), d) / dd
    return vec_add(line_start, vec_scale(d, t))


def dist_point_to_line(point, line_start, line_direction):
    """Perpendicular distance from a point to an infinite line."""
    foot = closest_point_on_line(point, line_start, line_direction)
    return norm(vec_sub(point, foot))


# --------------------------------------------------------------------------- #
# Axis line derivation
# --------------------------------------------------------------------------- #
def find_axis_line_from_face(face):
    """``(origin, direction)`` of the axis line of a B-Rep face."""
    surface_type = face.get("surface_type")
    if surface_type not in SURFACE_AXIS_SOURCES:
        raise AxisLineError(f"joint axis not supported for {surface_type!r}")
    origin_key, direction_key = SURFACE_AXIS_SOURCES[surface_type]
    origin = as_vec3(face[origin_key]) if origin_key in face \
        else as_vec3(face, origin_key)
    if direction_key is None:
        direction = _DEFAULT_SPHERE_AXIS
    elif direction_key in face:
        direction = normalize(as_vec3(face[direction_key]))
    else:
        direction = normalize(as_vec3(face, direction_key))
    return origin, direction


def find_axis_line_from_edge(edge):
    """``(origin, direction)`` of the axis line of a B-Rep edge."""
    if edge.get("is_degenerate"):
        raise AxisLineError("joint axis not supported for degenerate edges")
    curve_type = edge.get("curve_type")
    if curve_type not in CURVE_AXIS_SOURCES:
        raise AxisLineError(f"joint axis not supported for {curve_type!r}")
    origin_key, direction_key = CURVE_AXIS_SOURCES[curve_type]
    origin = as_vec3(edge[origin_key]) if origin_key in edge \
        else as_vec3(edge, origin_key)
    if direction_key is None:
        # A linear edge: the axis runs start -> end.
        end = as_vec3(edge["end_point"]) if "end_point" in edge \
            else as_vec3(edge, "end_point")
        direction = direction_between(origin, end)
    elif direction_key in edge:
        direction = normalize(as_vec3(edge[direction_key]))
    else:
        direction = normalize(as_vec3(edge, direction_key))
    return origin, direction


def find_axis_line(entity):
    """Dispatch on ``surface_type`` / ``curve_type`` to derive the axis line."""
    if "surface_type" in entity:
        return find_axis_line_from_face(entity)
    if "curve_type" in entity or entity.get("is_degenerate"):
        return find_axis_line_from_edge(entity)
    raise AxisLineError("entity has neither surface_type nor curve_type")


# --------------------------------------------------------------------------- #
# Axis comparison
# --------------------------------------------------------------------------- #
def joint_axis_error(axis_line1, axis_line2):
    """``(angle_degrees, distance)`` between two axis lines.

    The angle is direction-agnostic (folded into ``[0, 90]``) because a joint
    axis is an unoriented line.  The distance is the perpendicular distance
    from the first origin to the second line -- zero for colinear lines.
    """
    origin1, direction1 = axis_line1
    origin2, direction2 = axis_line2
    origin1 = as_vec3(origin1)
    origin2 = as_vec3(origin2)
    direction1 = as_vec3(direction1)
    direction2 = as_vec3(direction2)
    angle = angle_between(direction1, direction2)
    reversed_angle = angle_between(direction1, vec_scale(direction2, -1.0))
    angle_degs = math.degrees(min(angle, reversed_angle))
    if norm(direction2) == 0.0:
        distance = norm(vec_sub(origin1, origin2))
    else:
        distance = dist_point_to_line(origin1, origin2, direction2)
    return angle_degs, distance


def axis_lines_colinear(axis_line1, axis_line2, angle_tol_degs=10.0,
                        distance_tol=1e-2):
    """True when the two axis lines are colinear within the given tolerances."""
    angle_degs, distance = joint_axis_error(axis_line1, axis_line2)
    return angle_degs < angle_tol_degs and distance < distance_tol
