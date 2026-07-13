"""t2cadtd_isometric_projection — axonometric (isometric) projection of a solid.

Text2CAD (Yavartanoo et al., "Text to 3D CAD Generation via Technical Drawings",
SNU) uses an *isometric image* as the intermediate representation between the
text prompt and the orthographic technical drawings: text -> isometric ->
{top, front, side} -> 3D CAD. The paper fixes the isometric viewpoint precisely
(Experiments, "To render isometric images"):

    "the viewpoint is set at a 45-degree angle above the horizontal plane
     combined with a 45-degree rotation around the vertical axis. This
     perspective provides a clear and comprehensive view of the three-
     dimensional structure of the object without distortion."

and standardises scale (dataset section): "Each object is scaled so that its
longest edge measures precisely 2 units."

That viewpoint transform is a deterministic axonometric (parallel) projection —
no learned component. This module implements it for the axis-aligned box solids
used across :mod:`drawings.creft_projection`:

  * :func:`project_point` — orthographic axonometric projection of a 3D point to
    a 2D screen coordinate, parametrised by azimuth (rotation about the vertical
    Z axis) and elevation (angle above the horizontal), defaulting to the paper's
    45/45 viewpoint. A true-isometric elevation constant is also provided.
  * :func:`projected_axis_vectors` / :func:`axis_foreshortening` — the 2D vector
    (and its length) that each unit 3D axis maps to; the foreshortening factors
    are what :mod:`drawings.t2cadtd_iso_ortho_consistency` inverts.
  * :func:`project_box` — the eight projected corners of a box.
  * :func:`isometric_outline` — the convex-hull silhouette (a hexagon for a
    generic box) of the projected corners.
  * :func:`visible_faces` — the three box faces facing the camera, each labelled
    by the axis normal, i.e. the paper's claim that the isometric "combines all
    three critical perspectives (top, front, side)".
  * :func:`normalize_longest_edge` — the paper's longest-edge-to-2-units scaling.

Distinct from :mod:`drawings.creft_projection` (parallel *orthographic* views,
one axis dropped) and :mod:`generation.cadsmith_three_view` (which only stores
elevation/azimuth as metadata, computing no projection).

Pure stdlib, deterministic (no wall clock, no randomness).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from harnesscad.domain.drawings.creft_projection import Box

Point3 = Tuple[float, float, float]
Point2 = Tuple[float, float]

# The paper's fixed isometric viewpoint.
PAPER_AZIMUTH_DEG = 45.0
PAPER_ELEVATION_DEG = 45.0

# The classical *true isometric* elevation: atan(1/sqrt(2)) ~= 35.264 degrees,
# at which the three axes foreshorten equally (all projected edge lengths equal).
TRUE_ISO_ELEVATION_DEG = math.degrees(math.atan(1.0 / math.sqrt(2.0)))


def _screen_basis(azimuth_deg: float, elevation_deg: float
                  ) -> Tuple[Point3, Point3, Point3]:
    """Right, up and view-direction unit vectors for the given viewpoint.

    The camera orbits the origin: ``azimuth`` rotates about the vertical Z axis,
    ``elevation`` lifts the camera above the horizontal plane. The returned
    ``view`` vector points from the object toward the camera.
    """
    a = math.radians(azimuth_deg)
    e = math.radians(elevation_deg)
    ca, sa = math.cos(a), math.sin(a)
    ce, se = math.cos(e), math.sin(e)
    # View direction (object -> camera).
    view = (ca * ce, sa * ce, se)
    # Screen right: horizontal, perpendicular to view and to world-up.
    right = (-sa, ca, 0.0)
    # Screen up: completes a right-handed screen frame.
    up = (-ca * se, -sa * se, ce)
    return right, up, view


def project_point(point: Point3,
                  azimuth_deg: float = PAPER_AZIMUTH_DEG,
                  elevation_deg: float = PAPER_ELEVATION_DEG) -> Point2:
    """Project a 3D point to a 2D screen coordinate via parallel axonometry."""
    right, up, _ = _screen_basis(azimuth_deg, elevation_deg)
    u = point[0] * right[0] + point[1] * right[1] + point[2] * right[2]
    v = point[0] * up[0] + point[1] * up[1] + point[2] * up[2]
    return (u, v)


def projected_axis_vectors(azimuth_deg: float = PAPER_AZIMUTH_DEG,
                           elevation_deg: float = PAPER_ELEVATION_DEG
                           ) -> Dict[str, Point2]:
    """The 2D vector each *unit* 3D axis maps to under the projection."""
    return {
        "x": project_point((1.0, 0.0, 0.0), azimuth_deg, elevation_deg),
        "y": project_point((0.0, 1.0, 0.0), azimuth_deg, elevation_deg),
        "z": project_point((0.0, 0.0, 1.0), azimuth_deg, elevation_deg),
    }


def axis_foreshortening(azimuth_deg: float = PAPER_AZIMUTH_DEG,
                        elevation_deg: float = PAPER_ELEVATION_DEG
                        ) -> Dict[str, float]:
    """Projected length of each unit axis (its foreshortening factor).

    A unit-length edge along axis ``k`` draws with length
    ``axis_foreshortening()[k]`` in the isometric image. At
    :data:`TRUE_ISO_ELEVATION_DEG` all three factors are equal (true isometry).
    """
    vecs = projected_axis_vectors(azimuth_deg, elevation_deg)
    return {k: math.hypot(v[0], v[1]) for k, v in vecs.items()}


def _box_corners(box: Box) -> List[Point3]:
    xs = (box.x, box.xmax)
    ys = (box.y, box.ymax)
    zs = (box.z, box.zmax)
    return [(x, y, z) for x in xs for y in ys for z in zs]


def project_box(box: Box,
                azimuth_deg: float = PAPER_AZIMUTH_DEG,
                elevation_deg: float = PAPER_ELEVATION_DEG) -> List[Point2]:
    """Project the eight corners of a box; order is stable and deterministic."""
    return [project_point(c, azimuth_deg, elevation_deg)
            for c in _box_corners(box)]


def _cross(o: Point2, a: Point2, b: Point2) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points: Sequence[Point2]) -> List[Point2]:
    """Counter-clockwise convex hull (Andrew's monotone chain), deterministic.

    Collinear hull points are dropped. Returns the unique hull vertices; for a
    single point or a segment it returns the deduplicated input.
    """
    pts = sorted(set((float(p[0]), float(p[1])) for p in points))
    if len(pts) <= 2:
        return pts
    lower: List[Point2] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Point2] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def isometric_outline(box: Box,
                      azimuth_deg: float = PAPER_AZIMUTH_DEG,
                      elevation_deg: float = PAPER_ELEVATION_DEG) -> List[Point2]:
    """Convex-hull silhouette of the projected box (a hexagon for a generic box)."""
    return convex_hull(project_box(box, azimuth_deg, elevation_deg))


def bounding_box_2d(points: Sequence[Point2]) -> Tuple[float, float, float, float]:
    """Axis-aligned 2D bounds ``(umin, vmin, umax, vmax)`` of projected points."""
    if not points:
        return (0.0, 0.0, 0.0, 0.0)
    us = [p[0] for p in points]
    vs = [p[1] for p in points]
    return (min(us), min(vs), max(us), max(vs))


# The six faces of a box, each as (axis, sign, the two spanning axes).
_FACES: Tuple[Tuple[str, int, Tuple[str, str]], ...] = (
    ("x", +1, ("y", "z")), ("x", -1, ("y", "z")),
    ("y", +1, ("x", "z")), ("y", -1, ("x", "z")),
    ("z", +1, ("x", "y")), ("z", -1, ("x", "y")),
)

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def visible_faces(azimuth_deg: float = PAPER_AZIMUTH_DEG,
                  elevation_deg: float = PAPER_ELEVATION_DEG
                  ) -> List[Tuple[str, int]]:
    """The three box faces facing the camera, as ``(axis, sign)`` pairs.

    A face is visible when its outward normal points toward the camera (positive
    dot with the view direction). For the paper's 45/45 viewpoint these are the
    +X, +Y and +Z faces — the three perspectives the isometric "combines".
    The pair of axes each face spans is exactly one orthographic view's plane.
    """
    _, _, view = _screen_basis(azimuth_deg, elevation_deg)
    out: List[Tuple[str, int]] = []
    for axis, sign, _spans in _FACES:
        normal = [0.0, 0.0, 0.0]
        normal[_AXIS_INDEX[axis]] = float(sign)
        dot = normal[0] * view[0] + normal[1] * view[1] + normal[2] * view[2]
        if dot > 1e-12:
            out.append((axis, sign))
    return out


def face_spanning_axes(axis: str) -> Tuple[str, str]:
    """The two axes a face whose normal is ``axis`` spans (its in-plane axes)."""
    for a, _sign, spans in _FACES:
        if a == axis:
            return spans
    raise ValueError("unknown axis %r" % (axis,))


def normalize_longest_edge(box: Box, target: float = 2.0) -> Box:
    """Uniformly scale ``box`` so its longest edge equals ``target`` (paper: 2).

    Scaling is about the origin and preserves the box's low corner position
    proportionally; only relative proportions matter for the drawing.
    """
    if target <= 0.0:
        raise ValueError("target must be positive")
    longest = max(box.dx, box.dy, box.dz)
    if longest <= 0.0:
        raise ValueError("box has no positive extent")
    s = target / longest
    return Box(box.x * s, box.y * s, box.z * s,
               box.dx * s, box.dy * s, box.dz * s)
