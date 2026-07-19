"""Fillet-feasibility preflight predicate (deterministic, stdlib-only).

Derived from kerf (MIT, Copyright (c) 2026 Imran Paruk).

This module supplies the feasibility predicate and supported-input contract
for an analytic rolling-ball fillet. The full rolling-ball fillet surface
construction (quarter-cylinder fillet face for planar+planar, quarter-torus
fillet face for planar+cylindrical, sewing and body validation) is
DEFERRED -- this module answers, before any kernel is invoked, "can a
rolling-ball fillet of radius r be placed on this edge at all, and if not,
why not, and what is the largest radius that would fit?".

Supported behavior
------------------
* The supported-input taxonomy.  The predicate accepts exactly two face-pair
  configurations at the edge (classified by isinstance on the face surface;
  here by the analytic face dataclass type):

      1. planar+planar       -- both supports are planes meeting at a
                                straight convex edge.
      2. planar+cylindrical  -- one support is a plane, the other a
                                cylinder, meeting at a circular rim edge.

  Every other pairing is refused with the message "edge supports must be
  planar+planar or planar+cylindrical; got <A> + <B>".  The predicate never
  raises on contract violations; it returns a structured ``{ok: False, reason}``
  result.  This module mirrors that with :class:`FilletFeasibility`.

* The radius feasibility rule.  The rolling-ball radius r must satisfy
  ``r > 0`` and ``r < min(perpendicular extent of each adjacent support
  measured from the edge)`` so the contact lines lie strictly inside both
  supports.  Kerf measures those extents as the span of the support from
  the edge to its far boundary along the in-face direction perpendicular to
  the edge (for its axis-aligned box this is ``hi[axis] - edge_at[axis]``;
  the contact line at ``edge_at + r`` must stay strictly inside, i.e.
  ``r < extent - tol``).  Kerf additionally requires ``r < edge length``
  for the straight-edge case ("fillet would consume the entire edge") and,
  for the planar+cylindrical rim case, ``r < cylinder_radius`` and
  ``r < cylinder_height`` ("rolling ball does not fit") plus
  ``r < cap_extent`` (kerf's cap is a full disc, so its extent equals the
  cylinder radius; for a rim bounded by a polygon -- e.g. a hole rim in a
  planar face -- the extent is the clearance from the rim circle to the
  polygon boundary).

* Convexity along the edge.  Kerf's contract requires the supports to meet
  at a convex angle from the solid's interior (its box/cylinder primitives
  guarantee this structurally).  The predicate here makes the check
  explicit using the harness edge-convexity rule -- the sign of
  ``dot(cross(n_a, n_b), tangent)`` with outward normals sampled on the
  edge -- and also reports the unsigned dihedral angle between the outward
  normals.

* The structured refusal/result contract.  Kerf's ``FilletResult`` carries
  ``ok`` and a human-readable ``reason`` plus diagnostics (``kind``,
  ``radius``, per-case limits).  :class:`FilletFeasibility` carries
  ``feasible``, a machine-checkable ``reason_code``, kerf's ``reason``
  wording, the detected ``case``, the convexity label and dihedral angle,
  the computed per-support ``limits`` and the ``max_feasible_radius``
  (an open upper bound: any r strictly below it is radius-feasible).

Adaptation notes
----------------
The harness has no full B-rep ``Body``/``Edge``/``Face`` types, so the
preflight consumes a minimal analytic description mirroring exactly what
kerf's predicate reads off its topology: the edge as a point polyline
(:class:`EdgePreflight`) and the two adjacent supports as analytic faces
(:class:`PlanarFace`: point + outward normal + boundary polygon;
:class:`CylindricalFace`: axis point + direction + radius + height extent
+ material side).  Kerf restricts planar+planar to axis-aligned box edges
(a primitive-recognition limit, not a geometric one); this port checks
straightness of the edge instead and measures the perpendicular extents
against the supplied boundary polygons, which reproduces kerf's AABB-span
measurement exactly on rectangular supports.

Deterministic: pure arithmetic, no randomness, no clock.  Pure stdlib.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple, Union

from harnesscad.domain.geometry.topology.edge_convexity import (
    CONVEX,
    classify_edge_convexity,
    dihedral_angle,
)

__all__ = [
    "Vec3",
    "PlanarFace",
    "CylindricalFace",
    "EdgePreflight",
    "FilletFeasibility",
    "CASE_PLANAR_PLANAR",
    "CASE_PLANAR_CYLINDRICAL",
    "CASE_UNSUPPORTED",
    "classify_face",
    "classify_support_pair",
    "supported_contract",
    "check_fillet_feasibility",
    "main",
]

Vec3 = Tuple[float, float, float]

# Supported-input taxonomy (kerf fillet_solid_edge dispatch).
CASE_PLANAR_PLANAR = "planar+planar"
CASE_PLANAR_CYLINDRICAL = "planar+cylindrical"
CASE_UNSUPPORTED = "unsupported"

# Structured reason codes (machine-checkable companions to kerf's
# human-readable reason strings).
REASON_OK = ""
REASON_NONPOSITIVE_RADIUS = "nonpositive-radius"
REASON_UNSUPPORTED_FACE_PAIR = "unsupported-face-pair"
REASON_EDGE_NOT_STRAIGHT = "edge-not-straight"
REASON_EDGE_NOT_CAP_RIM = "edge-not-cap-rim"
REASON_EDGE_NOT_CONVEX = "edge-not-convex"
REASON_RADIUS_EXCEEDS_EDGE_LENGTH = "radius-exceeds-edge-length"
REASON_RADIUS_EXCEEDS_SUPPORT_EXTENT = "radius-exceeds-support-extent"
REASON_RADIUS_EXCEEDS_CYLINDER_RADIUS = "radius-exceeds-cylinder-radius"
REASON_RADIUS_EXCEEDS_CYLINDER_HEIGHT = "radius-exceeds-cylinder-height"
REASON_DEGENERATE_INPUT = "degenerate-input"

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------


def _v(p: Sequence[float]) -> Vec3:
    if len(p) != 3:
        raise ValueError("expected a 3-component point, got %d" % len(p))
    return (float(p[0]), float(p[1]), float(p[2]))


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: Vec3) -> float:
    return math.sqrt(_dot(a, a))


def _unit(a: Vec3) -> Vec3:
    n = _norm(a)
    if n < _EPS:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def _point_segment_distance(p: Vec3, a: Vec3, b: Vec3) -> float:
    """Distance from point ``p`` to segment ``a``-``b`` (all 3D)."""
    ab = _sub(b, a)
    denom = _dot(ab, ab)
    if denom < _EPS:
        return _norm(_sub(p, a))
    t = _dot(_sub(p, a), ab) / denom
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    closest = _add(a, _scale(ab, t))
    return _norm(_sub(p, closest))


def _point_line_distance(p: Vec3, origin: Vec3, direction: Vec3) -> float:
    """Distance from point ``p`` to the infinite line origin + t*direction."""
    d = _unit(direction)
    w = _sub(p, origin)
    along = _dot(w, d)
    perp = _sub(w, _scale(d, along))
    return _norm(perp)


# ---------------------------------------------------------------------------
# Preflight input types (analytic stand-ins for kerf's Face/Edge topology)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanarFace:
    """A planar support: a point on the plane, the OUTWARD unit normal of
    the solid at that face, and the face's boundary polygon (>= 3 vertices,
    all on the plane).  Mirrors kerf's ``Plane`` surface plus the loop kerf
    reads the support extent from."""

    origin: Vec3
    normal: Vec3
    boundary: Tuple[Vec3, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", _v(self.origin))
        object.__setattr__(self, "normal", _unit(_v(self.normal)))
        object.__setattr__(
            self, "boundary", tuple(_v(p) for p in self.boundary)
        )
        if len(self.boundary) < 3:
            raise ValueError("a planar face boundary needs >= 3 vertices")


@dataclass(frozen=True)
class CylindricalFace:
    """A cylindrical support: axis point + direction, radius, and the
    height extent of the face along the axis.  ``outward_radial`` is True
    when the solid material is inside the cylinder (boss / cylinder body:
    outward normals point away from the axis) and False for a hole (the
    material surrounds the cylinder; outward normals point toward the
    axis).  Mirrors kerf's ``CylinderSurface`` plus the height kerf reads
    from the cap-to-cap distance."""

    axis_point: Vec3
    axis_dir: Vec3
    radius: float
    height: float
    outward_radial: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis_point", _v(self.axis_point))
        object.__setattr__(self, "axis_dir", _unit(_v(self.axis_dir)))
        object.__setattr__(self, "radius", float(self.radius))
        object.__setattr__(self, "height", float(self.height))
        if self.radius <= 0.0:
            raise ValueError("cylinder radius must be positive")
        if self.height <= 0.0:
            raise ValueError("cylinder height must be positive")


@dataclass(frozen=True)
class EdgePreflight:
    """The candidate edge as an ordered point polyline (a straight segment
    may be given by its two endpoints; a rim circle by a closed sampled
    ring whose first and last points coincide).

    ``forward`` mirrors kerf's coedge orientation flag: the polyline must
    traverse the edge the way face ``a``'s loop does (counter-clockwise
    about face a's outward normal for an outer loop, clockwise for an
    inner loop such as a hole rim); set ``forward=False`` when the
    supplied polyline runs the other way."""

    points: Tuple[Vec3, ...]
    forward: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", tuple(_v(p) for p in self.points))
        if len(self.points) < 2:
            raise ValueError("an edge polyline needs >= 2 points")


# ---------------------------------------------------------------------------
# Result contract (kerf FilletResult, feasibility subset)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilletFeasibility:
    """Structured feasibility verdict -- kerf's ``{ok, reason, diagnostics}``
    refusal contract restricted to the preflight.

    ``max_feasible_radius`` is an OPEN upper bound: any radius strictly
    below it (by more than the tolerance) is radius-feasible for this edge;
    it is 0.0 when no radius fits or the case is unsupported.  ``limits``
    lists every individual bound kerf enforces for the detected case."""

    feasible: bool
    reason_code: str
    reason: str
    case: str
    convexity: str = ""
    dihedral_deg: float = 0.0
    max_feasible_radius: float = 0.0
    limits: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "feasible": self.feasible,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "case": self.case,
            "convexity": self.convexity,
            "dihedral_deg": self.dihedral_deg,
            "max_feasible_radius": self.max_feasible_radius,
            "limits": dict(self.limits),
        }


def _refuse(
    code: str,
    reason: str,
    case: str,
    convexity: str = "",
    dihedral_deg: float = 0.0,
    max_feasible_radius: float = 0.0,
    limits: Optional[Dict[str, float]] = None,
) -> FilletFeasibility:
    return FilletFeasibility(
        feasible=False,
        reason_code=code,
        reason=reason,
        case=case,
        convexity=convexity,
        dihedral_deg=dihedral_deg,
        max_feasible_radius=max_feasible_radius,
        limits=limits or {},
    )


# ---------------------------------------------------------------------------
# Supported-input taxonomy (kerf's isinstance dispatch + contract text)
# ---------------------------------------------------------------------------

FaceType = Union[PlanarFace, CylindricalFace]


def classify_face(face: FaceType) -> str:
    """Classify a support as ``planar`` or ``cylindrical`` -- kerf's
    ``_is_planar_face`` / ``_is_cylindrical_face`` isinstance checks over
    the analytic dataclasses."""
    if isinstance(face, PlanarFace):
        return "planar"
    if isinstance(face, CylindricalFace):
        return "cylindrical"
    return type(face).__name__


def classify_support_pair(face_a: FaceType, face_b: FaceType) -> str:
    """Map a face pair to kerf's supported-case labels."""
    ka = classify_face(face_a)
    kb = classify_face(face_b)
    if ka == "planar" and kb == "planar":
        return CASE_PLANAR_PLANAR
    if (ka == "planar" and kb == "cylindrical") or (
        ka == "cylindrical" and kb == "planar"
    ):
        return CASE_PLANAR_CYLINDRICAL
    return CASE_UNSUPPORTED


def supported_contract() -> str:
    """Human-readable supported-input contract (kerf
    ``edge_supported_contract``, adapted to the preflight)."""
    return (
        "fillet feasibility preflight supports two edge configurations:\n"
        "  1. planar+planar -- both faces incident to the edge are planar\n"
        "     supports meeting at a convex straight edge; the radius must\n"
        "     leave the contact lines strictly inside both supports\n"
        "     (r < min perpendicular extent) and r < edge length.\n"
        "  2. planar+cylindrical -- one support is a plane, the other a\n"
        "     cylinder, meeting at a circular rim; the radius must satisfy\n"
        "     0 < r < min(cylinder_radius, cylinder_height, cap_extent).\n"
        "All other edge configurations (other surface pairs, concave or\n"
        "smooth edges, or radii exceeding the local limits) return a\n"
        "structured {feasible: false, reason_code, reason} rather than\n"
        "raising.  Rolling-ball fillet surface construction is deferred."
    )


# ---------------------------------------------------------------------------
# Extent measurements (kerf's contact-line / rolling-ball fit checks)
# ---------------------------------------------------------------------------


def _planar_perpendicular_extent(
    face: PlanarFace, edge_point: Vec3, edge_tangent: Vec3
) -> float:
    """Perpendicular extent of a planar support from a straight edge.

    Kerf measures the span of the support from the edge to its far
    boundary along the in-face direction perpendicular to the edge (for
    its axis-aligned box: ``hi - edge_at``); the contact line at
    ``edge_at + r`` must lie strictly inside.  Here: project the boundary
    vertices onto ``d = unit(cross(normal, tangent))`` oriented into the
    face, and take the farthest projection.  Exact for rectangular
    supports (kerf's only planar+planar case)."""
    d = _unit(_cross(face.normal, edge_tangent))
    if d == (0.0, 0.0, 0.0):
        return 0.0
    centroid = (
        sum(p[0] for p in face.boundary) / len(face.boundary),
        sum(p[1] for p in face.boundary) / len(face.boundary),
        sum(p[2] for p in face.boundary) / len(face.boundary),
    )
    if _dot(_sub(centroid, edge_point), d) < 0.0:
        d = _scale(d, -1.0)
    return max(_dot(_sub(p, edge_point), d) for p in face.boundary)


def _cap_extent(
    plane: PlanarFace, cyl: CylindricalFace, tol: float
) -> Optional[float]:
    """Plane-side extent for the rim case (kerf's ``cap_extent``).

    For kerf's cap rim (material inside the cylinder) the cap is a full
    disc bounded by the rim itself, so the extent equals the cylinder
    radius -- the contact circle at radius R - r must keep R - r > 0.
    For a hole rim (material outside the cylinder) the contact circle at
    radius R + r moves toward the planar boundary, so the extent is the
    in-plane clearance from the rim circle to the boundary polygon.
    Returns None when the axis is parallel to the plane (degenerate: the
    circular rim does not exist)."""
    if cyl.outward_radial:
        return cyl.radius
    denom = _dot(cyl.axis_dir, plane.normal)
    if abs(denom) < 1e-9:
        return None
    s = _dot(_sub(plane.origin, cyl.axis_point), plane.normal) / denom
    centre = _add(cyl.axis_point, _scale(cyl.axis_dir, s))
    n = len(plane.boundary)
    clearance = min(
        _point_segment_distance(
            centre, plane.boundary[i], plane.boundary[(i + 1) % n]
        )
        for i in range(n)
    ) - cyl.radius
    return clearance


# ---------------------------------------------------------------------------
# Edge-shape recognition (kerf's _is_axis_aligned_edge / _cap_rim_edge_info)
# ---------------------------------------------------------------------------


def _straight_edge_info(
    edge: EdgePreflight, tol: float
) -> Optional[Tuple[Vec3, Vec3, float]]:
    """Return ``(start_point, unit_tangent, length)`` iff the polyline is a
    straight segment (every interior point within tolerance of the chord),
    otherwise None.  Generalises kerf's ``_is_axis_aligned_edge`` Line3
    check (axis alignment was a box-primitive restriction)."""
    p0 = edge.points[0]
    p1 = edge.points[-1]
    chord = _sub(p1, p0)
    length = _norm(chord)
    if length <= tol:
        return None
    for p in edge.points[1:-1]:
        if _point_line_distance(p, p0, chord) > tol * 100.0:
            return None
    return p0, _unit(chord), length


def _rim_edge_matches(
    edge: EdgePreflight, plane: PlanarFace, cyl: CylindricalFace, tol: float
) -> bool:
    """True iff the polyline is the closed circular rim where the plane
    meets the cylinder.  Ports kerf's ``_cap_rim_edge_info`` checks: the
    edge curve's radius must match the cylinder radius (kerf slack:
    100 * tol), the curve must span the full circle (here: the polyline is
    a closed ring), and it must lie in the cap plane."""
    slack = tol * 100.0
    if len(edge.points) < 4:
        return False
    if _norm(_sub(edge.points[0], edge.points[-1])) > slack:
        return False
    for p in edge.points:
        if abs(_point_line_distance(p, cyl.axis_point, cyl.axis_dir)
               - cyl.radius) > slack:
            return False
        if abs(_dot(_sub(p, plane.origin), plane.normal)) > slack:
            return False
    return True


def _cylinder_outward_normal(cyl: CylindricalFace, p: Vec3) -> Vec3:
    """Outward unit normal of the material's cylindrical face at ``p``."""
    d = cyl.axis_dir
    w = _sub(p, cyl.axis_point)
    radial = _sub(w, _scale(d, _dot(w, d)))
    n = _unit(radial)
    if not cyl.outward_radial:
        n = _scale(n, -1.0)
    return n


# ---------------------------------------------------------------------------
# The feasibility predicate (kerf fillet_solid_edge, checks-only)
# ---------------------------------------------------------------------------


def check_fillet_feasibility(
    edge: EdgePreflight,
    face_a: FaceType,
    face_b: FaceType,
    radius: float,
    tol: float = 1e-6,
) -> FilletFeasibility:
    """Preflight a rolling-ball fillet of ``radius`` on ``edge`` between
    supports ``face_a`` and ``face_b``.

    Never raises on contract violations; always returns a structured
    :class:`FilletFeasibility` (kerf's never-raise refusal contract).
    The check order mirrors ``fillet_solid_edge``: radius validation,
    case classification, edge-shape recognition, then the per-case radius
    limits in kerf's order."""
    # --- 0. Input validation (kerf step 0) --------------------------------
    if not isinstance(radius, (int, float)) or isinstance(radius, bool) \
            or radius <= 0.0:
        return _refuse(
            REASON_NONPOSITIVE_RADIUS,
            "radius must be a positive number, got %r" % (radius,),
            classify_support_pair(face_a, face_b),
        )
    radius = float(radius)

    # --- 1. Supported-input taxonomy (kerf dispatch) ----------------------
    case = classify_support_pair(face_a, face_b)
    if case == CASE_UNSUPPORTED:
        return _refuse(
            REASON_UNSUPPORTED_FACE_PAIR,
            "edge supports must be planar+planar or planar+cylindrical; "
            "got %s + %s"
            % (type(face_a).__name__, type(face_b).__name__),
            CASE_UNSUPPORTED,
        )

    if case == CASE_PLANAR_PLANAR:
        return _check_planar_planar(edge, face_a, face_b, radius, tol)
    return _check_planar_cylindrical(edge, face_a, face_b, radius, tol)


def _check_planar_planar(
    edge: EdgePreflight,
    face_a: PlanarFace,
    face_b: PlanarFace,
    radius: float,
    tol: float,
) -> FilletFeasibility:
    info = _straight_edge_info(edge, tol)
    if info is None:
        return _refuse(
            REASON_EDGE_NOT_STRAIGHT,
            "edge is not a straight segment (planar+planar fillet "
            "requires the straight intersection edge of the two planes)",
            CASE_PLANAR_PLANAR,
        )
    edge_point, tangent, edge_len = info

    # Convexity along the edge (kerf contract: supports must meet at a
    # convex angle from the solid's interior).
    label = classify_edge_convexity(
        face_a.normal, face_b.normal, tangent, forward=edge.forward,
    )
    dihedral_deg = math.degrees(dihedral_angle(face_a.normal, face_b.normal))
    if label != CONVEX:
        return _refuse(
            REASON_EDGE_NOT_CONVEX,
            "supports must meet at a convex angle from the solid's "
            "interior; edge classified %s" % label,
            CASE_PLANAR_PLANAR,
            convexity=label,
            dihedral_deg=dihedral_deg,
        )

    ext_a = _planar_perpendicular_extent(face_a, edge_point, tangent)
    ext_b = _planar_perpendicular_extent(face_b, edge_point, tangent)
    limits = {
        "support_a_extent": ext_a,
        "support_b_extent": ext_b,
        "edge_length": edge_len,
    }
    if ext_a <= tol or ext_b <= tol:
        return _refuse(
            REASON_DEGENERATE_INPUT,
            "a support has no perpendicular extent from the edge "
            "(degenerate boundary)",
            CASE_PLANAR_PLANAR,
            convexity=label,
            dihedral_deg=dihedral_deg,
            limits=limits,
        )
    max_feasible = min(ext_a, ext_b, edge_len)

    # Kerf check order: radius vs edge length first (fillet_solid_edge),
    # then contact-line-inside-support (in _box_filleted_edge_body:
    # contact >= extent - tol is refused, i.e. r must be < extent - tol).
    if radius >= edge_len:
        return _refuse(
            REASON_RADIUS_EXCEEDS_EDGE_LENGTH,
            "radius %s >= edge length %s; fillet would consume the "
            "entire edge (unsupported)" % (radius, edge_len),
            CASE_PLANAR_PLANAR,
            convexity=label,
            dihedral_deg=dihedral_deg,
            max_feasible_radius=max_feasible,
            limits=limits,
        )
    for name, ext in (("a", ext_a), ("b", ext_b)):
        if radius >= ext - tol:
            return _refuse(
                REASON_RADIUS_EXCEEDS_SUPPORT_EXTENT,
                "radius %s exceeds support %s extent (contact reaches "
                "%.6g, max %.6g)" % (radius, name, radius, ext),
                CASE_PLANAR_PLANAR,
                convexity=label,
                dihedral_deg=dihedral_deg,
                max_feasible_radius=max_feasible,
                limits=limits,
            )

    return FilletFeasibility(
        feasible=True,
        reason_code=REASON_OK,
        reason="",
        case=CASE_PLANAR_PLANAR,
        convexity=label,
        dihedral_deg=dihedral_deg,
        max_feasible_radius=max_feasible,
        limits=limits,
    )


def _check_planar_cylindrical(
    edge: EdgePreflight,
    face_a: FaceType,
    face_b: FaceType,
    radius: float,
    tol: float,
) -> FilletFeasibility:
    if isinstance(face_a, PlanarFace):
        plane, cyl = face_a, face_b
        plane_is_a = True
    else:
        plane, cyl = face_b, face_a
        plane_is_a = False
    assert isinstance(plane, PlanarFace) and isinstance(cyl, CylindricalFace)

    if not _rim_edge_matches(edge, plane, cyl, tol):
        return _refuse(
            REASON_EDGE_NOT_CAP_RIM,
            "edge is not a cap-rim circular edge of the cylinder",
            CASE_PLANAR_CYLINDRICAL,
        )

    # Convexity at a sample point on the rim.
    p0 = edge.points[0]
    tangent = _unit(_sub(edge.points[1], edge.points[0]))
    n_cyl = _cylinder_outward_normal(cyl, p0)
    if plane_is_a:
        n_a, n_b = plane.normal, n_cyl
    else:
        n_a, n_b = n_cyl, plane.normal
    label = classify_edge_convexity(n_a, n_b, tangent, forward=edge.forward)
    dihedral_deg = math.degrees(dihedral_angle(n_a, n_b))
    if label != CONVEX:
        return _refuse(
            REASON_EDGE_NOT_CONVEX,
            "supports must meet at a convex angle from the solid's "
            "interior; edge classified %s" % label,
            CASE_PLANAR_CYLINDRICAL,
            convexity=label,
            dihedral_deg=dihedral_deg,
        )

    cap_extent = _cap_extent(plane, cyl, tol)
    if cap_extent is None or cap_extent <= tol:
        return _refuse(
            REASON_DEGENERATE_INPUT,
            "cap support has no extent from the rim (degenerate rim "
            "clearance)",
            CASE_PLANAR_CYLINDRICAL,
            convexity=label,
            dihedral_deg=dihedral_deg,
        )
    limits = {
        "cylinder_radius": cyl.radius,
        "cylinder_height": cyl.height,
        "cap_extent": cap_extent,
    }
    max_feasible = min(cyl.radius, cyl.height, cap_extent)

    # Kerf check order (fillet_solid_edge planar+cylindrical branch):
    # radius vs cylinder radius, then vs cylinder height, then the cap
    # contact circle must stay inside the cap support.
    if radius >= cyl.radius:
        return _refuse(
            REASON_RADIUS_EXCEEDS_CYLINDER_RADIUS,
            "radius %s >= cylinder radius %s; rolling ball does not fit"
            % (radius, cyl.radius),
            CASE_PLANAR_CYLINDRICAL,
            convexity=label,
            dihedral_deg=dihedral_deg,
            max_feasible_radius=max_feasible,
            limits=limits,
        )
    if radius >= cyl.height:
        return _refuse(
            REASON_RADIUS_EXCEEDS_CYLINDER_HEIGHT,
            "radius %s >= cylinder height %s; rolling ball does not fit"
            % (radius, cyl.height),
            CASE_PLANAR_CYLINDRICAL,
            convexity=label,
            dihedral_deg=dihedral_deg,
            max_feasible_radius=max_feasible,
            limits=limits,
        )
    if radius >= cap_extent - tol:
        return _refuse(
            REASON_RADIUS_EXCEEDS_SUPPORT_EXTENT,
            "radius %s exceeds cap support extent (contact reaches %.6g, "
            "max %.6g)" % (radius, radius, cap_extent),
            CASE_PLANAR_CYLINDRICAL,
            convexity=label,
            dihedral_deg=dihedral_deg,
            max_feasible_radius=max_feasible,
            limits=limits,
        )

    return FilletFeasibility(
        feasible=True,
        reason_code=REASON_OK,
        reason="",
        case=CASE_PLANAR_CYLINDRICAL,
        convexity=label,
        dihedral_deg=dihedral_deg,
        max_feasible_radius=max_feasible,
        limits=limits,
    )


# ---------------------------------------------------------------------------
# Selfcheck
# ---------------------------------------------------------------------------


def _circle_polyline(
    centre: Vec3, radius: float, n: int, ccw: bool = True
) -> Tuple[Vec3, ...]:
    """Closed sampled circle in the z = centre.z plane; CCW about +Z when
    ``ccw`` else CW (inner-loop winding)."""
    pts = []
    for i in range(n + 1):
        theta = 2.0 * math.pi * (i % n) / n
        if not ccw:
            theta = -theta
        pts.append((
            centre[0] + radius * math.cos(theta),
            centre[1] + radius * math.sin(theta),
            centre[2],
        ))
    return tuple(pts)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.geometry.features."
             "fillet_feasibility",
        description="Fillet-feasibility preflight predicate; rolling-ball "
                    "construction is deliberately deferred.",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="run deterministic synthetic-geometry checks "
                             "of the feasibility predicate.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    # --- planar+planar: box edge, plane pair at 90 degrees, extents 1.0 --
    # Box [0,1]x[0,1]x[0,2]; filleted edge along +Z at (x=1, y=1), so the
    # perpendicular extents of both supports are 1.0 and edge length is 2.
    face_px = PlanarFace(
        origin=(1.0, 0.5, 1.0), normal=(1.0, 0.0, 0.0),
        boundary=((1.0, 0.0, 0.0), (1.0, 1.0, 0.0),
                  (1.0, 1.0, 2.0), (1.0, 0.0, 2.0)),
    )
    face_py = PlanarFace(
        origin=(0.5, 1.0, 1.0), normal=(0.0, 1.0, 0.0),
        boundary=((0.0, 1.0, 0.0), (1.0, 1.0, 0.0),
                  (1.0, 1.0, 2.0), (0.0, 1.0, 2.0)),
    )
    # Polyline forward per face_px's outer loop (CCW about +X).
    box_edge = EdgePreflight(
        points=((1.0, 1.0, 0.0), (1.0, 1.0, 1.0), (1.0, 1.0, 2.0)),
    )
    ok = check_fillet_feasibility(box_edge, face_px, face_py, 0.2)
    assert ok.feasible, ok.reason
    assert ok.case == CASE_PLANAR_PLANAR
    assert ok.convexity == CONVEX
    assert abs(ok.dihedral_deg - 90.0) < 1e-9
    assert abs(ok.max_feasible_radius - 1.0) < 1e-9
    assert abs(ok.limits["support_a_extent"] - 1.0) < 1e-9
    assert abs(ok.limits["support_b_extent"] - 1.0) < 1e-9
    print("[selfcheck] planar+planar r=0.2 feasible "
          "(max_feasible=%.3f, dihedral=%.1f deg)"
          % (ok.max_feasible_radius, ok.dihedral_deg))

    bad = check_fillet_feasibility(box_edge, face_px, face_py, 1.5)
    assert not bad.feasible
    assert bad.reason_code == REASON_RADIUS_EXCEEDS_SUPPORT_EXTENT, bad
    assert abs(bad.max_feasible_radius - 1.0) < 1e-9
    print("[selfcheck] planar+planar r=1.5 refused: %s (max_feasible=%.3f)"
          % (bad.reason_code, bad.max_feasible_radius))

    # Kerf's edge-length limit fires first when r also exceeds the edge.
    long_bad = check_fillet_feasibility(box_edge, face_px, face_py, 2.5)
    assert not long_bad.feasible
    assert long_bad.reason_code == REASON_RADIUS_EXCEEDS_EDGE_LENGTH
    print("[selfcheck] planar+planar r=2.5 refused: %s"
          % long_bad.reason_code)

    # Concave inner corner (a pocket wall: face a's outward normal points
    # the other way) is refused per contract.
    face_px_in = PlanarFace(
        origin=face_px.origin, normal=(-1.0, 0.0, 0.0),
        boundary=face_px.boundary,
    )
    concave = check_fillet_feasibility(box_edge, face_px_in, face_py, 0.2)
    assert not concave.feasible
    assert concave.reason_code == REASON_EDGE_NOT_CONVEX
    print("[selfcheck] concave planar+planar refused: %s"
          % concave.reason_code)

    # --- planar+cylindrical: hole rim in a unit cube's top face ----------
    # Cube [0,1]^3 with a through hole of radius 0.3 at the centre of the
    # top face.  cap_extent = 0.5 - 0.3 = 0.2 (rim clearance to the square
    # boundary), so feasibility flips at r = 0.2.
    top = PlanarFace(
        origin=(0.5, 0.5, 1.0), normal=(0.0, 0.0, 1.0),
        boundary=((0.0, 0.0, 1.0), (1.0, 0.0, 1.0),
                  (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)),
    )
    hole = CylindricalFace(
        axis_point=(0.5, 0.5, 0.0), axis_dir=(0.0, 0.0, 1.0),
        radius=0.3, height=1.0, outward_radial=False,
    )
    # Hole rim is an inner loop of the top face: CW about +Z.
    rim = EdgePreflight(
        points=_circle_polyline((0.5, 0.5, 1.0), 0.3, 24, ccw=False),
    )
    ok2 = check_fillet_feasibility(rim, top, hole, 0.1)
    assert ok2.feasible, ok2.reason
    assert ok2.case == CASE_PLANAR_CYLINDRICAL
    assert ok2.convexity == CONVEX
    assert abs(ok2.max_feasible_radius - 0.2) < 1e-9
    assert abs(ok2.limits["cap_extent"] - 0.2) < 1e-9
    bad2 = check_fillet_feasibility(rim, top, hole, 0.25)
    assert not bad2.feasible
    assert bad2.reason_code == REASON_RADIUS_EXCEEDS_SUPPORT_EXTENT, bad2
    print("[selfcheck] planar+cylindrical hole rim: r=0.1 feasible, "
          "r=0.25 refused (%s); bound flips at %.3f"
          % (bad2.reason_code, ok2.max_feasible_radius))

    # Kerf's exact cap-rim case: cylinder body (boss), R=0.5, H=1.0 --
    # bound is min(R, H, cap_extent=R) = 0.5, rolling-ball-does-not-fit
    # refusal above it.
    cap_pts = _circle_polyline((0.0, 0.0, 1.0), 0.5, 24, ccw=True)
    cap = PlanarFace(
        origin=(0.0, 0.0, 1.0), normal=(0.0, 0.0, 1.0), boundary=cap_pts[:-1],
    )
    barrel = CylindricalFace(
        axis_point=(0.0, 0.0, 0.0), axis_dir=(0.0, 0.0, 1.0),
        radius=0.5, height=1.0, outward_radial=True,
    )
    cap_rim = EdgePreflight(points=cap_pts)
    ok3 = check_fillet_feasibility(cap_rim, cap, barrel, 0.4)
    assert ok3.feasible, ok3.reason
    assert abs(ok3.max_feasible_radius - 0.5) < 1e-9
    bad3 = check_fillet_feasibility(cap_rim, cap, barrel, 0.6)
    assert not bad3.feasible
    assert bad3.reason_code == REASON_RADIUS_EXCEEDS_CYLINDER_RADIUS
    print("[selfcheck] planar+cylindrical cap rim: r=0.4 feasible, "
          "r=0.6 refused (%s)" % bad3.reason_code)

    # --- unsupported face pair: two cylinders -----------------------------
    other_cyl = CylindricalFace(
        axis_point=(0.0, 0.0, 0.0), axis_dir=(1.0, 0.0, 0.0),
        radius=0.5, height=1.0,
    )
    unsup = check_fillet_feasibility(cap_rim, barrel, other_cyl, 0.1)
    assert not unsup.feasible
    assert unsup.reason_code == REASON_UNSUPPORTED_FACE_PAIR
    assert unsup.case == CASE_UNSUPPORTED
    assert "planar+planar or planar+cylindrical" in unsup.reason
    print("[selfcheck] cylinder+cylinder refused: %s" % unsup.reason_code)

    # --- degenerate radius -------------------------------------------------
    zero = check_fillet_feasibility(box_edge, face_px, face_py, 0.0)
    assert not zero.feasible
    assert zero.reason_code == REASON_NONPOSITIVE_RADIUS
    print("[selfcheck] r=0 refused: %s" % zero.reason_code)

    assert "planar+cylindrical" in supported_contract()
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
