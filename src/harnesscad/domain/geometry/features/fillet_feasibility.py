"""Rolling-ball fillet feasibility preflight (deterministic, stdlib-only).

WHAT THIS MODULE IS
-------------------
A PREDICATE, and nothing else.  Given an edge, the two supports that meet
at it, and a candidate rolling-ball radius, it answers one question before
any modelling kernel is asked to do work:

    can a ball of this radius roll along this edge in contact with both
    supports, without running off either of them?

It answers with a verdict object.  It does NOT build a fillet: no fillet
surface, no trimmed supports, no sewn body.  Constructing the quarter
cylinder / quarter torus that the rolling ball sweeps is a separate,
deliberately unimplemented concern, and the verdict carries no geometry --
only the diagnostic fields listed on :class:`FilletFeasibility`.

THE GEOMETRY BEHIND THE PREDICATE
---------------------------------
A rolling ball of radius ``r`` seated in the crease between two supports
touches each support along a contact track offset from the edge by ``r``
measured inside that support, perpendicular to the edge.  Feasibility is
therefore a statement about clearance:

* the crease must actually be a crease -- the supports must meet convexly
  as seen from the solid's material side, since a ball placed against a
  concave or tangent junction is not a fillet at all;
* each contact track must land strictly inside its own support, so ``r``
  must stay below that support's perpendicular clearance from the edge;
* the ball must not outgrow the edge it rides along, nor (for a circular
  rim on a cylinder) the cylinder's own radius or height, since a ball
  larger than those cannot complete the sweep.

Two support pairings are handled, because those are the two for which the
clearance quantities above are elementary:

    planar + planar       two planes meeting along a straight segment;
    planar + cylindrical  a plane meeting a cylinder along a circular rim.

Anything else is out of contract.

REFUSAL DISCIPLINE
------------------
An infeasible or out-of-contract request is REFUSED, never raised and
never quietly approximated.  Every refusal carries a stable machine
readable ``reason_code`` alongside human wording, so callers can branch on
the cause (bad radius, wrong support pair, wrong edge shape, wrong
convexity, or which specific bound was exceeded).  Malformed dataclass
construction -- a two-vertex "polygon", a negative cylinder radius -- still
raises, because that is a caller bug rather than an infeasible request.

``max_feasible_radius`` is reported as an OPEN upper bound: it is the
smallest of the bounds that apply to the detected case, and any radius
strictly below it (by more than ``tol``) passes the radius tests.  It is
0.0 when the case is unsupported or no radius could fit.

INPUTS
------
The harness has no boundary-representation body type, so the supports are
described analytically and minimally: :class:`PlanarFace` (a plane point,
its outward normal, and the bounding polygon that limits its clearance),
:class:`CylindricalFace` (axis, radius, height, and which side holds
material), and :class:`EdgePreflight` (the edge sampled as an ordered
polyline, plus the orientation flag needed to read convexity).  Convexity
itself is delegated to
:mod:`harnesscad.domain.geometry.topology.edge_convexity`.

Deterministic: closed-form arithmetic only -- no randomness, no clock, no
iteration to a tolerance.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

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

# Detected support-pair configurations.
CASE_PLANAR_PLANAR = "planar+planar"
CASE_PLANAR_CYLINDRICAL = "planar+cylindrical"
CASE_UNSUPPORTED = "unsupported"

# Stable refusal codes.  Callers branch on these; the prose is advisory.
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

_TINY = 1e-12
# Sampled polylines carry discretisation error far larger than the modelling
# tolerance; shape recognition is therefore given this much slack over `tol`.
_SHAPE_SLACK = 100.0


# ---------------------------------------------------------------------------
# Minimal vector arithmetic on plain 3-tuples
# ---------------------------------------------------------------------------


def _as_point(raw: Sequence[float]) -> Vec3:
    if len(raw) != 3:
        raise ValueError("expected a 3-component point, got %d" % len(raw))
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def _diff(head: Vec3, tail: Vec3) -> Vec3:
    return (head[0] - tail[0], head[1] - tail[1], head[2] - tail[2])


def _walk(base: Vec3, direction: Vec3, distance: float) -> Vec3:
    """Point reached by stepping ``distance`` along ``direction`` from base."""
    return (
        base[0] + direction[0] * distance,
        base[1] + direction[1] * distance,
        base[2] + direction[2] * distance,
    )


def _flip(vec: Vec3) -> Vec3:
    return (-vec[0], -vec[1], -vec[2])


def _dot3(u: Vec3, w: Vec3) -> float:
    return u[0] * w[0] + u[1] * w[1] + u[2] * w[2]


def _cross3(u: Vec3, w: Vec3) -> Vec3:
    return (
        u[1] * w[2] - u[2] * w[1],
        u[2] * w[0] - u[0] * w[2],
        u[0] * w[1] - u[1] * w[0],
    )


def _length(vec: Vec3) -> float:
    return math.sqrt(_dot3(vec, vec))


def _normalise(vec: Vec3) -> Vec3:
    """Unit vector, or the zero vector when ``vec`` has no usable direction."""
    mag = _length(vec)
    if mag < _TINY:
        return (0.0, 0.0, 0.0)
    return (vec[0] / mag, vec[1] / mag, vec[2] / mag)


def _is_zero(vec: Vec3) -> bool:
    return _length(vec) < _TINY


def _offset_from_line(probe: Vec3, anchor: Vec3, along: Vec3) -> float:
    """Perpendicular distance from ``probe`` to the infinite line
    ``anchor + t * along``."""
    axis = _normalise(along)
    spoke = _diff(probe, anchor)
    return _length(_diff(spoke, _walk((0.0, 0.0, 0.0), axis,
                                      _dot3(spoke, axis))))


def _offset_from_segment(probe: Vec3, start: Vec3, end: Vec3) -> float:
    """Perpendicular distance from ``probe`` to the finite segment
    ``start``-``end`` (clamped to the endpoints)."""
    span = _diff(end, start)
    span_sq = _dot3(span, span)
    if span_sq < _TINY:
        return _length(_diff(probe, start))
    where = _dot3(_diff(probe, start), span) / span_sq
    where = min(1.0, max(0.0, where))
    return _length(_diff(probe, _walk(start, span, where)))


# ---------------------------------------------------------------------------
# Analytic descriptions of the edge and its two supports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanarFace:
    """A planar support.

    ``origin`` is any point of the plane, ``normal`` the OUTWARD normal of
    the solid there (it is normalised on construction), and ``boundary``
    the face's bounding polygon -- at least three coplanar vertices.  The
    polygon is what limits how far a contact track may travel before it
    leaves the support.
    """

    origin: Vec3
    normal: Vec3
    boundary: Tuple[Vec3, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", _as_point(self.origin))
        object.__setattr__(self, "normal", _normalise(_as_point(self.normal)))
        object.__setattr__(self, "boundary",
                           tuple(_as_point(p) for p in self.boundary))
        if len(self.boundary) < 3:
            raise ValueError("a planar face boundary needs >= 3 vertices")


@dataclass(frozen=True)
class CylindricalFace:
    """A cylindrical support of the given ``radius`` and axial ``height``.

    ``outward_radial`` records which side holds material: True for a solid
    barrel or boss (outward normals point away from the axis) and False for
    a drilled hole (material surrounds the surface, so outward normals
    point back toward the axis).
    """

    axis_point: Vec3
    axis_dir: Vec3
    radius: float
    height: float
    outward_radial: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis_point", _as_point(self.axis_point))
        object.__setattr__(self, "axis_dir",
                           _normalise(_as_point(self.axis_dir)))
        object.__setattr__(self, "radius", float(self.radius))
        object.__setattr__(self, "height", float(self.height))
        if self.radius <= 0.0:
            raise ValueError("cylinder radius must be positive")
        if self.height <= 0.0:
            raise ValueError("cylinder height must be positive")

    def outward_normal_at(self, probe: Vec3) -> Vec3:
        """Outward unit normal of this support at a point on its surface."""
        spoke = _diff(probe, self.axis_point)
        radial = _diff(spoke, _walk((0.0, 0.0, 0.0), self.axis_dir,
                                    _dot3(spoke, self.axis_dir)))
        outward = _normalise(radial)
        return outward if self.outward_radial else _flip(outward)


@dataclass(frozen=True)
class EdgePreflight:
    """The candidate edge, sampled as an ordered polyline.

    A straight edge may be given by its two endpoints alone; a circular rim
    should be a closed ring whose first and last samples coincide.
    ``forward`` states whether the polyline runs the way the FIRST support's
    loop traverses the edge; pass ``forward=False`` when it runs the other
    way, so that convexity is read with the correct sign.
    """

    points: Tuple[Vec3, ...]
    forward: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "points",
                           tuple(_as_point(p) for p in self.points))
        if len(self.points) < 2:
            raise ValueError("an edge polyline needs >= 2 points")


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilletFeasibility:
    """Outcome of the preflight -- diagnostics only, never geometry.

    ``max_feasible_radius`` is an OPEN upper bound: any radius strictly
    below it (by more than the tolerance) clears every radius bound of the
    detected case.  It is 0.0 when nothing fits or the case is unsupported.
    ``limits`` names each individual bound that applied.
    """

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


@dataclass(frozen=True)
class _Bound:
    """One scalar ceiling the radius must stay under.

    The radius clears the bound when ``radius < value - slack``; ``slack``
    is the tolerance margin that keeps a contact track strictly (not just
    barely) inside its support, and is zero for bounds where touching the
    limit is already the failure.  ``explain`` turns the offending radius
    into human wording.
    """

    key: str
    value: float
    slack: float
    code: str
    explain: Callable[[float], str]

    def rejects(self, radius: float) -> bool:
        return radius >= self.value - self.slack


# ---------------------------------------------------------------------------
# Support-pair recognition
# ---------------------------------------------------------------------------

FaceType = Union[PlanarFace, CylindricalFace]


def classify_face(face: FaceType) -> str:
    """Name a support's surface kind: ``planar``, ``cylindrical``, or, for
    anything this module does not model, the type's own name."""
    if isinstance(face, PlanarFace):
        return "planar"
    if isinstance(face, CylindricalFace):
        return "cylindrical"
    return type(face).__name__


def classify_support_pair(face_a: FaceType, face_b: FaceType) -> str:
    """Reduce an unordered pair of supports to one of the case labels."""
    kinds = sorted((classify_face(face_a), classify_face(face_b)))
    if kinds == ["planar", "planar"]:
        return CASE_PLANAR_PLANAR
    if kinds == ["cylindrical", "planar"]:
        return CASE_PLANAR_CYLINDRICAL
    return CASE_UNSUPPORTED


def supported_contract() -> str:
    """Prose statement of what the preflight will and will not accept."""
    return (
        "fillet feasibility preflight supports two edge configurations:\n"
        "  1. planar+planar -- two planar supports meeting convexly along a\n"
        "     straight edge; the radius must stay under the perpendicular\n"
        "     clearance of each support and under the edge length.\n"
        "  2. planar+cylindrical -- a planar support meeting a cylindrical\n"
        "     one along a circular rim; the radius must stay under the\n"
        "     cylinder radius, the cylinder height and the rim clearance of\n"
        "     the planar support.\n"
        "Any other configuration -- a different surface pairing, a concave\n"
        "or tangent junction, an edge of the wrong shape, or a radius past\n"
        "one of the bounds above -- is refused with a structured\n"
        "{feasible: false, reason_code, reason} verdict rather than an\n"
        "exception.  The predicate reports feasibility only; it does not\n"
        "construct the rolling-ball fillet surface."
    )


# ---------------------------------------------------------------------------
# Clearance measurements
# ---------------------------------------------------------------------------


def _planar_clearance(face: PlanarFace, seat: Vec3, tangent: Vec3) -> float:
    """How far a planar support reaches away from a straight edge.

    The contact track sits parallel to the edge, displaced into the face
    along ``inward = normal x tangent``.  The clearance is the largest
    displacement any boundary vertex reaches in that direction, so a radius
    below it keeps the track inside the polygon.  For a rectangular support
    this is exactly the side length perpendicular to the edge.
    """
    inward = _normalise(_cross3(face.normal, tangent))
    if _is_zero(inward):
        return 0.0
    reach = [_dot3(_diff(vertex, seat), inward) for vertex in face.boundary]
    # `inward` may have come out pointing away from the material; the face
    # lies wholly on one side of the edge, so the sign of the total reach
    # tells us which.
    return max(reach) if sum(reach) >= 0.0 else -min(reach)


def _rim_clearance(plane: PlanarFace,
                   cyl: CylindricalFace) -> Optional[float]:
    """How far the planar support reaches away from a circular rim.

    When material fills the cylinder (a barrel with a cap), the cap is the
    disc bounded by the rim itself and its contact circle shrinks toward
    the axis, so the reach available is the cylinder radius.  When the
    cylinder is a hole, the cap surrounds it and the contact circle grows
    outward, so the reach is the gap between the rim circle and the nearest
    boundary edge of the polygon.  ``None`` when the axis lies parallel to
    the plane, in which case no circular rim exists at all.
    """
    if cyl.outward_radial:
        return cyl.radius
    tilt = _dot3(cyl.axis_dir, plane.normal)
    if abs(tilt) < 1e-9:
        return None
    travel = _dot3(_diff(plane.origin, cyl.axis_point), plane.normal) / tilt
    hub = _walk(cyl.axis_point, cyl.axis_dir, travel)
    corners = plane.boundary
    nearest_wall = min(
        _offset_from_segment(hub, corners[i], corners[(i + 1) % len(corners)])
        for i in range(len(corners))
    )
    return nearest_wall - cyl.radius


# ---------------------------------------------------------------------------
# Edge-shape recognition
# ---------------------------------------------------------------------------


def _as_straight_edge(edge: EdgePreflight,
                      tol: float) -> Optional[Tuple[Vec3, Vec3, float]]:
    """``(seat, unit tangent, length)`` when the polyline is a straight
    segment -- every intermediate sample lying on the chord -- else None."""
    first, last = edge.points[0], edge.points[-1]
    chord = _diff(last, first)
    span = _length(chord)
    if span <= tol:
        return None
    wobble = tol * _SHAPE_SLACK
    for sample in edge.points[1:-1]:
        if _offset_from_line(sample, first, chord) > wobble:
            return None
    return first, _normalise(chord), span


def _is_rim_of(edge: EdgePreflight, plane: PlanarFace, cyl: CylindricalFace,
               tol: float) -> bool:
    """True when the polyline traces the whole circle where the cylinder
    meets the plane: a closed ring, every sample at the cylinder radius
    from the axis and on the plane."""
    if len(edge.points) < 4:
        return False
    slop = tol * _SHAPE_SLACK
    if _length(_diff(edge.points[0], edge.points[-1])) > slop:
        return False
    for sample in edge.points:
        off_axis = _offset_from_line(sample, cyl.axis_point, cyl.axis_dir)
        if abs(off_axis - cyl.radius) > slop:
            return False
        if abs(_dot3(_diff(sample, plane.origin), plane.normal)) > slop:
            return False
    return True


# ---------------------------------------------------------------------------
# Verdict assembly
# ---------------------------------------------------------------------------


def _refusal(code: str, reason: str, case: str, convexity: str = "",
             dihedral_deg: float = 0.0, max_feasible_radius: float = 0.0,
             limits: Optional[Dict[str, float]] = None) -> FilletFeasibility:
    return FilletFeasibility(
        feasible=False,
        reason_code=code,
        reason=reason,
        case=case,
        convexity=convexity,
        dihedral_deg=dihedral_deg,
        max_feasible_radius=max_feasible_radius,
        limits=dict(limits) if limits else {},
    )


def _adjudicate(radius: float, bounds: List[_Bound], case: str,
                convexity: str, dihedral_deg: float) -> FilletFeasibility:
    """Test the radius against every bound of a case, in order.

    Both branches funnel through here, so the reported ``limits`` and
    ``max_feasible_radius`` are always derived from exactly the bounds that
    were enforced -- they cannot drift apart.
    """
    limits = {bound.key: bound.value for bound in bounds}
    ceiling = min(bound.value for bound in bounds)
    for bound in bounds:
        if bound.rejects(radius):
            return _refusal(bound.code, bound.explain(radius), case,
                            convexity=convexity, dihedral_deg=dihedral_deg,
                            max_feasible_radius=ceiling, limits=limits)
    return FilletFeasibility(
        feasible=True,
        reason_code=REASON_OK,
        reason="",
        case=case,
        convexity=convexity,
        dihedral_deg=dihedral_deg,
        max_feasible_radius=ceiling,
        limits=limits,
    )


def _convexity_of(n_a: Vec3, n_b: Vec3, tangent: Vec3,
                  forward: bool) -> Tuple[str, float]:
    """Convexity label and unsigned dihedral angle (degrees) at the edge."""
    label = classify_edge_convexity(n_a, n_b, tangent, forward=forward)
    return label, math.degrees(dihedral_angle(n_a, n_b))


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------


def check_fillet_feasibility(edge: EdgePreflight, face_a: FaceType,
                             face_b: FaceType, radius: float,
                             tol: float = 1e-6) -> FilletFeasibility:
    """Decide whether a rolling-ball fillet of ``radius`` can be seated on
    ``edge`` between the supports ``face_a`` and ``face_b``.

    Returns a :class:`FilletFeasibility` in every case -- an infeasible or
    out-of-contract request is refused with a reason code, not raised.  No
    fillet geometry is produced or implied.

    The candidate is screened in widening order: is the radius a usable
    number, is the support pair one we model, is the edge the shape that
    pair implies, do the supports meet convexly there, and finally does the
    radius clear every clearance bound of the detected case.
    """
    if isinstance(radius, bool) or not isinstance(radius, (int, float)) \
            or radius <= 0.0:
        return _refusal(
            REASON_NONPOSITIVE_RADIUS,
            "radius must be a positive number, got %r" % (radius,),
            classify_support_pair(face_a, face_b),
        )
    radius = float(radius)

    case = classify_support_pair(face_a, face_b)
    if case == CASE_PLANAR_PLANAR:
        return _preflight_crease(edge, face_a, face_b, radius, tol)
    if case == CASE_PLANAR_CYLINDRICAL:
        return _preflight_rim(edge, face_a, face_b, radius, tol)
    return _refusal(
        REASON_UNSUPPORTED_FACE_PAIR,
        "edge supports must be planar+planar or planar+cylindrical; got "
        "%s + %s" % (type(face_a).__name__, type(face_b).__name__),
        CASE_UNSUPPORTED,
    )


def _preflight_crease(edge: EdgePreflight, face_a: PlanarFace,
                      face_b: PlanarFace, radius: float,
                      tol: float) -> FilletFeasibility:
    """Two planes meeting along a straight edge."""
    straight = _as_straight_edge(edge, tol)
    if straight is None:
        return _refusal(
            REASON_EDGE_NOT_STRAIGHT,
            "edge is not a straight segment; two planar supports meet along "
            "the straight line of intersection of their planes",
            CASE_PLANAR_PLANAR,
        )
    seat, tangent, edge_length = straight

    label, dihedral_deg = _convexity_of(face_a.normal, face_b.normal, tangent,
                                        edge.forward)
    if label != CONVEX:
        return _refusal(
            REASON_EDGE_NOT_CONVEX,
            "a rolling ball can only be seated where the supports meet "
            "convexly from the material side; this edge is %s" % label,
            CASE_PLANAR_PLANAR,
            convexity=label, dihedral_deg=dihedral_deg,
        )

    reach_a = _planar_clearance(face_a, seat, tangent)
    reach_b = _planar_clearance(face_b, seat, tangent)
    if reach_a <= tol or reach_b <= tol:
        return _refusal(
            REASON_DEGENERATE_INPUT,
            "a support has no measurable reach away from the edge; its "
            "boundary polygon is degenerate for this edge",
            CASE_PLANAR_PLANAR,
            convexity=label, dihedral_deg=dihedral_deg,
            limits={"support_a_extent": reach_a,
                    "support_b_extent": reach_b,
                    "edge_length": edge_length},
        )

    # A ball wider than the edge it rides cannot complete the sweep, so that
    # bound is stated before the two clearance bounds.
    bounds = [
        _Bound(
            "edge_length", edge_length, 0.0,
            REASON_RADIUS_EXCEEDS_EDGE_LENGTH,
            lambda r: "radius %s is not shorter than the edge (%s); the "
                      "fillet would consume the whole edge"
                      % (r, edge_length),
        ),
        _Bound(
            "support_a_extent", reach_a, tol,
            REASON_RADIUS_EXCEEDS_SUPPORT_EXTENT,
            lambda r: "radius %s puts the contact track off support a, "
                      "which reaches only %.6g from the edge" % (r, reach_a),
        ),
        _Bound(
            "support_b_extent", reach_b, tol,
            REASON_RADIUS_EXCEEDS_SUPPORT_EXTENT,
            lambda r: "radius %s puts the contact track off support b, "
                      "which reaches only %.6g from the edge" % (r, reach_b),
        ),
    ]
    return _adjudicate(radius, bounds, CASE_PLANAR_PLANAR, label,
                       dihedral_deg)


def _preflight_rim(edge: EdgePreflight, face_a: FaceType, face_b: FaceType,
                   radius: float, tol: float) -> FilletFeasibility:
    """A plane meeting a cylinder along a circular rim."""
    plane_leads = isinstance(face_a, PlanarFace)
    plane = face_a if plane_leads else face_b
    cyl = face_b if plane_leads else face_a
    assert isinstance(plane, PlanarFace)
    assert isinstance(cyl, CylindricalFace)

    if not _is_rim_of(edge, plane, cyl, tol):
        return _refusal(
            REASON_EDGE_NOT_CAP_RIM,
            "edge is not the circular rim where the cylinder meets the "
            "plane",
            CASE_PLANAR_CYLINDRICAL,
        )

    # Convexity is uniform around a rim, so one sample settles it.
    sample = edge.points[0]
    tangent = _normalise(_diff(edge.points[1], sample))
    normal_cyl = cyl.outward_normal_at(sample)
    lead, trail = ((plane.normal, normal_cyl) if plane_leads
                   else (normal_cyl, plane.normal))
    label, dihedral_deg = _convexity_of(lead, trail, tangent, edge.forward)
    if label != CONVEX:
        return _refusal(
            REASON_EDGE_NOT_CONVEX,
            "a rolling ball can only be seated where the supports meet "
            "convexly from the material side; this rim is %s" % label,
            CASE_PLANAR_CYLINDRICAL,
            convexity=label, dihedral_deg=dihedral_deg,
        )

    reach = _rim_clearance(plane, cyl)
    if reach is None or reach <= tol:
        return _refusal(
            REASON_DEGENERATE_INPUT,
            "the planar support has no measurable reach away from the rim",
            CASE_PLANAR_CYLINDRICAL,
            convexity=label, dihedral_deg=dihedral_deg,
        )

    bounds = [
        _Bound(
            "cylinder_radius", cyl.radius, 0.0,
            REASON_RADIUS_EXCEEDS_CYLINDER_RADIUS,
            lambda r: "radius %s is not smaller than the cylinder radius "
                      "(%s); the ball cannot fit against the barrel"
                      % (r, cyl.radius),
        ),
        _Bound(
            "cylinder_height", cyl.height, 0.0,
            REASON_RADIUS_EXCEEDS_CYLINDER_HEIGHT,
            lambda r: "radius %s is not smaller than the cylinder height "
                      "(%s); the ball cannot fit along the barrel"
                      % (r, cyl.height),
        ),
        _Bound(
            "cap_extent", reach, tol,
            REASON_RADIUS_EXCEEDS_SUPPORT_EXTENT,
            lambda r: "radius %s puts the contact circle off the planar "
                      "support, which reaches only %.6g from the rim"
                      % (r, reach),
        ),
    ]
    return _adjudicate(radius, bounds, CASE_PLANAR_CYLINDRICAL, label,
                       dihedral_deg)


# ---------------------------------------------------------------------------
# Selfcheck
# ---------------------------------------------------------------------------


def _ring(centre: Vec3, radius: float, samples: int,
          ccw: bool = True) -> Tuple[Vec3, ...]:
    """Closed sampled circle in the plane z = centre[2], wound CCW about +Z
    when ``ccw`` and CW otherwise (the winding of an inner loop)."""
    turn = 2.0 * math.pi / samples
    ring = []
    for i in range(samples + 1):
        theta = turn * (i % samples) * (1.0 if ccw else -1.0)
        ring.append((centre[0] + radius * math.cos(theta),
                     centre[1] + radius * math.sin(theta),
                     centre[2]))
    return tuple(ring)


def _selfcheck() -> None:
    # --- planar+planar: the vertical edge of a 1 x 1 x 2 block -------------
    # Both supports reach 1.0 from the edge; the edge itself is 2.0 long.
    wall_x = PlanarFace(
        origin=(1.0, 0.5, 1.0), normal=(1.0, 0.0, 0.0),
        boundary=((1.0, 0.0, 0.0), (1.0, 1.0, 0.0),
                  (1.0, 1.0, 2.0), (1.0, 0.0, 2.0)),
    )
    wall_y = PlanarFace(
        origin=(0.5, 1.0, 1.0), normal=(0.0, 1.0, 0.0),
        boundary=((0.0, 1.0, 0.0), (1.0, 1.0, 0.0),
                  (1.0, 1.0, 2.0), (0.0, 1.0, 2.0)),
    )
    corner = EdgePreflight(points=((1.0, 1.0, 0.0), (1.0, 1.0, 1.0),
                                   (1.0, 1.0, 2.0)))

    seated = check_fillet_feasibility(corner, wall_x, wall_y, 0.2)
    assert seated.feasible, seated.reason
    assert seated.case == CASE_PLANAR_PLANAR
    assert seated.convexity == CONVEX
    assert abs(seated.dihedral_deg - 90.0) < 1e-9
    assert abs(seated.max_feasible_radius - 1.0) < 1e-9
    assert abs(seated.limits["support_a_extent"] - 1.0) < 1e-9
    assert abs(seated.limits["support_b_extent"] - 1.0) < 1e-9
    assert abs(seated.limits["edge_length"] - 2.0) < 1e-9
    print("[selfcheck] planar+planar r=0.2 feasible "
          "(bound %.3f, dihedral %.1f deg)"
          % (seated.max_feasible_radius, seated.dihedral_deg))

    too_wide = check_fillet_feasibility(corner, wall_x, wall_y, 1.5)
    assert not too_wide.feasible
    assert too_wide.reason_code == REASON_RADIUS_EXCEEDS_SUPPORT_EXTENT
    assert abs(too_wide.max_feasible_radius - 1.0) < 1e-9
    print("[selfcheck] planar+planar r=1.5 refused: %s (bound %.3f)"
          % (too_wide.reason_code, too_wide.max_feasible_radius))

    # Past the edge length too: the edge bound is the one reported.
    too_long = check_fillet_feasibility(corner, wall_x, wall_y, 2.5)
    assert not too_long.feasible
    assert too_long.reason_code == REASON_RADIUS_EXCEEDS_EDGE_LENGTH
    print("[selfcheck] planar+planar r=2.5 refused: %s"
          % too_long.reason_code)

    # Flip one outward normal and the same edge becomes an inner corner.
    pocket_wall = PlanarFace(origin=wall_x.origin, normal=(-1.0, 0.0, 0.0),
                             boundary=wall_x.boundary)
    inner = check_fillet_feasibility(corner, pocket_wall, wall_y, 0.2)
    assert not inner.feasible
    assert inner.reason_code == REASON_EDGE_NOT_CONVEX
    print("[selfcheck] concave crease refused: %s" % inner.reason_code)

    # A curved polyline is not a plane-plane intersection.
    bent = EdgePreflight(points=((1.0, 1.0, 0.0), (1.2, 1.0, 1.0),
                                 (1.0, 1.0, 2.0)))
    assert check_fillet_feasibility(bent, wall_x, wall_y, 0.1).reason_code \
        == REASON_EDGE_NOT_STRAIGHT

    # --- planar+cylindrical: a hole rim in the top of a unit cube ---------
    # Hole radius 0.3 centred in a 1 x 1 face: the cap reaches 0.5 - 0.3
    # = 0.2 from the rim, so feasibility flips there.
    top = PlanarFace(
        origin=(0.5, 0.5, 1.0), normal=(0.0, 0.0, 1.0),
        boundary=((0.0, 0.0, 1.0), (1.0, 0.0, 1.0),
                  (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)),
    )
    bore = CylindricalFace(axis_point=(0.5, 0.5, 0.0),
                           axis_dir=(0.0, 0.0, 1.0), radius=0.3, height=1.0,
                           outward_radial=False)
    bore_rim = EdgePreflight(points=_ring((0.5, 0.5, 1.0), 0.3, 24,
                                          ccw=False))
    hole_ok = check_fillet_feasibility(bore_rim, top, bore, 0.1)
    assert hole_ok.feasible, hole_ok.reason
    assert hole_ok.case == CASE_PLANAR_CYLINDRICAL
    assert hole_ok.convexity == CONVEX
    assert abs(hole_ok.max_feasible_radius - 0.2) < 1e-9
    assert abs(hole_ok.limits["cap_extent"] - 0.2) < 1e-9
    hole_bad = check_fillet_feasibility(bore_rim, top, bore, 0.25)
    assert not hole_bad.feasible
    assert hole_bad.reason_code == REASON_RADIUS_EXCEEDS_SUPPORT_EXTENT
    print("[selfcheck] hole rim: r=0.1 feasible, r=0.25 refused (%s); "
          "bound at %.3f"
          % (hole_bad.reason_code, hole_ok.max_feasible_radius))

    # --- planar+cylindrical: the cap rim of a solid barrel ----------------
    cap_ring = _ring((0.0, 0.0, 1.0), 0.5, 24, ccw=True)
    cap = PlanarFace(origin=(0.0, 0.0, 1.0), normal=(0.0, 0.0, 1.0),
                     boundary=cap_ring[:-1])
    barrel = CylindricalFace(axis_point=(0.0, 0.0, 0.0),
                             axis_dir=(0.0, 0.0, 1.0), radius=0.5,
                             height=1.0, outward_radial=True)
    cap_rim = EdgePreflight(points=cap_ring)
    cap_ok = check_fillet_feasibility(cap_rim, cap, barrel, 0.4)
    assert cap_ok.feasible, cap_ok.reason
    assert abs(cap_ok.max_feasible_radius - 0.5) < 1e-9
    cap_bad = check_fillet_feasibility(cap_rim, cap, barrel, 0.6)
    assert not cap_bad.feasible
    assert cap_bad.reason_code == REASON_RADIUS_EXCEEDS_CYLINDER_RADIUS
    print("[selfcheck] cap rim: r=0.4 feasible, r=0.6 refused (%s)"
          % cap_bad.reason_code)

    # A short barrel is limited by its height instead.
    stub = CylindricalFace(axis_point=(0.0, 0.0, 0.0),
                           axis_dir=(0.0, 0.0, 1.0), radius=0.5, height=0.2)
    stub_bad = check_fillet_feasibility(cap_rim, cap, stub, 0.3)
    assert stub_bad.reason_code == REASON_RADIUS_EXCEEDS_CYLINDER_HEIGHT

    # --- out of contract ---------------------------------------------------
    crossing = CylindricalFace(axis_point=(0.0, 0.0, 0.0),
                               axis_dir=(1.0, 0.0, 0.0), radius=0.5,
                               height=1.0)
    unsupported = check_fillet_feasibility(cap_rim, barrel, crossing, 0.1)
    assert not unsupported.feasible
    assert unsupported.reason_code == REASON_UNSUPPORTED_FACE_PAIR
    assert unsupported.case == CASE_UNSUPPORTED
    assert "planar+planar or planar+cylindrical" in unsupported.reason
    print("[selfcheck] cylinder+cylinder refused: %s"
          % unsupported.reason_code)

    for bad_radius in (0.0, -1.0, True):
        nope = check_fillet_feasibility(corner, wall_x, wall_y, bad_radius)
        assert not nope.feasible
        assert nope.reason_code == REASON_NONPOSITIVE_RADIUS
    print("[selfcheck] non-positive radii refused: %s"
          % REASON_NONPOSITIVE_RADIUS)

    # The verdict is diagnostics only -- no geometry rides along with it.
    assert set(seated.to_dict()) == {
        "feasible", "reason_code", "reason", "case", "convexity",
        "dihedral_deg", "max_feasible_radius", "limits"}
    assert "planar+cylindrical" in supported_contract()
    print("[selfcheck] OK")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.geometry.features."
             "fillet_feasibility",
        description="Rolling-ball fillet feasibility preflight; reports "
                    "whether a fillet could be placed, and builds nothing.",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="run the deterministic synthetic-geometry "
                             "checks of the feasibility predicate.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0
    _selfcheck()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
