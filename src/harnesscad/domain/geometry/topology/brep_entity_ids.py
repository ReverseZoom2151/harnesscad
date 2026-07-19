"""JoinABLe B-rep entity-type + discrete edge-convexity id tables (fact tables).

The harness already models B-rep surfaces analytically -- ``Plane``, ``Cylinder``,
``Cone``, ``Sphere``, ``Torus`` in
:mod:`harnesscad.domain.geometry.parametric.analytic_surfaces` -- and it labels
shared edges with a *continuous* three-way sign (``convex`` / ``concave`` /
``smooth`` strings) in
:mod:`harnesscad.domain.geometry.topology.edge_convexity`. What it did not have
is (a) an integer taxonomy spanning the full set of surface *and* curve entity
types a B-rep node can carry, and (b) the *discrete six-value* convexity
classification (including the ``None``, ``Non-manifold`` and ``Degenerate``
states that a continuous sign cannot express).

This module supplies both, transcribed verbatim (no invented ids) from the
JoinABLe reference implementation:

    resources/cad_repos/JoinABLe-main/JoinABLe-main/
        datasets/joint_graph_dataset.py  ->  entity_type_map (16 types, 0..15)
                                              convexity_type_map (6 states, 0..5)
        joint/joint_axis.py              ->  check_colinear_with_tolerance(
                                                 angle_tol_degs=10.0,
                                                 distance_tol=1e-2)

Source: JoinABLe, Willis et al. 2022, MIT License
(``Copyright (c) 2022 Autodesk, Inc``).

Entity types 0..15 cover eight surface families (plane, cylinder, cone, sphere,
torus, elliptical cylinder, elliptical cone, NURBS surface) and eight curve
families (line, arc, circle, ellipse, elliptical arc, infinite line, NURBS
curve, and a degenerate-edge sentinel). Convexity states 0..5 are
``None`` (assigned to faces), ``Convex``, ``Concave``, ``Smooth``,
``Non-manifold``, and ``Degenerate`` (assigned to degenerate edges).

Because these are published fact tables the ids are authoritative and must not
be renumbered. The harness's continuous ``edge_convexity`` labels map onto the
discrete states as a strict subset via :data:`EDGE_CONVEXITY_TO_ID`, and the
analytic surface classes map onto the surface ids via
:data:`ANALYTIC_SURFACE_TO_ID` -- the new ids (elliptical/NURBS surfaces, all
curve types, and the None/Non-manifold/Degenerate convexity states) are a
superset of what the harness already had.

Wiring points (reported, not performed -- those files are not owned here)
-------------------------------------------------------------------------
* :mod:`harnesscad.domain.geometry.topology.edge_convexity` -- its ``CONVEX`` /
  ``CONCAVE`` / ``SMOOTH`` string labels are three of the six discrete states
  here; ``classify_edge_convexity`` could return :class:`Convexity` ids, and its
  ``AttributedAdjacencyGraph`` could accept ``None`` / ``Non-manifold`` /
  ``Degenerate`` arcs by validating against :data:`convexity_name_to_id`.
* :mod:`harnesscad.domain.geometry.parametric.analytic_surfaces` -- its surface
  classes align with entity ids 0..4 via :data:`ANALYTIC_SURFACE_TO_ID`.

Public API
----------
``EntityType`` / ``Convexity``      -- the integer enums.
``entity_name_to_id`` / ``entity_id_to_name``  -- entity round-trip tables.
``convexity_name_to_id`` / ``convexity_id_to_name``  -- convexity round-trip.
``is_surface_type(t)`` / ``is_curve_type(t)``  -- entity family predicates.
``is_convex(c)`` / ``is_concave(c)`` / ``classify(...)``  -- convexity helpers.
``AXIS_ANGLE_TOL_DEG`` / ``AXIS_DISTANCE_TOL``  -- axis colinearity thresholds.
``axis_lines_colinear(...)``        -- the JoinABLe axis-hit test.
``ANALYTIC_SURFACE_TO_ID`` / ``EDGE_CONVEXITY_TO_ID``  -- harness bridges.
"""

from __future__ import annotations

import enum
from typing import Dict, Optional

__all__ = [
    "EntityType",
    "Convexity",
    "entity_name_to_id",
    "entity_id_to_name",
    "convexity_name_to_id",
    "convexity_id_to_name",
    "SURFACE_TYPES",
    "CURVE_TYPES",
    "is_surface_type",
    "is_curve_type",
    "is_convex",
    "is_concave",
    "classify",
    "AXIS_ANGLE_TOL_DEG",
    "AXIS_DISTANCE_TOL",
    "axis_lines_colinear",
    "ANALYTIC_SURFACE_TO_ID",
    "EDGE_CONVEXITY_TO_ID",
]


class EntityType(enum.IntEnum):
    """B-rep node entity types (JoinABLe ``entity_type_map``, ids 0..15).

    Ids 0..7 are surface families; ids 8..15 are curve families (15 is the
    degenerate-edge sentinel). Transcribed verbatim from JoinABLe
    ``datasets/joint_graph_dataset.py``.
    """

    PlaneSurfaceType = 0
    CylinderSurfaceType = 1
    ConeSurfaceType = 2
    SphereSurfaceType = 3
    TorusSurfaceType = 4
    EllipticalCylinderSurfaceType = 5
    EllipticalConeSurfaceType = 6
    NurbsSurfaceType = 7
    Line3DCurveType = 8
    Arc3DCurveType = 9
    Circle3DCurveType = 10
    Ellipse3DCurveType = 11
    EllipticalArc3DCurveType = 12
    InfiniteLine3DCurveType = 13
    NurbsCurve3DCurveType = 14
    Degenerate3DCurveType = 15  # special case for degenerate edges


class Convexity(enum.IntEnum):
    """Discrete edge-convexity states (JoinABLe ``convexity_type_map``, ids 0..5).

    ``NoneType`` (id 0) is assigned to faces; ``Degenerate`` (id 5) to degenerate
    edges. The middle four are the true per-edge classes. The member is spelled
    ``NoneType`` because ``None`` is a Python keyword; its canonical wire name is
    the string ``"None"`` (see :data:`convexity_name_to_id`).
    """

    NoneType = 0
    Convex = 1
    Concave = 2
    Smooth = 3
    Nonmanifold = 4  # wire name "Non-manifold"
    Degenerate = 5


# Wire names differ from Python-safe member names in two spots (None / Non-manifold).
_CONVEXITY_WIRE_NAME: Dict[Convexity, str] = {
    Convexity.NoneType: "None",
    Convexity.Convex: "Convex",
    Convexity.Concave: "Concave",
    Convexity.Smooth: "Smooth",
    Convexity.Nonmanifold: "Non-manifold",
    Convexity.Degenerate: "Degenerate",
}

#: canonical wire name -> integer id (matches JoinABLe convexity_type_map exactly).
convexity_name_to_id: Dict[str, int] = {
    name: int(member.value) for member, name in _CONVEXITY_WIRE_NAME.items()
}
#: integer id -> canonical wire name.
convexity_id_to_name: Dict[int, str] = {
    int(member.value): name for member, name in _CONVEXITY_WIRE_NAME.items()
}

#: entity name -> integer id (matches JoinABLe entity_type_map exactly).
entity_name_to_id: Dict[str, int] = {m.name: int(m.value) for m in EntityType}
#: integer id -> entity name.
entity_id_to_name: Dict[int, str] = {int(m.value): m.name for m in EntityType}

#: The eight surface families (ids 0..7).
SURFACE_TYPES = frozenset(
    {
        EntityType.PlaneSurfaceType,
        EntityType.CylinderSurfaceType,
        EntityType.ConeSurfaceType,
        EntityType.SphereSurfaceType,
        EntityType.TorusSurfaceType,
        EntityType.EllipticalCylinderSurfaceType,
        EntityType.EllipticalConeSurfaceType,
        EntityType.NurbsSurfaceType,
    }
)
#: The eight curve families (ids 8..15).
CURVE_TYPES = frozenset(m for m in EntityType if m not in SURFACE_TYPES)


def is_surface_type(entity_type: EntityType) -> bool:
    """Whether an entity type is a surface (face) family."""
    return EntityType(entity_type) in SURFACE_TYPES


def is_curve_type(entity_type: EntityType) -> bool:
    """Whether an entity type is a curve (edge) family."""
    return EntityType(entity_type) in CURVE_TYPES


def is_convex(convexity: Convexity) -> bool:
    """Whether a discrete convexity state is the convex edge class."""
    return Convexity(convexity) is Convexity.Convex


def is_concave(convexity: Convexity) -> bool:
    """Whether a discrete convexity state is the concave edge class."""
    return Convexity(convexity) is Convexity.Concave


def classify(name: str) -> Convexity:
    """Discrete :class:`Convexity` for a wire name (e.g. ``"Non-manifold"``).

    Accepts the exact JoinABLe wire names (``None``, ``Convex``, ``Concave``,
    ``Smooth``, ``Non-manifold``, ``Degenerate``); case-insensitive. Raises
    ``KeyError`` for an unknown state.
    """
    key = name.strip().lower()
    for member, wire in _CONVEXITY_WIRE_NAME.items():
        if wire.lower() == key:
            return member
    raise KeyError(f"unknown convexity state: {name!r}")


# Axis colinearity thresholds -- JoinABLe joint_axis.check_colinear_with_tolerance
# uses these defaults to decide whether two entity axes count as the same joint
# axis: angle < 10 degrees AND distance < 1e-2.
AXIS_ANGLE_TOL_DEG: float = 10.0
AXIS_DISTANCE_TOL: float = 1e-2


def axis_lines_colinear(
    angle_deg: float,
    distance: float,
    angle_tol_deg: float = AXIS_ANGLE_TOL_DEG,
    distance_tol: float = AXIS_DISTANCE_TOL,
) -> bool:
    """Whether two axis lines count as colinear (the JoinABLe axis-hit test).

    ``angle_deg`` is the (direction-reversal-minimised) angle between the two
    axis directions in degrees; ``distance`` is the perpendicular distance
    between the lines. Both must be strictly under tolerance, matching the source
    predicate ``angle_degs < angle_tol_degs and dist < distance_tol``.
    """
    return angle_deg < angle_tol_deg and distance < distance_tol


# --- bridges to the harness's existing partial vocabularies -----------------

# analytic_surfaces.py surface class name -> JoinABLe surface entity id.
ANALYTIC_SURFACE_TO_ID: Dict[str, int] = {
    "Plane": EntityType.PlaneSurfaceType,
    "Cylinder": EntityType.CylinderSurfaceType,
    "Cone": EntityType.ConeSurfaceType,
    "Sphere": EntityType.SphereSurfaceType,
    "Torus": EntityType.TorusSurfaceType,
}

# edge_convexity.py continuous label -> discrete convexity id (strict subset).
EDGE_CONVEXITY_TO_ID: Dict[str, int] = {
    "convex": Convexity.Convex,
    "concave": Convexity.Concave,
    "smooth": Convexity.Smooth,
}


def _selfcheck() -> int:
    """Prove real properties of both tables. Returns a process exit code."""
    problems = []

    # 1. Entity ids: 16 members, unique, contiguous 0..15.
    ent_ids = [int(m.value) for m in EntityType]
    if len(ent_ids) != len(set(ent_ids)):
        problems.append("duplicate entity ids")
    if sorted(ent_ids) != list(range(16)):
        problems.append(f"entity ids not contiguous 0..15: {sorted(ent_ids)}")
    if len(SURFACE_TYPES) != 8 or len(CURVE_TYPES) != 8:
        problems.append("surface/curve split is not 8/8")

    # 2. Entity name<->id round-trip over all 16.
    for member in EntityType:
        if entity_name_to_id[member.name] != int(member.value):
            problems.append(f"entity name->id failed for {member.name}")
        if entity_id_to_name[int(member.value)] != member.name:
            problems.append(f"entity id->name failed for {member.value}")

    # 3. Convexity classifier covers all 6 states, ids unique + contiguous 0..5.
    cvx_ids = [int(m.value) for m in Convexity]
    if len(cvx_ids) != len(set(cvx_ids)):
        problems.append("duplicate convexity ids")
    if sorted(cvx_ids) != list(range(6)):
        problems.append(f"convexity ids not contiguous 0..5: {sorted(cvx_ids)}")
    for wire, cid in convexity_name_to_id.items():
        if int(classify(wire).value) != cid:
            problems.append(f"classify round-trip failed for {wire!r}")
    # all six states reachable through classify by their wire names
    if len({classify(w) for w in convexity_name_to_id}) != 6:
        problems.append("classify does not cover all 6 states")

    # 4. Superset over the harness's continuous edge_convexity labels.
    for label in ("convex", "concave", "smooth"):
        if label not in EDGE_CONVEXITY_TO_ID:
            problems.append(f"edge_convexity label {label!r} not bridged")
    # the three NEW states absent from continuous edge_convexity:
    new_states = {"None", "Non-manifold", "Degenerate"}
    if not new_states <= set(convexity_name_to_id):
        problems.append("expected new discrete states missing")

    # 5. Superset over the harness's 5 analytic surface classes.
    for cls in ("Plane", "Cylinder", "Cone", "Sphere", "Torus"):
        if cls not in ANALYTIC_SURFACE_TO_ID:
            problems.append(f"analytic surface {cls!r} not bridged")

    # 6. Axis-hit thresholds match the cited source defaults.
    if AXIS_ANGLE_TOL_DEG != 10.0 or AXIS_DISTANCE_TOL != 1e-2:
        problems.append("axis thresholds do not match source (10 deg / 1e-2)")
    if not axis_lines_colinear(9.9, 9e-3):
        problems.append("axis_lines_colinear false negative")
    if axis_lines_colinear(10.0, 0.0) or axis_lines_colinear(0.0, 1e-2):
        problems.append("axis_lines_colinear not strict at the boundary")

    print(f"EntityType members: {len(EntityType)} (surfaces={len(SURFACE_TYPES)}, "
          f"curves={len(CURVE_TYPES)}), ids contiguous 0..15")
    print(f"entity name<->id round-trip: {len(EntityType)}/{len(EntityType)}")
    print(f"Convexity states: {len(Convexity)} covering "
          f"{sorted(convexity_name_to_id)}")
    print(f"classify covers all 6 states: yes")
    print(f"new discrete states beyond continuous edge_convexity: "
          f"{sorted(new_states)}")
    print(f"axis thresholds: angle<{AXIS_ANGLE_TOL_DEG} deg, dist<{AXIS_DISTANCE_TOL}")

    if problems:
        for p in problems:
            print("FAIL:", p)
        return 1
    print("OK")
    return 0


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="verify entity/convexity id uniqueness, round-trip, and supersets",
    )
    args = parser.parse_args(argv)

    if args.selfcheck:
        return _selfcheck()

    print("Entity types:")
    for member in EntityType:
        family = "surface" if is_surface_type(member) else "curve"
        print(f"  {int(member.value):>2}  {member.name:<32} ({family})")
    print("Convexity states:")
    for member in Convexity:
        print(f"  {int(member.value):>2}  {_CONVEXITY_WIRE_NAME[member]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
