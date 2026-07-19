"""Onshape sketch-constraint type ids (``ConstraintType``, a fact table).

The harness already models sketch constraints by *name*: HistCAD's 19-name
evaluation set lives in :mod:`harnesscad.domain.geometry.sketch.constraint_satisfaction`
(``CONSTRAINT_TYPES``), and the Onshape wire format in
:mod:`harnesscad.io.formats.onshape_json` carries an ``EntityType`` /
``SubnodeType`` taxonomy for *geometry* but no numeric taxonomy for the
*constraints* themselves. What was missing is the concrete integer id that
Onshape's FeatureScript assigns to every constraint kind -- the code a
generative sketch model quantises and predicts over.

This module supplies that table. The ids are Onshape's own API values: they run
0..29 contiguously plus the out-of-band ``Subnode = 101`` sentinel, and they are
reproduced here exactly, with no invented or renumbered entries.

Because these are third-party API constants they are authoritative and must not
be renumbered. The harness's HistCAD names are a strict *subset* of this
taxonomy; :data:`HISTCAD_TO_ID` records the mapping (including the two aliases
``minor_radius -> Minor_Diameter`` and ``major_radius -> Major_Diameter``, where
Onshape labels the dimension by diameter rather than radius).

Wiring point (reported, not performed -- these files are not owned here)
-----------------------------------------------------------------------
* :mod:`harnesscad.io.formats.onshape_json` -- its ``EntityType`` /
  ``SubnodeType`` enums could import :class:`ConstraintType` from here to gain a
  numeric constraint taxonomy alongside the geometry one. ``SubnodeType.SN_*``
  already reuses the ``101`` band; ``ConstraintType.Subnode = 101`` is the
  corresponding constraint-side sentinel.
* :mod:`harnesscad.domain.geometry.sketch.constraint_satisfaction` -- its
  ``CONSTRAINT_TYPES`` name tuple can be given integer ids by looking each name
  up through :data:`HISTCAD_TO_ID` / :func:`id_for_name`.

Public API
----------
``ConstraintType``       -- the integer enum (0..29, 101).
``name_to_id`` / ``id_to_name``  -- round-tripping lookup tables.
``id_for_name(name)`` / ``name_for_id(id)``  -- case-insensitive helpers.
``HISTCAD_TO_ID``        -- HistCAD constraint name -> ConstraintType id.
``has_parameters(t)``    -- whether a constraint carries numeric parameters.
"""

from __future__ import annotations

import enum
from typing import Dict

__all__ = [
    "ConstraintType",
    "name_to_id",
    "id_to_name",
    "id_for_name",
    "name_for_id",
    "HISTCAD_TO_ID",
    "PARAMETRIC_TYPES",
    "has_parameters",
]


class ConstraintType(enum.IntEnum):
    """Onshape sketch constraint types with their FeatureScript integer ids."""

    Coincident = 0
    Projected = 1
    Mirror = 2
    Distance = 3
    Horizontal = 4
    Parallel = 5
    Vertical = 6
    Tangent = 7
    Length = 8
    Perpendicular = 9
    Midpoint = 10
    Equal = 11
    Diameter = 12
    Offset = 13
    Radius = 14
    Concentric = 15
    Fix = 16
    Angle = 17
    Circular_Pattern = 18
    Pierce = 19
    Linear_Pattern = 20
    Centerline_Dimension = 21
    Intersected = 22
    Silhoutted = 23  # (sic) spelling as it appears in the Onshape taxonomy
    Quadrant = 24
    Normal = 25
    Minor_Diameter = 26
    Major_Diameter = 27
    Rho = 28
    Unknown = 29
    Subnode = 101  # out-of-band sentinel for implicit sub-entity references


#: name (as written in the enum) -> integer id.
name_to_id: Dict[str, int] = {m.name: int(m.value) for m in ConstraintType}

#: integer id -> canonical name.
id_to_name: Dict[int, str] = {int(m.value): m.name for m in ConstraintType}


def id_for_name(name: str) -> int:
    """Integer id for a constraint name (case-insensitive). Raises ``KeyError``."""
    key = name.strip().lower()
    for member in ConstraintType:
        if member.name.lower() == key:
            return int(member.value)
    raise KeyError(f"unknown constraint type name: {name!r}")


def name_for_id(type_id: int) -> str:
    """Canonical name for an integer id. Raises ``KeyError`` if unknown."""
    return ConstraintType(type_id).name


# HistCAD's 19 evaluation-side constraint names (see constraint_satisfaction.py)
# mapped onto this authoritative id table. ``minor_radius`` / ``major_radius``
# are the two aliases -- Onshape labels these dimensions by diameter, HistCAD by
# radius; they denote the same constraint family.
HISTCAD_TO_ID: Dict[str, int] = {
    "coincident": ConstraintType.Coincident,
    "horizontal": ConstraintType.Horizontal,
    "vertical": ConstraintType.Vertical,
    "parallel": ConstraintType.Parallel,
    "perpendicular": ConstraintType.Perpendicular,
    "concentric": ConstraintType.Concentric,
    "tangent": ConstraintType.Tangent,
    "normal": ConstraintType.Normal,
    "length": ConstraintType.Length,
    "distance": ConstraintType.Distance,
    "diameter": ConstraintType.Diameter,
    "radius": ConstraintType.Radius,
    "angle": ConstraintType.Angle,
    "minor_radius": ConstraintType.Minor_Diameter,
    "major_radius": ConstraintType.Major_Diameter,
    "fix": ConstraintType.Fix,
    "midpoint": ConstraintType.Midpoint,
    "equal": ConstraintType.Equal,
    "mirror": ConstraintType.Mirror,
}


#: Constraint types that carry numeric parameters in the Onshape schema.
PARAMETRIC_TYPES = frozenset(
    {
        ConstraintType.Angle,
        ConstraintType.Distance,
        ConstraintType.Length,
        ConstraintType.Offset,
        ConstraintType.Diameter,
        ConstraintType.Radius,
    }
)


def has_parameters(constraint_type: ConstraintType) -> bool:
    """Whether a constraint type carries numeric parameters."""
    return ConstraintType(constraint_type) in PARAMETRIC_TYPES


def _selfcheck() -> int:
    """Prove real properties of the table. Returns a process exit code."""
    problems = []

    # 1. Every id is unique.
    ids = [int(m.value) for m in ConstraintType]
    if len(ids) != len(set(ids)):
        problems.append("duplicate integer ids in ConstraintType")

    # 2. Name<->id round-trips totally over the whole enum.
    for member in ConstraintType:
        rid = id_for_name(member.name)
        if rid != int(member.value):
            problems.append(f"name->id failed for {member.name}")
        if name_for_id(rid) != member.name:
            problems.append(f"id->name failed for {member.value}")

    # 3. Contiguous 0..29 core plus the 101 sentinel, as Onshape assigns them.
    core = sorted(i for i in ids if i < 100)
    if core != list(range(0, 30)):
        problems.append(f"core ids not contiguous 0..29: {core}")
    if 101 not in ids:
        problems.append("missing Subnode = 101 sentinel")

    # 4. Superset over the harness's existing HistCAD names: every one maps.
    histcad_names = (
        "coincident", "horizontal", "vertical", "parallel", "perpendicular",
        "concentric", "tangent", "normal", "length", "distance", "diameter",
        "radius", "angle", "minor_radius", "major_radius", "fix", "midpoint",
        "equal", "mirror",
    )
    for name in histcad_names:
        if name not in HISTCAD_TO_ID:
            problems.append(f"HistCAD name {name!r} has no id mapping")

    total = len(ConstraintType)
    print(f"ConstraintType members: {total}")
    print(f"core ids 0..29 contiguous + Subnode=101 sentinel: yes")
    print(f"name<->id round-trip: {total}/{total}")
    print(f"HistCAD names covered (superset): {len(histcad_names)}/{len(histcad_names)}")
    print(f"parametric types: {sorted(t.name for t in PARAMETRIC_TYPES)}")

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
        help="verify id uniqueness, name<->id round-trip, and HistCAD superset",
    )
    args = parser.parse_args(argv)

    if args.selfcheck:
        return _selfcheck()

    for member in ConstraintType:
        print(f"{int(member.value):>3}  {member.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
