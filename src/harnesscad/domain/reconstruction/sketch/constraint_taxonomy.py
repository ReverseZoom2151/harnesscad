"""Sketch primitive & constraint taxonomy.

This is the *taxonomy* layer that a relational sketch representation is built
on. Two catalogs are provided:

Primitives (parameters + observed corpus frequencies)
    Each primitive type carries a fixed number of degrees of freedom (DOF) and an
    observed dataset frequency::

        Point   dof 2    8.58 %
        Line    dof 4   68.47 %
        Circle  dof 3    9.97 %
        Arc     dof 5    9.45 %
        Ellipse dof 5    0.08 %
        Spline  dof None (variable # control points)  2.57 %

    Note this is a strict superset of the DOF table in :mod:`cisp.ops`
    (``PRIMITIVE_DOF`` there only has point/line/circle/rectangle) -- it adds the
    arc, ellipse and spline as well.

Constraints (parameter schemata + observed corpus frequencies)
    Every constraint type is described by one or more *schemata* -- ordered tuples
    of parameter names. Parameters named ``local0``, ``local1``, ``local2`` are
    references to primitives (the edge's member nodes); the rest are numeric /
    enumerated / boolean quantities (``length``, ``angle``, ``direction``,
    ``halfSpace0`` ...). The number of ``local#`` references is the constraint's
    *member arity*, which fixes whether the constraint is a self-loop (arity 1), a
    plain edge (arity 2) or a hyperedge (arity >= 3) in the constraint graph.

The constraint schemata::

    (local0)                                              Horizontal, Vertical
    (local0, local1)                                      Coincident, Horizontal,
                                                          Vertical, Parallel,
                                                          Perpendicular, Tangent,
                                                          Midpoint, Equal, Offset,
                                                          Concentric
    (local0, local1, local2)                              Mirror
    (local0, length)                                      Diameter, Radius
    (local0, direction, length)                           Length
    (local0, local1, direction, halfSpace0,              Distance
             halfSpace1, length)
    (local0, local1, aligned, clockwise, angle)           Angle

Degrees of freedom removed per constraint follow standard 2D geometric-constraint
conventions (a pairwise coincidence pins a point = 2 DOF; an angular / dimensional
relation removes 1). ``mirror`` removes a variable amount (equal to the mirrored
primitive's own DOF) and ``projected`` is external geometry, so both carry
``dof_removed = None``.

Pure stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Parameter-name prefix marking a reference to a primitive (an edge member node).
LOCAL_PREFIX = "local"


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PrimitiveSpec:
    """One primitive type in the taxonomy.

    ``dof`` is the number of degrees of freedom the primitive contributes to the
    sketch (``None`` for spline, whose DOF depend on its control-point count).
    ``frequency_pct`` is the observed corpus frequency.
    """

    name: str
    dof: Optional[int]
    frequency_pct: float


PRIMITIVE_SPECS: Dict[str, PrimitiveSpec] = {
    "point": PrimitiveSpec("point", 2, 8.58),
    "line": PrimitiveSpec("line", 4, 68.47),
    "circle": PrimitiveSpec("circle", 3, 9.97),
    "arc": PrimitiveSpec("arc", 5, 9.45),
    "ellipse": PrimitiveSpec("ellipse", 5, 0.08),
    "spline": PrimitiveSpec("spline", None, 2.57),
}

# Convenience DOF lookup (excludes spline's variable DOF).
PRIMITIVE_DOF: Dict[str, Optional[int]] = {
    n: s.dof for n, s in PRIMITIVE_SPECS.items()
}


def primitive_dof(name: str) -> int:
    """DOF contributed by a primitive type.

    Raises ``KeyError`` for an unknown type and ``ValueError`` for ``spline``
    (variable DOF -- the caller must supply a control-point count separately).
    """
    if name not in PRIMITIVE_SPECS:
        raise KeyError(f"unknown primitive type '{name}'")
    dof = PRIMITIVE_SPECS[name].dof
    if dof is None:
        raise ValueError(f"primitive '{name}' has variable DOF")
    return dof


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConstraintSpec:
    """One constraint type in the taxonomy.

    ``schemas`` is a tuple of parameter-name tuples. ``dof_removed``
    is the nominal DOF a single application removes (``None`` for the variable
    ``mirror`` and the external ``projected``). ``frequency_pct`` is the observed
    corpus frequency (``None`` when the type is not among the most-common set).
    """

    name: str
    schemas: Tuple[Tuple[str, ...], ...]
    dof_removed: Optional[int]
    frequency_pct: Optional[float]
    is_external: bool = False

    # -- schema-derived helpers --------------------------------------------
    @staticmethod
    def _local_count(schema: Tuple[str, ...]) -> int:
        return sum(1 for p in schema if p.startswith(LOCAL_PREFIX))

    @property
    def member_arities(self) -> Tuple[int, ...]:
        """Sorted distinct member arities (# ``local#`` refs) across schemata."""
        return tuple(sorted({self._local_count(s) for s in self.schemas}))

    @property
    def min_members(self) -> int:
        return self.member_arities[0]

    @property
    def max_members(self) -> int:
        return self.member_arities[-1]

    @property
    def is_dimensional(self) -> bool:
        """True when some schema carries a numeric quantity (``length``/``angle``)."""
        return any(
            ("length" in s or "angle" in s) for s in self.schemas
        )

    def allows_loop(self) -> bool:
        """True when this constraint can act on a single primitive (a self-loop)."""
        return 1 in self.member_arities

    def allows_hyperedge(self) -> bool:
        """True when this constraint can act on three or more primitives."""
        return self.max_members >= 3

    def numeric_params(self) -> Tuple[str, ...]:
        """Distinct non-``local#`` parameter names across all schemata (ordered)."""
        seen: list[str] = []
        for s in self.schemas:
            for p in s:
                if not p.startswith(LOCAL_PREFIX) and p not in seen:
                    seen.append(p)
        return tuple(seen)


# Parameter schemata + nominal DOF removed + corpus frequencies.
CONSTRAINT_SPECS: Dict[str, ConstraintSpec] = {
    "coincident": ConstraintSpec(
        "coincident", (("local0", "local1"),), 2, 42.17),
    "projected": ConstraintSpec(
        "projected", (("local0", "local1"),), None, 9.71, is_external=True),
    "distance": ConstraintSpec(
        "distance",
        (("local0", "local1", "direction", "halfSpace0", "halfSpace1", "length"),),
        1, 6.72),
    "horizontal": ConstraintSpec(
        "horizontal", (("local0",), ("local0", "local1")), 1, 6.45),
    "mirror": ConstraintSpec(
        "mirror", (("local0", "local1", "local2"),), None, 5.54),
    "vertical": ConstraintSpec(
        "vertical", (("local0",), ("local0", "local1")), 1, 4.78),
    "parallel": ConstraintSpec(
        "parallel", (("local0", "local1"),), 1, 4.37),
    "length": ConstraintSpec(
        "length", (("local0", "direction", "length"),), 1, 3.68),
    "perpendicular": ConstraintSpec(
        "perpendicular", (("local0", "local1"),), 1, 3.24),
    "tangent": ConstraintSpec(
        "tangent", (("local0", "local1"),), 1, 2.94),
    # Types present in the schemata but not in the reported top set.
    "midpoint": ConstraintSpec(
        "midpoint", (("local0", "local1"),), 2, None),
    "equal": ConstraintSpec(
        "equal", (("local0", "local1"),), 1, None),
    "offset": ConstraintSpec(
        "offset", (("local0", "local1"),), 1, None),
    "concentric": ConstraintSpec(
        "concentric", (("local0", "local1"),), 2, None),
    "diameter": ConstraintSpec(
        "diameter", (("local0", "length"),), 1, None),
    "radius": ConstraintSpec(
        "radius", (("local0", "length"),), 1, None),
    "angle": ConstraintSpec(
        "angle", (("local0", "local1", "aligned", "clockwise", "angle"),), 1, None),
}


def constraint_dof(name: str) -> int:
    """Nominal DOF removed by a constraint.

    Raises ``KeyError`` for an unknown type and ``ValueError`` for the variable
    (``mirror``) / external (``projected``) types whose removal is not a fixed
    constant.
    """
    if name not in CONSTRAINT_SPECS:
        raise KeyError(f"unknown constraint type '{name}'")
    dof = CONSTRAINT_SPECS[name].dof_removed
    if dof is None:
        raise ValueError(f"constraint '{name}' has variable/external DOF")
    return dof


def classify_edge(n_members: int) -> str:
    """Classify a constraint edge by its number of member nodes.

    ``1`` -> ``'loop'`` (a scale/orientation constraint on a single primitive),
    ``2`` -> ``'edge'``, ``>=3`` -> ``'hyperedge'`` (e.g. a mirror's axis member).
    """
    if n_members < 1:
        raise ValueError("a constraint edge must have at least one member")
    if n_members == 1:
        return "loop"
    if n_members == 2:
        return "edge"
    return "hyperedge"


def constraints_by_frequency() -> Tuple[ConstraintSpec, ...]:
    """The constraint types with a known frequency, ordered by descending frequency."""
    ranked = [s for s in CONSTRAINT_SPECS.values() if s.frequency_pct is not None]
    ranked.sort(key=lambda s: s.frequency_pct, reverse=True)  # type: ignore[arg-type]
    return tuple(ranked)


def primitives_by_frequency() -> Tuple[PrimitiveSpec, ...]:
    """The primitive types ordered by descending frequency."""
    ranked = list(PRIMITIVE_SPECS.values())
    ranked.sort(key=lambda s: s.frequency_pct, reverse=True)
    return tuple(ranked)
