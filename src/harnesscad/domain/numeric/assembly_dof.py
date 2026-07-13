"""CadQuery assembly constraint algebra and 6-DOF well-posedness analysis.

CadQuery's ``Assembly`` (``cadquery/assembly.py`` + ``occ_impl/solver.py``)
positions rigid parts in space by solving a set of geometric *constraints*.  The
numerics are delegated to SciPy/CasADi, but the *algebra* is fully
deterministic: each constraint kind has a fixed arity, marker types, parameter
type, and -- crucially for design intent -- a fixed number of relative degrees
of freedom it removes.  This module builds that algebra plus a Grubler-style
mobility analysis that classifies an assembly as under-, well-, or
over-constrained without ever running the solver.

This is distinct from the harness's existing DOF work, which is all *2D sketch*
DOF over point/line/circle primitives (:mod:`reconstruction.sgraphs2_dof_mask`,
:mod:`numeric.opencad_constraint_jacobian`, :mod:`constraints`).  Assembly DOF is
*3D rigid-body*: every part is a free body with 6 DOF (3 translation + 3
rotation), and constraints couple whole parts.

Constraint kinds mirror ``solver.py`` exactly:

* Unary (act on one part): ``Fixed`` (6), ``FixedPoint`` (3), ``FixedAxis`` (2),
  ``FixedRotation`` (3).
* Binary (couple two parts): ``Point`` (3), ``Axis`` (2), ``PointInPlane`` (1),
  ``PointOnLine`` (2), ``Plane`` = ``Axis`` + ``Point`` (5).

The DOF-removed figures are the standard mechanical-assembly values for these
mates and match the residual structure of the reference cost functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "UNARY_KINDS",
    "BINARY_KINDS",
    "DOF_REMOVED",
    "ConstraintError",
    "AssemblyConstraint",
    "AssemblyDOF",
    "DOFReport",
]

# arity classification straight from solver.py's Literal sets
UNARY_KINDS = frozenset({"Fixed", "FixedPoint", "FixedAxis", "FixedRotation"})
BINARY_KINDS = frozenset({"Plane", "Point", "Axis", "PointInPlane", "PointOnLine"})

# relative DOF each constraint removes (out of the 6 DOF of a rigid body)
DOF_REMOVED: Dict[str, int] = {
    # unary
    "Fixed": 6,
    "FixedPoint": 3,
    "FixedAxis": 2,
    "FixedRotation": 3,
    # binary
    "Point": 3,
    "Axis": 2,
    "PointInPlane": 1,
    "PointOnLine": 2,
    "Plane": 5,  # Axis (2) + Point (3)
}

# parameter presence: which kinds accept a scalar/tuple param (from ConstraintInvariants)
_PARAM_KINDS = {
    "Point": "scalar",
    "Axis": "scalar",
    "PointInPlane": "scalar",
    "PointOnLine": "scalar",
    "Plane": "scalar",
    "FixedPoint": "triple",
    "FixedAxis": "triple",
    "FixedRotation": "triple",
    "Fixed": "none",
}

DOF_PER_PART = 6


class ConstraintError(ValueError):
    """Raised for an invalid constraint (unknown kind, wrong arity, bad part)."""


@dataclass(frozen=True)
class AssemblyConstraint:
    """A single assembly constraint between one or two named parts."""

    kind: str
    parts: Tuple[str, ...]
    param: Optional[object] = None

    @property
    def arity(self) -> int:
        return len(self.parts)

    @property
    def dof_removed(self) -> int:
        return DOF_REMOVED[self.kind]

    def signature(self) -> Tuple:
        """Identity used for redundant-constraint detection (ignores param value)."""
        return (self.kind, frozenset(self.parts))


@dataclass
class DOFReport:
    """Result of :meth:`AssemblyDOF.analyze`."""

    n_parts: int
    total_dof: int
    removed: int
    mobility: int          # 6*n_parts - removed
    status: str            # "well" | "under" | "over" | "empty"
    grounded: bool         # at least one unary anchor present
    per_part_removed: Dict[str, int]
    redundant: List[Tuple[str, Tuple[str, ...]]]
    notes: List[str]

    def is_well_constrained(self) -> bool:
        return self.status == "well"


class AssemblyDOF(object):
    """Builds an assembly constraint graph and analyses its degrees of freedom."""

    def __init__(self) -> None:
        self._parts: List[str] = []
        self._part_set: set = set()
        self._constraints: List[AssemblyConstraint] = []

    # ---- construction --------------------------------------------------
    def add_part(self, name: str) -> "AssemblyDOF":
        if name in self._part_set:
            raise ConstraintError(f"duplicate part {name!r}")
        self._parts.append(name)
        self._part_set.add(name)
        return self

    def constrain(
        self, kind: str, parts: Sequence[str], param: object = None
    ) -> "AssemblyDOF":
        """Add a validated constraint. ``parts`` is 1 name (unary) or 2 (binary)."""
        if kind not in DOF_REMOVED:
            raise ConstraintError(f"unknown constraint kind {kind!r}")
        parts = tuple(parts)
        expected = 1 if kind in UNARY_KINDS else 2
        if len(parts) != expected:
            raise ConstraintError(
                f"{kind} requires {expected} part(s), got {len(parts)}"
            )
        for p in parts:
            if p not in self._part_set:
                raise ConstraintError(f"constraint references unknown part {p!r}")
        self._constraints.append(AssemblyConstraint(kind, parts, param))
        return self

    # ---- analysis ------------------------------------------------------
    def analyze(self) -> DOFReport:
        n = len(self._parts)
        total_dof = DOF_PER_PART * n
        notes: List[str] = []

        # per-part removed accounting + redundant detection
        per_part: Dict[str, int] = {p: 0 for p in self._parts}
        seen: Dict[Tuple, AssemblyConstraint] = {}
        redundant: List[Tuple[str, Tuple[str, ...]]] = []
        removed = 0
        grounded = False

        for c in self._constraints:
            sig = c.signature()
            if sig in seen:
                redundant.append((c.kind, c.parts))
                # a duplicate contributes redundancy, not fresh DOF removal
                continue
            seen[sig] = c
            removed += c.dof_removed
            if c.kind in UNARY_KINDS:
                grounded = True
            # attribute removal to the involved parts (split for binary)
            share = c.dof_removed / max(1, len(c.parts))
            for p in c.parts:
                per_part[p] += share

        # cap over-removal at the part level for a local over-constraint note
        for p, r in per_part.items():
            if r > DOF_PER_PART + 1e-9:
                notes.append(
                    f"part {p!r} has {r:.0f} DOF removed (> {DOF_PER_PART}); "
                    "locally over-constrained"
                )

        mobility = total_dof - removed

        if n == 0:
            status = "empty"
        elif mobility > 0:
            status = "under"
        elif mobility == 0:
            status = "well"
        else:
            status = "over"

        if redundant:
            notes.append(
                f"{len(redundant)} redundant (duplicate) constraint(s) detected"
            )
            if status == "well":
                # duplicates over a well-posed core still make it over-specified
                status = "over"

        if not grounded and n > 0 and status in ("well", "over"):
            notes.append(
                "no unary anchor (Fixed/FixedPoint/...): the whole assembly "
                "retains 6 rigid-body DOF and floats freely"
            )

        return DOFReport(
            n_parts=n,
            total_dof=total_dof,
            removed=removed,
            mobility=mobility,
            status=status,
            grounded=grounded,
            per_part_removed=per_part,
            redundant=redundant,
            notes=notes,
        )

    @property
    def constraints(self) -> List[AssemblyConstraint]:
        return list(self._constraints)

    @property
    def parts(self) -> List[str]:
        return list(self._parts)
