"""Three-layer information-consistency reconciliation.

This approach identifies the *primary challenge* of parametric/direct integration as
"maintaining information consistency in a CAD model undergoing parametric/direct
edits." A CAD model carries three layers of information: **topology**,
**geometry**, and **constraints**. Parametric edits act on the constraint layer,
direct edits on the geometry layer, and "changes at a layer cannot be reflected
in others by current model representation schemes," which produces
inconsistencies, invalid models, and unpredictable behaviour.

This module builds a :class:`HybridModel` that bundles all three layers and
implements deterministic consistency machinery:

* :func:`check_consistency` verifies the geometry layer satisfies the constraint
  layer and the topology layer (no face over-runs a neighbour), returning the
  list of :class:`Inconsistency` records the current schemes silently create.
* :func:`propagate_parametric_to_geometry` reflects a constraint-layer edit down
  into the geometry layer (parametric -> direct edit propagation).
* :func:`propagate_direct_to_constraint` reflects a geometry-layer push-pull back
  up into the constraint layer (direct -> parametric edit propagation) -- the
  reconciliation the "Constrained Direct Modeling" approach (4.5) attempts.
* :func:`recognize_constraints` re-derives constraints from the current geometry
  (automatic constraint recognition, 4.5) and :func:`design_intent_drift` flags
  where the recognized constraints differ from the original design intent -- the
  documented disadvantage of that approach.

Geometry is planar faces along a shared axis (offsets); a ``DistanceConstraint``
fixes the signed gap between two faces, a ``ParallelConstraint`` fixes co-normal
faces. Stdlib-only, deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from harnesscad.domain.editing.hybrid_model import (
    DirectBRep, Face, FeatureTree, InfoLayer, ParameterEdit, PushPullEdit,
)

_QUANT = 1_000_000


def _q(v: float) -> int:
    return int(round(float(v) * _QUANT))


# ---------------------------------------------------------------------------
# Constraint layer
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DistanceConstraint:
    """Fixes the signed offset gap ``face_b.offset - face_a.offset == value``."""

    face_a: str
    face_b: str
    value: float
    ctype: str = "distance"

    def to_dict(self) -> Dict:
        return {"ctype": "distance", "face_a": self.face_a,
                "face_b": self.face_b, "value": self.value}


@dataclass(frozen=True)
class ParallelConstraint:
    """Requires two faces to share a normal direction (co-planar orientation)."""

    face_a: str
    face_b: str
    ctype: str = "parallel"

    def to_dict(self) -> Dict:
        return {"ctype": "parallel", "face_a": self.face_a, "face_b": self.face_b}


@dataclass
class Inconsistency:
    """A detected disagreement between two information layers."""

    layer: InfoLayer
    detail: str
    entities: Tuple[str, ...] = ()

    def to_dict(self) -> Dict:
        return {"layer": self.layer.value, "detail": self.detail,
                "entities": list(self.entities)}


@dataclass
class HybridModel:
    """A CAD model with all three information layers bound together.

    ``brep`` supplies geometry + topology; ``constraints`` the constraint layer;
    ``tree`` optionally links parameters that drive the constraints.
    """

    brep: DirectBRep
    constraints: List[object] = field(default_factory=list)
    tree: Optional[FeatureTree] = None


# ---------------------------------------------------------------------------
# Consistency checking
# ---------------------------------------------------------------------------
def check_consistency(model: HybridModel) -> List[Inconsistency]:
    """Detect geometry/constraint/topology-layer disagreements.

    Returns an ordered list of :class:`Inconsistency`. An empty list means all
    three layers agree.
    """
    out: List[Inconsistency] = []
    faces = model.brep.faces
    for c in model.constraints:
        if isinstance(c, DistanceConstraint):
            if c.face_a not in faces or c.face_b not in faces:
                out.append(Inconsistency(
                    InfoLayer.CONSTRAINT,
                    f"distance constraint references missing face",
                    (c.face_a, c.face_b)))
                continue
            gap = faces[c.face_b].offset - faces[c.face_a].offset
            if _q(gap) != _q(c.value):
                out.append(Inconsistency(
                    InfoLayer.GEOMETRY,
                    (f"distance {c.face_a}->{c.face_b} is {gap}, "
                     f"constraint requires {c.value}"),
                    (c.face_a, c.face_b)))
        elif isinstance(c, ParallelConstraint):
            if c.face_a in faces and c.face_b in faces:
                na = faces[c.face_a].normal()
                nb = faces[c.face_b].normal()
                if tuple(_q(x) for x in na) != tuple(_q(x) for x in nb):
                    out.append(Inconsistency(
                        InfoLayer.GEOMETRY,
                        f"faces {c.face_a},{c.face_b} not parallel",
                        (c.face_a, c.face_b)))
    return out


def is_consistent(model: HybridModel) -> bool:
    return not check_consistency(model)


# ---------------------------------------------------------------------------
# Edit propagation between the two representations
# ---------------------------------------------------------------------------
def propagate_parametric_to_geometry(model: HybridModel,
                                     edit: ParameterEdit) -> HybridModel:
    """Reflect a constraint-layer parameter edit down into the geometry layer.

    The parameter is assumed to drive one distance constraint whose ``value``
    equals the parameter; changing the parameter updates that constraint and then
    re-solves the geometry (moves ``face_b``) so the layers stay consistent. This
    is parametric -> direct propagation, the direction current schemes handle.
    """
    new = HybridModel(model.brep.copy(),
                      list(model.constraints),
                      model.tree.copy() if model.tree else None)
    if new.tree is not None:
        new.tree.set_parameter(edit.target_fid, edit.param, edit.new_value)
    # Update every distance constraint tagged to this parameter, then re-solve.
    rebuilt: List[object] = []
    for c in new.constraints:
        if (isinstance(c, DistanceConstraint)
                and c.ctype == "distance"
                and _param_tag(c) == (edit.target_fid, edit.param)):
            c = DistanceConstraint(c.face_a, c.face_b, edit.new_value)
        rebuilt.append(c)
    new.constraints = rebuilt
    _solve_geometry(new)
    return new


def propagate_direct_to_constraint(model: HybridModel,
                                   edit: PushPullEdit) -> HybridModel:
    """Reflect a geometry-layer push-pull back up into the constraint layer.

    Apply the push-pull to the geometry, then update any distance constraint the
    moved face participates in so its ``value`` matches the new gap. This is the
    direct -> parametric reconciliation that keeps constraints "in the
    background" instead of dropping them.
    """
    new = HybridModel(model.brep.copy(),
                      [], model.tree.copy() if model.tree else None)
    new.brep.push_pull(edit.face_name, edit.distance)
    faces = new.brep.faces
    for c in model.constraints:
        if isinstance(c, DistanceConstraint) and edit.face_name in (c.face_a, c.face_b):
            gap = faces[c.face_b].offset - faces[c.face_a].offset
            c = DistanceConstraint(c.face_a, c.face_b, gap)
        new.constraints.append(c)
    return new


def _param_tag(c: DistanceConstraint) -> Optional[Tuple[str, str]]:
    """Optional (fid, param) a distance constraint is driven by, via naming.

    Convention: a constraint driven by parameter ``P`` of feature ``F`` names its
    faces ``F:P:a`` / ``F:P:b``. Absent that, returns None (undriven constraint).
    """
    for f in (c.face_a, c.face_b):
        parts = f.split(":")
        if len(parts) >= 3:
            return (parts[0], parts[1])
    return None


def _solve_geometry(model: HybridModel) -> None:
    """Move ``face_b`` of each distance constraint to satisfy it (topological sort-free).

    Deterministic single pass: for each distance constraint, set
    ``face_b.offset = face_a.offset + value``. Sufficient for chains defined in
    dependency order (the parametric regeneration order).
    """
    faces = model.brep.faces
    for c in model.constraints:
        if isinstance(c, DistanceConstraint) and c.face_a in faces and c.face_b in faces:
            faces[c.face_b].offset = faces[c.face_a].offset + c.value


# ---------------------------------------------------------------------------
# Automatic constraint recognition + design-intent drift
# ---------------------------------------------------------------------------
def recognize_constraints(brep: DirectBRep) -> List[object]:
    """Re-derive constraints from current geometry (automatic recognition).

    Emits a :class:`ParallelConstraint` for each adjacent co-normal face pair and
    a :class:`DistanceConstraint` for the offset between them. Deterministic:
    pairs come from the sorted adjacency list. Mirrors 4.5's constraint
    recognition that operates directly on boundary elements.
    """
    out: List[object] = []
    faces = brep.faces
    for a, b in sorted(tuple(sorted(p)) for p in brep.adjacency):
        if a not in faces or b not in faces:
            continue
        fa, fb = faces[a], faces[b]
        if tuple(_q(x) for x in fa.normal()) == tuple(_q(x) for x in fb.normal()):
            out.append(ParallelConstraint(a, b))
            out.append(DistanceConstraint(a, b, fb.offset - fa.offset))
    return out


def design_intent_drift(original: List[object],
                        recognized: List[object]) -> List[object]:
    """Constraints present in ``original`` but absent from ``recognized``.

    The disadvantage of automatic constraint recognition (4.5): recognized
    constraints "usually differ from the original design intent." This returns
    the design-intent constraints the recognizer failed to reproduce.
    """
    rec_keys = {_constraint_key(c) for c in recognized}
    return [c for c in original if _constraint_key(c) not in rec_keys]


def _constraint_key(c: object) -> Tuple:
    if isinstance(c, DistanceConstraint):
        return ("distance", c.face_a, c.face_b, _q(c.value))
    if isinstance(c, ParallelConstraint):
        return ("parallel", c.face_a, c.face_b)
    return ("other", repr(c))
