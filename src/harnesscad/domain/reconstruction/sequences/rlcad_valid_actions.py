"""RLCAD valid-action generation (Algorithm 1) + revolve command-sequence validity.

RLCAD observes that "not all actions are valid, and invalid actions can interfere
with the agent's learning" (Sec. 5.3). Its **Algorithm 1** enumerates the valid
action set ``A_valid`` from the target face-adjacency graph so the RL action space
is exactly the feasible operations:

    1. partition faces into planar P and non-planar S
    2. for each group of *parallel* planar faces, every distinct pair yields a
       candidate ``ValidExtrude(p_i, p_j)``
    3. every revolve-eligible non-planar face yields ``ValidRevolve(s)``

Extrusion needs a pair of parallel, non-coplanar planes (start/end faces, Sec.4.1
+ Appendix A.2). Revolution needs a single surface of revolution -- cylinder,
cone, sphere or torus (Sec. 4.2, Appendix A.2); a plane or a free-form surface
from which "a stable rotation axis or sketch cannot be inferred" is deliberately
rejected.

This module is the deterministic feasibility core: it takes a lightweight
description of each face (surface type, and for planes an outward normal + signed
plane offset) and returns the valid extrude/revolve action tuples in the
:mod:`reconstruction.rlcad_command_spec` encoding. Boolean-op expansion is a
separate degree of freedom. Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from harnesscad.domain.reconstruction.tokens.rlcad_command_spec import (
    BOOLEAN_OPS, EXTRUDE, NEWBODY, REVOLVE, Action, ExtrudeCommand,
    RevolveCommand,
)

# Surface-of-revolution types the plugin can parse into an axis+profile
# (Sec. 4.2 / Appendix A.2). Anything else is rejected for revolution.
PLANE = "plane"
REVOLVABLE_TYPES: Tuple[str, ...] = ("cylinder", "cone", "sphere", "torus")

_EPS = 1e-9


@dataclass(frozen=True)
class FaceInfo:
    """Minimal per-face description drawn from the B-Rep face-adjacency graph.

    * ``surface_type`` -- ``"plane"`` or one of :data:`REVOLVABLE_TYPES` (others
      are simply non-revolvable).
    * ``normal`` -- outward unit-ish normal, required for planar faces (used to
      group parallel planes).
    * ``offset`` -- signed plane offset ``n . x`` used to reject coplanar pairs.
    """

    face_id: int
    surface_type: str
    normal: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset: float = 0.0

    @property
    def is_planar(self) -> bool:
        return self.surface_type == PLANE

    @property
    def is_revolvable(self) -> bool:
        return self.surface_type in REVOLVABLE_TYPES


def _unit(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    n = math.sqrt(sum(c * c for c in v))
    if n < _EPS:
        raise ValueError("planar face needs a non-zero normal")
    return (v[0] / n, v[1] / n, v[2] / n)


def _parallel(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> bool:
    """True if two normals are parallel (same or opposite direction)."""
    ua, ub = _unit(a), _unit(b)
    dot = sum(x * y for x, y in zip(ua, ub))
    return abs(abs(dot) - 1.0) < 1e-6


def partition_faces(faces: Sequence[FaceInfo]) -> Tuple[List[FaceInfo], List[FaceInfo]]:
    """Algorithm 1, line 1: split into planar ``P`` and non-planar ``S``."""
    planar = [f for f in faces if f.is_planar]
    nonplanar = [f for f in faces if not f.is_planar]
    return planar, nonplanar


def group_parallel_faces(planar: Sequence[FaceInfo]) -> List[List[FaceInfo]]:
    """Group planar faces by parallel normal (Algorithm 1's GroupParallelFaces).

    Groups are returned deterministically ordered by their first member's id.
    """
    groups: List[List[FaceInfo]] = []
    for f in sorted(planar, key=lambda x: x.face_id):
        placed = False
        for g in groups:
            if _parallel(g[0].normal, f.normal):
                g.append(f)
                placed = True
                break
        if not placed:
            groups.append([f])
    return groups


def valid_extrude(p_i: FaceInfo, p_j: FaceInfo) -> bool:
    """Feasibility of an extrusion between two planar faces.

    Requires parallel and *non-coplanar* planes (distinct offsets); the two
    faces define the start sketch and the parallel end face at a positive
    distance (Sec. 4.1).
    """
    if p_i.face_id == p_j.face_id:
        return False
    if not (p_i.is_planar and p_j.is_planar):
        return False
    if not _parallel(p_i.normal, p_j.normal):
        return False
    # Non-coplanar: distinct plane offsets (normalise offset by normal parity).
    ui, uj = _unit(p_i.normal), _unit(p_j.normal)
    dot = sum(x * y for x, y in zip(ui, uj))
    oj = p_j.offset if dot > 0 else -p_j.offset
    return abs(p_i.offset - oj) > _EPS


def valid_revolve(s: FaceInfo) -> bool:
    """Feasibility of a revolution on a single non-planar face (Sec. 4.2)."""
    return s.is_revolvable


def generate_valid_actions(faces: Sequence[FaceInfo],
                           boolean_ops: Sequence[str] = (NEWBODY,)) -> List[Action]:
    """Algorithm 1: enumerate ``A_valid`` as command-spec action tuples.

    For each requested boolean op, emits one action per feasible extrude pair
    and per revolve-eligible face. With the default single op the result is the
    pure geometric action set; pass all :data:`BOOLEAN_OPS` to expand the space.
    Deterministically ordered.
    """
    for op in boolean_ops:
        if op not in BOOLEAN_OPS:
            raise ValueError(f"unknown boolean op: {op!r}")

    planar, nonplanar = partition_faces(faces)
    actions: List[Action] = []

    for op in boolean_ops:
        # Extrude candidates from parallel planar groups.
        for group in group_parallel_faces(planar):
            ordered = sorted(group, key=lambda x: x.face_id)
            for i in range(len(ordered)):
                for j in range(len(ordered)):
                    if i == j:
                        continue
                    p_i, p_j = ordered[i], ordered[j]
                    if valid_extrude(p_i, p_j):
                        actions.append(
                            (p_i.face_id, p_j.face_id, op, EXTRUDE))
        # Revolve candidates from revolve-eligible non-planar faces.
        for s in sorted(nonplanar, key=lambda x: x.face_id):
            if valid_revolve(s):
                actions.append((s.face_id, s.face_id, op, REVOLVE))
    return actions


@dataclass
class ValidActionSet:
    """Convenience wrapper exposing the enumerated feasible commands."""

    faces: Sequence[FaceInfo]
    boolean_ops: Sequence[str] = (NEWBODY,)
    _actions: List[Action] = field(default_factory=list, init=False)

    def __post_init__(self):
        self._actions = generate_valid_actions(self.faces, self.boolean_ops)

    @property
    def actions(self) -> List[Action]:
        return list(self._actions)

    def __len__(self) -> int:
        return len(self._actions)

    def extrude_commands(self) -> List[ExtrudeCommand]:
        return [ExtrudeCommand(a[0], a[1], a[2])
                for a in self._actions if a[3] == EXTRUDE]

    def revolve_commands(self) -> List[RevolveCommand]:
        return [RevolveCommand(a[0], a[2])
                for a in self._actions if a[3] == REVOLVE]

    def action_space_size(self) -> int:
        """RLCAD defines the action-space dimension by the number of valid actions."""
        return len(self._actions)


def revolve_sequence_valid(commands: Sequence, faces: Sequence[FaceInfo]) -> bool:
    """Command-sequence validity focused on revolve ops (Sec. 4 / 5.3).

    Every revolve command must reference a revolve-eligible face present in the
    graph; every extrude must reference a feasible parallel, non-coplanar pair.
    """
    by_id: Dict[int, FaceInfo] = {f.face_id: f for f in faces}
    for c in commands:
        if isinstance(c, RevolveCommand):
            f = by_id.get(c.face)
            if f is None or not valid_revolve(f):
                return False
        elif isinstance(c, ExtrudeCommand):
            fi, fe = by_id.get(c.start_face), by_id.get(c.end_face)
            if fi is None or fe is None or not valid_extrude(fi, fe):
                return False
        else:
            return False
    return True
