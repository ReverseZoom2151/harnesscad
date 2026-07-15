"""Mechanism mobility, kinematic-tree validity, and mate-aware clash classification.

Deterministic mechanism analysis distilled from three assembly papers -- the piece
that lets the harness answer "does this assembly actually move, and is it well-posed?"
without any model call (the "scissors test" of Adler et al.).

* **AADvark** (Adler, Russo, Cafarella, 2026) and **ASSEMCAD** (Dong et al., 2026,
  axiom F-01) motivate the classical *Kutzbach-Gruebler* mobility criterion: for a
  mechanism of ``n`` links (including ground) joined by ``j`` joints, the number of
  independent degrees of freedom in spatial (3D) mechanisms is::

      M = 6 * (n - 1) - sum_i (6 - f_i)

  where ``f_i`` is the freedom of joint ``i``. This is what tells a static assembly
  (M = 0) apart from a functional mechanism (M >= 1, e.g. a pair of scissors: two
  bodies + one revolute joint => M = 1). A planar variant uses the factor 3.

* **ArtiCAD** (Shui et al., 2026), Sec. 3 -- restricts articulated assemblies to a
  *kinematic tree* ``T = (P, J, g)``: ``N`` parts joined by ``N - 1`` typed joints,
  rooted at a ground part, acyclic so "each part has a unique parent, so the solver
  always receives a well-posed problem." Their per-joint DOF table
  ``delta(tau)`` (Fixed=0, Revolute/Slider=1, Cylindrical=2, Ball=3) sums to the
  tree's total DOF. :func:`tree_dof` reproduces this; :func:`validate_kinematic_tree`
  enforces the acyclic, single-root, unique-parent structure.

* **ASSEMCAD** Sec. 4.4 -- verification checks *connectivity* (BFS from the root
  component) and *mate-aware clash classification* (their Eq. 3): geometric overlap is
  a clash **unless** the two parts are joined by a contact mate (gear_mesh, press_fit,
  thread_engage), where interference is physically intended.

Consumes the joint freedom / DOF-removed tables from
:mod:`harnesscad.domain.geometry.assembly.mates`. Stdlib only, fully deterministic.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

from harnesscad.domain.geometry.assembly.joint_taxonomy import (
    JOINT_DOF_REMOVED,
    SPATIAL_DOF,
    UnknownJointKindError,
    is_known_joint_kind,
    joint_freedom,
)
from harnesscad.domain.geometry.assembly.mates import (
    CONTACT_MATES,
    MATE_DOF_REMOVED,
    MATE_TYPES,
)

__all__ = [
    "JOINT_FREEDOM",
    "kutzbach_mobility",
    "tree_dof",
    "validate_kinematic_tree",
    "assembly_connectivity",
    "classify_clash",
    "MobilityReport",
    "mate_freedom",
]

# Per-joint kinematic freedom f_i (ArtiCAD delta(tau); ASSEMCAD kinematic axioms).
# Derived from the single authoritative :mod:`joint_taxonomy` so the full joint
# set (revolute/slider/ball/cylindrical/planar/gear/press-fit/thread/snap and the
# ASSEMCAD port contacts) is covered and can never drift from the DOF verifier.
JOINT_FREEDOM: Mapping[str, int] = {
    kind: SPATIAL_DOF - removed for kind, removed in JOINT_DOF_REMOVED.items()
}


def mate_freedom(mate_kind: str) -> int:
    """Freedom (6 - DOF removed) left by an ASSEMCAD mate type."""
    if mate_kind not in MATE_DOF_REMOVED:
        raise UnknownJointKindError(f"unknown mate type {mate_kind!r}")
    return SPATIAL_DOF - MATE_DOF_REMOVED[mate_kind]


def _freedom_of(joint_kind: str) -> int:
    """Freedom of any taxonomy joint / mate kind. Raises on an unknown kind."""
    return joint_freedom(joint_kind)


def kutzbach_mobility(n_links: int, joint_kinds: Sequence[str],
                      *, planar: bool = False) -> int:
    """Kutzbach-Gruebler mobility of a mechanism.

    ``n_links`` counts every rigid body *including* ground. ``joint_kinds`` names each
    joint (from :data:`JOINT_FREEDOM` or an ASSEMCAD mate type). ``planar`` selects the
    3-DOF planar formula instead of the 6-DOF spatial one.

    A closed four-bar linkage (4 links, 4 revolute joints) gives M = 1 in the planar
    form; two bodies + one revolute gives M = 1 in either form.
    """
    if n_links < 1:
        raise ValueError("a mechanism needs at least one link (ground)")
    dof_per_body = 3 if planar else 6
    total = dof_per_body * (n_links - 1)
    for jk in joint_kinds:
        f = _freedom_of(jk)
        constrained = dof_per_body - f
        if constrained < 0:
            constrained = 0
        total -= constrained
    return total


@dataclass(frozen=True)
class MobilityReport:
    """Result of :func:`validate_kinematic_tree`."""

    valid: bool
    total_dof: int
    root: Optional[str]
    errors: tuple[str, ...]
    reachable: tuple[str, ...]


def tree_dof(joint_kinds: Iterable[str]) -> int:
    """Total DOF of a kinematic *tree*: the plain sum of per-joint freedoms (ArtiCAD).

    Valid only for acyclic assemblies (one joint per parent edge); for general graphs
    with loops use :func:`kutzbach_mobility`.
    """
    return sum(_freedom_of(jk) for jk in joint_kinds)


def validate_kinematic_tree(
    parts: Sequence[str],
    joints: Sequence[tuple[str, str, str]],
    root: Optional[str] = None,
) -> MobilityReport:
    """Validate an articulated assembly as a rooted kinematic tree (ArtiCAD Sec. 3).

    ``joints`` are ``(parent, child, joint_kind)`` triples. Enforces: every endpoint is
    a known part; exactly ``N - 1`` joints for ``N`` parts; each non-root part has a
    unique parent; the structure is connected and acyclic (a tree). Returns a
    :class:`MobilityReport` whose ``total_dof`` is the summed joint freedom.
    """
    errors: list[str] = []
    part_set = set(parts)
    if len(part_set) != len(parts):
        errors.append("duplicate part names")

    for (parent, child, kind) in joints:
        if parent not in part_set:
            errors.append(f"joint references unknown parent {parent!r}")
        if child not in part_set:
            errors.append(f"joint references unknown child {child!r}")
        if not is_known_joint_kind(kind):
            errors.append(f"joint has unknown type {kind!r}")

    n = len(part_set)
    if n and len(joints) != n - 1:
        errors.append(
            f"a tree over {n} parts needs {n - 1} joints, got {len(joints)}"
        )

    # Unique-parent check.
    parent_of: dict[str, str] = {}
    for (parent, child, _kind) in joints:
        if child in parent_of:
            errors.append(f"part {child!r} has more than one parent")
        else:
            parent_of[child] = parent

    # Determine / validate the root.
    resolved_root = root
    if resolved_root is None:
        roots = [p for p in part_set if p not in parent_of]
        if len(roots) == 1:
            resolved_root = roots[0]
        elif len(roots) == 0 and n:
            errors.append("no root: every part has a parent (cycle present)")
        elif len(roots) > 1:
            errors.append(f"multiple roots: {sorted(roots)}")
    elif resolved_root not in part_set:
        errors.append(f"declared root {resolved_root!r} is not a part")
    elif resolved_root in parent_of:
        errors.append(f"declared root {resolved_root!r} has a parent")

    # Connectivity / acyclicity from the root over undirected joint edges.
    reachable: tuple[str, ...] = ()
    if resolved_root in part_set:
        adj: dict[str, list[str]] = {p: [] for p in part_set}
        for (parent, child, _k) in joints:
            if parent in adj and child in adj:
                adj[parent].append(child)
                adj[child].append(parent)
        seen = {resolved_root}
        order = [resolved_root]
        dq = deque([resolved_root])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    order.append(v)
                    dq.append(v)
        reachable = tuple(order)
        if len(seen) != n:
            errors.append(
                f"assembly is not connected: {n - len(seen)} part(s) unreachable "
                f"from root {resolved_root!r}"
            )

    dof = tree_dof(kind for (_p, _c, kind) in joints) if not errors else 0
    return MobilityReport(
        valid=not errors,
        total_dof=dof,
        root=resolved_root if resolved_root in part_set else None,
        errors=tuple(errors),
        reachable=reachable,
    )


def assembly_connectivity(
    parts: Sequence[str],
    mate_edges: Sequence[tuple[str, str]],
    root: Optional[str] = None,
) -> tuple[bool, tuple[str, ...]]:
    """BFS connectivity of an assembly graph (ASSEMCAD Sec. 4.4).

    Returns ``(connected, reachable_parts)`` where reachability is measured from
    ``root`` (or the first part if ``root`` is ``None``). Unlike the tree validator
    this accepts general graphs (loops allowed).
    """
    part_set = set(parts)
    if not part_set:
        return True, ()
    start = root if root is not None else parts[0]
    if start not in part_set:
        raise ValueError(f"root {start!r} is not a part")
    adj: dict[str, list[str]] = {p: [] for p in part_set}
    for (a, b) in mate_edges:
        if a in adj and b in adj:
            adj[a].append(b)
            adj[b].append(a)
    seen = {start}
    order = [start]
    dq = deque([start])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                order.append(v)
                dq.append(v)
    return len(seen) == len(part_set), tuple(order)


def classify_clash(
    part_a: str,
    part_b: str,
    overlap_volume: float,
    mates: Sequence[tuple[str, str, str]],
    *,
    tol: float = 1e-9,
) -> str:
    """Mate-aware clash classification (ASSEMCAD Eq. 3).

    ``mates`` are ``(part_i, part_j, mate_kind)`` triples. Returns one of
    ``"clear"`` (overlap below tolerance), ``"expected"`` (overlap explained by a
    contact mate between the pair), or ``"clash"`` (unexplained interference).
    """
    if overlap_volume <= tol:
        return "clear"
    pair = frozenset((part_a, part_b))
    for (pi, pj, kind) in mates:
        if frozenset((pi, pj)) == pair and kind in CONTACT_MATES:
            return "expected"
    return "clash"
