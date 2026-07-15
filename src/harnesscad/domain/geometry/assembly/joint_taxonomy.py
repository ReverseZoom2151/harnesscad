"""The single authoritative joint / mate taxonomy for the assembly stack.

Historically three tables described "how many degrees of freedom does this joint
remove" and they had drifted apart:

* ``eval.verifiers.assembly.MATE_DOF`` validated a CISP ``Mate`` against only
  five mechanical joints (rigid/revolute/slider/cylindrical/planar).
* ``domain.geometry.assembly.mobility.JOINT_FREEDOM`` knew nine kinematic joints
  (adding ball/spherical/free/prismatic).
* ``domain.geometry.assembly.mates.MATE_DOF_REMOVED`` knew seven ASSEMCAD typed
  mates (adding gear/press-fit/thread/snap and the port-frame contacts).

The consequence was a silent hole: ``Mate(kind="ball")`` was a legal op that the
DOF verifier treated as *unknown* and simply excluded from the mobility count --
a check with no rule, silently passed. This module removes that hole by being the
*one* place the mapping lives. Every consumer (the verifier, ``mates`` and
``mobility``) imports from here, so the three tables can never drift again.

The rule is uniform: a kind either has a DOF rule (and the verifier / mobility
analysis applies it) or it is **rejected** with a typed
:class:`UnknownJointKindError` -- the same "refuse, don't silently pass"
discipline the rest of the harness enforces. Stdlib only, fully deterministic.

Model
-----
Two rigid bodies have six relative degrees of freedom (3 translation + 3
rotation). Each joint / mate *removes* some of them; :data:`SPATIAL_DOF` is that
count of six. A joint's *freedom* is ``SPATIAL_DOF - removed``. Canonical kinds and
their DOF-removed values:

======================  =======  =======  =========================================
kind                    removed  freedom  meaning
======================  =======  =======  =========================================
fixed                   6        0        welded / bonded -- no relative motion
revolute                5        1        hinge -- one rotation
prismatic               5        1        slider -- one translation
cylindrical             4        2        rotate *and* slide about one axis
planar                  3        3        slide in a plane + spin about its normal
ball                    3        3        spherical -- three rotations
free                    0        6        unconstrained
gear                    5        1        meshed pair -- one coupled rotation
press_fit               6        0        interference fit -- fully constrained
thread                  5        1        helical / screw pair -- one coupled screw
snap                    6        0        snap seat -- fully constrained
face_to_face            3        3        coincident plane (ASSEMCAD port contact)
coaxial                 4        2        collinear axes (ASSEMCAD port contact)
coaxial_face            5        1        coaxial + axial seat -> revolute
======================  =======  =======  =========================================
"""

from __future__ import annotations

from typing import Mapping, Optional

__all__ = [
    "SPATIAL_DOF",
    "UnknownJointKindError",
    "CANONICAL_DOF_REMOVED",
    "JOINT_ALIASES",
    "JOINT_DOF_REMOVED",
    "CANONICAL_JOINT_KINDS",
    "ALL_JOINT_KINDS",
    "normalize_kind",
    "is_known_joint_kind",
    "canonical_kind",
    "joint_dof_removed",
    "joint_freedom",
    "dof_removed_or_none",
    "freedom_or_none",
]

#: Relative degrees of freedom between two free rigid bodies.
SPATIAL_DOF: int = 6


class UnknownJointKindError(ValueError):
    """Raised when a joint / mate kind has no DOF rule in the taxonomy.

    Subclasses :class:`ValueError` so existing ``except ValueError`` /
    ``assertRaises(ValueError)`` call sites keep working while callers that want
    to be specific can catch this type.
    """


#: Canonical joint / mate kind -> DOF *removed* from the six relative DOF.
CANONICAL_DOF_REMOVED: Mapping[str, int] = {
    # --- classical kinematic joints -------------------------------------- #
    "fixed": 6,
    "revolute": 5,
    "prismatic": 5,
    "cylindrical": 4,
    "planar": 3,
    "ball": 3,
    "free": 0,
    # --- functional / manufactured mates --------------------------------- #
    "gear": 5,
    "press_fit": 6,
    "thread": 5,
    "snap": 6,
    # --- ASSEMCAD port-frame contact mates ------------------------------- #
    "face_to_face": 3,
    "coaxial": 4,
    "coaxial_face": 5,
}

#: Accepted spelling / synonym -> canonical kind. Every alias resolves to a key
#: of :data:`CANONICAL_DOF_REMOVED`.
JOINT_ALIASES: Mapping[str, str] = {
    "rigid": "fixed",
    "weld": "fixed",
    "bonded": "fixed",
    "hinge": "revolute",
    "slider": "prismatic",
    "spherical": "ball",
    "gear_mesh": "gear",
    "thread_engage": "thread",
    "screw": "thread",
    "helical": "thread",
    "snap_to_face": "snap",
}


def _build_flat() -> Mapping[str, int]:
    flat: dict[str, int] = dict(CANONICAL_DOF_REMOVED)
    for alias, canon in JOINT_ALIASES.items():
        if canon not in CANONICAL_DOF_REMOVED:  # pragma: no cover - guards typos
            raise AssertionError(
                f"alias {alias!r} points at unknown canonical {canon!r}")
        flat[alias] = CANONICAL_DOF_REMOVED[canon]
    return flat


#: Flat lookup: every accepted spelling (canonical *and* alias) -> DOF removed.
#: All keys are lower-case; look up via :func:`joint_dof_removed` to get case /
#: whitespace normalisation and rejection of unknown kinds.
JOINT_DOF_REMOVED: Mapping[str, int] = _build_flat()

#: The canonical kind names, in declaration order.
CANONICAL_JOINT_KINDS: tuple[str, ...] = tuple(CANONICAL_DOF_REMOVED)

#: Every accepted spelling (canonical + alias).
ALL_JOINT_KINDS: frozenset[str] = frozenset(JOINT_DOF_REMOVED)


def _clean(kind: str) -> str:
    return str(kind).strip().lower()


def normalize_kind(kind: str) -> str:
    """Return the canonical spelling of ``kind`` (lower/stripped, alias-resolved).

    Raises :class:`UnknownJointKindError` when ``kind`` has no DOF rule.
    """
    k = _clean(kind)
    if k in CANONICAL_DOF_REMOVED:
        return k
    if k in JOINT_ALIASES:
        return JOINT_ALIASES[k]
    raise UnknownJointKindError(
        f"unknown joint/mate kind {kind!r}; known kinds: "
        f"{', '.join(sorted(ALL_JOINT_KINDS))}")


# Backwards-compatible name used by mates.py's port-mate validation.
canonical_kind = normalize_kind


def is_known_joint_kind(kind: str) -> bool:
    """True when ``kind`` (any accepted spelling) has a DOF rule."""
    return _clean(kind) in JOINT_DOF_REMOVED


def joint_dof_removed(kind: str) -> int:
    """DOF removed by ``kind`` (0..6). Raises :class:`UnknownJointKindError`."""
    return CANONICAL_DOF_REMOVED[normalize_kind(kind)]


def joint_freedom(kind: str) -> int:
    """Freedom (``SPATIAL_DOF - removed``) of ``kind``. Raises on unknown."""
    return SPATIAL_DOF - joint_dof_removed(kind)


def dof_removed_or_none(kind: str) -> Optional[int]:
    """DOF removed by ``kind``, or ``None`` when the kind is unknown.

    The lenient companion to :func:`joint_dof_removed` for call sites that must
    degrade gracefully (e.g. a verifier that reports a diagnostic instead of
    raising, or a backend that refuses an op) rather than raise.
    """
    k = _clean(kind)
    return JOINT_DOF_REMOVED.get(k)


def freedom_or_none(kind: str) -> Optional[int]:
    """Freedom of ``kind``, or ``None`` when the kind is unknown (lenient)."""
    removed = dof_removed_or_none(kind)
    return None if removed is None else SPATIAL_DOF - removed
