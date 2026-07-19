"""Extended CAD command set: sketch / extrude / **revolve**.

The mainstream CAD command vocabularies (see
:mod:`reconstruction.deepcad_command_spec`) are *extrude-only*. The contribution of
this command set is a vocabulary that also carries a **revolution** operation. This
module is the deterministic, network-agnostic representation of that extended DSL,
whose grammar is::

    M := G ; [X]
    X := E | R
    E := add_extrude(F, F, O)
    R := add_revolve(F, O)
    F := face ID
    O := newbody | intersection | union | subtraction

Each modeling command is executed against a current geometry ``G`` and combined
with the existing body through a Boolean operation ``O``. The extrude operation
``E`` takes a **start face and a distinct end face** (a pair of parallel,
non-coplanar planes); the revolve operation ``R`` takes a **single** revolve-
eligible face that the kernel geometrically parses into an axis, angle and
profile.

The command set also defines a flat *action* encoding for a reinforcement-learning
policy: ``a = (f_s, f_e, o_t, a_t)`` where ``o_t`` is the Boolean op, ``a_t`` the
action type (extrude/revolve), and -- crucially -- ``f_s != f_e`` for extrusion but
``f_s == f_e`` for revolution: the start and end faces differ for extrusion but
coincide for revolution. This module encodes/decodes that 4-tuple, checks its
structural validity, and serialises the DSL. The learned policy is out of scope.
Pure stdlib, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

# --- Boolean operation vocabulary O ----------------------------------------
NEWBODY = "newbody"
INTERSECTION = "intersection"
UNION = "union"
SUBTRACTION = "subtraction"

BOOLEAN_OPS: Tuple[str, ...] = (NEWBODY, INTERSECTION, UNION, SUBTRACTION)
BOOLEAN_INDEX = {name: i for i, name in enumerate(BOOLEAN_OPS)}

# --- action-type vocabulary a_t --------------------------------------------
EXTRUDE = "extrude"
REVOLVE = "revolve"

ACTION_TYPES: Tuple[str, ...] = (EXTRUDE, REVOLVE)
ACTION_TYPE_INDEX = {name: i for i, name in enumerate(ACTION_TYPES)}


def _check_op(op: str) -> str:
    if op not in BOOLEAN_INDEX:
        raise ValueError(f"unknown boolean op: {op!r}")
    return op


@dataclass(frozen=True)
class ExtrudeCommand:
    """``E := add_extrude(F, F, O)`` -- start face, distinct end face, boolean op."""

    start_face: int
    end_face: int
    op: str = NEWBODY

    def __post_init__(self):
        _check_op(self.op)
        if self.start_face == self.end_face:
            raise ValueError(
                "extrude requires distinct start/end faces (they differ for "
                "extrusion but coincide for revolution)")

    @property
    def action_type(self) -> str:
        return EXTRUDE

    def to_dsl(self) -> str:
        return f"add_extrude({self.start_face}, {self.end_face}, {self.op})"


@dataclass(frozen=True)
class RevolveCommand:
    """``R := add_revolve(F, O)`` -- one revolve-eligible face + a boolean op.

    The single face is geometrically parsed (by the kernel) into the rotation
    axis, angle and profile; in the flat action encoding this shows up as
    ``f_s == f_e``.
    """

    face: int
    op: str = NEWBODY

    def __post_init__(self):
        _check_op(self.op)

    @property
    def action_type(self) -> str:
        return REVOLVE

    def to_dsl(self) -> str:
        return f"add_revolve({self.face}, {self.op})"


Command = Union[ExtrudeCommand, RevolveCommand]

# --- flat 4-tuple action encoding a = (f_s, f_e, o_t, a_t) ------------------
Action = Tuple[int, int, str, str]


def encode_action(command: Command) -> Action:
    """Encode a command to the policy action 4-tuple ``(f_s, f_e, o_t, a_t)``."""
    if isinstance(command, ExtrudeCommand):
        return (command.start_face, command.end_face, command.op, EXTRUDE)
    if isinstance(command, RevolveCommand):
        # Revolve: f_s == f_e (the single face is both slots).
        return (command.face, command.face, command.op, REVOLVE)
    raise TypeError(f"not an RLCAD command: {command!r}")


def decode_action(action: Action) -> Command:
    """Inverse of :func:`encode_action`; validates the ``f_s``/``f_e`` invariant."""
    if not is_valid_action(action):
        raise ValueError(f"invalid action tuple: {action!r}")
    f_s, f_e, o_t, a_t = action
    if a_t == EXTRUDE:
        return ExtrudeCommand(f_s, f_e, o_t)
    return RevolveCommand(f_s, o_t)


def is_valid_action(action: Action) -> bool:
    """Structural validity of an action 4-tuple.

    * op must be one of the four boolean ops, action type one of the two;
    * extrude requires ``f_s != f_e``; revolve requires ``f_s == f_e``.
    """
    if len(action) != 4:
        return False
    f_s, f_e, o_t, a_t = action
    if o_t not in BOOLEAN_INDEX or a_t not in ACTION_TYPE_INDEX:
        return False
    if not isinstance(f_s, int) or not isinstance(f_e, int):
        return False
    if a_t == EXTRUDE:
        return f_s != f_e
    return f_s == f_e  # REVOLVE


def action_to_index(action: Action) -> Tuple[int, int, int, int]:
    """Integer-only view ``(f_s, f_e, op_idx, action_type_idx)`` for embeddings."""
    if not is_valid_action(action):
        raise ValueError(f"invalid action tuple: {action!r}")
    f_s, f_e, o_t, a_t = action
    return (f_s, f_e, BOOLEAN_INDEX[o_t], ACTION_TYPE_INDEX[a_t])


# --- full model M := G ; [X] -----------------------------------------------
@dataclass(frozen=True)
class ModelSequence:
    """A whole model ``M`` = initial geometry id ``G`` plus a list of ops ``[X]``."""

    initial_geometry: int
    commands: Tuple[Command, ...] = ()

    def to_dsl(self) -> str:
        head = f"G{self.initial_geometry}"
        if not self.commands:
            return head
        body = "; ".join(c.to_dsl() for c in self.commands)
        return f"{head}; {body}"

    def action_types(self) -> Tuple[str, ...]:
        return tuple(c.action_type for c in self.commands)

    def revolve_count(self) -> int:
        return sum(1 for c in self.commands if isinstance(c, RevolveCommand))


def validate_sequence(seq: ModelSequence) -> bool:
    """Every command encodes to a structurally valid action tuple."""
    return all(is_valid_action(encode_action(c)) for c in seq.commands)
