"""Sequence-validity grammar for CADParser command workflows.

The paper notes CAD command sequences are "under such strict constraints" that a
free decoder frequently emits invalid programs (their Recall = failure ratio).
This module makes those constraints explicit as a deterministic finite-state
acceptor over the :mod:`reconstruction.cadparser_schema` vocabulary, so a
candidate sequence can be validated (and step-wise legal continuations enumerated
for constrained decoding / masking) without any learned model.

Unlike :mod:`grammar_fsa` (DeepCAD's sketch+extrusion-only token stream), this
grammar covers CADParser's full operation set: arcs, revolution with its explicit
axis, extrusion/revolution *cut* variants, and the fillet/chamfer edge features.

Rules enforced (derived from Table 3 and the construction workflow semantics):

  * A workflow is ``<SOS>`` ... ``<EOS>`` then zero or more ``<PAD>``.
  * A solid must be created before it can be cut or edited: the first shaping
    operation may not be a cut (``Ec``/``Rc``) nor an edge feature (``F``/``Cf``).
  * Sketch curves (``L``/``A``/``C``) accumulate a profile; a profile must be
    non-empty before an extrusion or revolution consumes it.
  * A revolution/revolution-cut (``R``/``Rc``) must be immediately preceded by its
    axis ``Ax`` (Table 3 lists ``Ax`` as a distinct command feeding ``R``).
  * ``Ax`` is only meaningful directly before ``R``/``Rc``.
  * Fillet/chamfer (``F``/``Cf``) act on an existing solid, not on an open profile.
"""

from __future__ import annotations

from enum import Enum

from harnesscad.domain.reconstruction.cadparser_schema import PAD, SOS, EOS


CURVES = frozenset({"L", "A", "C"})
SOLID_MAKERS = frozenset({"E", "R"})          # create/extend a solid
CUTS = frozenset({"Ec", "Rc"})                # subtract from an existing solid
REVOLVES = frozenset({"R", "Rc"})             # need an axis first
EDGE_FEATURES = frozenset({"F", "Cf"})        # act on an existing solid


class State(str, Enum):
    START = "start"       # before <SOS>
    EMPTY = "empty"       # solid stream open, no profile, no solid yet
    PROFILE = "profile"   # a sketch profile is being drawn
    SOLID = "solid"       # at least one solid exists, no open profile
    SOLID_PROFILE = "solid_profile"  # a solid exists AND a new profile is open
    AXIS = "axis"         # an axis was just declared (solid exists)
    AXIS_EMPTY = "axis_empty"  # an axis was just declared (no solid yet)
    PAD = "pad"           # after <EOS>, only <PAD> allowed
    DEAD = "dead"


def _has_solid(state: State) -> bool:
    return state in (State.SOLID, State.SOLID_PROFILE, State.AXIS)


def allowed(state: State) -> frozenset[str]:
    """Legal next tokens from ``state`` (for decode-time masking)."""
    if state is State.START:
        return frozenset({SOS})
    if state is State.EMPTY:
        # open a profile, or declare an axis (revolution path); may not cut/edit.
        return CURVES | {"Ax"}
    if state is State.PROFILE:
        # extrude the profile, or declare an axis to revolve it; R/Rc need the axis
        return CURVES | {"E", "Ax"}
    if state is State.SOLID:
        return CURVES | EDGE_FEATURES | {"Ax", EOS}
    if state is State.SOLID_PROFILE:
        # extrude or extrude-cut the profile, or declare an axis to revolve it
        return CURVES | {"E", "Ec", "Ax"}
    if state is State.AXIS:
        return REVOLVES
    if state is State.AXIS_EMPTY:
        return frozenset({"R"})  # no solid yet -> only additive revolution
    if state is State.PAD:
        return frozenset({PAD})
    return frozenset()


def transition(state: State, token: str) -> State:
    """Advance the acceptor; an illegal token drives it to :attr:`State.DEAD`."""
    if token not in allowed(state):
        return State.DEAD
    if state is State.START:
        return State.EMPTY
    if token == EOS:
        return State.PAD
    if token == PAD:
        return State.PAD
    if token in CURVES:
        return State.SOLID_PROFILE if _has_solid(state) else State.PROFILE
    if token == "Ax":
        return State.AXIS if _has_solid(state) else State.AXIS_EMPTY
    if token in SOLID_MAKERS or token in CUTS:
        return State.SOLID
    if token in EDGE_FEATURES:
        return State.SOLID
    return State.DEAD


def run(tokens: list[str]) -> tuple[State, tuple[str, ...]]:
    """Validate a token stream; return ``(final_state, issues)``.

    ``issues`` is empty iff the stream is accepted (ends in ``PAD`` or ``SOLID``
    having produced ``<EOS>``). The first illegal step is reported as
    ``illegal:<index>:<token>``.
    """
    state = State.START
    for index, token in enumerate(tokens):
        state = transition(state, token)
        if state is State.DEAD:
            return state, (f"illegal:{index}:{token}",)
    issues: tuple[str, ...] = ()
    if state not in (State.PAD, State.SOLID):
        issues = (f"unterminated:{state.value}",)
    return state, issues


def is_valid(tokens: list[str]) -> bool:
    """True iff ``tokens`` is an accepted CADParser command sequence."""
    return not run(tokens)[1]
