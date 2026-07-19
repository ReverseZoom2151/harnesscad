"""Refinement round/turn state machine for sketch refinement.

Implements the deterministic transition function and rollout structure:

  * ``apply_action(D, a)`` -- the typed transition for a single edit, with the
     shared-control-point semantics ("If multiple curves share the same
    control point, moving the point would modify all the curves, and deleting
    the point would delete all the curves").
  * ``apply_actions(D, A)`` -- ``A(D) = (a_n o ... o a_1)(D)``.
  * :class:`Round` -- a tuple ``r_i = (D_i, m_i, A_i, D'_i)`` where ``D'_i = A_i(D_i)``.
  * :class:`Rollout` / :class:`RefinementSession` -- a sequence of rounds with the
    invariant ``D_1 = {}`` and ``D_i = D'_{i-1}``, plus a win check
    ``Delta(D'_n, D*) < theta``.

Pure stdlib, deterministic; no rendering and no learned maker/designer model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

from harnesscad.domain.editing.sketch_edit_schema import (
    Curve,
    DeletePoint,
    Design,
    MakeCurve,
    Message,
    MoveCurve,
    MovePoint,
    RemoveCurve,
)

Action = object  # one of the typed edit dataclasses from mrcad_schema


def apply_action(design: Design, action: Action) -> Design:
    """Apply one typed edit and return the resulting design (immutable)."""
    if isinstance(action, MakeCurve):
        return design.add(action.curve)
    if isinstance(action, RemoveCurve):
        return design.remove(action.curve)
    if isinstance(action, MoveCurve):
        if action.curve not in design.curves:
            return design
        moved = action.curve.translate(action.vector)
        return Design(
            tuple(moved if c == action.curve else c for c in design.curves)
        )
    if isinstance(action, MovePoint):
        # Shared-point semantics: every curve touching `old` is updated.
        return Design(
            tuple(c.replace_point(action.old, action.new) for c in design.curves)
        )
    if isinstance(action, DeletePoint):
        p = (float(action.point[0]), float(action.point[1]))
        return Design(tuple(c for c in design.curves if not c.has_point(p)))
    raise TypeError(f"unknown action type: {type(action).__name__}")


def apply_actions(design: Design, actions: Sequence[Action]) -> Design:
    """Compose a sequence of actions left-to-right: ``(a_n o ... o a_1)(D)``."""
    d = design
    for a in actions:
        d = apply_action(d, a)
    return d


@dataclass(frozen=True)
class Round:
    """A single round ``r_i = (D_i, m_i, A_i, D'_i)``."""

    index: int
    design: Design           # D_i (state before the maker acts)
    message: Message         # m_i (designer's multimodal instruction)
    actions: Tuple[Action, ...]  # A_i
    result: Design           # D'_i = A_i(D_i)

    @property
    def is_generation(self) -> bool:
        """Round 1 is the generation round; rounds 2+ are refinement."""
        return self.index == 1


@dataclass
class Rollout:
    """A sequence of rounds ``R = [r_1, ..., r_n]`` with ``D_1 = {}``."""

    rounds: Tuple[Round, ...] = ()

    def __len__(self) -> int:
        return len(self.rounds)

    def __iter__(self):
        return iter(self.rounds)

    @property
    def final_design(self) -> Design:
        return self.rounds[-1].result if self.rounds else Design.empty()

    def designs(self) -> Tuple[Design, ...]:
        """The trajectory of resulting designs ``[D'_1, ..., D'_n]``."""
        return tuple(r.result for r in self.rounds)

    def validate(self) -> bool:
        """Check the rollout invariants: ``D_1 = {}`` and ``D_i = D'_{i-1}``."""
        prev: Optional[Design] = None
        for i, r in enumerate(self.rounds, start=1):
            if r.index != i:
                return False
            expected = Design.empty() if prev is None else prev
            if r.design != expected:
                return False
            if r.result != apply_actions(r.design, r.actions):
                return False
            prev = r.result
        return True


class RefinementSession:
    """Stateful driver for a rollout enforcing ``D_i = D'_{i-1}``.

    Starts from the empty design ``D_1 = {}``. Each :meth:`play_round` supplies a
    designer message and a maker action sequence; the transition is applied and a
    :class:`Round` is appended.
    """

    def __init__(self, initial: Optional[Design] = None) -> None:
        self._initial = initial if initial is not None else Design.empty()
        self._rounds: list[Round] = []

    @property
    def current(self) -> Design:
        return self._rounds[-1].result if self._rounds else self._initial

    @property
    def rounds(self) -> Tuple[Round, ...]:
        return tuple(self._rounds)

    def play_round(
        self, message: Message, actions: Sequence[Action]
    ) -> Design:
        before = self.current
        acts = tuple(actions)
        result = apply_actions(before, acts)
        self._rounds.append(
            Round(len(self._rounds) + 1, before, message, acts, result)
        )
        return result

    def rollout(self) -> Rollout:
        return Rollout(tuple(self._rounds))


def won(
    design: Design,
    target: Design,
    threshold: float,
    distance: Callable[[Design, Design], float],
) -> bool:
    """A game is won if ``Delta(D'_n, D*) < theta``."""
    return distance(design, target) < threshold
