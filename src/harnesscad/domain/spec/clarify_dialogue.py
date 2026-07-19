"""clarify_dialogue -- a two-round proactive clarification MDP.

The clarifying agent is modelled as a finite-horizon MDP
``M = (S, A, R)`` where a state ``s = (p, h)`` couples the prompt with the
conversation history and the action space is::

    A = {ACCEPT} u {ASK(u) : u in U}

Under the assumption that the user answers any *clear* question correctly, the
optimal policy collapses to **two rounds**:

  * Round 1 -- the agent either ACCEPTs the prompt (emitting the standardized
    specification unchanged) or ASKs a single batched set of targeted
    clarification questions.
  * Round 2 -- after receiving the user's answers, the agent deterministically
    ACCEPTs and emits the corrected, self-consistent specification.

This module is the deterministic state machine implementing that reduction. It
consumes :mod:`clarify_ambiguity` for the audit and a pluggable *user
simulator* callback (deterministic; instead of an LLM user simulator, we
supply an answer-oracle interface) to close the loop. No LLM, no wall clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from harnesscad.domain.spec.clarify_ambiguity import (
    CADSpec,
    ClarQuestion,
    CONFLICTING,
    Feature,
    audit,
)

# Actions.
ACCEPT = "ACCEPT"
ASK = "ASK"

# States.
START = "start"
AWAIT_ANSWERS = "await_answers"
DONE = "done"


#: A user simulator answers a question ``key`` with a corrected value.
#: It receives (question, spec) and returns a value (float, tuple, or None).
UserSimulator = Callable[[ClarQuestion, CADSpec], object]


def oracle_from_truth(truth: CADSpec) -> UserSimulator:
    """Build a deterministic user simulator that answers from a ground-truth spec.

    Mirrors the assumption that the user "can provide correct answers to
    any asked question as long as the question itself is clear".
    """

    def answer(q: ClarQuestion, _spec: CADSpec) -> object:
        return _lookup(truth, q.key)

    return answer


@dataclass
class Turn:
    """One recorded interaction turn."""

    action: str  # ACCEPT | ASK
    questions: Tuple[str, ...] = ()
    answers: Tuple[Tuple[str, object], ...] = ()


@dataclass
class DialogueResult:
    """Outcome of a completed clarification dialogue."""

    rounds: int
    turns: List[Turn]
    corrected: CADSpec
    asked_keys: Tuple[str, ...]
    is_misleading: bool

    def interaction_cost(self) -> int:
        """C(h): number of question-asking rounds (0 or 1 here)."""
        return sum(1 for t in self.turns if t.action == ASK)


class ClarificationDialogue:
    """Deterministic two-round proactive clarification state machine."""

    def __init__(self, spec: CADSpec) -> None:
        self.spec = spec
        self.state = START
        self.turns: List[Turn] = []
        self._pending: List[ClarQuestion] = []

    # -- round 1 -------------------------------------------------------- #
    def step_round1(self) -> Turn:
        """Audit the prompt; ACCEPT it or ASK the minimal question batch."""
        if self.state != START:
            raise RuntimeError("round 1 already taken")
        report = audit(self.spec)
        if not report.is_misleading:
            self.state = DONE
            turn = Turn(ACCEPT)
            self.turns.append(turn)
            return turn
        self._pending = list(report.questions)
        self.state = AWAIT_ANSWERS
        turn = Turn(ASK, tuple(q.text for q in self._pending))
        self.turns.append(turn)
        return turn

    # -- round 2 -------------------------------------------------------- #
    def step_round2(self, user: UserSimulator) -> Turn:
        """Collect answers via ``user`` and deterministically ACCEPT."""
        if self.state != AWAIT_ANSWERS:
            raise RuntimeError("no questions are pending")
        answers: List[Tuple[str, object]] = []
        for q in self._pending:
            val = user(q, self.spec)
            answers.append((q.key, val))
            _apply(self.spec, q.key, val)
        self.state = DONE
        turn = Turn(ACCEPT, tuple(q.text for q in self._pending), tuple(answers))
        self.turns.append(turn)
        return turn

    # -- driver --------------------------------------------------------- #
    def run(self, user: Optional[UserSimulator]) -> DialogueResult:
        """Run the full two-round policy and return the corrected spec."""
        t1 = self.step_round1()
        asked = tuple(self._pending_keys())
        if t1.action == ASK:
            if user is None:
                raise ValueError("prompt is ambiguous but no user simulator given")
            self.step_round2(user)
        return DialogueResult(
            rounds=len(self.turns),
            turns=list(self.turns),
            corrected=self.spec,
            asked_keys=asked,
            is_misleading=(t1.action == ASK),
        )

    def _pending_keys(self) -> List[str]:
        return [q.key for q in self._pending]


def run_dialogue(prompt: CADSpec, truth: CADSpec) -> DialogueResult:
    """Convenience: clarify ``prompt`` using ``truth`` as the answer oracle.

    The input ``prompt`` is copied so the caller's spec is not mutated.
    """
    dlg = ClarificationDialogue(prompt.copy())
    return dlg.run(oracle_from_truth(truth))


# --------------------------------------------------------------------------- #
# spec <-> key application
# --------------------------------------------------------------------------- #

def _lookup(spec: CADSpec, key: str) -> object:
    if key == "setup.workplane":
        return spec.workplane
    if key == "setup.origin":
        return spec.origin
    if key == "build.extrude_direction":
        return spec.extrude_direction
    if key == "build.extrude_distance":
        return _resolve(spec.extrude_distance)
    feat, param = _split(key)
    for f in spec.features:
        if (f.name or f.kind) == feat:
            return _resolve(f.params.get(param))
    return None


def _apply(spec: CADSpec, key: str, value: object) -> None:
    """Write the user-supplied corrected value into the spec, clearing conflicts."""
    if key == "setup.workplane":
        spec.workplane = value
        return
    if key == "setup.origin":
        spec.origin = None if value is None else tuple(value)
        return
    if key == "build.extrude_direction":
        spec.extrude_direction = value
        return
    if key == "build.extrude_distance":
        spec.extrude_distance = value
        return
    feat, param = _split(key)
    for f in spec.features:
        if (f.name or f.kind) == feat:
            f.params[param] = value
            return


def _split(key: str) -> Tuple[str, str]:
    parts = key.split(".")
    return parts[0], parts[-1]


def _resolve(raw: object) -> object:
    """A single scalar from a possibly-conflicting stated value (first one)."""
    if isinstance(raw, (list, tuple, set)):
        vals = [v for v in raw if v is not None]
        return vals[0] if vals else None
    return raw
