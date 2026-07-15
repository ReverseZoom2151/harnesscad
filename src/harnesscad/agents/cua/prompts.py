"""prompts — the observation-prompt schema and the verbatim templates worth keeping.

The os_computer_use agent's per-step prompt is the single most reusable artefact
in the batch, because it forces the model into a shape that is CHECKABLE rather
than free prose. Kept verbatim as :data:`OBSERVATION_TEMPLATE`, its structure is:

    The objective is: [restate the objective]
    On the screen, I see: [an EXHAUSTIVE enumeration of everything relevant]
    This means the objective is: [complete|not complete]
    The next step is to [click|type|run ...] [the single next step]
        in order to [what you EXPECT to happen].

Four properties make it worth porting as a schema, not just a string:

    1. **Restate the objective** every step — cheap defence against the model
       drifting off task over a long loop.
    2. **Exhaustive enumeration** before deciding — the model must look before it
       acts, and an empty enumeration is a tell that it did not.
    3. **Termination as a REQUIRED binary field** — "complete | not complete".
       Not an afterthought the loop infers from silence; a field the model must
       fill, which :func:`parse_termination` turns into a hard bool.
    4. **The falsifiable "in order to [prediction]".** The model states what the
       next screenshot should show. The NEXT observation can then check whether it
       came true — a free, per-step self-consistency signal that needs no oracle.
       :func:`predicted_outcome` extracts it so a loop can carry it forward.

This module is prompt DATA plus pure parsers. No model is called. Absolute
imports; the CAD system prompt is kept minimal and geometry-first on purpose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


# --- verbatim templates (data) ----------------------------------------------
#: The os_computer_use observation prompt, kept EXACTLY. ``{objective}`` is the
#: only substitution; everything else is the model's to fill.
OBSERVATION_TEMPLATE = (
    "This image shows the current display of the computer. Please respond in the "
    "following format:\n"
    "The objective is: [put the objective here]\n"
    "On the screen, I see: [an extensive list of everything that might be "
    "relevant to the objective including windows, icons, menus, apps, and UI "
    "elements]\n"
    "This means the objective is: [complete|not complete]\n\n"
    "(Only continue if the objective is not complete.)\n"
    "The next step is to [click|type|run the shell command] [put the next single "
    "step here] in order to [put what you expect to happen here]."
)

#: The bare system line the reference agent used. Kept for provenance.
GENERIC_SYSTEM = "You are an AI assistant with computer use abilities."

#: The array-nudge lesson from agents/cua/models.py, restated for the action
#: model: emit the WHOLE ordered step, not a fragment.
ACTION_PREFIX = ("I will now use tool calls to take these actions, or use the "
                 "stop command if the objective is complete.")

#: A CAD-specific system prompt. Geometry-first, and it bakes in the two traps the
#: harness already knows about: the decimal separator, and never saving.
CAD_SYSTEM = (
    "You are driving a real CAD application through its GUI to build ONE part to "
    "a written brief. You are graded only on the geometry the application "
    "actually produces, measured through its own kernel -- not on what you say. "
    "Rules that are not negotiable:\n"
    "- Type dimensions EXACTLY as written. A value like 37.5 is thirty-seven "
    "point five; never let a locale turn the '.' into a thousands separator.\n"
    "- You NEVER save, export, delete, discard, or exit. The harness owns all "
    "file I/O.\n"
    "- Prefer parameter dialogs (type numbers) over dragging in the viewport.\n"
    "- If a step needs a pick inside the 3D view that you cannot ground exactly, "
    "say so and stop; an honest refusal beats a wrong click."
)


def render_observation_prompt(objective: str) -> str:
    """The observation prompt with the objective restated up front, so the model
    sees the goal before the template it must fill. The template's own
    ``[put the objective here]`` slot is preserved verbatim."""
    return "OBJECTIVE: %s\n\n%s" % (objective.strip(), OBSERVATION_TEMPLATE)


# --- the schema + parsers ---------------------------------------------------
# Anchored to line starts (re.M): "The objective is:" and "This means the
# objective is:" both contain the substring "the objective is:", so the objective
# field is pinned to the START of its line to avoid matching the termination line.
# Captures stop at the newline ([^\n]) so a field never bleeds into the next line.
_OBJECTIVE = re.compile(r"^[ \t]*the objective is:[ \t]*([^\n]*)", re.I | re.M)
_SEE = re.compile(r"^[ \t]*on the screen,? i see:[ \t]*([^\n]*)", re.I | re.M)
_COMPLETE = re.compile(r"^[ \t]*this means the objective is:[ \t]*([^\n]*)",
                       re.I | re.M)
_NEXT = re.compile(r"^[ \t]*the next step is to[ \t]+([^\n]*)", re.I | re.M)
_IN_ORDER_TO = re.compile(r"\bin order to\b\s*(.+)", re.I)

_TERMINAL_YES = re.compile(r"\b(is\s+)?complete\b", re.I)
_TERMINAL_NO = re.compile(r"\bnot\s+complete\b", re.I)


class ObservationError(ValueError):
    """A model reply did not follow the required observation shape."""


@dataclass(frozen=True)
class Observation:
    """One parsed, validated observation. ``complete`` is a hard bool; ``prediction``
    is the falsifiable "in order to ..." the NEXT screenshot can be judged against.
    """

    objective: str
    seen: str
    complete: bool
    next_step: str
    prediction: str

    def to_dict(self) -> dict:
        return {"objective": self.objective, "seen": self.seen,
                "complete": self.complete, "next_step": self.next_step,
                "prediction": self.prediction}


def parse_termination(text: str) -> bool:
    """The REQUIRED binary field -> bool. 'not complete' beats 'complete' because
    the former CONTAINS the latter as a substring; a reply with neither raises,
    because termination is never inferred from silence."""
    if _TERMINAL_NO.search(text):
        return False
    if _TERMINAL_YES.search(text):
        return True
    raise ObservationError("no termination field ('complete' | 'not complete') "
                           "found; termination must be stated, not inferred")


def predicted_outcome(text: str) -> Optional[str]:
    """The 'in order to [prediction]' clause of the next step, or None if absent.
    This is the falsifiable claim about the next screenshot."""
    m = _IN_ORDER_TO.search(text)
    return m.group(1).strip().rstrip(".") if m else None


def _one(pattern: re.Pattern, text: str, field: str, required: bool = True
         ) -> str:
    m = pattern.search(text)
    if m:
        # The capture is already newline-free; may be empty (an unfilled field).
        return m.group(1).strip()
    if required:
        raise ObservationError("missing required field: %s" % field)
    return ""


def parse_observation(text: str) -> Observation:
    """Parse a model reply into a validated :class:`Observation`, RAISING when a
    required field is absent. This is what makes the template a contract:

    * objective restated (else the model has drifted),
    * a non-empty enumeration (else it did not look),
    * termination present as a binary field,
    * a next step WITH an 'in order to' prediction (unless already complete).
    """
    objective = _one(_OBJECTIVE, text, "objective")
    seen = _one(_SEE, text, "on-screen enumeration")
    if not seen:
        raise ObservationError("empty enumeration: the model must list what it "
                               "sees before deciding")
    complete = parse_termination(_one(_COMPLETE, text, "termination"))
    next_step = _one(_NEXT, text, "next step", required=not complete)
    prediction = predicted_outcome(next_step) or ""
    if not complete and not prediction:
        raise ObservationError("next step has no 'in order to [prediction]'; the "
                               "step must state a falsifiable expected outcome")
    return Observation(objective=objective, seen=seen, complete=complete,
                       next_step=next_step, prediction=prediction)


def prediction_held(previous: Observation, next_seen: str) -> Optional[bool]:
    """Cheap self-consistency: did the previous step's prediction show up in the
    next screenshot's enumeration? Word-overlap heuristic; None when there was no
    prediction to test. This is a SIGNAL for the grader, never a hard gate."""
    if not previous.prediction:
        return None
    pred_words = {w for w in re.findall(r"[a-z0-9]+", previous.prediction.lower())
                  if len(w) > 3}
    if not pred_words:
        return None
    seen_words = set(re.findall(r"[a-z0-9]+", next_seen.lower()))
    overlap = pred_words & seen_words
    return len(overlap) >= max(1, len(pred_words) // 3)
