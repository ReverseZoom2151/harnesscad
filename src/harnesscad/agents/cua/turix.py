"""turix — TuriX's step-verification, skill-file (SOP) schema, and Brain rules.

TuriX is a desktop computer-use agent whose three most portable ideas are all
model-free in structure, so they come across as deterministic data + policy:

1. **Two-screenshot step-eval.** TuriX does not trust that an action worked because
   it was issued; it captures a screenshot BEFORE and AFTER the action and judges the
   step by the DIFFERENCE. Ported here as :func:`evaluate_step`: given the before/after
   observations (any hashable-into-dict state — a SetOfMarks element list, a measured
   metrics dict) it reports whether the screen CHANGED and whether the step's stated
   expectation was met. The lesson is the structure: a step with no observable effect
   is a *failed* step, not a completed one.

2. **The SOP / skill-file schema.** TuriX stores reusable "Standard Operating
   Procedures" — named, triggered, ordered step lists — that a planner recalls instead
   of re-deriving a routine. :class:`SOP` + :class:`SkillLibrary` port that as a typed,
   JSON-persistable store, keyed by a trigger substring. (This is the *procedure*
   analogue of experience.DialogFeatureMemory's per-feature recipe: a whole multi-step
   routine, not one dialog.)

3. **Brain rules.** TuriX's "Brain" carries two rules that stop the two most common
   desktop-agent misfires, both reproduced as pure policy:
   * **loading-grace-period** (:func:`settle_index`): do NOT judge the screen while it
     is still loading; wait until the observation has been STABLE for a grace period,
     so a mid-load frame is never mistaken for a finished state.
   * **detail-must-be-explicit** (:func:`require_explicit`): the Brain does not invent
     missing details; an instruction to build to a dimension MUST state that dimension
     explicitly, or it is refused rather than guessed.

Pure stdlib, import-safe, deterministic. No screenshot, no model, no app.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# --- 1. two-screenshot step evaluation --------------------------------------

@dataclass(frozen=True)
class StepEval:
    """The verdict on one action, judged by BEFORE vs AFTER (TuriX's core check).

    ``changed`` is whether the observation differed at all — a step that changed
    nothing is suspect. ``expectation_met`` is whether the caller's expected tokens
    appeared in the after-state (``None`` when no expectation was supplied). ``ok`` is
    the overall call: something changed AND (no expectation, or it was met).
    """

    changed: bool
    expectation_met: Optional[bool]
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return {"changed": self.changed, "expectation_met": self.expectation_met,
                "ok": self.ok, "detail": self.detail}


def _canonical(state: Any) -> str:
    """A stable string for an observation, so equality is order-independent."""
    try:
        return json.dumps(state, sort_keys=True, default=str)
    except TypeError:
        return repr(state)


def evaluate_step(before: Any, after: Any,
                  expectation: Optional[str] = None) -> StepEval:
    """Judge a step by comparing the before/after observations (TuriX's two-shot eval).

    ``expectation`` is an optional natural-language claim about the result ("the Pad
    dialog is open"); it is checked by word-overlap against the after-state's text — a
    cheap self-consistency signal, never a hard oracle (the geometry grade is that).
    A step that produced NO change is ``ok=False`` even if it "looks" done.
    """
    changed = _canonical(before) != _canonical(after)
    expectation_met: Optional[bool] = None
    if expectation:
        want = {w for w in re.findall(r"[a-z0-9]+", expectation.lower()) if len(w) > 2}
        after_text = _canonical(after).lower()
        after_words = set(re.findall(r"[a-z0-9]+", after_text))
        if want:
            hit = want & after_words
            expectation_met = len(hit) >= max(1, len(want) // 2)
        else:
            expectation_met = None
    ok = changed and (expectation_met is None or expectation_met)
    if not changed:
        detail = "no observable change between the two screenshots; step had no effect"
    elif expectation_met is False:
        detail = "screen changed but the expected result did not appear"
    else:
        detail = "step produced the expected observable change"
    return StepEval(changed=changed, expectation_met=expectation_met, ok=ok,
                    detail=detail)


# --- 2. the SOP / skill-file schema -----------------------------------------

@dataclass(frozen=True)
class SOPStep:
    """One line of a Standard Operating Procedure: an action, its target, a value."""

    action: str
    target: str = ""
    value: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return {"action": self.action, "target": self.target,
                "value": self.value, "note": self.note}

    @classmethod
    def from_dict(cls, d: dict) -> "SOPStep":
        return cls(action=d["action"], target=d.get("target", ""),
                   value=d.get("value", ""), note=d.get("note", ""))


@dataclass(frozen=True)
class SOP:
    """A reusable, triggered routine — TuriX's skill file.

    ``trigger`` is the substring of an instruction that recalls this SOP; ``steps`` is
    the ordered recipe. This is a whole multi-step procedure (open dialog, type,
    confirm), distinct from a single dialog recipe.
    """

    name: str
    trigger: str
    steps: Tuple[SOPStep, ...] = ()
    note: str = ""

    def matches(self, instruction: str) -> bool:
        return bool(self.trigger) and self.trigger.lower() in instruction.lower()

    def to_dict(self) -> dict:
        return {"name": self.name, "trigger": self.trigger,
                "steps": [s.to_dict() for s in self.steps], "note": self.note}

    @classmethod
    def from_dict(cls, d: dict) -> "SOP":
        return cls(name=d["name"], trigger=d.get("trigger", ""),
                   steps=tuple(SOPStep.from_dict(s) for s in d.get("steps", [])),
                   note=d.get("note", ""))


class SkillLibrary:
    """A store of :class:`SOP` skill files, recalled by instruction trigger."""

    def __init__(self, sops: Optional[Sequence[SOP]] = None) -> None:
        self._sops: List[SOP] = list(sops or [])

    def add(self, sop: SOP) -> None:
        self._sops.append(sop)

    def all(self) -> List[SOP]:
        return list(self._sops)

    def get(self, name: str) -> Optional[SOP]:
        for s in self._sops:
            if s.name == name:
                return s
        return None

    def recall(self, instruction: str) -> Optional[SOP]:
        """The first SOP whose trigger occurs in the instruction (TuriX's lookup).

        Longest trigger wins on multiple matches, so a specific routine beats a
        generic one; ties are broken by name for determinism."""
        hits = [s for s in self._sops if s.matches(instruction)]
        if not hits:
            return None
        hits.sort(key=lambda s: (-len(s.trigger), s.name))
        return hits[0]

    def to_dict(self) -> dict:
        return {"sops": [s.to_dict() for s in self._sops]}

    @classmethod
    def from_dict(cls, d: dict) -> "SkillLibrary":
        return cls([SOP.from_dict(s) for s in d.get("sops", [])])

    def __len__(self) -> int:
        return len(self._sops)


# --- 3. Brain rules ----------------------------------------------------------

#: TuriX waits this many stable observations before trusting the screen has settled.
DEFAULT_GRACE = 2


def settle_index(observations: Sequence[Any], grace: int = DEFAULT_GRACE) -> Optional[int]:
    """The loading-grace-period rule: the index at which the screen has SETTLED.

    Returns the index of the first observation after which the state stayed IDENTICAL
    for ``grace`` consecutive further observations (i.e. loading has stopped). Returns
    ``None`` if the stream never stabilises for that long — the honest "still loading,
    do not judge yet" answer. ``grace`` of 0 settles immediately.
    """
    if grace < 0:
        raise ValueError("grace must be >= 0")
    n = len(observations)
    if n == 0:
        return None
    canon = [_canonical(o) for o in observations]
    for i in range(n):
        # stable if the next `grace` observations all equal observation i.
        if i + grace < n and all(canon[i] == canon[j]
                                 for j in range(i + 1, i + grace + 1)):
            return i
        if grace == 0:
            return i
    return None


def is_settled(observations: Sequence[Any], grace: int = DEFAULT_GRACE) -> bool:
    """Whether the observation stream has settled within a grace period."""
    return settle_index(observations, grace) is not None


#: A dimension mention: a number, optionally with a unit, that makes a build
#: instruction explicit enough to act on without inventing values.
_EXPLICIT_NUMBER = re.compile(r"\d+(?:\.\d+)?\s*(?:mm|cm|m|in|inch|deg|°)?", re.I)
#: Words that signal a MEASURED build whose details must be present.
_DIMENSION_WORDS = re.compile(
    r"\b(mm|cm|in|inch|long|wide|tall|thick|high|deep|radius|diameter|"
    r"length|width|height|dimension|size|by)\b", re.I)


def require_explicit(instruction: str, *, min_numbers: int = 1
                     ) -> Tuple[bool, List[str]]:
    """The detail-must-be-explicit rule: refuse an under-specified build instruction.

    TuriX's Brain does not guess missing details. If an instruction TALKS about
    dimensions (long/wide/thick/mm/...) but does not STATE enough explicit numbers, it
    is refused with the reason, rather than acted on with invented values. An
    instruction that mentions no dimensions at all is left to the planner (not this
    rule's job). Returns ``(ok, reasons)``.
    """
    reasons: List[str] = []
    mentions_dims = bool(_DIMENSION_WORDS.search(instruction))
    numbers = _EXPLICIT_NUMBER.findall(instruction)
    n_numbers = sum(1 for tok in numbers if re.search(r"\d", tok))
    if mentions_dims and n_numbers < min_numbers:
        reasons.append(
            "instruction refers to dimensions but states %d explicit value(s); "
            "at least %d required (details must be explicit, never inferred)"
            % (n_numbers, min_numbers))
    return (not reasons), reasons
