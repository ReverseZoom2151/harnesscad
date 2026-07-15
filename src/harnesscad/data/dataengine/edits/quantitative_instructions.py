"""Quantitative edit-instruction parsing and progressive refinement (PR-CAD).

Mined from *PR-CAD: Progressive Refinement for Unified Controllable and Faithful
Text-to-CAD Generation*. PR-CAD unifies generation and editing behind an iterative
"progressive refinement" loop driven by two instruction modalities: **qualitative**
("make the base thicker") and **quantitative** ("reduce the radius by 6 mm"). The
quantitative branch is deterministic: an instruction names a parameter, an
operation (increase / reduce / set), and a magnitude, which applies as an exact
delta. This module ports that branch plus the add/modify/delete refinement history.

*   :func:`parse_instruction` -- turn an English edit sentence into a structured
    :class:`EditInstruction` (quantitative when a magnitude is present, else
    qualitative).
*   :func:`apply_instruction` / :class:`RefinementSession` -- apply quantitative
    edits to a parameter dict and record the progressive-refinement history.

Stdlib-only, deterministic. Qualitative instructions are parsed but not applied
(they need a model); quantitative ones apply exactly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = [
    "EditInstruction",
    "parse_instruction",
    "apply_instruction",
    "RefinementSession",
]

_INCREASE = ("increase", "raise", "enlarge", "extend", "thicken", "widen", "grow")
_REDUCE = ("reduce", "decrease", "lower", "shrink", "narrow", "shorten")
_SET = ("set", "make", "change")
_ADD = ("add", "insert", "create")
_DELETE = ("delete", "remove", "cut")

_NUM_UNIT = re.compile(
    r"(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m|deg|degrees|%)?", re.IGNORECASE
)
# a plausible parameter word: radius, height, base, width, depth, thickness, ...
_PARAM = re.compile(
    r"\b(radius|diameter|height|width|depth|length|thickness|base|angle|"
    r"fillet|chamfer|hole|count|spacing|offset)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class EditInstruction:
    """A parsed edit instruction.

    ``edit_type`` is one of ``add``/``delete``/``modify``. Quantitative modify
    instructions carry ``operation`` (``increase``/``reduce``/``set``), ``amount``
    and ``unit``; qualitative ones set ``qualitative=True`` and leave ``amount`` None.
    """

    text: str
    edit_type: str
    target: Optional[str] = None
    operation: Optional[str] = None
    amount: Optional[float] = None
    unit: Optional[str] = None
    qualitative: bool = False


def _first_keyword(words, table) -> Optional[str]:
    for w in words:
        if w in table:
            return w
    return None


def parse_instruction(text: str) -> EditInstruction:
    """Parse an English edit sentence into a structured :class:`EditInstruction`."""
    low = text.lower()
    words = re.findall(r"[a-z%]+", low)
    target_m = _PARAM.search(low)
    target = target_m.group(1).lower() if target_m else None

    if _first_keyword(words, _ADD):
        return EditInstruction(text=text, edit_type="add", target=target)
    if _first_keyword(words, _DELETE):
        return EditInstruction(text=text, edit_type="delete", target=target)

    num_m = _NUM_UNIT.search(low)
    has_number = num_m is not None and num_m.group("value") is not None \
        and re.search(r"\d", low) is not None

    if _first_keyword(words, _INCREASE):
        operation = "increase"
    elif _first_keyword(words, _REDUCE):
        operation = "reduce"
    elif _first_keyword(words, _SET):
        operation = "set"
    else:
        operation = "set"

    if has_number:
        # "to N" means set; "by N" means increase/reduce
        value = float(num_m.group("value"))
        unit = num_m.group("unit")
        if re.search(r"\bto\b", low):
            operation = "set"
        return EditInstruction(text=text, edit_type="modify", target=target,
                              operation=operation, amount=value, unit=unit)
    # no magnitude -> qualitative
    return EditInstruction(text=text, edit_type="modify", target=target,
                          operation=operation, qualitative=True)


def apply_instruction(
    params: Dict[str, float], instruction: EditInstruction
) -> Dict[str, float]:
    """Apply a quantitative modify instruction to a parameter dict (returns a copy).

    Raises ``ValueError`` for non-applicable (qualitative / add / delete) or when
    the target parameter is missing.
    """
    if instruction.edit_type != "modify" or instruction.qualitative:
        raise ValueError("only quantitative modify instructions are applicable")
    if instruction.target is None or instruction.target not in params:
        raise ValueError(f"unknown target parameter: {instruction.target!r}")
    out = dict(params)
    cur = out[instruction.target]
    amt = instruction.amount
    if amt is None:
        raise ValueError("quantitative instruction has no amount")
    if instruction.operation == "increase":
        out[instruction.target] = cur + amt
    elif instruction.operation == "reduce":
        out[instruction.target] = cur - amt
    elif instruction.operation == "set":
        out[instruction.target] = amt
    else:
        raise ValueError(f"unknown operation {instruction.operation!r}")
    return out


class RefinementSession:
    """Tracks a progressive-refinement history over a parameter dict."""

    def __init__(self, params: Dict[str, float]) -> None:
        self.params = dict(params)
        self.history: List[EditInstruction] = []

    def refine(self, text: str) -> Dict[str, float]:
        """Parse and apply one instruction, recording it. Returns the new params."""
        inst = parse_instruction(text)
        self.params = apply_instruction(self.params, inst)
        self.history.append(inst)
        return dict(self.params)

    def num_steps(self) -> int:
        return len(self.history)
