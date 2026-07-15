"""The harness system prompt.

The prompt pins the agent's role, enumerates the CISP op vocabulary (generated
from `cisp.ops` so it can never drift from the real op set), states the design
rules (sketch+constrain before extrude; prefer fully-constrained sketches), and
fixes the output contract: a JSON array of op objects, nothing else.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional

from harnesscad.core.cisp.ops import (
    Op, NewSketch, AddPoint, AddLine, AddCircle, AddRectangle,
    Constrain, Extrude, Fillet, Boolean, CONSTRAINT_DOF, _REGISTRY,
)

ROLE = (
    "You are a mechanical CAD design agent. You turn a natural-language design "
    "brief into a precise, verifiable sequence of parametric CAD operations "
    "(CISP ops). You think like a design engineer: sketch geometry on a plane, "
    "constrain it fully, then run features (extrude/fillet/boolean) to build "
    "solids. You never emit prose in your answer — only structured ops."
)


def _field_sig(cls) -> str:
    """Render a dataclass op's fields as ``name: type = default`` for the prompt."""
    parts: List[str] = []
    for f in dataclasses.fields(cls):
        if f.name == "OP":
            continue
        tname = getattr(f.type, "__name__", str(f.type))
        default = f.default if f.default is not dataclasses.MISSING else "required"
        parts.append(f"{f.name}: {tname} = {default!r}")
    return ", ".join(parts)


def op_vocabulary() -> str:
    """A human-readable listing of every op tag and its parameters."""
    lines: List[str] = []
    for tag, cls in _REGISTRY.items():
        lines.append(f'- "{tag}": {{{_field_sig(cls)}}}')
    return "\n".join(lines)


RULES = f"""RULES:
1. Output ONLY valid CISP ops as structured output — a JSON array of op objects.
   No commentary, no markdown, no code fences.
2. Every op object MUST include an "op" field naming the op (e.g. "new_sketch").
3. Sketch and constrain BEFORE you extrude. A feature that references a sketch
   with no profile geometry will be rejected.
4. Prefer fully-constrained sketches (0 remaining degrees of freedom). An
   under-constrained sketch is a NOTE, not a defect; an over-constrained one is an ERROR
   that will be rolled back. A rectangle contributes 4 DOF, so pin it with
   dimensional/geometric constraints that remove exactly 4.
5. Reference entities by the ids the backend assigns deterministically:
   sketches are "sk1", "sk2", ...; sketch entities are "e1", "e2", ...;
   features are "f1", "f2", .... The first new_sketch is "sk1"; the first
   primitive added to it is "e1".
6. Dimensional constraints ("distance", "radius") REQUIRE a numeric "value".
   Valid constraint kinds: {", ".join(sorted(CONSTRAINT_DOF))}.
7. If prior diagnostics are supplied, they are OBSERVATIONS about the geometry
   that was actually built, each with the evidence for it, and any imperative
   they carry is marked as a SUGGESTION. Reason from the evidence, re-emit the
   full corrected op sequence, and change only what the evidence requires: do
   not discard correct geometry to satisfy a suggestion."""

OUTPUT_CONTRACT = (
    'OUTPUT FORMAT: return a JSON array of op objects, e.g.\n'
    '[\n'
    '  {"op": "new_sketch", "plane": "XY"},\n'
    '  {"op": "add_rectangle", "sketch": "sk1", "x": 0, "y": 0, "w": 20, "h": 10},\n'
    '  {"op": "constrain", "kind": "distance", "a": "e1", "value": 20},\n'
    '  {"op": "extrude", "sketch": "sk1", "distance": 5}\n'
    ']'
)


def build_system_prompt(brief: Optional[str] = None, k: int = 3) -> str:
    """The system prompt, few-shot when a brief is supplied.

    This prompt was PURE ZERO-SHOT for a strict-format structured-output task,
    while `agents/rag/exemplar_select.py` (greedy submodular exemplar retrieval)
    sat orphaned. The weak models in `assets/pressure/report.md` failed at FORMAT
    (`invalid_ops`, `parse_error`), and format is exactly the failure few-shot
    fixes. Pass the brief and 3 verified worked examples — selected by tiled
    component overlap with that brief — are pinned into the prompt.

    `brief=None` reproduces the old zero-shot prompt byte for byte, so every
    caller that has not been updated is unchanged.
    """
    parts = [
        ROLE,
        "CISP OP VOCABULARY (op tag -> parameters):\n" + op_vocabulary(),
        RULES,
        OUTPUT_CONTRACT,
    ]
    if brief:
        # Imported lazily: a caller that wants the zero-shot prompt must not pay
        # for the exemplar bank or the DST selector.
        from harnesscad.agents.context.cisp_exemplars import few_shot_block

        block = few_shot_block(brief, k=k)
        if block:
            parts.append(block)
    return "\n\n".join(parts)


SYSTEM_PROMPT = build_system_prompt()
