"""CADReview error generator — deterministically inject the eight error types.

CADReview's dataset is built by *mutating* correct CAD programs: annotators take
a known-good program and introduce exactly one error of a known type in a known
block, producing an (erroneous program, error type, block ID) triple. That
mutation step is a deterministic program transformation, so it is fully
reproducible without any annotator — this module is that fault injector.

Given a correct OpenSCAD program, :func:`inject` applies one of the eight
mutations from :mod:`cadreview_taxonomy` and returns the mutated source together
with the ground-truth :class:`InjectedError` (error type + block ID, following
the same block-ID convention that :mod:`cadreview_detect` reports). This gives a
self-checking benchmark: inject a known error, run the detector, and the
detector's ``(type, block_id)`` must equal the injected ground truth — which is
exactly the "Acc" metric the paper scores (:mod:`cadreview_review`).

Mutations are seeded (``random.Random(seed)``) so a given (program, type, seed)
always yields the same erroneous program. :func:`injectable_types` reports which
of the eight mutations a program can support (e.g. a program with no control
flow cannot host a Logic error). Pure stdlib; source in, source out.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from programs.cadreview_blocks import Block, segment
from programs.cadreview_taxonomy import (
    CONSTANT_ERROR, ErrorType, LOGIC_ERROR, MISSING_BLOCK, POSITION_ERROR,
    PRIMITIVE_ERROR, REDUNDANT_BLOCK, ROTATION_ERROR, SIZE_ERROR,
)

_PRIMS = ("cube", "sphere", "cylinder", "polyhedron", "square", "circle",
          "polygon")
_PRIM_SWAP = {"cube": "cylinder", "cylinder": "sphere", "sphere": "cube",
              "square": "circle", "circle": "square"}
_NUM_RE = re.compile(r"-?\d+\.?\d*")
_DISTINCTIVE = "translate([137, 149, 151]) cube([1, 1, 1]);"


@dataclass
class InjectedError:
    """Ground truth for one injected error: type + block ID + mutated source."""

    error_type: ErrorType
    block_id: Optional[int]
    source: str

    def to_dict(self) -> dict:
        return {
            "error_type": self.error_type.label,
            "error_id": self.error_type.id,
            "block_id": self.block_id,
            "source": self.source,
        }


def _reassemble(blocks: List[Block]) -> str:
    return "\n".join(b.text for b in blocks) + "\n"


def _bump(value: str) -> str:
    """Return a numeric literal guaranteed different from ``value``."""
    try:
        if "." in value or "e" in value.lower():
            v = float(value)
            nv = v + 5.0 if v <= 0 else v + 5.0
            return repr(round(nv, 6))
        v = int(value)
        return str(v + 7)
    except ValueError:
        return value


def _edit_call_number(text: str, name: str, rng: random.Random) -> Optional[str]:
    """Bump one numeric argument inside the first ``name(...)`` call of ``text``."""
    m = re.search(r"\b" + re.escape(name) + r"\s*\(", text)
    if not m:
        return None
    i = m.end() - 1
    depth = 0
    end = None
    for j in range(i, len(text)):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                end = j
                break
    if end is None:
        return None
    args = text[i + 1:end]
    hits = list(_NUM_RE.finditer(args))
    if not hits:
        return None
    pick = hits[rng.randrange(len(hits))]
    new_args = args[:pick.start()] + _bump(pick.group()) + args[pick.end():]
    return text[:i + 1] + new_args + text[end:]


def _first_primitive_block(blocks: List[Block]) -> Optional[int]:
    for idx, b in enumerate(blocks):
        if any(c in _PRIMS for c in b.calls):
            return idx
    return None


def injectable_types(src: str) -> List[ErrorType]:
    """Which of the eight error types can be injected into ``src``."""
    blocks = segment(src)
    out: List[ErrorType] = []
    has_prim = _first_primitive_block(blocks) is not None
    non_macro = [b for b in blocks if b.kind != "assignment"]
    if has_prim:
        out += [PRIMITIVE_ERROR, SIZE_ERROR, ROTATION_ERROR, POSITION_ERROR]
    if any(b.kind == "assignment" and _NUM_RE.search(b.text) for b in blocks):
        out.append(CONSTANT_ERROR)
    if any(b.kind == "control_flow" and _NUM_RE.search(b.text) for b in blocks):
        out.append(LOGIC_ERROR)
    if len(non_macro) >= 1:
        out += [MISSING_BLOCK, REDUNDANT_BLOCK]
    return out


def inject(src: str, error_type: ErrorType, seed: int = 0) -> Optional[InjectedError]:
    """Inject ``error_type`` into ``src``; return the ground-truth mutation.

    Returns None when the program cannot host that error (see
    :func:`injectable_types`). Deterministic in ``seed``."""
    rng = random.Random(seed)
    blocks = segment(src)

    if error_type.id == PRIMITIVE_ERROR.id:
        pos = _first_primitive_block(blocks)
        if pos is None:
            return None
        b = blocks[pos]
        prim = next(c for c in b.calls if c in _PRIMS)
        repl = _PRIM_SWAP.get(prim, "sphere")
        new_text = re.sub(r"\b" + re.escape(prim) + r"\b", repl, b.text, count=1)
        blocks[pos] = Block(b.id, b.kind, b.head, b.calls, new_text)
        return InjectedError(PRIMITIVE_ERROR, pos, _reassemble(blocks))

    if error_type.id == SIZE_ERROR.id:
        pos = _first_primitive_block(blocks)
        if pos is None:
            return None
        b = blocks[pos]
        prim = next(c for c in b.calls if c in _PRIMS)
        new_text = _edit_call_number(b.text, prim, rng)
        if new_text is None:
            return None
        blocks[pos] = Block(b.id, b.kind, b.head, b.calls, new_text)
        return InjectedError(SIZE_ERROR, pos, _reassemble(blocks))

    if error_type.id == ROTATION_ERROR.id:
        pos = _first_primitive_block(blocks)
        if pos is None:
            return None
        b = blocks[pos]
        if "rotate" in b.calls:
            new_text = _edit_call_number(b.text, "rotate", rng)
        else:
            new_text = "rotate([0, 0, 90]) " + b.text
        if new_text is None:
            return None
        blocks[pos] = Block(b.id, b.kind, b.head, b.calls, new_text)
        return InjectedError(ROTATION_ERROR, pos, _reassemble(blocks))

    if error_type.id == POSITION_ERROR.id:
        pos = _first_primitive_block(blocks)
        if pos is None:
            return None
        b = blocks[pos]
        if "translate" in b.calls:
            new_text = _edit_call_number(b.text, "translate", rng)
        else:
            new_text = "translate([25, 0, 0]) " + b.text
        if new_text is None:
            return None
        blocks[pos] = Block(b.id, b.kind, b.head, b.calls, new_text)
        return InjectedError(POSITION_ERROR, pos, _reassemble(blocks))

    if error_type.id == CONSTANT_ERROR.id:
        for pos, b in enumerate(blocks):
            if b.kind == "assignment" and _NUM_RE.search(b.text):
                hit = _NUM_RE.search(b.text)
                new_text = b.text[:hit.start()] + _bump(hit.group()) + b.text[hit.end():]
                blocks[pos] = Block(b.id, b.kind, b.head, b.calls, new_text)
                return InjectedError(CONSTANT_ERROR, pos, _reassemble(blocks))
        return None

    if error_type.id == LOGIC_ERROR.id:
        for pos, b in enumerate(blocks):
            if b.kind == "control_flow":
                new_text = None
                for name in ("for", "if", "intersection_for"):
                    new_text = _edit_call_number(b.text, name, rng)
                    if new_text is not None:
                        break
                if new_text is None and _NUM_RE.search(b.text):
                    hit = _NUM_RE.search(b.text)
                    new_text = b.text[:hit.start()] + _bump(hit.group()) + b.text[hit.end():]
                if new_text is not None:
                    blocks[pos] = Block(b.id, b.kind, b.head, b.calls, new_text)
                    return InjectedError(LOGIC_ERROR, pos, _reassemble(blocks))
        return None

    if error_type.id == MISSING_BLOCK.id:
        candidates = [i for i, b in enumerate(blocks) if b.kind != "assignment"]
        if not candidates:
            return None
        pos = candidates[rng.randrange(len(candidates))]
        removed_id = blocks[pos].id
        del blocks[pos]
        return InjectedError(MISSING_BLOCK, removed_id, _reassemble(blocks))

    if error_type.id == REDUNDANT_BLOCK.id:
        if not blocks:
            return None
        new_id = len(blocks)  # appended at the end -> id == old count
        extra = Block(new_id, "transform", "translate",
                      ["translate", "cube"], _DISTINCTIVE)
        blocks.append(extra)
        return InjectedError(REDUNDANT_BLOCK, new_id, _reassemble(blocks))

    return None


def inject_all(src: str, seed: int = 0) -> List[InjectedError]:
    """One injected sample per injectable error type (a mini benchmark)."""
    out: List[InjectedError] = []
    for i, et in enumerate(injectable_types(src)):
        e = inject(src, et, seed=seed + i)
        if e is not None:
            out.append(e)
    return out
