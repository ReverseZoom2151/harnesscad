"""CADReview correction pass — turn a detected error into a code edit.

Detection (:mod:`cadreview_detect`) names *what* is wrong and *where*; the
paper's second stage then *corrects* the program, mapping the geometric error to
a concrete code modification (swap the primitive, fix the rotation angle,
re-insert the missing block, ...). The learned code editor does this from images;
this module implements the deterministic, reference-grounded corrector: when the
correct reference program is known, every one of the eight error types has an
exact, mechanical fix — restore the offending block to its reference form,
remove a redundant block, or re-insert a missing one.

:func:`correct` takes the candidate program, its reference, and a
:class:`cadreview_detect.Review`, and returns a :class:`Correction` carrying (a)
one human-readable :class:`FixSuggestion` per detected error — the geometric-op
-to-code-edit mapping from the taxonomy's ``fix_action`` — and (b) a repaired
program. The repaired program is block-level: only the offending blocks are
changed, everything else is preserved. The guaranteeing property (exercised by
the tests) is round-trip: ``detect(correct(cand, ref).source, ref)`` reports
``No error``. Pure stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from harnesscad.domain.programs.review.blocks import Block, segment
from harnesscad.domain.programs.review.detect import Detection, Review
from harnesscad.domain.programs.review.taxonomy import MISSING_BLOCK, REDUNDANT_BLOCK

_INPLACE = {
    "primitive_error", "rotation_error", "position_error",
    "size_error", "logic_error", "constant_error",
}


@dataclass
class FixSuggestion:
    """A concrete, human-readable correction for one detected error."""

    error_type: str
    block_id: Optional[int]
    fix_action: str
    instruction: str

    def to_dict(self) -> dict:
        return {
            "error_type": self.error_type,
            "block_id": self.block_id,
            "fix_action": self.fix_action,
            "instruction": self.instruction,
        }


@dataclass
class Correction:
    """The result of correcting a program against its reference."""

    suggestions: List[FixSuggestion] = field(default_factory=list)
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "suggestions": [s.to_dict() for s in self.suggestions],
            "source": self.source,
        }


def _reassemble(blocks: List[Block]) -> str:
    return "\n".join(b.text for b in blocks) + "\n"


def _instruction(det: Detection, ref_block: Optional[Block]) -> str:
    et = det.error_type
    ref_text = ref_block.text.strip() if ref_block else ""
    if et.id == "primitive_error":
        return (f"Block {det.block_id}: wrong primitive. Restore the reference "
                f"primitive -> `{ref_text}`.")
    if et.id == "rotation_error":
        return (f"Block {det.block_id}: rotation is wrong. Set the rotate() "
                f"transform to the reference -> `{ref_text}`.")
    if et.id == "position_error":
        return (f"Block {det.block_id}: position is wrong. Set the translate() "
                f"offset to the reference -> `{ref_text}`.")
    if et.id == "size_error":
        return (f"Block {det.block_id}: dimension is wrong. Restore the "
                f"reference size -> `{ref_text}`.")
    if et.id == "logic_error":
        return (f"Block {det.block_id}: control-flow condition is wrong. Restore "
                f"the reference condition -> `{ref_text}`.")
    if et.id == "constant_error":
        return (f"Block {det.block_id}: global constant is wrong. Restore the "
                f"reference value -> `{ref_text}`.")
    if et.id == MISSING_BLOCK.id:
        return (f"Re-insert the missing block (reference Block {det.block_id}) "
                f"-> `{ref_text}`.")
    if et.id == REDUNDANT_BLOCK.id:
        return f"Remove the redundant Block {det.block_id}."
    return f"Block {det.block_id}: review against the reference design."


def correct(candidate_src: str, reference_src: str, review: Review) -> Correction:
    """Correct ``candidate_src`` toward ``reference_src`` per ``review``.

    Applies one edit per detection: in-place restore for parametric errors,
    removal for a redundant block, insertion for a missing block. Returns the
    repaired source plus per-error fix suggestions."""
    ref_blocks = segment(reference_src)
    ref_by_id = {b.id: b for b in ref_blocks}
    work = segment(candidate_src)

    dets = list(review.detections) if not review.ok else []
    suggestions: List[FixSuggestion] = []

    # Emit suggestions (order = detection order) before mutating indices.
    for det in dets:
        ref_block = ref_by_id.get(det.block_id) if det.error_type.id in _INPLACE \
            or det.error_type.id == MISSING_BLOCK.id else None
        suggestions.append(FixSuggestion(
            error_type=det.error_type.label,
            block_id=det.block_id,
            fix_action=det.error_type.fix_action,
            instruction=_instruction(det, ref_block),
        ))

    # 1. In-place restores (index-stable).
    for det in dets:
        if det.error_type.id in _INPLACE and det.block_id is not None:
            ref_block = ref_by_id.get(det.block_id)
            for k, b in enumerate(work):
                if b.id == det.block_id and ref_block is not None:
                    work[k] = Block(b.id, ref_block.kind, ref_block.head,
                                    list(ref_block.calls), ref_block.text)
                    break

    # 2. Remove redundant blocks (highest id first to keep positions stable).
    redundant_ids = sorted(
        (d.block_id for d in dets
         if d.error_type.id == REDUNDANT_BLOCK.id and d.block_id is not None),
        reverse=True)
    for rid in redundant_ids:
        work = [b for b in work if b.id != rid]

    # 3. Insert missing blocks (lowest id first).
    missing_ids = sorted(
        d.block_id for d in dets
        if d.error_type.id == MISSING_BLOCK.id and d.block_id is not None)
    for mid in missing_ids:
        ref_block = ref_by_id.get(mid)
        if ref_block is None:
            continue
        pos = min(mid, len(work))
        work.insert(pos, Block(mid, ref_block.kind, ref_block.head,
                               list(ref_block.calls), ref_block.text))

    # Renumber ids so the output re-segments cleanly.
    renum = [Block(i, b.kind, b.head, b.calls, b.text) for i, b in enumerate(work)]
    return Correction(suggestions=suggestions, source=_reassemble(renum))
