"""Review report + diagnostic scorer.

This ties the pipeline together into the artifact the review task emits, and the
metric it is scored by.

* REVIEW REPORT -- the review is returned as a structured record with exactly
  four fields: ``"error type"``, ``"erroneous code block ID"``, ``"feedback"``,
  and ``"correct code"``. :class:`ReviewReport` is that record.
  :func:`build_report` runs detection (:mod:`cadreview_detect`) + correction
  (:mod:`cadreview_correct`) and fills it in, generating a concise (<=75-word)
  natural-language ``feedback`` string in a consistent house style ("The 3D
  model shows a deviation in the rotation of the component ... Please correct
  the rotation in Block 2 ..."). For a program that already matches the
  reference, it draws from ten predefined "no error" feedback lines, selected
  deterministically by seed.

* DIAGNOSTIC SCORER -- the accuracy metric ("Acc") and the error-diagnostic
  reward V_d both credit a review only when BOTH the error type AND the
  erroneous block ID are correct. :func:`diagnostic_reward` is V_d;
  :func:`score_dataset` aggregates it into Acc over a benchmark, alongside the
  looser type-only and block-only accuracies for diagnosis.

Pure stdlib; deterministic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from harnesscad.domain.programs.review.correct import Correction, correct
from harnesscad.domain.programs.review.detect import Detection, Review, detect
from harnesscad.domain.programs.review.taxonomy import ErrorType, NO_ERROR

#: Ten predefined feedback lines for a correct program.
PREDEFINED_FEEDBACK: Tuple[str, ...] = (
    "The 3D rendering captures the essence of the design blueprint with "
    "remarkable precision and fidelity.",
    "The 3D model matches the design drawing perfectly, with no deviations in "
    "key features like frames and recesses.",
    "The OpenSCAD-generated 3D model matches the original design drawing "
    "perfectly in all aspects.",
    "The 3D model mirrors the design drawing with exceptional clarity, "
    "maintaining all specified features.",
    "The implementation of the design in OpenSCAD results in a highly accurate "
    "and detailed 3D model.",
    "The alignment between the 3D rendering and the design drawing is precise, "
    "with all features correctly placed.",
    "The design intent is fully realized in the 3D model, with precise "
    "implementation of all structural elements.",
    "The faithful replication of the design drawing in the 3D model indicates "
    "precise coding and attention to detail.",
    "The correspondence between the design plan and the 3D model is seamless, "
    "with no misalignment or deviation.",
    "A careful analysis shows the 3D model to be a perfect reproduction of the "
    "design drawing.",
)

# Per-error phrasing for the feedback house style.
_ANOMALY = {
    "primitive_error": "an incorrect geometric primitive",
    "rotation_error": "a deviation in the rotation of the component",
    "position_error": "a deviation in the position of the component",
    "size_error": "a deviation in the size of the component",
    "logic_error": "an incorrect control-flow path",
    "constant_error": "an incorrect global constant",
    "missing_block": "a missing component",
    "redundant_block": "an extra, unintended component",
}
_FIX = {
    "primitive_error": "replace it with the correct primitive",
    "rotation_error": "correct the rotation to ensure proper orientation",
    "position_error": "correct the translation to restore the intended position",
    "size_error": "restore the correct dimension",
    "logic_error": "correct the loop/conditional so the intended parts are built",
    "constant_error": "restore the correct constant value",
    "missing_block": "re-insert the missing block",
    "redundant_block": "remove the redundant block",
}


def _cap_words(text: str, limit: int = 75) -> str:
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]).rstrip(".,;") + "."


def generate_feedback(det: Detection, seed: int = 0) -> str:
    """A concise (<=75-word) feedback string in a consistent house style."""
    if det.error_type.id == NO_ERROR.id:
        rng = random.Random(seed)
        return PREDEFINED_FEEDBACK[rng.randrange(len(PREDEFINED_FEEDBACK))]
    anomaly = _ANOMALY.get(det.error_type.id, "a deviation in a component")
    fix = _FIX.get(det.error_type.id, "correct the component")
    where = f"Block {det.block_id}" if det.block_id is not None else "the program"
    text = (f"The 3D model shows {anomaly} compared to the design drawing. "
            f"Specifically, {det.detail} in {where}. "
            f"Please {fix} in {where} to match the reference design.")
    return _cap_words(text)


@dataclass
class ReviewReport:
    """The structured review record (output schema)."""

    error_type: str
    block_id: Optional[int]
    feedback: str
    correct_code: str
    suggestions: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        # Keys mirror the required JSON output exactly.
        return {
            "error type": self.error_type,
            "erroneous code block ID": self.block_id,
            "feedback": self.feedback,
            "correct code": self.correct_code,
            "suggestions": list(self.suggestions),
        }


def build_report(candidate_src: str, reference_src: str, seed: int = 0
                 ) -> ReviewReport:
    """Full review: detect the error, correct it, and format the report."""
    review = detect(candidate_src, reference_src)
    fix: Correction = correct(candidate_src, reference_src, review)
    det = review.primary
    return ReviewReport(
        error_type=det.error_type.label,
        block_id=det.block_id,
        feedback=generate_feedback(det, seed=seed),
        correct_code=fix.source if not review.ok else candidate_src,
        suggestions=[s.to_dict() for s in fix.suggestions],
    )


# --------------------------------------------------------------------------- #
# Diagnostic scorer (Acc / V_d)
# --------------------------------------------------------------------------- #
def _norm_type(t) -> Optional[str]:
    if t is None:
        return None
    if isinstance(t, ErrorType):
        return t.id
    from harnesscad.domain.programs.review.taxonomy import by_id, from_label
    s = str(t)
    return (by_id(s) or from_label(s)).id if (by_id(s) or from_label(s)) else s


def diagnostic_reward(pred_type, pred_block, gold_type, gold_block) -> int:
    """V_d: 1 iff BOTH error type AND block ID are correct."""
    type_ok = _norm_type(pred_type) == _norm_type(gold_type)
    block_ok = pred_block == gold_block
    return 1 if (type_ok and block_ok) else 0


def score_dataset(preds: Sequence[Tuple], golds: Sequence[Tuple]) -> dict:
    """Aggregate accuracy over a benchmark.

    ``preds`` / ``golds`` are sequences of ``(error_type, block_id)`` tuples.
    Returns Acc (both correct -- the rigorous metric), plus type-only and
    block-only accuracy for diagnosis."""
    if len(preds) != len(golds):
        raise ValueError("preds and golds must be the same length")
    n = len(golds)
    if n == 0:
        return {"n": 0, "acc": 0.0, "type_acc": 0.0, "block_acc": 0.0}
    both = types = blocks = 0
    for (pt, pb), (gt, gb) in zip(preds, golds):
        t_ok = _norm_type(pt) == _norm_type(gt)
        b_ok = pb == gb
        types += int(t_ok)
        blocks += int(b_ok)
        both += int(t_ok and b_ok)
    return {
        "n": n,
        "acc": both / n,
        "type_acc": types / n,
        "block_acc": blocks / n,
    }
