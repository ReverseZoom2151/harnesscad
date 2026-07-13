"""Geometry-constrained local-edit prompts for GeoCAD (Zhang et al. 2025, Sec. 3.2).

GeoCAD fine-tunes an LLM in two stages. The deterministic, buildable artefact of each
stage is its *prompt construction*, which this module implements on top of the
existing FlexCAD masking (:mod:`reconstruction.flexcad_text`,
:mod:`dataengine.flexcad_masking`). The LLM training itself is out of scope.

**Stage 2 -- instruction fine-tuning (Fig. 4).** Like FlexCAD, a local loop is masked
and the model must predict it from the remaining parts. GeoCAD's distinct contribution
is that the prompt *incorporates the geometric instruction as an explicit constraint*
(paper Sec. 3.2):

    "FlexCAD's training process has one critical limitation: its prompts lack
     geometric constraints during training. Consequently, once trained, FlexCAD
     struggles to follow geometric instructions. In light of this, ... our prompts
     incorporate the geometric instructions as constraints when fine-tuning LLMs."

**Stage 1 -- CAD-text alignment (Fig. 3).** A single local part plus its augmented
copies are presented with their (shared, invariant) geometric instruction, asking the
model to reproduce the initial and augmented parts.

Everything here is deterministic string assembly; no wall clock, no learned model.
"""

from __future__ import annotations

from dataclasses import dataclass

from harnesscad.domain.reconstruction.translate.flexcad_text import (
    CADModel,
    LEVEL_LOOP,
    MaskTarget,
    mask_field,
)


@dataclass(frozen=True)
class Stage2Prompt:
    """A geometry-constrained local-infill training example (Fig. 4)."""

    prompt: str
    answer: str
    instruction: str
    mask_token: str


def build_stage2_prompt(model: CADModel, se: int, face: int, loop: int,
                        instruction: str) -> Stage2Prompt:
    """Mask one local loop and build a prompt carrying its geometric instruction.

    The masked-out loop's serialisation is the training ``answer``; the surrounding
    CAD text with a ``[loopmask]`` in place plus the geometric ``instruction`` form
    the input the LLM must complete. Contrast FlexCAD, whose prompt omits the
    instruction.
    """
    target = MaskTarget(LEVEL_LOOP, se=se, face=face, loop=loop)
    res = mask_field(model, target)
    remaining = " ".join(res.instruction)
    answer = " ".join(res.answer)
    prompt = (
        "Complete the masked local part of the following CAD model. "
        f"The masked part must be {instruction}.\n"
        f"CAD model: {remaining}\n"
        "Masked part:"
    )
    return Stage2Prompt(prompt, answer, instruction, res.mask[0])


@dataclass(frozen=True)
class Stage1Prompt:
    """A CAD-text alignment example over a part and its augmentations (Fig. 3)."""

    prompt: str
    answers: tuple[str, ...]
    instruction: str


def build_stage1_prompt(part_serialisations: list[str],
                        instruction: str) -> Stage1Prompt:
    """Build a stage-1 alignment prompt for a part and its augmented copies.

    All ``part_serialisations`` (the initial part first, then its augmentations)
    share the same invariant ``instruction``; the model is asked to emit each.
    """
    if not part_serialisations:
        raise ValueError("need at least the initial part serialisation")
    prompt = (
        f"The following local CAD parts are all {instruction}. "
        f"Generate {len(part_serialisations)} such part(s):"
    )
    return Stage1Prompt(prompt, tuple(part_serialisations), instruction)
