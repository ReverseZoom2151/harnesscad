"""Masked-infill training-pair constructor for FlexCAD (Zhang et al. 2024, Sec. 3.2).

A FlexCAD prompt template is a pair of an **instruction** -- a fixed natural-language
lead-in plus the CAD text whose hierarchy-aware field has been replaced by a mask
token -- and an **answer** containing the tokens of the masked field (paper Sec. 3.2,
"Prompt Template Design"). During unified training, at each epoch a template is
*uniformly sampled* over the seven hierarchies; the LLM predicts the masked field
autoregressively and a cross-entropy loss against the answer is back-propagated.

Beyond the seven controllable templates, FlexCAD supports **unconditional
generation** by adding a template whose instruction is simply
``"Below is a description of a CAD sequence:"`` and whose answer is the whole CAD
text (paper appendix A.5).

This module deterministically builds those (instruction, answer) training pairs from
:mod:`reconstruction.flexcad_text` models via the masking scheme in
:mod:`dataengine.flexcad_masking`. Sampling uses a seeded ``random.Random`` -- no
wall clock. The actual LLM fine-tuning is out of scope.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from harnesscad.domain.reconstruction.translate.flexcad_text import (
    CADModel,
    LEVEL_CAD,
    LEVEL_SE,
    LEVEL_SKETCH,
    LEVEL_EXTRUSION,
    LEVEL_FACE,
    LEVEL_LOOP,
    LEVEL_CURVE,
    MaskResult,
    MaskTarget,
    infill,
    parse_tokens,
    serialize,
    tokenize,
)
from harnesscad.data.dataengine.augment.flexcad_masking import (
    SAMPLING_LEVELS,
    mask,
    sample_target,
)

# Per-hierarchy instruction lead-ins (paper Fig. 4 style).
INSTRUCTIONS: dict[str, str] = {
    LEVEL_CAD: "Below is a CAD sequence; predict its sketch-extrusions:",
    LEVEL_SE: "Below is a CAD sequence; predict the masked sketch-extrusion:",
    LEVEL_SKETCH: "Below is a CAD sequence; predict the masked sketch:",
    LEVEL_EXTRUSION: "Below is a CAD sequence; predict the masked extrusion:",
    LEVEL_FACE: "Below is a CAD sequence; predict the masked face:",
    LEVEL_LOOP: "Below is a CAD sequence; predict the masked loop:",
    LEVEL_CURVE: "Below is a CAD sequence; predict the masked curves:",
}

# Unconditional-generation template (appendix A.5).
UNCONDITIONAL_INSTRUCTION = "Below is a description of a CAD sequence:"


@dataclass(frozen=True)
class InfillPair:
    """One masked-infill training pair.

    ``instruction`` = lead-in text + masked CAD text; ``answer`` = the masked
    field's tokens (as text); ``level`` names the hierarchy; ``mask`` is the mask
    token run so the pair can be re-infilled and verified.
    """

    instruction: str
    answer: str
    level: str
    mask: tuple[str, ...]

    def instruction_tokens(self) -> tuple[str, ...]:
        """Just the CAD-text portion of the instruction (lead-in stripped)."""
        lead = INSTRUCTIONS.get(self.level, UNCONDITIONAL_INSTRUCTION)
        body = self.instruction[len(lead):].strip()
        return tuple(body.split())


def pair_from_target(m: CADModel, target: MaskTarget) -> InfillPair:
    """Build the (instruction, answer) pair for masking one addressed field."""
    result: MaskResult = mask(m, target)
    lead = INSTRUCTIONS[target.level]
    instruction = lead + " " + " ".join(result.instruction)
    answer = " ".join(result.answer)
    return InfillPair(instruction, answer, target.level, result.mask)


def unconditional_pair(m: CADModel) -> InfillPair:
    """Unconditional-generation pair: the answer is the whole CAD text."""
    text = serialize(m)
    instruction = UNCONDITIONAL_INSTRUCTION + " " + text
    return InfillPair(instruction, text, "unconditional", tuple())


def sample_pair(m: CADModel, rng: random.Random,
                level: str | None = None) -> InfillPair:
    """Sample one training pair by drawing a hierarchy uniformly then a field.

    Deterministic given ``rng``. Mirrors the paper's per-epoch uniform template
    sampling over the seven hierarchies.
    """
    target = sample_target(m, rng, level=level)
    return pair_from_target(m, target)


def build_epoch(models: list[CADModel], seed: int,
                include_unconditional: bool = False) -> list[InfillPair]:
    """Build one epoch's worth of training pairs, one per model.

    For each model a hierarchy is uniformly sampled and its field masked. When
    ``include_unconditional`` is set, an extra unconditional pair is appended per
    model (paper appendix A.5). Fully deterministic in ``seed``.
    """
    rng = random.Random(seed)
    pairs: list[InfillPair] = []
    for m in models:
        pairs.append(sample_pair(m, rng))
        if include_unconditional:
            pairs.append(unconditional_pair(m))
    return pairs


def verify_pair(m: CADModel, pair: InfillPair) -> bool:
    """Check the pair re-infills to the original model's tokens (round-trip).

    Confirms the constructed (instruction, answer) is a faithful, loss-consistent
    reconstruction target: swapping the answer back for the mask run reproduces
    exactly :func:`tokenize` of ``m``.
    """
    if pair.level == "unconditional":
        return pair.answer == serialize(m)
    instr = pair.instruction_tokens()
    answer = tuple(pair.answer.split())
    try:
        rebuilt = infill(instr, answer, pair.mask)
    except ValueError:
        return False
    return rebuilt == tokenize(m)
