"""Text2CAD beginner-to-expert prompt-level taxonomy (Khan et al., NeurIPS 2024).

Text2CAD's central data-annotation contribution is a *multi-level-of-abstraction*
prompt scheme: every CAD model in the DeepCAD dataset is annotated with four text
prompts of increasing geometric detail, aimed at users of increasing skill (paper
Sec. 3, "Multi-level Design Instructions"):

    L0  Abstract      -- abstract shape description of the final model (from a VLM);
                         shape only, no construction steps.
    L1  Beginner      -- simplified description for laypersons / preliminary design;
                         plain construction steps, no measurements or jargon.
    L2  Intermediate  -- generalized geometric description; abstracts some detail,
                         balancing comprehensibility with technical accuracy.
    L3  Expert        -- detailed geometric description with *relative* values;
                         precise geometry and relative measurements for practitioners.

Figure 4 further shows that each level highlights specific *design aspects*: the
abstract/beginner levels are dominated by **shape descriptions** (teal), whereas the
intermediate/expert levels progressively add **sketch** (yellow) and **extrusion**
(red) parametric detail.

This module is the deterministic, LLM-free taxonomy: the four ordered levels, the
design aspects each includes, the numeric-precision each carries, and a
content-based classifier that maps a free-text prompt back to its most likely
abstraction level (used, e.g., to bucket a prompt corpus or to check that a
generated prompt sits at the intended level). No wall clock, no randomness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class PromptLevelError(ValueError):
    """Raised for unknown level codes/indices or malformed input."""


# --- numeric-precision ladder ----------------------------------------------
PRECISION_NONE = "none"                 # no measurements at all
PRECISION_GENERALIZED = "generalized"   # qualitative / rounded ("a small circle")
PRECISION_PRECISE = "precise"           # exact relative values ("radius 0.25")

_PRECISION_ORDER = (PRECISION_NONE, PRECISION_GENERALIZED, PRECISION_PRECISE)


# --- design aspects highlighted in a prompt (Fig. 4 colour coding) ----------
ASPECT_SHAPE = "shape"          # teal: abstract shape description
ASPECT_SKETCH = "sketch"        # yellow: 2D sketch (curves/loops/faces) detail
ASPECT_EXTRUSION = "extrusion"  # red: extrusion detail


@dataclass(frozen=True)
class PromptLevel:
    """One abstraction level in the Text2CAD taxonomy."""

    code: str            # "L0".."L3"
    index: int           # 0..3 (monotone with detail)
    name: str            # human name ("Abstract", ...)
    audience: str        # intended user skill level
    aspects: tuple[str, ...]      # design aspects this level highlights
    precision: str       # one of _PRECISION_ORDER
    uses_jargon: bool    # technical vocabulary permitted?
    relative_values: bool  # carries explicit relative measurements?
    source: str          # "vlm" (L0) or "llm" (L1..L3)

    def includes(self, aspect: str) -> bool:
        """Does this level highlight ``aspect`` (shape/sketch/extrusion)?"""
        return aspect in self.aspects

    @property
    def precision_rank(self) -> int:
        return _PRECISION_ORDER.index(self.precision)


# The four levels, ordered from most abstract (L0) to most detailed (L3).
LEVELS: tuple[PromptLevel, ...] = (
    PromptLevel(
        code="L0", index=0, name="Abstract",
        audience="any (glance-level shape recognition)",
        aspects=(ASPECT_SHAPE,),
        precision=PRECISION_NONE, uses_jargon=False, relative_values=False,
        source="vlm",
    ),
    PromptLevel(
        code="L1", index=1, name="Beginner",
        audience="laypersons / preliminary design stage",
        aspects=(ASPECT_SHAPE,),
        precision=PRECISION_NONE, uses_jargon=False, relative_values=False,
        source="llm",
    ),
    PromptLevel(
        code="L2", index=2, name="Intermediate",
        audience="designers wanting generalized technical description",
        aspects=(ASPECT_SHAPE, ASPECT_SKETCH, ASPECT_EXTRUSION),
        precision=PRECISION_GENERALIZED, uses_jargon=True, relative_values=False,
        source="llm",
    ),
    PromptLevel(
        code="L3", index=3, name="Expert",
        audience="practitioners performing the CAD modelling task",
        aspects=(ASPECT_SHAPE, ASPECT_SKETCH, ASPECT_EXTRUSION),
        precision=PRECISION_PRECISE, uses_jargon=True, relative_values=True,
        source="llm",
    ),
)

_BY_CODE: dict[str, PromptLevel] = {lv.code: lv for lv in LEVELS}
_BY_INDEX: dict[int, PromptLevel] = {lv.index: lv for lv in LEVELS}

N_LEVELS = len(LEVELS)  # == 4


def level(code: str) -> PromptLevel:
    """Look up a level by its code ("L0".."L3")."""
    key = code.upper()
    if key not in _BY_CODE:
        raise PromptLevelError(f"unknown level code: {code!r}")
    return _BY_CODE[key]


def level_by_index(index: int) -> PromptLevel:
    """Look up a level by its 0..3 detail index."""
    if index not in _BY_INDEX:
        raise PromptLevelError(f"level index out of range: {index}")
    return _BY_INDEX[index]


def ordered_codes() -> tuple[str, ...]:
    """The level codes in increasing-detail order."""
    return tuple(lv.code for lv in LEVELS)


def is_more_detailed(a: str, b: str) -> bool:
    """True iff level ``a`` is strictly more detailed (higher index) than ``b``."""
    return level(a).index > level(b).index


# --- content-based level classification ------------------------------------
# Signals of parametric precision in a prompt's text.
_COORD_RE = re.compile(r"\(\s*-?\d*\.?\d+\s*,\s*-?\d*\.?\d+\s*(?:,\s*-?\d*\.?\d+\s*)?\)")
_NUMBER_RE = re.compile(r"-?\d*\.?\d+")
# Jargon / technical vocabulary that distinguishes L2+ from beginner text.
_JARGON = (
    "extrude", "extrusion", "normal", "coordinate system", "euler",
    "translation", "loop", "profile", "sketch plane", "boolean",
    "quantize", "quantized", "scale factor", "sketch scale",
)
# Words signalling explicit relative measurement (L3).
_RELATIVE = ("radius", "diameter", "distance", "units", "degrees", "angle")


@dataclass(frozen=True)
class LevelSignals:
    """Extracted textual features used by :func:`classify_prompt_level`."""

    n_coordinates: int
    n_numbers: int
    jargon_terms: tuple[str, ...]
    relative_terms: tuple[str, ...]

    @property
    def has_coordinates(self) -> bool:
        return self.n_coordinates > 0

    @property
    def has_jargon(self) -> bool:
        return bool(self.jargon_terms)

    @property
    def has_relative(self) -> bool:
        return bool(self.relative_terms)


def extract_signals(text: str) -> LevelSignals:
    """Deterministically extract the level-discriminating features from ``text``."""
    lowered = text.lower()
    coords = _COORD_RE.findall(text)
    # Count numbers that are *not* already inside a coordinate tuple.
    without_coords = _COORD_RE.sub(" ", text)
    numbers = _NUMBER_RE.findall(without_coords)
    jargon = tuple(term for term in _JARGON if term in lowered)
    relative = tuple(term for term in _RELATIVE if term in lowered)
    return LevelSignals(
        n_coordinates=len(coords),
        n_numbers=len(numbers),
        jargon_terms=jargon,
        relative_terms=relative,
    )


def classify_prompt_level(text: str) -> str:
    """Return the most likely level code ("L0".."L3") for a free-text prompt.

    Heuristic ladder mirroring the taxonomy's defining features:

      * explicit coordinates / relative measurements + jargon  -> Expert (L3);
      * jargon (sketch/extrusion vocabulary) but no exact values -> Intermediate (L2);
      * multi-clause construction steps (contains a verb like "draw"/"create"
        and multiple sentences) but no jargon                    -> Beginner (L1);
      * otherwise a bare shape phrase                            -> Abstract (L0).
    """
    signals = extract_signals(text)
    lowered = text.lower()
    step_verbs = ("draw", "create", "sketch", "make", "extrude", "build", "form")
    has_steps = any(v in lowered for v in step_verbs)

    if signals.has_jargon and (
        signals.has_coordinates or (signals.has_relative and signals.n_numbers > 0)
    ):
        return "L3"
    if signals.has_jargon:
        return "L2"
    if has_steps:
        return "L1"
    return "L0"


def level_matches(text: str, expected_code: str) -> bool:
    """True iff a prompt classifies to ``expected_code`` (round-trip QC check)."""
    return classify_prompt_level(text) == level(expected_code).code
