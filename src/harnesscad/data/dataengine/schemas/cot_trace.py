"""Multimodal CoT trace schema + format-conformance checker (CAD-RL, 2026).

CAD-RL's CoT-based Cold Start (Sec. 3.2) fine-tunes the model to emit a
*two-part* generation: a reasoning span followed by the executable CADQuery
code, delimited by special tokens (e.g. ``<Think> ... </Think>``). The joint
reasoning+code sequence is what the autoregressive cold-start objective (Eq. 1)
is trained on. Inputs are multimodal (Sec. 3.1): a non-expert natural-language
prompt, an expert structured design-language specification, and an optional
reference image.

This module provides:

  * :class:`MultimodalIntent` -- the input schema (natural-language prompt,
    structured design language, optional reference-image reference), with a
    ``modality_flags`` view mirroring the paper's ablation axes (Ref img / text
    settings).
  * :class:`CoTTrace` -- the two-part output schema (``think`` reasoning span +
    ``code`` CADQuery span).
  * :func:`parse_cot` / :func:`check_conformance` -- a strict conformance checker
    for the ``<Think>...</Think>`` + code format, returning a structured report
    of every violated rule so a data-engine filter can drop or repair
    non-conformant cold-start examples.

The checker is a deterministic *format* verifier; it does not judge reasoning
quality or execute the code. This is unlike ``dataengine.cot_records`` (a
provenance / split-leakage audit over already-parsed records) -- here we parse
and validate the raw two-part textual format itself. Pure stdlib.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

THINK_OPEN = "<Think>"
THINK_CLOSE = "</Think>"

# Natural-language vs. structured design-language input settings (paper Table 1).
INPUT_NATURAL = "natural_language"
INPUT_STRUCTURED = "structured_design_language"


@dataclass(frozen=True)
class MultimodalIntent(object):
    """A CAD-RL multimodal input example.

    ``natural_language`` is the non-expert prompt; ``structured`` is the
    expert-level design-language specification; ``reference_image`` optionally
    names a rendered reference. At least one textual modality must be present.
    """

    natural_language: str = ""
    structured: str = ""
    reference_image: Optional[str] = None

    def __post_init__(self):
        if not self.natural_language.strip() and not self.structured.strip():
            raise ValueError("at least one textual modality is required")

    @property
    def modality_flags(self):
        """Boolean view of which modalities are present (ablation axes)."""
        return {
            INPUT_NATURAL: bool(self.natural_language.strip()),
            INPUT_STRUCTURED: bool(self.structured.strip()),
            "reference_image": self.reference_image is not None,
        }


@dataclass(frozen=True)
class CoTTrace(object):
    """Two-part CAD-RL output: a reasoning span and a CADQuery code span."""

    think: str
    code: str


@dataclass(frozen=True)
class ConformanceReport(object):
    """Result of :func:`check_conformance`."""

    conformant: bool
    violations: tuple = field(default_factory=tuple)
    trace: Optional[CoTTrace] = None


def parse_cot(text: str) -> Optional[CoTTrace]:
    """Parse ``<Think>reasoning</Think>code`` into a :class:`CoTTrace`.

    Returns ``None`` if the delimiter structure is malformed. The code span is
    everything after ``</Think>``.
    """
    open_idx = text.find(THINK_OPEN)
    close_idx = text.find(THINK_CLOSE)
    if open_idx == -1 or close_idx == -1 or close_idx < open_idx:
        return None
    # Reject nested / repeated delimiters.
    if text.count(THINK_OPEN) != 1 or text.count(THINK_CLOSE) != 1:
        return None
    think = text[open_idx + len(THINK_OPEN):close_idx]
    code = text[close_idx + len(THINK_CLOSE):]
    return CoTTrace(think=think.strip(), code=code.strip())


def check_conformance(text: str) -> ConformanceReport:
    """Strictly validate the two-part CoT format.

    Rules: exactly one well-ordered ``<Think>``/``</Think>`` pair, a non-empty
    reasoning span, a non-empty code span, and the reasoning must precede the
    code (guaranteed by parsing). Every failed rule is reported.
    """
    violations = []
    trace = parse_cot(text)
    if trace is None:
        if text.count(THINK_OPEN) != 1 or text.count(THINK_CLOSE) != 1:
            violations.append("delimiter_count")
        else:
            violations.append("delimiter_order")
        return ConformanceReport(False, tuple(violations), None)
    if not trace.think:
        violations.append("empty_reasoning")
    if not trace.code:
        violations.append("empty_code")
    return ConformanceReport(len(violations) == 0, tuple(violations), trace)


def is_conformant(text: str) -> bool:
    """Convenience boolean wrapper over :func:`check_conformance`."""
    return check_conformance(text).conformant
