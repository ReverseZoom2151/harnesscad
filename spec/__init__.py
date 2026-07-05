"""spec — the NL-requirements front-of-pipeline for HarnessCAD.

This package turns a natural-language *brief* into a *formal spec*: a typed
:class:`~spec.formalize.RequirementSet` of countable / parametric asks pulled
from the prompt ("4 mounting holes", "100 mm long", "aluminium", "+/- 0.1"),
which then

  * seeds a machine-verifiable :class:`contract.Contract` (via
    :func:`~spec.formalize.to_contract`), and
  * drives a :class:`checks_requirements.RequirementsCheck` that asserts the
    *built* model actually contains what the brief asked for.

A companion :class:`~spec.interview.RequirementsInterview` plans the targeted
clarifying questions to ask when a brief under-specifies material, tolerance,
load, envelope or quantity.

Everything degrades gracefully with no LLM and no third-party dependencies: the
formaliser and interviewer both ship deterministic heuristic defaults, and an
injected :class:`llm.base.LLM` is used only when provided.
"""

from __future__ import annotations

from spec.formalize import (
    Requirement,
    RequirementSet,
    formalize,
    to_contract,
    requirement_schema,
)
from spec.interview import Question, RequirementsInterview

__all__ = [
    "Requirement",
    "RequirementSet",
    "formalize",
    "to_contract",
    "requirement_schema",
    "Question",
    "RequirementsInterview",
]
