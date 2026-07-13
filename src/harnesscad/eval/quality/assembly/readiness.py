"""Assembly requirement completeness, intervention routing, and readiness.

The paper-derived D/L/N/G/F categories are represented explicitly:
dimensions, layout/constraints, number of elements, element geometry, and
function.  The module is CAD-kernel independent so the same scorecard can sit
in front of FreeCAD, CadQuery, or another backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from statistics import fmean
from typing import Iterable, Mapping


class RequirementField(str, Enum):
    DIMENSIONS = "dimensions"
    LAYOUT = "layout_constraints"
    ELEMENT_COUNT = "element_count"
    GEOMETRY = "element_geometry"
    FUNCTION = "function"


@dataclass(frozen=True)
class AssemblyRequirements:
    values: Mapping[RequirementField, tuple[str, ...]]

    @classmethod
    def from_mapping(
        cls, values: Mapping[RequirementField | str, Iterable[str]]
    ) -> "AssemblyRequirements":
        normalized: dict[RequirementField, tuple[str, ...]] = {}
        for key, entries in values.items():
            field = key if isinstance(key, RequirementField) else RequirementField(key)
            normalized[field] = tuple(
                text.strip() for text in entries if text and text.strip()
            )
        return cls(normalized)

    @property
    def missing(self) -> tuple[RequirementField, ...]:
        return tuple(field for field in RequirementField if not self.values.get(field))

    @property
    def completeness(self) -> float:
        return (len(RequirementField) - len(self.missing)) / len(RequirementField)

    @property
    def complete(self) -> bool:
        return not self.missing

    def questions(self) -> tuple[str, ...]:
        prompts = {
            RequirementField.DIMENSIONS: "What dimensions and tolerances are required?",
            RequirementField.LAYOUT: "How are parts located and constrained?",
            RequirementField.ELEMENT_COUNT: "How many instances of each part are needed?",
            RequirementField.GEOMETRY: "What geometry and detailed features must each part have?",
            RequirementField.FUNCTION: "What must the assembly and moving parts do?",
        }
        return tuple(prompts[field] for field in self.missing)


class InterventionMode(str, Enum):
    PROMPT = "prompt"
    CODE = "code"
    DIRECT_CAD = "direct_cad"


@dataclass(frozen=True)
class CorrectionAttempt:
    mode: InterventionMode
    quality_before: float
    quality_after: float
    effort: float = 1.0
    note: str = ""

    def __post_init__(self) -> None:
        if not 0 <= self.quality_before <= 1 or not 0 <= self.quality_after <= 1:
            raise ValueError("quality values must be in [0, 1]")
        if self.effort < 0:
            raise ValueError("effort cannot be negative")

    @property
    def improvement(self) -> float:
        return self.quality_after - self.quality_before

    @property
    def efficiency(self) -> float:
        return self.improvement / max(self.effort, 1e-12)


@dataclass(frozen=True)
class HandoffDecision:
    mode: InterventionMode
    reason: str


@dataclass(frozen=True)
class HandoffPolicy:
    maximum_prompt_attempts: int = 3
    minimum_prompt_improvement: float = 0.03
    maximum_prompt_effort: float = 5.0
    code_to_cad_threshold: int = 2

    def decide(self, attempts: Iterable[CorrectionAttempt]) -> HandoffDecision:
        history = tuple(attempts)
        prompt = tuple(item for item in history if item.mode is InterventionMode.PROMPT)
        code = tuple(item for item in history if item.mode is InterventionMode.CODE)
        prompt_effort = sum(item.effort for item in prompt)
        recent_gain = prompt[-1].improvement if prompt else None
        if len(code) >= self.code_to_cad_threshold:
            return HandoffDecision(
                InterventionMode.DIRECT_CAD,
                "code correction limit reached; use direct CAD editing",
            )
        if len(prompt) >= self.maximum_prompt_attempts:
            return HandoffDecision(
                InterventionMode.CODE,
                "prompt correction limit reached",
            )
        if prompt_effort >= self.maximum_prompt_effort:
            return HandoffDecision(
                InterventionMode.CODE,
                "prompt effort budget exhausted",
            )
        if recent_gain is not None and recent_gain < self.minimum_prompt_improvement:
            return HandoffDecision(
                InterventionMode.CODE,
                "latest prompt produced diminishing returns",
            )
        return HandoffDecision(InterventionMode.PROMPT, "prompt iteration remains efficient")


class ReadinessAspect(str, Enum):
    GROSS_SHAPE = "gross_shape"
    DIMENSIONS = "dimensions"
    PLACEMENT = "placement"
    DETAILED_FEATURES = "detailed_features"
    FUNCTION = "function"


@dataclass(frozen=True)
class AspectResult:
    aspect: ReadinessAspect
    passed: int
    total: int
    critical_failure: bool = False
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.total < 1 or not 0 <= self.passed <= self.total:
            raise ValueError("aspect counts must satisfy 0 <= passed <= total")

    @property
    def score(self) -> float:
        return self.passed / self.total


@dataclass(frozen=True)
class AssemblyReadiness:
    score: float
    production_ready: bool
    aspect_scores: Mapping[ReadinessAspect, float]
    blockers: tuple[str, ...]


def assess_readiness(
    results: Iterable[AspectResult],
    *,
    minimum_aspect_score: float = 0.8,
) -> AssemblyReadiness:
    if not 0 <= minimum_aspect_score <= 1:
        raise ValueError("minimum_aspect_score must be in [0, 1]")
    by_aspect: dict[ReadinessAspect, AspectResult] = {}
    for result in results:
        if result.aspect in by_aspect:
            raise ValueError(f"duplicate readiness aspect: {result.aspect.value}")
        by_aspect[result.aspect] = result
    missing = tuple(aspect for aspect in ReadinessAspect if aspect not in by_aspect)
    blockers = [f"missing:{aspect.value}" for aspect in missing]
    for aspect, result in by_aspect.items():
        if result.critical_failure:
            blockers.append(f"critical:{aspect.value}")
        elif result.score < minimum_aspect_score:
            blockers.append(f"below_threshold:{aspect.value}")
    scores = {aspect: result.score for aspect, result in by_aspect.items()}
    score = fmean(scores.values()) if scores else 0.0
    return AssemblyReadiness(
        score=score,
        production_ready=not blockers,
        aspect_scores=scores,
        blockers=tuple(blockers),
    )
