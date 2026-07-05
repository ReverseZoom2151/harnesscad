"""Deterministic multi-view annotation jobs and type-specific quality gates.

The module deliberately keeps rendering and language models behind callables.
That makes the orchestration, filtering, confidence accounting, and acceptance
rules usable in tests and in offline dataset production without requiring a
particular VLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from statistics import fmean
from typing import Callable, Iterable, Mapping, Sequence


class AnnotationKind(str, Enum):
    CAPTION = "caption"
    TAG = "tag"
    DIMENSION = "dimension"
    FEATURE = "feature"


@dataclass(frozen=True)
class Candidate:
    value: str
    confidence: float
    view: str
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not self.value.strip():
            raise ValueError("candidate value cannot be empty")


@dataclass(frozen=True)
class QualityPolicy:
    minimum_confidence: float = 0.7
    minimum_views: int = 2
    minimum_agreement: float = 0.5
    forbidden_terms: tuple[str, ...] = ()
    required_evidence: bool = False


@dataclass(frozen=True)
class Scorecard:
    kind: AnnotationKind
    value: str
    confidence: float
    agreement: float
    supporting_views: tuple[str, ...]
    accepted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class AnnotationBatch:
    model_id: str
    scorecards: tuple[Scorecard, ...]
    rejected_candidates: tuple[Candidate, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    @property
    def accepted(self) -> tuple[Scorecard, ...]:
        return tuple(card for card in self.scorecards if card.accepted)


def _normalise(value: str) -> str:
    return " ".join(value.casefold().split())


def score_candidates(
    kind: AnnotationKind,
    candidates: Iterable[Candidate],
    policy: QualityPolicy,
) -> tuple[Scorecard, ...]:
    """Group equivalent answers and produce an auditable quality scorecard."""
    groups: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        groups.setdefault(_normalise(candidate.value), []).append(candidate)

    total_views = len({item.view for group in groups.values() for item in group})
    cards: list[Scorecard] = []
    for key, group in sorted(groups.items()):
        views = tuple(sorted({item.view for item in group}))
        confidence = fmean(item.confidence for item in group)
        agreement = len(views) / max(total_views, 1)
        reasons: list[str] = []
        if confidence < policy.minimum_confidence:
            reasons.append("confidence_below_threshold")
        if len(views) < policy.minimum_views:
            reasons.append("insufficient_view_support")
        if agreement < policy.minimum_agreement:
            reasons.append("insufficient_cross_view_agreement")
        if any(term.casefold() in key for term in policy.forbidden_terms):
            reasons.append("forbidden_term")
        if policy.required_evidence and any(not item.evidence for item in group):
            reasons.append("missing_evidence")
        cards.append(
            Scorecard(
                kind=kind,
                value=group[0].value.strip(),
                confidence=confidence,
                agreement=agreement,
                supporting_views=views,
                accepted=not reasons,
                reasons=tuple(reasons),
            )
        )
    return tuple(cards)


class MultiViewAnnotationJob:
    """Run injected annotators over views and apply kind-specific policies."""

    def __init__(
        self,
        annotators: Mapping[
            AnnotationKind, Callable[[str, object], Sequence[Candidate]]
        ],
        policies: Mapping[AnnotationKind, QualityPolicy],
    ) -> None:
        self._annotators = dict(annotators)
        self._policies = dict(policies)

    def run(
        self,
        model_id: str,
        views: Mapping[str, object],
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> AnnotationBatch:
        if not views:
            raise ValueError("at least one view is required")
        cards: list[Scorecard] = []
        rejected: list[Candidate] = []
        for kind, annotator in self._annotators.items():
            policy = self._policies.get(kind, QualityPolicy())
            candidates: list[Candidate] = []
            for view_name, payload in sorted(views.items()):
                for item in annotator(view_name, payload):
                    if item.view != view_name:
                        raise ValueError("annotator candidate view does not match input view")
                    candidates.append(item)
            kind_cards = score_candidates(kind, candidates, policy)
            cards.extend(kind_cards)
            rejected_values = {
                _normalise(card.value) for card in kind_cards if not card.accepted
            }
            rejected.extend(
                item
                for item in candidates
                if _normalise(item.value) in rejected_values
            )
        return AnnotationBatch(
            model_id=model_id,
            scorecards=tuple(cards),
            rejected_candidates=tuple(rejected),
            metadata=dict(metadata or {}),
        )
