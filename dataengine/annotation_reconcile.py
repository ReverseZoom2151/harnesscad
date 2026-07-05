"""Reconcile image and point-cloud descriptions into an auditable decision."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


_TOKEN = re.compile(r"[a-z0-9]+")
_CONTRADICTIONS = (
    frozenset(("cube", "cylinder")),
    frozenset(("round", "square")),
    frozenset(("solid", "hollow")),
    frozenset(("single", "multiple")),
)


@dataclass(frozen=True)
class ModalityDescription:
    modality: str
    text: str
    confidence: float
    provenance: str = ""

    def __post_init__(self) -> None:
        if not self.modality or not self.text.strip():
            raise ValueError("modality and description text are required")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class Reconciliation:
    route: str
    merged_text: str
    conflicts: tuple[str, ...]
    modalities: tuple[str, ...]
    confidence: float


def reconcile_descriptions(
    descriptions: Iterable[ModalityDescription],
    *,
    minimum_modalities: int = 2,
    minimum_confidence: float = 0.7,
) -> Reconciliation:
    items = tuple(sorted(descriptions, key=lambda item: item.modality))
    if not items:
        return Reconciliation("manual_review", "", ("missing_all_modalities",), (), 0.0)
    tokens = {
        item.modality: frozenset(_TOKEN.findall(item.text.casefold())) for item in items
    }
    conflicts: set[str] = set()
    for pair in _CONTRADICTIONS:
        owners = {
            word: tuple(name for name, values in tokens.items() if word in values)
            for word in pair
        }
        if all(owners[word] for word in pair):
            conflicts.add("contradiction:" + "/".join(sorted(pair)))
    modalities = tuple(item.modality for item in items)
    if len(set(modalities)) < minimum_modalities:
        conflicts.add("insufficient_modalities")
    confidence = min(item.confidence for item in items)
    if confidence < minimum_confidence:
        conflicts.add("low_confidence")
    merged = " | ".join(f"{item.modality}: {item.text.strip()}" for item in items)
    return Reconciliation(
        "auto_pass" if not conflicts else "manual_review",
        merged,
        tuple(sorted(conflicts)),
        modalities,
        confidence,
    )
