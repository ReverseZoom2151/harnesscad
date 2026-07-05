"""Balanced instruction taxonomy and deterministic de-duplication."""

from __future__ import annotations

from dataclasses import dataclass
import random
import re
from typing import Callable, Iterable, Sequence


CATEGORIES = (
    "foodstuffs", "clothing", "travel_goods", "brushware",
    "textile_piecegoods", "furnishing", "household_goods", "tools_hardware",
    "packages_containers", "clocks_watches", "adornment", "transport",
    "production_equipment", "medical_laboratory", "machines_appliances",
    "graphic_symbols",
)
STYLES = (
    "imperative", "descriptive", "conversational", "technical",
    "functional", "comparative", "constraint_led", "minimal",
)
LENGTH_BUCKETS = ("very_short", "short", "medium", "long", "very_long")


@dataclass(frozen=True, order=True)
class InstructionSlot:
    category: str
    style: str
    length_bucket: str


@dataclass(frozen=True)
class InstructionSample:
    text: str
    slot: InstructionSlot


def quota_matrix(quota: int = 1) -> tuple[InstructionSlot, ...]:
    if quota < 1:
        raise ValueError("quota must be positive")
    return tuple(
        slot
        for category in CATEGORIES
        for style in STYLES
        for length in LENGTH_BUCKETS
        for slot in (InstructionSlot(category, style, length),) * quota
    )


def seeded_slots(quota: int, seed: int) -> tuple[InstructionSlot, ...]:
    slots = list(quota_matrix(quota))
    random.Random(seed).shuffle(slots)
    return tuple(slots)


_WORDS = re.compile(r"[a-z0-9]+")


def normalized_name(text: str) -> str:
    return " ".join(_WORDS.findall(text.casefold()))


def deduplicate(
    candidates: Iterable[InstructionSample],
    *,
    similarity: Callable[[str, str], float],
    threshold: float = 0.8,
    maximum_name_frequency: int = 1,
) -> tuple[InstructionSample, ...]:
    if not 0 <= threshold <= 1 or maximum_name_frequency < 1:
        raise ValueError("invalid deduplication policy")
    kept: list[InstructionSample] = []
    names: dict[str, int] = {}
    for candidate in candidates:
        name = normalized_name(candidate.text)
        if names.get(name, 0) >= maximum_name_frequency:
            continue
        if any(similarity(candidate.text, prior.text) >= threshold for prior in kept):
            continue
        kept.append(candidate)
        names[name] = names.get(name, 0) + 1
    return tuple(kept)


def slot_coverage(samples: Sequence[InstructionSample]) -> float:
    expected = set(quota_matrix())
    represented = {sample.slot for sample in samples}
    return len(represented & expected) / len(expected)
