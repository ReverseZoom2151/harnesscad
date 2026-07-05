"""Typed confidence and selective-correction context for CAD command sequences."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CommandConfidence:
    index: int
    command: str
    type_confidence: float
    arguments: Mapping[str, float]

    def __post_init__(self) -> None:
        values = (self.type_confidence, *self.arguments.values())
        if self.index < 0 or not self.command:
            raise ValueError("command index must be nonnegative and command nonempty")
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError("confidence values must be in [0, 1]")

    @property
    def minimum(self) -> float:
        return min((self.type_confidence, *self.arguments.values()))

    @property
    def mean(self) -> float:
        return fmean((self.type_confidence, *self.arguments.values()))


@dataclass(frozen=True)
class SequenceConfidence:
    commands: tuple[CommandConfidence, ...]
    minimum: float
    mean: float
    low_confidence: tuple[str, ...]
    correction_context: str


def assess_sequence_confidence(
    commands: Sequence[CommandConfidence], *, threshold: float = 0.7
) -> SequenceConfidence:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    ordered = tuple(sorted(commands, key=lambda item: item.index))
    if len({item.index for item in ordered}) != len(ordered):
        raise ValueError("command indices must be unique")
    low: list[str] = []
    for item in ordered:
        if item.type_confidence < threshold:
            low.append(f"{item.index}:{item.command}:type")
        low.extend(
            f"{item.index}:{item.command}:arg:{name}"
            for name, value in sorted(item.arguments.items())
            if value < threshold
        )
    values = [value for item in ordered for value in (item.type_confidence, *item.arguments.values())]
    context = (
        "Review only these uncertain decisions: " + ", ".join(low)
        if low
        else "No selective correction required."
    )
    return SequenceConfidence(
        ordered,
        min(values) if values else 1.0,
        fmean(values) if values else 1.0,
        tuple(low),
        context,
    )
