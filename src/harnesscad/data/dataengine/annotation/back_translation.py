"""Bounded reverse-description verification for generated CAD sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


def lcs_length(left: Sequence[T], right: Sequence[T]) -> int:
    row = [0] * (len(right) + 1)
    for a in left:
        previous = 0
        for index, b in enumerate(right, start=1):
            saved = row[index]
            row[index] = (
                previous + 1
                if a == b
                else max(row[index], row[index - 1])
            )
            previous = saved
    return row[-1]


def ordered_recovery_ratio(reference: Sequence[T], recovered: Sequence[T]) -> float:
    """Paper-exact ``LCS / len(reference)`` with an empty-sequence convention."""
    if not reference:
        return 1.0 if not recovered else 0.0
    return lcs_length(reference, recovered) / len(reference)


@dataclass(frozen=True)
class ReverseAttempt:
    round: int
    description: str
    recovered: tuple[object, ...]
    ratio: float
    accepted: bool
    feedback: str


@dataclass(frozen=True)
class ReverseVerification:
    accepted: bool
    threshold: float
    attempts: tuple[ReverseAttempt, ...]

    @property
    def final_description(self) -> str:
        return self.attempts[-1].description if self.attempts else ""


def verify_with_reflection(
    reference: Sequence[T],
    initial_description: str,
    reverse: Callable[[str], Sequence[T]],
    reflect: Callable[[str, str, float], str],
    *,
    threshold: float = 0.9,
    maximum_reflections: int = 2,
) -> ReverseVerification:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if maximum_reflections < 0:
        raise ValueError("maximum_reflections cannot be negative")
    description = initial_description
    attempts: list[ReverseAttempt] = []
    for round_index in range(maximum_reflections + 1):
        recovered = tuple(reverse(description))
        ratio = ordered_recovery_ratio(reference, recovered)
        accepted = ratio >= threshold
        feedback = (
            "accepted"
            if accepted
            else f"ordered recovery {ratio:.3f} is below {threshold:.3f}"
        )
        attempts.append(
            ReverseAttempt(
                round_index, description, recovered, ratio, accepted, feedback
            )
        )
        if accepted:
            break
        if round_index < maximum_reflections:
            description = reflect(description, feedback, ratio)
    return ReverseVerification(attempts[-1].accepted, threshold, tuple(attempts))
