"""Directional before/after annotations with inversion and leakage checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


LEVELS = ("global", "part", "face")


@dataclass(frozen=True)
class DirectionalAnnotation:
    level: str
    subject: str
    before: str
    after: str
    change: str
    inverse_change: str

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"unknown annotation level: {self.level}")
        if not all(str(v).strip() for v in (
            self.subject, self.before, self.after, self.change, self.inverse_change
        )):
            raise ValueError("annotation fields cannot be empty")

    def inverted(self) -> "DirectionalAnnotation":
        return DirectionalAnnotation(
            self.level, self.subject, self.after, self.before,
            self.inverse_change, self.change,
        )


def validate_annotations(
    annotations: Iterable[DirectionalAnnotation],
    *,
    forbidden_tokens: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return stable diagnostics rather than silently accepting label leakage."""
    items = tuple(annotations)
    errors: list[str] = []
    seen = set()
    forbidden = tuple(t.casefold() for t in forbidden_tokens if t)
    for i, item in enumerate(items):
        key = (item.level, item.subject)
        if key in seen:
            errors.append(f"duplicate:{item.level}:{item.subject}")
        seen.add(key)
        text = " ".join((item.before, item.after, item.change, item.inverse_change)).casefold()
        if any(token in text for token in forbidden):
            errors.append(f"leakage:{i}")
        if item.inverted().inverted() != item:
            errors.append(f"non_involutive:{i}")
    return tuple(errors)
