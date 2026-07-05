"""Coarse/refine scheduling for comparatively expensive physical checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GenerationPhase(str, Enum):
    COARSE = "coarse"
    REFINE = "refine"


@dataclass(frozen=True)
class PhysicsSchedule:
    every_n: int = 10
    first_refine_iteration: int = 0

    def __post_init__(self) -> None:
        if self.every_n <= 0 or self.first_refine_iteration < 0:
            raise ValueError("every_n must be positive and first iteration non-negative")

    def should_run(self, phase: GenerationPhase, iteration: int) -> bool:
        if iteration < 0:
            raise ValueError("iteration must be non-negative")
        return (
            phase is GenerationPhase.REFINE
            and iteration >= self.first_refine_iteration
            and (iteration - self.first_refine_iteration) % self.every_n == 0
        )
