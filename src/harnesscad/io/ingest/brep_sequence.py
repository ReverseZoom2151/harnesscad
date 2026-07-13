"""Stable serialization for history-free B-rep edit sequences."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping


@dataclass(frozen=True)
class BrepEditStep:
    operation: str
    parameters: Mapping[str, object]

    def to_dict(self) -> dict:
        return {"operation": self.operation, "parameters": dict(sorted(self.parameters.items()))}


@dataclass(frozen=True)
class BrepEditSequence:
    source_digest: str
    instruction: str
    steps: tuple[BrepEditStep, ...]

    @classmethod
    def build(cls, source_digest: str, instruction: str,
              steps: Iterable[BrepEditStep]) -> "BrepEditSequence":
        if not source_digest or not instruction.strip():
            raise ValueError("source digest and instruction are required")
        return cls(source_digest, instruction.strip(), tuple(steps))

    def to_dict(self) -> dict:
        return {
            "source_digest": self.source_digest,
            "instruction": self.instruction,
            "steps": [step.to_dict() for step in self.steps],
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
