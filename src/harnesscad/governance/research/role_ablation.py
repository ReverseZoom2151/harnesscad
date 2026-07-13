"""Small deterministic helper for reporting agent-role ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class AblationResult:
    removed_role: str
    deltas: Mapping[str, float]

    @property
    def harmful(self) -> bool:
        return any(delta < 0 for delta in self.deltas.values())


def compare_role_ablation(
    baseline: Mapping[str, float],
    without_role: Mapping[str, float],
    role: str,
) -> AblationResult:
    """Compute ``without_role - baseline`` for an identical metric set."""
    if not role.strip():
        raise ValueError("role is required")
    if baseline.keys() != without_role.keys():
        raise ValueError("baseline and ablation must contain identical metrics")
    return AblationResult(
        removed_role=role,
        deltas={key: without_role[key] - baseline[key] for key in sorted(baseline)},
    )
