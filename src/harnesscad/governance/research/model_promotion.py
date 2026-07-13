"""Evidence gate for promoting an adapted model over a baseline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    reasons: tuple[str, ...]
    improvement: float


def promotion_gate(*, baseline_quality: float, candidate_quality: float,
                   candidate_peak_memory: int, memory_ceiling: int,
                   minimum_improvement: float = 0.0,
                   evidence_count: int, minimum_evidence: int = 1):
    improvement = candidate_quality - baseline_quality
    reasons = []
    if evidence_count < minimum_evidence:
        reasons.append("insufficient-evidence")
    if improvement < minimum_improvement:
        reasons.append("quality-threshold")
    if candidate_peak_memory > memory_ceiling:
        reasons.append("memory-ceiling")
    return PromotionDecision(not reasons, tuple(reasons), improvement)
