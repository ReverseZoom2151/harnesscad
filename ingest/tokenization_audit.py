"""Fidelity and overflow gates for continuous B-rep tokenization."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TokenizationAudit:
    segment_count: int
    max_segments: int
    max_deviation: float
    trim_deviation: float
    max_join_gap: float
    tolerance: float
    issues: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return not self.issues


def audit_tokenization(*, reference_points=(), encoded_points=(), joins=(),
                       trim_deviation: float = 0.0, segment_count: int,
                       max_segments: int = 100, tolerance: float = 1e-6):
    if tolerance <= 0 or max_segments <= 0 or segment_count < 0:
        raise ValueError("invalid audit limits")
    if len(reference_points) != len(encoded_points):
        raise ValueError("reference and encoded samples must align")
    deviations = [math.dist(tuple(a), tuple(b))
                  for a, b in zip(reference_points, encoded_points)]
    gaps = [math.dist(tuple(a), tuple(b)) for a, b in joins]
    maximum, join_gap = max(deviations, default=0.0), max(gaps, default=0.0)
    issues = []
    if segment_count > max_segments:
        issues.append("sequence-overflow")
    if maximum > tolerance:
        issues.append("geometry-deviation")
    if trim_deviation > tolerance:
        issues.append("trim-deviation")
    if join_gap > tolerance:
        issues.append("continuity-gap")
    return TokenizationAudit(segment_count, max_segments, maximum, trim_deviation,
                             join_gap, tolerance, tuple(issues))
