"""Frontend-neutral confidence overlays for CAD findings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class ConfidenceOverlay:
    target: str
    confidence: float
    level: str
    label: str
    reason: str
    source: str

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "confidence": self.confidence,
            "level": self.level,
            "label": self.label,
            "reason": self.reason,
            "source": self.source,
        }


def confidence_level(value: float) -> str:
    if not 0.0 <= value <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if value < 0.5:
        return "low"
    if value < 0.8:
        return "medium"
    return "high"


def build_overlays(findings: Iterable[Mapping[str, object]]) -> List[ConfidenceOverlay]:
    """Normalize verifier/model findings into stable presentation records."""
    overlays = []
    for finding in findings:
        confidence = float(finding.get("confidence", 1.0))
        target = str(
            finding.get("target") or finding.get("where") or finding.get("feature_id")
            or "model"
        )
        label = str(finding.get("label") or finding.get("code") or "finding")
        reason = str(finding.get("reason") or finding.get("message") or "")
        source = str(finding.get("source") or "verifier")
        overlays.append(ConfidenceOverlay(
            target, confidence, confidence_level(confidence), label, reason, source
        ))
    return sorted(
        overlays,
        key=lambda item: (item.target, item.label, item.source, item.confidence),
    )
