"""Release gate over externally detected sensitive image regions."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable


SENSITIVE=frozenset({"face","person","license_plate","logo","location"})

@dataclass(frozen=True)
class PrivacyRegion:
    kind: str; redacted: bool; confidence: float=1.0

@dataclass(frozen=True)
class PrivacyDecision:
    releasable: bool; reasons: tuple[str,...]


def release_gate(regions: Iterable[PrivacyRegion], *, manually_verified: bool) -> PrivacyDecision:
    reasons=[]
    for i,r in enumerate(regions):
        if r.kind in SENSITIVE and not r.redacted: reasons.append(f"unredacted:{i}:{r.kind}")
    if not manually_verified: reasons.append("manual_verification_required")
    return PrivacyDecision(not reasons,tuple(reasons))
