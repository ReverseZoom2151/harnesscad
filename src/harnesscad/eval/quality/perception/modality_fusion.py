"""Policy-only modality fusion with auditable missing/conflict routes."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class FusionDecision:
    route: str; modalities: tuple[str,...]; reasons: tuple[str,...]

def fusion_policy(inputs,*,required=(),conflicts=()):
    present=tuple(sorted(k for k,v in inputs.items() if v is not None))
    reasons=[f"missing:{x}" for x in required if x not in present]
    reasons.extend(f"conflict:{a}/{b}" for a,b in conflicts
                   if inputs.get(a) is not None and inputs.get(b) is not None and inputs[a]!=inputs[b])
    return FusionDecision("manual_review" if reasons else "fuse",present,tuple(sorted(reasons)))
