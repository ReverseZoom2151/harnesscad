"""Evidence-calibrated stage policy that refuses cross-domain extrapolation."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class StageRecommendation:
    weight: float; evidence_key: tuple[str,str,str]; reason: str
def recommend_stage(stage,provider,family,evidence):
    key=(stage,provider,family)
    rows=evidence.get(key)
    if not rows:raise LookupError("no calibration for stage/provider/family")
    best=max(rows,key=lambda x:(x["objective"],-x["weight"]))
    return StageRecommendation(best["weight"],key,"best calibrated objective")
