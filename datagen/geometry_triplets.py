"""Select executable minimum-distance code and assign geometric quality tiers."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib

@dataclass(frozen=True)
class GeometryTriplet:
    id: str; prompt: str; code: str; target_digest: str; cd: float
    tier: str; candidate_index: int; generator: str

def quality_tier(cd):
    return "high" if cd<1e-4 else ("medium" if cd<1e-3 else "hard")

def select_triplet(prompt,candidates,target,*,execute,distance,generator="injected"):
    scored=[]
    for i,code in enumerate(candidates):
        try: shape=execute(code); cd=float(distance(shape,target))
        except Exception: continue
        scored.append((cd,i,code))
    if not scored:return None
    cd,i,code=min(scored,key=lambda x:(x[0],x[1]))
    target_digest=hashlib.sha256(repr(target).encode()).hexdigest()
    identity=hashlib.sha256(f"{prompt}\0{code}\0{target_digest}".encode()).hexdigest()
    return GeometryTriplet(identity,prompt,code,target_digest,cd,quality_tier(cd),i,generator)
