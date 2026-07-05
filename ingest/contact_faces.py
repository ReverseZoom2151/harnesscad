"""Bidirectional sampled contact evidence over injected parametric faces."""
from __future__ import annotations
from dataclasses import dataclass
from math import dist

@dataclass(frozen=True)
class ContactEvidence:
    left_id: str; right_id: str; left_support: int; right_support: int
    samples_per_side: int; min_distance: float; contact: bool

def _support(samples,target,projector,tolerance,normal_dot):
    supported=0;minimum=float("inf")
    for point,normal in samples:
        closest,target_normal=projector(target,point);d=dist(point,closest);minimum=min(minimum,d)
        dot=sum(a*b for a,b in zip(normal,target_normal))
        supported+=d<=tolerance and dot<normal_dot
    return supported,minimum
def contact_evidence(left_id,left,right_id,right,*,sampler,projector,
                     tolerance=.1,normal_dot=0,min_support=1):
    if tolerance<0 or min_support<1:raise ValueError("invalid contact policy")
    left_samples,right_samples=tuple(sampler(left)),tuple(sampler(right))
    ls,ld=_support(left_samples,right,projector,tolerance,normal_dot)
    rs,rd=_support(right_samples,left,projector,tolerance,normal_dot)
    count=max(len(left_samples),len(right_samples))
    return ContactEvidence(left_id,right_id,ls,rs,count,min(ld,rd),
                           ls>=min_support or rs>=min_support)
