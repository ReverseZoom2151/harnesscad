"""Step-scheduled validity-first neighbor selection with Pareto evidence."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class GuidedCandidate:
    index: int; value: object; valid: bool; geometry: float; regularization: float; score: float
def guided_step(current,step,*,guidance_steps,neighbors,is_valid,geometry,regularization,omega=1):
    if step not in set(guidance_steps):return current,()
    rows=[]
    for i,value in enumerate(neighbors(current,step)):
        g=float(geometry(value));r=float(regularization(value,current))
        rows.append(GuidedCandidate(i,value,bool(is_valid(value)),g,r,g+omega*r))
    if not rows:return current,()
    winner=min(rows,key=lambda x:(not x.valid,x.score,x.index))
    return winner.value,tuple(rows)
def pareto_evidence(rows):
    values=tuple(rows)
    return tuple(x for x in values if x.valid and not any(
        y.valid and y.geometry<=x.geometry and y.regularization<=x.regularization and
        (y.geometry<x.geometry or y.regularization<x.regularization)
        for y in values if y is not x))
