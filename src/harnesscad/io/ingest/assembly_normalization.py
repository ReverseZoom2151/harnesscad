"""One invertible condition-derived transform shared by an assembly pair."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class AssemblyTransform:
    center: tuple[float,float,float]; scale: float; extent: float=3
    def __post_init__(self):
        if self.scale<=0 or self.extent<=0:raise ValueError("invalid transform")
    def apply(self,p):return tuple((x-c)*self.scale for x,c in zip(p,self.center))
    def invert(self,p):return tuple(x/self.scale+c for x,c in zip(p,self.center))
def fit_condition_transform(points,extent=3):
    pts=tuple(points)
    if not pts:raise ValueError("condition points required")
    mins=tuple(min(p[i] for p in pts) for i in range(3));maxs=tuple(max(p[i] for p in pts) for i in range(3))
    center=tuple((a+b)/2 for a,b in zip(mins,maxs));radius=max((b-a)/2 for a,b in zip(mins,maxs))
    return AssemblyTransform(center,extent/(radius or 1),extent)
