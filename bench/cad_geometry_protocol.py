"""Versioned normalization, squared Chamfer, and corpus aggregates."""
from __future__ import annotations
from dataclasses import dataclass
from math import dist
from statistics import median

@dataclass(frozen=True)
class GeometryProtocol:
    sample_count: int=2048; seed: int=0; center: bool=True; unit_scale: bool=True
    version: str="cad-geometry-v1"
    def __post_init__(self):
        if self.sample_count<=0 or not self.version: raise ValueError("invalid protocol")

def normalize(points, protocol=GeometryProtocol()):
    pts=[tuple(map(float,p)) for p in points]
    if not pts:return ()
    dims=len(pts[0]); center=tuple(sum(p[i] for p in pts)/len(pts) for i in range(dims))
    shifted=[tuple(p[i]-center[i] for i in range(dims)) for p in pts] if protocol.center else pts
    scale=max((sum(v*v for v in p)**.5 for p in shifted),default=1) if protocol.unit_scale else 1
    scale=scale or 1
    return tuple(tuple(v/scale for v in p) for p in shifted)

def squared_chamfer(a,b):
    x,y=tuple(a),tuple(b)
    if not x or not y:return None
    directed=lambda p,q:sum(min(dist(i,j)**2 for j in q) for i in p)/len(p)
    return directed(x,y)+directed(y,x)

def evaluate_geometry(outcomes):
    values=tuple(outcomes); valid=[float(v) for v in values if v is not None]
    return {"n":len(values),"valid":len(valid),
            "invalidity_ratio":(len(values)-len(valid))/len(values) if values else None,
            "mean_cd":sum(valid)/len(valid) if valid else None,
            "median_cd":median(valid) if valid else None}
