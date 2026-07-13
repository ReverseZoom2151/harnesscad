"""Deterministic suppression and scoring for scan-point B-rep labels."""
from dataclasses import dataclass
from math import dist

@dataclass(frozen=True)
class PointPrediction:
    id: str
    point: tuple[float, float, float]
    confidence: float
    label: str

def suppress(points, radius):
    if radius < 0: raise ValueError("radius must be non-negative")
    kept = []
    for item in sorted(points, key=lambda x: (-x.confidence, x.id)):
        if all(dist(item.point, other.point) > radius for other in kept):
            kept.append(item)
    return tuple(kept)

def boundary_first(boundaries, junctions, radius):
    b = suppress(boundaries, radius)
    eligible = [j for j in junctions if any(dist(j.point, x.point) <= radius for x in b)]
    return b, suppress(eligible, radius)

def precision_recall_f1(actual, expected):
    a, e = set(actual), set(expected); n = len(a & e)
    p = n/len(a) if a else float(not e); r = n/len(e) if e else float(not a)
    return {"precision": p, "recall": r, "f1": 2*p*r/(p+r) if p+r else 0.0}
