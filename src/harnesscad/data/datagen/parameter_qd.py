"""Quality-diversity archive for injected parametric CAD evaluators."""
from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class ArchiveEntry:
    parameters: tuple[float, ...]
    descriptor: tuple[float, ...]
    evidence: dict

class NoveltyArchive:
    def __init__(self, epsilon):
        if epsilon < 0: raise ValueError("epsilon must be non-negative")
        self.epsilon = float(epsilon); self.entries = []
    def consider(self, parameters, evaluate):
        evidence = dict(evaluate(tuple(parameters)))
        reason = _quality_reason(evidence)
        descriptor = tuple(map(float, evidence.get("descriptor", ())))
        if not reason and any(_distance(descriptor, x.descriptor) < self.epsilon
                              for x in self.entries): reason = "not-novel"
        accepted = not reason
        if accepted: self.entries.append(ArchiveEntry(tuple(parameters), descriptor, evidence))
        return {"accepted": accepted, "reason": reason, "evidence": evidence}

def _distance(a, b):
    if len(a) != len(b): return math.inf
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

def _quality_reason(e):
    if not e.get("valid"): return "invalid"
    if e.get("solid_count") != 1: return "not-one-solid"
    if not e.get("watertight"): return "not-watertight"
    bounds=e.get("bounds")
    if not bounds or len(bounds)!=6: return "missing-bounds"
    extent=max(bounds[3+i]-bounds[i] for i in range(3))
    if not 60 <= extent <= 200: return "extent-out-of-range"
    if any(bounds[i] < -100 or bounds[3+i] > 100 for i in range(3)):
        return "outside-cube"
    return ""

def fill_archive(propose, evaluate, *, target=15, budget=100, epsilon=0.0):
    archive=NoveltyArchive(epsilon); attempts=[]
    for index in range(budget):
        if len(archive.entries) >= target: break
        params=tuple(propose(index)); attempts.append(archive.consider(params,evaluate))
    return {"entries": tuple(archive.entries), "attempts": tuple(attempts),
            "termination": "target" if len(archive.entries)>=target else "budget"}
