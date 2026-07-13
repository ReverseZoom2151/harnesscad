"""Configurable KnitCAD-style complexity filters and rejection counts."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class KnitLimits:
    max_faces: int=70; max_contacts: int=10; max_edges_per_face: int=40; require_single_solid: bool=True
def filter_record(record,limits=KnitLimits()):
    reasons=[]
    if record["faces"]>limits.max_faces:reasons.append("too-many-faces")
    if record["contacts"]>limits.max_contacts:reasons.append("too-many-contacts")
    if record["max_edges_per_face"]>limits.max_edges_per_face:reasons.append("too-many-edges")
    if limits.require_single_solid and record["solids"]!=1:reasons.append("multiple-solids")
    return tuple(reasons)
def rejection_distribution(records,limits=KnitLimits()):
    counts={}
    for record in records:
        for reason in filter_record(record,limits):counts[reason]=counts.get(reason,0)+1
    return dict(sorted(counts.items()))
