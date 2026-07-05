"""Directed assembly pair records, reversal, and duplicate audits."""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Mapping

@dataclass(frozen=True)
class AssemblyPairRecord:
    id: str; source_pair_id: str; condition_id: str; target_id: str
    condition_faces: tuple[str,...]; target_faces: tuple[str,...]
    prompt: str; split: str; transform: Mapping[str,object]
    source: str=""; license: str=""; mapping_kind: str="one_to_one"
    def __post_init__(self):
        if self.mapping_kind not in {"one_to_one","ambiguous"}:raise ValueError("mapping kind")

def reverse_pair(record,new_id):
    return replace(record,id=new_id,condition_id=record.target_id,target_id=record.condition_id,
                   condition_faces=record.target_faces,target_faces=record.condition_faces)
def audit_pairs(records):
    seen={};duplicates=[];leaks=[]
    for r in records:
        key=(r.condition_id,r.target_id,tuple(sorted(r.condition_faces)),tuple(sorted(r.target_faces)))
        if key in seen:duplicates.append((seen[key],r.id))
        else:seen[key]=r.id
        lineage=tuple(sorted((r.condition_id,r.target_id)))
        owner=seen.setdefault(("split",lineage),r.split)
        if owner!=r.split:leaks.append(r.source_pair_id)
    return {"duplicates":tuple(sorted(duplicates)),"split_leakage":tuple(sorted(set(leaks)))}
