"""CoT/code/geometry provenance and split leakage audit."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib

@dataclass(frozen=True)
class CoTRecord:
    id: str; prompt: str; plan: tuple[tuple[str,str],...]; code: str
    geometry_digest: str; cd: float; executable: bool; manually_reviewed: bool
    split: str; source_id: str
    @property
    def lineage_digest(self):
        return hashlib.sha256(f"{self.source_id}\0{self.geometry_digest}".encode()).hexdigest()

def cot_leakage(records):
    seen={}; out=[]
    for r in records:
        for key in (r.source_id,r.geometry_digest,r.lineage_digest):
            old=seen.setdefault(key,r.split)
            if old!=r.split:out.append(key)
    return tuple(sorted(set(out)))
