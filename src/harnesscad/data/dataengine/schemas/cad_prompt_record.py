"""CAD-image prompt lineage, configuration, ratings, and license audit."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping

@dataclass(frozen=True)
class CADPromptRecord:
    id: str; prompt: str; render_id: str; render_source: str; render_license: str
    provider: str; weight: float; seed: int; config: Mapping[str,object]
    output_digest: str; split: str
def audit_prompt_records(records):
    owners={};leaks=[];licenses=[]
    for r in records:
        old=owners.setdefault(r.render_id,r.split)
        if old!=r.split:leaks.append(r.render_id)
        if not r.render_license:licenses.append(r.id)
    return {"render_leakage":tuple(sorted(set(leaks))),"missing_license":tuple(sorted(licenses))}
