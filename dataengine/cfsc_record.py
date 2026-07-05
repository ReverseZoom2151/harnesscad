"""Prompt/script/artifact records and lineage-safe split auditing."""

from __future__ import annotations
from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class CFSCRecord:
    id: str; prompt: str; script: str; artifact_digest: str
    family: str; template_digest: str; parameters: tuple[tuple[str,object],...]
    dimension: str; annotation_mode: str; legal: bool; built: bool; roundtrip: bool
    split: str=""

    def __post_init__(self):
        if self.dimension not in {"2d","3d"}:raise ValueError("dimension must be 2d/3d")
        if self.annotation_mode not in {"none","drafting"}:raise ValueError("bad annotation mode")

    @property
    def prompt_digest(self):return hashlib.sha256(" ".join(self.prompt.casefold().split()).encode()).hexdigest()


def audit_leakage(records):
    seen={}; leaks=[]
    for r in records:
        for key in (r.prompt_digest,r.template_digest,r.artifact_digest):
            prior=seen.setdefault(key,r.split)
            if prior!=r.split:leaks.append((key,prior,r.split))
    return tuple(sorted(set(leaks)))
