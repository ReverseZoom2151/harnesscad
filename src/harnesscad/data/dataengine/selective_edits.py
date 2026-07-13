"""Candidate evidence and deterministic human selection records."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class EditCandidateRecord:
    id: str; sequence: object; render_digest: str; valid: bool; score: float
@dataclass(frozen=True)
class SelectiveEdit:
    source_id: str; instruction: str; candidates: tuple[EditCandidateRecord,...]
    selected_id: str; reviewers: tuple[str,...]

def create_selection(source_id,instruction,candidates,selected_id,reviewers):
    ordered=tuple(sorted(candidates,key=lambda x:x.id))
    if selected_id not in {x.id for x in ordered}:raise ValueError("selected candidate absent")
    if len(set(reviewers))<1:raise ValueError("reviewer required")
    return SelectiveEdit(source_id,instruction,ordered,selected_id,tuple(sorted(set(reviewers))))
