"""Injected individual-description and directional assembly-prompt fusion."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class AssemblyCaption:
    condition: str; target: str; prompt: str; reverse_prompt: str; consistent: bool
def caption_assembly(condition,target,*,describe,fuse):
    a,b=describe(condition),describe(target)
    forward=fuse(a,b);reverse=fuse(b,a)
    if not forward.strip() or not reverse.strip():raise ValueError("empty prompt")
    return AssemblyCaption(a,b,forward.strip(),reverse.strip(),forward.strip()!=reverse.strip())
