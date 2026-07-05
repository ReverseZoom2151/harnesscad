"""Typed five-stage CAD planning and strict reasoning/code envelopes."""
from __future__ import annotations
from dataclasses import dataclass
import re

STAGES=("description","coordinates","sketch","extrusion","implementation")

@dataclass(frozen=True)
class CADPlan:
    description: str; coordinates: str; sketch: str; extrusion: str; implementation: str
    def __post_init__(self):
        if any(not getattr(self,s).strip() for s in STAGES): raise ValueError("all planning stages are required")

@dataclass(frozen=True)
class CADEnvelope:
    plan: CADPlan; code: str

_ENV=re.compile(r"^\s*<think>(.*?)</think>\s*```python\s*(.*?)\s*```\s*$",re.S|re.I)

def parse_envelope(text: str) -> CADEnvelope:
    match=_ENV.match(text)
    if not match: raise ValueError("expected think block followed by python fence")
    sections={}
    for line in match.group(1).splitlines():
        if ":" in line:
            key,value=line.split(":",1); key=key.strip().casefold()
            if key in STAGES: sections[key]=value.strip()
    if set(sections)!=set(STAGES): raise ValueError("missing or duplicate planning stages")
    code=match.group(2).strip()
    if not code: raise ValueError("code block is empty")
    return CADEnvelope(CADPlan(**sections),code)
