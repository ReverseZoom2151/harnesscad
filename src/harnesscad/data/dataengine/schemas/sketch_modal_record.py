"""Lineage-safe paired primitive/image sketch records."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class SketchModalRecord:
    id: str; parent_id: str; full_entities: tuple; partial_entities: tuple
    full_render_digest: str; partial_render_digest: str
    prefix_ratio: float; constraints: tuple; units: str; frame: object
    condition: str; split: str
    def __post_init__(self):
        if not 0<=self.prefix_ratio<=1:raise ValueError("prefix ratio")
        if self.partial_entities!=self.full_entities[:len(self.partial_entities)]:
            raise ValueError("partial entities must be an ordered prefix")
        if len(self.partial_entities)>len(self.full_entities):raise ValueError("oversized prefix")
