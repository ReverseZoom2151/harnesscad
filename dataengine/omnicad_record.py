"""Aligned multimodal CAD records with explicit frames and provenance."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping

@dataclass(frozen=True)
class ViewAsset:
    name: str; camera: tuple[float,...]; digest: str
@dataclass(frozen=True)
class PointNormal:
    point: tuple[float,float,float]; normal: tuple[float,float,float]
@dataclass(frozen=True)
class OmniCADRecord:
    id: str; command_digest: str; parent_id: str; text: str
    views: tuple[ViewAsset,...]; points: tuple[PointNormal,...]
    units: str; coordinate_frame: str; split: str
    provenance: Mapping[str,str]=field(default_factory=dict)
    def __post_init__(self):
        if not self.id or not self.command_digest or not self.units or not self.coordinate_frame:
            raise ValueError("identity, commands, units and frame are required")
        if len({v.name for v in self.views})!=len(self.views):raise ValueError("duplicate view names")
    @property
    def modalities(self):
        return frozenset(name for name,present in
                         (("text",bool(self.text)),("image",bool(self.views)),("point",bool(self.points)))
                         if present)
