"""Validated drafting annotations with stable entity-reference remapping."""

from __future__ import annotations
from dataclasses import dataclass, replace
from math import isfinite
from typing import Mapping


@dataclass(frozen=True)
class Annotation:
    id: str
    entity_ids: tuple[str, ...]
    value: float | str
    unit: str = ""

    def __post_init__(self):
        if not self.id or not self.entity_ids:
            raise ValueError("annotation identity and references are required")
        if isinstance(self.value, float) and not isfinite(self.value):
            raise ValueError("annotation value must be finite")


@dataclass(frozen=True)
class Linear(Annotation): pass
@dataclass(frozen=True)
class Radius(Annotation): pass
@dataclass(frozen=True)
class Angle(Annotation): pass
@dataclass(frozen=True)
class Tolerance(Annotation):
    lower: float = 0.0
    upper: float = 0.0
    def __post_init__(self):
        super().__post_init__()
        if self.lower > self.upper: raise ValueError("lower tolerance exceeds upper")
@dataclass(frozen=True)
class ChamferCallout(Annotation): pass
@dataclass(frozen=True)
class SurfaceRoughness(Annotation):
    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.value, (int, float)) or self.value < 0:
            raise ValueError("roughness must be non-negative")


def remap_annotations(items, entity_map: Mapping[str, str], *, drop_missing=False):
    out=[]
    for item in items:
        if any(entity not in entity_map for entity in item.entity_ids):
            if drop_missing: continue
            raise KeyError(f"missing entity mapping for {item.id}")
        out.append(replace(item, entity_ids=tuple(entity_map[x] for x in item.entity_ids)))
    return tuple(out)
