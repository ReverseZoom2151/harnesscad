"""Neutral DXF-like document contract; concrete serializers remain optional."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping, Protocol


@dataclass(frozen=True)
class Layer:
    name: str
    color: int = 7
    line_type: str = "CONTINUOUS"

    def __post_init__(self):
        if not self.name or not 1 <= self.color <= 255:
            raise ValueError("invalid layer")


@dataclass(frozen=True)
class Entity:
    kind: str
    values: Mapping[str, object]
    layer: str = "0"


@dataclass(frozen=True)
class DraftAnnotation:
    kind: str
    entity_ids: tuple[str, ...]
    value: float | str
    unit: str = ""


@dataclass(frozen=True)
class DxfDocument:
    units: str
    layers: tuple[Layer, ...]
    entities: Mapping[str, Entity]
    annotations: tuple[DraftAnnotation, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.units not in {"mm", "cm", "m", "in", "ft"}:
            raise ValueError("unsupported units")
        names = {layer.name for layer in self.layers} | {"0"}
        if any(entity.layer not in names for entity in self.entities.values()):
            raise ValueError("entity references unknown layer")
        ids = set(self.entities)
        if any(not set(note.entity_ids) <= ids for note in self.annotations):
            raise ValueError("annotation references unknown entity")


class DxfSerializer(Protocol):
    def serialize(self, document: DxfDocument) -> bytes: ...


class DxfParser(Protocol):
    def parse(self, payload: bytes) -> DxfDocument: ...
