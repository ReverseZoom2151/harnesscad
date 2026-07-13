"""Generic hierarchy and task-linked visual anomaly assets."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class VisionTask(str, Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"


@dataclass(frozen=True)
class Box:
    x1: float; y1: float; x2: float; y2: float
    label: str


@dataclass(frozen=True)
class Mask:
    label: str
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class AnomalyAsset:
    id: str
    domain: str
    system: str
    part: str
    anomaly: str
    tasks: frozenset[VisionTask]
    width: int
    height: int
    source: str
    source_kind: str = "real"
    normal: bool = False
    labels: tuple[str, ...] = ()
    boxes: tuple[Box, ...] = ()
    masks: tuple[Mask, ...] = ()
    group_id: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if not all((self.id, self.domain, self.system, self.part, self.source)):
            raise ValueError("identity, hierarchy, and source are required")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        if self.source_kind not in {"real", "synthetic", "converted"}:
            raise ValueError("invalid source kind")
        if VisionTask.DETECTION in self.tasks and not self.normal and not self.boxes:
            raise ValueError("detection asset requires boxes")
        if VisionTask.SEGMENTATION in self.tasks and not self.normal and not self.masks:
            raise ValueError("segmentation asset requires masks")
        if self.normal and self.anomaly not in {"", "normal"}:
            raise ValueError("normal asset cannot declare an anomaly")


def validate_hierarchy(asset: AnomalyAsset, parents: Mapping[str, str]) -> tuple[str, ...]:
    expected = ((asset.system, asset.domain), (asset.part, asset.system))
    return tuple(f"parent:{child}" for child, parent in expected
                 if parents.get(child) != parent)
