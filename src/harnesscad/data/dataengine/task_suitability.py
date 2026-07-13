"""Auditable routing by visibility and annotation availability."""

from __future__ import annotations
from dataclasses import dataclass
from harnesscad.data.dataengine.anomaly_schema import VisionTask


@dataclass(frozen=True)
class TaskRoute:
    tasks: frozenset[VisionTask]; reasons: tuple[str,...]


def route_tasks(*, visibility: float, has_boxes=False, has_masks=False) -> TaskRoute:
    if not 0<=visibility<=1: raise ValueError("visibility must be in [0,1]")
    tasks={VisionTask.CLASSIFICATION}; reasons=["classification_baseline"]
    if visibility>=.4 and has_boxes: tasks.add(VisionTask.DETECTION); reasons.append("visible_boxes")
    if visibility>=.7 and has_masks: tasks.add(VisionTask.SEGMENTATION); reasons.append("high_visibility_masks")
    return TaskRoute(frozenset(tasks),tuple(reasons))
