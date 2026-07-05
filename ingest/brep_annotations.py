"""Persistent, kernel-independent annotations for B-rep entities."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from typing import Iterable, Literal, Mapping, Sequence

Point3 = tuple[float, float, float]
EntityKind = Literal["face", "edge", "vertex"]


def _quantized(values: Sequence[float], tolerance: float) -> tuple[int, ...]:
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    return tuple(round(value / tolerance) for value in values)


@dataclass(frozen=True)
class LocalFrame:
    origin: Point3
    x_axis: Point3
    y_axis: Point3
    z_axis: Point3

    def __post_init__(self) -> None:
        for axis in (self.x_axis, self.y_axis, self.z_axis):
            if math.isclose(sum(v * v for v in axis), 0.0):
                raise ValueError("frame axes must be non-zero")


@dataclass(frozen=True)
class EntityRecord:
    """Portable entity summary supplied by any geometry kernel adapter."""

    kind: EntityKind
    anchor: Point3
    geometry_signature: tuple[float, ...]
    topology_signature: tuple[int, ...]
    samples: tuple[Point3, ...] = ()
    source_id: str | None = None

    def signature(self, tolerance: float = 1e-6) -> tuple[object, ...]:
        return (
            self.kind,
            _quantized(self.geometry_signature, tolerance),
            self.topology_signature,
        )

    def derived_id(self, tolerance: float = 1e-6) -> str:
        payload = json.dumps(self.signature(tolerance), separators=(",", ":"))
        return f"{self.kind[0]}_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"

    def distance_to(self, point: Point3) -> float:
        candidates = (self.anchor, *self.samples)
        return min(math.dist(point, candidate) for candidate in candidates)


@dataclass(frozen=True)
class EntityAnnotation:
    entity_id: str
    kind: EntityKind
    labels: tuple[str, ...] = ()
    frame: LocalFrame | None = None
    attributes: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ExternalTag:
    point: Point3
    labels: tuple[str, ...]
    source: str = ""


@dataclass(frozen=True)
class Assignment:
    tag_index: int
    entity_id: str
    distance: float
    threshold: float


@dataclass(frozen=True)
class AnnotationIssue:
    code: Literal["unassigned", "ambiguous", "conflict"]
    tag_indices: tuple[int, ...]
    entity_ids: tuple[str, ...]
    message: str


@dataclass(frozen=True)
class AssignmentResult:
    assignments: tuple[Assignment, ...]
    annotations: tuple[EntityAnnotation, ...]
    issues: tuple[AnnotationIssue, ...]


def persist_entity_ids(
    previous: Mapping[str, EntityRecord],
    current: Iterable[EntityRecord],
    *,
    signature_tolerance: float = 1e-6,
) -> dict[str, EntityRecord]:
    """Carry IDs forward by signature; derive deterministic IDs for new entities."""
    prior_by_signature: dict[tuple[object, ...], list[str]] = {}
    for entity_id, entity in previous.items():
        prior_by_signature.setdefault(entity.signature(signature_tolerance), []).append(entity_id)
    result: dict[str, EntityRecord] = {}
    for entity in current:
        candidates = prior_by_signature.get(entity.signature(signature_tolerance), [])
        unused = [candidate for candidate in sorted(candidates) if candidate not in result]
        entity_id = unused[0] if unused else entity.derived_id(signature_tolerance)
        if entity_id in result:
            suffix = 2
            base = entity_id
            while f"{base}_{suffix}" in result:
                suffix += 1
            entity_id = f"{base}_{suffix}"
        result[entity_id] = entity
    return result


def assign_external_tags(
    entities: Mapping[str, EntityRecord],
    tags: Sequence[ExternalTag],
    *,
    thresholds: Sequence[float] = (1e-4, 1e-3, 1e-2),
    ambiguity_epsilon: float = 1e-9,
    existing: Mapping[str, EntityAnnotation] | None = None,
) -> AssignmentResult:
    """Assign tags at the first threshold that reaches an unambiguous entity."""
    ordered_thresholds = tuple(sorted(set(thresholds)))
    if not ordered_thresholds or ordered_thresholds[0] < 0:
        raise ValueError("thresholds must contain non-negative values")
    assignments: list[Assignment] = []
    issues: list[AnnotationIssue] = []
    labels_by_entity: dict[str, list[str]] = {
        entity_id: list(annotation.labels)
        for entity_id, annotation in (existing or {}).items()
    }

    for index, tag in enumerate(tags):
        distances = sorted(
            ((entity.distance_to(tag.point), entity_id) for entity_id, entity in entities.items()),
            key=lambda item: (item[0], item[1]),
        )
        assigned = False
        for threshold in ordered_thresholds:
            reached = [item for item in distances if item[0] <= threshold]
            if not reached:
                continue
            nearest = reached[0][0]
            tied = tuple(entity_id for distance, entity_id in reached
                         if abs(distance - nearest) <= ambiguity_epsilon)
            if len(tied) > 1:
                issues.append(AnnotationIssue(
                    "ambiguous", (index,), tied,
                    f"tag {index} is equidistant from {len(tied)} entities",
                ))
                assigned = True
                break
            entity_id = tied[0]
            assignments.append(Assignment(index, entity_id, nearest, threshold))
            current_labels = labels_by_entity.setdefault(entity_id, [])
            overlap = set(current_labels) & set(tag.labels)
            if current_labels and tag.labels and not overlap:
                issues.append(AnnotationIssue(
                    "conflict", (index,), (entity_id,),
                    f"tag {index} labels conflict with existing labels on {entity_id}",
                ))
            for label in tag.labels:
                if label not in current_labels:
                    current_labels.append(label)
            assigned = True
            break
        if not assigned:
            issues.append(AnnotationIssue(
                "unassigned", (index,), (),
                f"tag {index} is outside the largest proximity threshold",
            ))

    annotations: list[EntityAnnotation] = []
    for entity_id in sorted(labels_by_entity):
        prior = (existing or {}).get(entity_id)
        entity = entities.get(entity_id)
        kind = entity.kind if entity else prior.kind  # type: ignore[union-attr]
        annotations.append(replace(
            prior or EntityAnnotation(entity_id, kind),
            labels=tuple(labels_by_entity[entity_id]),
        ))
    return AssignmentResult(tuple(assignments), tuple(annotations), tuple(issues))
