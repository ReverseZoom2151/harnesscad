"""Decompose CAD annotation into scoped, auditable worker tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence


@dataclass(frozen=True)
class AnnotationTask:
    task_id: str
    artifact_id: str
    kind: str
    prompt: str
    requires_expert: bool
    evidence_fields: tuple[str, ...]


def decompose_annotation(
    artifact_id: str,
    available_fields: Iterable[str],
    *,
    include_intent: bool = True,
) -> List[AnnotationTask]:
    """Create independent tasks whose answers can be consensus-checked."""
    if not artifact_id:
        raise ValueError("artifact_id is required")
    fields = {str(field) for field in available_fields}
    specs = [
        ("family", "Classify the mechanical part family.", False, ("thumbnail",)),
        ("process", "Choose the likely manufacturing process.", True,
         ("geometry", "material")),
        ("features", "Label recognizable CAD feature types.", False, ("op_stream",)),
        ("quality", "Flag invalid, degenerate or suspicious geometry.", True,
         ("geometry", "verifier_report")),
    ]
    if include_intent:
        specs.append((
            "intent", "Describe the functional design intent and supporting evidence.",
            True, ("op_stream", "requirements"),
        ))
    tasks = []
    for index, (kind, prompt, expert, required) in enumerate(specs, 1):
        usable = tuple(field for field in required if field in fields)
        if not usable:
            continue
        tasks.append(AnnotationTask(
            f"{artifact_id}:{index:02d}:{kind}", artifact_id, kind, prompt,
            expert, usable,
        ))
    return tasks
