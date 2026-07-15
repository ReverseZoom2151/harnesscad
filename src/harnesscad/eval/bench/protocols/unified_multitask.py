"""Unified multi-task, multi-modal CAD scorecard and CAD-QA metric (UniCAD).

Mined from *UniCAD: A Unified Benchmark and Universal Model for Multi-Modal
Multi-Task CAD*. UniCAD's model is trained, but its **benchmark protocol** is a
deterministic aggregation: it defines standardized tasks -- ``textCAD``,
``image/sketchCAD``, ``point-cloudCAD`` and ``CAD QA`` -- each with a task-specific
metric, and reports a single unified comparison across them.

This module ports:

*   :data:`UNICAD_TASKS` -- the task set, each with a metric name and orientation
    (higher- or lower-is-better);
*   :func:`normalise_score` -- fold a raw task score into ``[0, 1]`` (lower-is-better
    metrics like Chamfer distance are inverted through a supplied worst-case cap);
*   :func:`unified_score` -- the mean normalised score across tasks; and
*   :func:`cad_qa_accuracy` -- exact-match accuracy for the CAD-QA task, over
    whitespace/case-normalised answers.

Deterministic, stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

__all__ = [
    "UNICAD_TASKS",
    "TaskSpec",
    "normalise_score",
    "unified_score",
    "cad_qa_accuracy",
]


@dataclass(frozen=True)
class TaskSpec:
    """A UniCAD task: its name, the metric it reports, and metric orientation."""

    name: str
    metric: str
    higher_is_better: bool


#: The four UniCAD task families (paper Sec. 1).
UNICAD_TASKS: Tuple[TaskSpec, ...] = (
    TaskSpec("textcad", "chamfer_distance", higher_is_better=False),
    TaskSpec("imagecad", "chamfer_distance", higher_is_better=False),
    TaskSpec("sketchcad", "chamfer_distance", higher_is_better=False),
    TaskSpec("pointcloudcad", "chamfer_distance", higher_is_better=False),
    TaskSpec("cadqa", "accuracy", higher_is_better=True),
)

_TASKS_BY_NAME = {t.name: t for t in UNICAD_TASKS}


def normalise_score(task: str, raw: float, worst: float = 1.0) -> float:
    """Fold a raw task score into ``[0, 1]`` (1 = best).

    For higher-is-better metrics, the raw score is clamped to ``[0, worst]`` and
    divided by ``worst``. For lower-is-better metrics (e.g. Chamfer distance), it is
    inverted: ``1 - min(raw, worst)/worst``. ``worst`` is the worst tolerated value.
    """
    if task not in _TASKS_BY_NAME:
        raise ValueError(f"unknown task {task!r}")
    if worst <= 0:
        raise ValueError("worst must be positive")
    if raw < 0:
        raise ValueError("raw score must be non-negative")
    spec = _TASKS_BY_NAME[task]
    capped = min(raw, worst)
    if spec.higher_is_better:
        return capped / worst
    return 1.0 - capped / worst


def unified_score(
    raw_scores: Mapping[str, float], worst: Mapping[str, float]
) -> float:
    """Mean normalised score across all reported tasks (the unified comparison).

    Only tasks present in ``raw_scores`` are aggregated; each needs a ``worst`` cap.
    """
    if not raw_scores:
        raise ValueError("need at least one task score")
    total = 0.0
    for task, raw in raw_scores.items():
        if task not in worst:
            raise ValueError(f"no worst-case cap supplied for task {task!r}")
        total += normalise_score(task, raw, worst[task])
    return total / len(raw_scores)


def _norm_answer(a: str) -> str:
    return " ".join(a.strip().lower().split())


def cad_qa_accuracy(
    predictions: Sequence[str], references: Sequence[str]
) -> float:
    """Exact-match accuracy for CAD-QA over case/whitespace-normalised answers."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must be the same length")
    if not predictions:
        raise ValueError("need at least one QA pair")
    hits = sum(1 for p, r in zip(predictions, references)
               if _norm_answer(p) == _norm_answer(r))
    return hits / len(predictions)
