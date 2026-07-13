"""Deterministic edit-context selection and sampled orthographic projections."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from harnesscad.domain.editing.brep import FaceDescriptor, canonicalize_faces


def select_edit_context(
    faces: Iterable[FaceDescriptor],
    selected: Iterable[int],
    adjacency: Mapping[int, Iterable[int]],
    *,
    rings: int = 1,
) -> tuple[int, ...]:
    if rings < 0:
        raise ValueError("rings must be non-negative")
    count = len(canonicalize_faces(faces))
    current = {i for i in selected if 0 <= i < count}
    result = set(current)
    for _ in range(rings):
        current = {
            neighbour for face in current for neighbour in adjacency.get(face, ())
            if 0 <= neighbour < count
        } - result
        result.update(current)
    return tuple(sorted(result))


def projected_bbox(
    points: Iterable[Sequence[float]], view: str
) -> tuple[float, float, float, float] | None:
    axes = {"front": (0, 2), "right": (1, 2), "top": (0, 1)}
    if view not in axes:
        raise ValueError(f"unknown view: {view}")
    a, b = axes[view]
    projected = [(float(p[a]), float(p[b])) for p in points]
    if not projected:
        return None
    xs, ys = zip(*projected)
    return min(xs), min(ys), max(xs), max(ys)


def best_view(points: Iterable[Sequence[float]]) -> str:
    cached = tuple(tuple(p) for p in points)
    choices = []
    for order, name in enumerate(("front", "right", "top")):
        box = projected_bbox(cached, name)
        area = 0.0 if box is None else (box[2] - box[0]) * (box[3] - box[1])
        choices.append((-area, order, name))
    return min(choices)[2]
