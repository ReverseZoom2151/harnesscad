"""Complexity strata and leakage-safe deterministic B-rep dataset splits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class BRepComplexity:
    face_count: int
    planar_fraction: float
    trimmed_fraction: float
    max_curve_segments: int

    @property
    def stratum(self) -> str:
        if self.max_curve_segments > 100 or self.trimmed_fraction >= 0.5:
            return "hard"
        if self.face_count > 40 or self.planar_fraction < 0.8:
            return "moderate"
        return "simple"


def complexity(*, face_count, planar_faces, trimmed_faces, curve_segments):
    if face_count < 0 or not 0 <= planar_faces <= face_count or not 0 <= trimmed_faces <= face_count:
        raise ValueError("invalid face counts")
    return BRepComplexity(face_count,
                          planar_faces / face_count if face_count else 0.0,
                          trimmed_faces / face_count if face_count else 0.0,
                          max(curve_segments, default=0))


def grouped_split(records, *, key=lambda item: item["id"],
                  group=lambda item: item.get("family", item["id"])):
    """Assign whole groups to deterministic 70/15/15 buckets."""
    groups = {}
    for item in records:
        groups.setdefault(str(group(item)), []).append(item)
    out = {"train": [], "validation": [], "test": []}
    for group_id in sorted(groups):
        bucket = int(hashlib.sha256(group_id.encode()).hexdigest()[:8], 16) % 100
        name = "train" if bucket < 70 else "validation" if bucket < 85 else "test"
        out[name].extend(sorted(groups[group_id], key=key))
    return {name: tuple(items) for name, items in out.items()}
