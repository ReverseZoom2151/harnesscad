"""Compile-and-distance selection of candidates against an input point cloud."""

from __future__ import annotations

from dataclasses import dataclass
from bench.compiler_judge import symmetric_chamfer


@dataclass(frozen=True)
class PointCloudCandidate:
    index: int
    candidate: object
    valid: bool
    distance: float | None
    error: str = ""


def select_pointcloud_candidate(cloud, provider, compiler, sampler, *,
                                count=10, sample_count=1024, seed=0):
    attempts = []
    for index in range(count):
        try:
            candidate = provider(cloud, seed+index)
            shape = compiler(candidate)
            sampled = sampler(shape, sample_count, seed)
            attempts.append(PointCloudCandidate(
                index, candidate, True, symmetric_chamfer(sampled, cloud)))
        except Exception as exc:
            attempts.append(PointCloudCandidate(
                index, None, False, None, f"{type(exc).__name__}: {exc}"))
    valid = [item for item in attempts if item.valid]
    winner = min(valid, key=lambda item: (item.distance, item.index)) if valid else None
    return {"attempts": tuple(attempts), "winner": winner}
