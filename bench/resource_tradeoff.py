"""Quality/resource records and deterministic Pareto-frontier selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceResult:
    name: str
    quality: float
    peak_memory_bytes: int
    latency_seconds: float
    batch_size: int = 1
    oom: bool = False


def pareto_frontier(results):
    values = tuple(results)
    frontier = []
    for candidate in values:
        dominated = any(
            not other.oom
            and other.quality >= candidate.quality
            and other.peak_memory_bytes <= candidate.peak_memory_bytes
            and other.latency_seconds <= candidate.latency_seconds
            and (other.quality > candidate.quality
                 or other.peak_memory_bytes < candidate.peak_memory_bytes
                 or other.latency_seconds < candidate.latency_seconds)
            for other in values if other is not candidate
        )
        if not candidate.oom and not dominated:
            frontier.append(candidate)
    return tuple(sorted(frontier, key=lambda item: (-item.quality,
                                                     item.peak_memory_bytes,
                                                     item.latency_seconds,
                                                     item.name)))
