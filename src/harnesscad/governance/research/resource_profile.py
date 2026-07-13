"""Injected, portable resource profiling with explicit measurement provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ResourceSampler(Protocol):
    def start(self) -> None: ...
    def stop(self) -> dict: ...


@dataclass(frozen=True)
class ResourceProfile:
    peak_memory_bytes: int
    elapsed_seconds: float
    trainable_parameters: int | None
    frozen_parameters: int | None
    batch_size: int
    provenance: str


def profile(call, sampler: ResourceSampler, *, trainable_parameters=None,
            frozen_parameters=None, batch_size=1, provenance="injected"):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    sampler.start()
    try:
        result = call()
    finally:
        measured = sampler.stop()
    report = ResourceProfile(
        max(0, int(measured.get("peak_memory_bytes", 0))),
        max(0.0, float(measured.get("elapsed_seconds", 0))),
        trainable_parameters, frozen_parameters, batch_size, provenance,
    )
    return result, report
