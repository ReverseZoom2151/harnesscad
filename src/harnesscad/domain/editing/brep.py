"""Provider seam, stable face identities, and validity-first candidate search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, Sequence


@dataclass(frozen=True)
class FaceDescriptor:
    centroid: tuple[float, float, float]
    normal: tuple[float, float, float]
    area: float
    surface: str = "unknown"
    source_id: str = ""

    def signature(self, digits: int = 8) -> tuple:
        return (
            self.surface.casefold(),
            round(self.area, digits),
            *(round(v, digits) for v in self.centroid),
            *(round(v, digits) for v in self.normal),
            self.source_id,
        )


def canonicalize_faces(faces: Iterable[FaceDescriptor]) -> tuple[FaceDescriptor, ...]:
    """Return a kernel-order-independent face sequence."""
    return tuple(sorted(faces, key=FaceDescriptor.signature))


@dataclass(frozen=True)
class EditCandidate:
    shape: Any
    operation: str
    parameters: dict[str, Any] = field(default_factory=dict)
    valid: bool = False
    verifier_score: float = 0.0
    ordinal: int = 0


class EditProvider(Protocol):
    """History-free proposal provider over an imported shape and instruction."""

    def propose(
        self, shape: Any, instruction: str, faces: Sequence[FaceDescriptor], k: int
    ) -> Iterable[EditCandidate]: ...


def rank_candidates(candidates: Iterable[EditCandidate], k: int) -> tuple[EditCandidate, ...]:
    if k < 0:
        raise ValueError("k must be non-negative")
    ranked = sorted(
        candidates,
        key=lambda c: (
            not c.valid, -float(c.verifier_score), c.operation,
            tuple(sorted((str(a), repr(b)) for a, b in c.parameters.items())),
            c.ordinal,
        ),
    )
    return tuple(ranked[:k])


def generate_candidates(
    provider: EditProvider,
    shape: Any,
    instruction: str,
    faces: Iterable[FaceDescriptor],
    k: int,
    *,
    is_valid: Callable[[Any], bool],
    score: Callable[[Any, str], float],
) -> tuple[EditCandidate, ...]:
    """Generate a deterministic pool and independently verify every proposal."""
    canonical = canonicalize_faces(faces)
    proposed = list(provider.propose(shape, instruction, canonical, k))
    checked = [
        EditCandidate(
            item.shape, item.operation, dict(item.parameters),
            bool(is_valid(item.shape)), float(score(item.shape, instruction)), i,
        )
        for i, item in enumerate(proposed)
    ]
    return rank_candidates(checked, k)
