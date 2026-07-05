"""GeometryBackend — the swappable kernel interface.

Everything above this line (ops, DAG, verifiers, loop) is kernel-agnostic. A
backend turns an op stream into geometry + a content digest. v0 ships a stub
(no dependencies); a CadQuery/OCCT backend follows, and a Rust kernel (Fornjot/
Truck/Cadmium) can be dropped in later behind this same protocol.

The digest is load-bearing: replaying the same ops must yield the same digest
(the "deterministic replay" invariant from the CISP spec).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol, runtime_checkable

from cisp.ops import Op
from verify import Diagnostic


@dataclass
class ApplyResult:
    ok: bool
    created: List[str] = field(default_factory=list)
    diagnostics: List[Diagnostic] = field(default_factory=list)


@runtime_checkable
class GeometryBackend(Protocol):
    def reset(self) -> None:
        """Discard all state and return to an empty model."""

    def apply(self, op: Op) -> ApplyResult:
        """Apply one op. On invalid references, return ok=False WITHOUT mutating
        (block-and-correct); the loop rejects the op and returns diagnostics."""

    def regenerate(self) -> List[Diagnostic]:
        """Rebuild derived geometry from the current op state (no-op for
        incremental backends). Returns any regen diagnostics."""

    def query(self, q: str) -> dict:
        """Read-only queries: 'summary', 'sketch_dof', 'feature_count', ..."""

    def export(self, fmt: str):
        """Export the current model in `fmt` (e.g. 'step')."""

    def state_digest(self) -> str:
        """Content hash of the current model — stable across identical replays."""
