"""CISP result types — the shape the agent sees back from applyOps.

Mirrors the LSP-inspired CISP contract from the blueprint: applyOps returns
{ ok, applied, diagnostics, digest, rejected }. `digest` is the deterministic
model hash (the replay invariant); `rejected` carries the op that was blocked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from verify import Diagnostic


@dataclass
class ApplyOpsResult:
    ok: bool
    applied: int
    digest: str
    diagnostics: List[Diagnostic] = field(default_factory=list)
    rejected: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "applied": self.applied,
            "digest": self.digest,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "rejected": self.rejected,
        }
