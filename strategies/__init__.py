"""strategies — reliability patterns from HARNESS_BLUEPRINT.md (sec.4, 8, 10).

These compose the existing harness spine (Planner -> HarnessSession -> verify ->
MemoryStore) without touching it. Each strategy spends extra inference compute to
lift the single-shot success probability `p` toward 1:

  - **best_of_n** (sec.4, "Best-of-N + verifier"): generate N candidate op-plans,
    apply each through a FRESH session, and let the deterministic verifier pick the
    winner. Rationale: `P(success) = 1 - (1 - p)^N`.
  - **ReflexionLoop** (sec.8/sec.10, "Read-Act-Reflect-Write / Reflexion"): on a
    failed verify, synthesize an actionable insight, WRITE it to semantic memory,
    RECALL prior insights into the next attempt's context, retry.

Absolute imports, stdlib only. The harness (loop.py, agent/, memory/) is imported
and injected, never edited.
"""

from __future__ import annotations

from strategies.best_of_n import (
    BestOfNResult,
    Candidate,
    best_of_n,
    default_scorer,
)
from strategies.reflexion import (
    ReflexionAttempt,
    ReflexionLoop,
    ReflexionResult,
    heuristic_reflect,
)

__all__ = [
    "best_of_n",
    "default_scorer",
    "BestOfNResult",
    "Candidate",
    "ReflexionLoop",
    "ReflexionResult",
    "ReflexionAttempt",
    "heuristic_reflect",
]
