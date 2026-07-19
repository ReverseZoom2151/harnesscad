"""strategies — reliability patterns from docs/blueprint.md (sec.4, 8, 10).

These compose the existing harness spine (Planner -> HarnessSession -> verify ->
MemoryStore) without touching it. Each strategy spends extra inference compute to
lift the single-shot success probability `p` toward 1:

  - **best_of_n** (sec.4, "Best-of-N + verifier"): generate N candidate op-plans,
    apply each through a FRESH session, and let the deterministic verifier pick the
    winner. Rationale: `P(success) = 1 - (1 - p)^N`.
  - **ReflexionLoop** (sec.8/sec.10, "Read-Act-Reflect-Write / Reflexion"): on a
    failed verify, synthesize an actionable insight, WRITE it to semantic memory,
    RECALL prior insights into the next attempt's context, retry.
  - **select_by_consensus** (execution-based self-consistency):
    re-select among N already-generated candidates with NO oracle at all -- probe
    each by an injected geometric `measure`, cluster by quantized signature, and
    keep a representative of the largest agreeing cluster. Where best_of_n needs
    the deterministic verifier to rank, this needs only a measurement channel, so
    it applies when no ground truth or feasibility signal exists. Its MONOTONE
    rule never regresses below candidate[0] without >=2-agreement evidence.

Absolute imports, stdlib only. The harness (loop.py, agent/, memory/) is imported
and injected, never edited.
"""

from __future__ import annotations

from harnesscad.eval.reliability.strategies.best_of_n import (
    BestOfNResult,
    Candidate,
    best_of_n,
    default_scorer,
)
from harnesscad.eval.reliability.strategies.exec_consensus import (
    ConsensusResult,
    select_by_consensus,
)
from harnesscad.eval.reliability.strategies.reflexion import (
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
    "ConsensusResult",
    "select_by_consensus",
    "ReflexionLoop",
    "ReflexionResult",
    "ReflexionAttempt",
    "heuristic_reflect",
]
