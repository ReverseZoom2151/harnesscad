"""Dynamic utility retrieval over a dual-track memory (Memory-Augmented RL Agent).

Mined from *Memory-Augmented Reinforcement Learning Agent for CAD Generation*. The
paper keeps a **case library** (past successful intent -> trajectory -> geometric
feedback) and a **skill library** (parameterised reusable op-templates), and argues
that retrieval by *semantic similarity alone* falls into "retrieval traps":
examples that look relevant but are geometrically infeasible in the current
context. Its fix is a **dynamic utility retrieval** that reranks candidates by a
utility estimate updated online from execution feedback -- shifting recall "from
examples that merely look similar to examples that are more likely to succeed".

This module ports the deterministic mechanism (not the trained policy):

*   :class:`MemoryEntry` -- a case or skill with a running utility estimate.
*   :meth:`utility` -- an incremental value updated by execution reward
    (``U <- U + lr * (reward - U)``), a standard tabular value update.
*   :func:`retrieval_score` -- ``alpha * similarity + beta * utility``.
*   :class:`DualTrackMemory` -- separate case/skill tracks, utility-reranked
    retrieval, and online utility updates from feedback.

Deterministic: identical inputs and feedback order yield identical rankings (ties
break by insertion order). Stdlib-only, no model calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "MemoryEntry",
    "retrieval_score",
    "jaccard_similarity",
    "DualTrackMemory",
]


def jaccard_similarity(a: str, b: str) -> float:
    """Token-set Jaccard similarity of two text keys (a lightweight stand-in)."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class MemoryEntry:
    """A case or skill entry carrying a running utility estimate.

    ``utility`` starts optimistic-neutral at ``0.0`` and is nudged toward observed
    execution rewards. ``count`` tracks how many updates it has received.
    """

    key: str
    track: str  # "case" or "skill"
    payload: object = None
    utility: float = 0.0
    count: int = 0
    order: int = 0

    def update(self, reward: float, lr: float = 0.5) -> None:
        """Incremental value update ``U <- U + lr * (reward - U)``."""
        if not 0.0 < lr <= 1.0:
            raise ValueError("lr must be in (0, 1]")
        self.utility += lr * (reward - self.utility)
        self.count += 1


def retrieval_score(
    similarity: float, utility: float, alpha: float = 0.5, beta: float = 0.5
) -> float:
    """Combined score ``alpha * similarity + beta * utility``."""
    return alpha * similarity + beta * utility


class DualTrackMemory:
    """A case library and a skill library with utility-reranked retrieval."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.5) -> None:
        if alpha < 0 or beta < 0:
            raise ValueError("alpha and beta must be non-negative")
        self.alpha = alpha
        self.beta = beta
        self._entries: Dict[Tuple[str, str], MemoryEntry] = {}
        self._counter = 0

    def add(self, key: str, track: str, payload: object = None) -> MemoryEntry:
        """Register (or return existing) a case/skill entry."""
        if track not in ("case", "skill"):
            raise ValueError("track must be 'case' or 'skill'")
        idx = (track, key)
        if idx in self._entries:
            return self._entries[idx]
        entry = MemoryEntry(key=key, track=track, payload=payload, order=self._counter)
        self._counter += 1
        self._entries[idx] = entry
        return entry

    def record_feedback(
        self, key: str, track: str, reward: float, lr: float = 0.5
    ) -> None:
        """Update an entry's utility from an execution reward (online RL signal)."""
        idx = (track, key)
        if idx not in self._entries:
            raise KeyError(f"no such entry: {track}:{key}")
        self._entries[idx].update(reward, lr)

    def retrieve(
        self, query: str, track: str, top_k: int = 3
    ) -> List[Tuple[MemoryEntry, float]]:
        """Top-``k`` entries of a track, ranked by utility-aware retrieval score.

        Ties break by insertion order, so retrieval is deterministic.
        """
        if track not in ("case", "skill"):
            raise ValueError("track must be 'case' or 'skill'")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        scored: List[Tuple[MemoryEntry, float]] = []
        for entry in self._entries.values():
            if entry.track != track:
                continue
            sim = jaccard_similarity(query, entry.key)
            score = retrieval_score(sim, entry.utility, self.alpha, self.beta)
            scored.append((entry, score))
        scored.sort(key=lambda es: (-es[1], es[0].order))
        return scored[:top_k]
