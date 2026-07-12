"""Reinforced-decay salience for a memory store (MemoryBank forgetting curve).

Mined from CoMeT's ``MemoryNode`` schema, whose usage-driven salience fields
carry an explicit, un-implemented policy in their doc strings:

    strength: 'Salience strength S; retention R = exp(-Δt_days / (S * τ)).
               Bumped on recall hit.'
    last_recall_at: 'Timestamp of most recent retrieval hit (resets the decay
               clock).'
    recall_count: 'Cumulative retrieval-hit count, folded in by the dream
               reinforced-decay pass.'

That is the Ebbinghaus forgetting curve as used in MemoryBank: every node's
retention decays exponentially with the time since it was last recalled, at a
rate set by its own strength ``S`` -- and each recall *reinforces* the node
(``S += 1``) and resets its decay clock, so frequently-used knowledge fades
slowly and stale knowledge fades fast. CoMeT names the fields and the formula
but never ships the arithmetic; this module implements it as a deterministic,
clockless sweep (all times are caller-supplied day numbers -- no wall clock, no
randomness), suitable for pruning or down-weighting a growing CAD knowledge
base.

The transferable core:

  * ``retention`` -- the exp-decay curve R = exp(-Δt / (S·τ)) in [0, 1].
  * ``reinforce`` -- one recall hit: S += bump, clock reset, count += 1.
  * ``time_to_retention`` -- invert the curve: when will R fall to a floor?
  * ``decay_sweep`` -- rank a set by current retention and split it into
    ``retained`` / ``forgotten`` at a threshold, deterministically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "Salience",
    "SweepResult",
    "retention",
    "reinforce",
    "time_to_retention",
    "decay_sweep",
]


@dataclass(frozen=True)
class Salience:
    """Usage-driven salience state for one memory node.

    ``last_recall_day`` is an absolute day number on the caller's own clock
    (e.g. days since store creation). All decay is measured relative to it, so
    the module never reads a wall clock and results are reproducible.
    """

    node_id: str
    strength: float = 1.0
    last_recall_day: float = 0.0
    recall_count: int = 0

    def __post_init__(self) -> None:
        if self.strength <= 0.0:
            raise ValueError("strength must be > 0")


def retention(strength: float, elapsed_days: float, tau: float = 5.0) -> float:
    """Ebbinghaus retention R = exp(-Δt / (S·τ)), clamped to [0, 1].

    ``elapsed_days`` is the time since the node was last recalled; negative
    elapsed (a recall in the caller's future) is treated as 0 -> R = 1.
    Larger ``strength`` or ``tau`` means slower forgetting.
    """
    if strength <= 0.0:
        raise ValueError("strength must be > 0")
    if tau <= 0.0:
        raise ValueError("tau must be > 0")
    if elapsed_days <= 0.0:
        return 1.0
    r = math.exp(-elapsed_days / (strength * tau))
    if r < 0.0:
        return 0.0
    if r > 1.0:
        return 1.0
    return r


def reinforce(sal: Salience, now_day: float, bump: float = 1.0) -> Salience:
    """Apply one recall hit: strengthen, reset the decay clock, bump the count.

    Returns a new :class:`Salience` (the input is immutable). ``now_day`` sets
    the fresh ``last_recall_day``; a recall dated before the current one is
    ignored for the clock (the clock only moves forward) but still strengthens
    and counts.
    """
    if bump < 0.0:
        raise ValueError("bump must be >= 0")
    return replace(
        sal,
        strength=sal.strength + bump,
        last_recall_day=max(sal.last_recall_day, now_day),
        recall_count=sal.recall_count + 1,
    )


def time_to_retention(strength: float, floor: float, tau: float = 5.0) -> float:
    """Days until retention decays to ``floor`` (invert the curve).

    Δt = -S·τ·ln(floor). A ``floor`` of 0 never arrives -> ``math.inf``; a
    ``floor`` of 1 is immediate -> 0.0.
    """
    if strength <= 0.0:
        raise ValueError("strength must be > 0")
    if tau <= 0.0:
        raise ValueError("tau must be > 0")
    if not 0.0 <= floor <= 1.0:
        raise ValueError("floor must be in [0, 1]")
    if floor == 0.0:
        return math.inf
    if floor >= 1.0:
        return 0.0
    return -strength * tau * math.log(floor)


@dataclass
class SweepResult:
    retained: List[Tuple[str, float]]  # (node_id, retention), strongest first
    forgotten: List[Tuple[str, float]]  # (node_id, retention), weakest first

    def to_dict(self) -> dict:
        return {
            "retained": [[n, round(r, 6)] for n, r in self.retained],
            "forgotten": [[n, round(r, 6)] for n, r in self.forgotten],
        }


def decay_sweep(
    saliences: Sequence[Salience],
    now_day: float,
    *,
    tau: float = 5.0,
    forget_threshold: float = 0.2,
    keep_min: int = 0,
) -> SweepResult:
    """Score every node's current retention and split at ``forget_threshold``.

    A node whose retention is below the threshold is a *forget* candidate.
    ``keep_min`` protects the strongest N nodes from forgetting regardless of
    their score (a floor on working-set size, mirroring the orchestrator's
    passive-first guarantee). ``retained`` is sorted strongest-first,
    ``forgotten`` weakest-first; ties break by ``node_id`` so the partition is a
    pure function of the input set.
    """
    if not 0.0 <= forget_threshold <= 1.0:
        raise ValueError("forget_threshold must be in [0, 1]")
    if keep_min < 0:
        raise ValueError("keep_min must be >= 0")

    scored = [
        (s.node_id, retention(s.strength, now_day - s.last_recall_day, tau))
        for s in saliences
    ]
    # Strongest first; ties by node_id.
    scored.sort(key=lambda kv: (-kv[1], kv[0]))

    retained: List[Tuple[str, float]] = []
    forgotten: List[Tuple[str, float]] = []
    for rank, (nid, r) in enumerate(scored):
        if rank < keep_min or r >= forget_threshold:
            retained.append((nid, r))
        else:
            forgotten.append((nid, r))

    forgotten.sort(key=lambda kv: (kv[1], kv[0]))
    return SweepResult(retained=retained, forgotten=forgotten)
