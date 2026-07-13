"""CADMorph verification stage: shape-distance selection with a cross-round queue.

From CADMorph (Ma et al., NeurIPS 2025), Section 3.4, "Verification", plus the
overall objective in Section 3.1. Given the candidate sequences produced by the
generation stage, the verifier keeps the one whose *rendered shape* lies closest
to the target shape ``S'``.

Two deterministic mechanisms from the paper are built here:

  * **Distance-to-target selection** (paper Eq. 4). CADMorph embeds each
    candidate and the target into a shared latent space and picks the minimum
    L2 distance. With voxelised tSDFs as the shared representation
    (:mod:`geometry.cadmorph_tsdf`) the distance is an ordinary shape metric;
    this module treats the distance as a pluggable callable so any proxy works.

  * **Cross-iteration priority queue ``Q``** (paper Eq. 4 and ablation B).
    Rather than choosing only among the current round's candidates, CADMorph
    keeps a priority queue of the ``X`` best candidates seen across *all* rounds
    and always returns the global best. This "rescues high-quality candidates
    generated in earlier rounds and attenuates occasional noisy candidates."
    Removing the queue (ablation B) costs a large IoU drop, so it is a real,
    load-bearing piece of the algorithm — and it is pure bookkeeping.

The overall CADMorph objective (paper Eq. 1) trades shape fidelity against
structure preservation::

    C' = argmin_C  D_geometry(F(C), S')  +  lambda * R_structure(C, C_orig)

We expose :func:`edit_distance` (segment-level Levenshtein) as ``R_structure``
and :func:`objective` as the weighted sum, so the verifier can prefer the
candidate that both matches the target *and* stays closest to the original
sequence — the paper's "smallest possible edits" behaviour (Figure 5).

Stdlib-only and deterministic; ties break by candidate order to keep selection
reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, List, Optional, Sequence, Tuple, TypeVar

C = TypeVar("C")


# --------------------------------------------------------------------------- #
# Structure preservation (R_structure)
# --------------------------------------------------------------------------- #
def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein (insert/delete/substitute) distance over two segment lists.

    Serves as ``R_structure`` in the CADMorph objective: how far a candidate
    sequence diverges from the original (paper's "Edit Dist." metric, Table 1).
    """
    a, b = list(a), list(b)
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def objective(geom_distance: float, candidate: Sequence, original: Sequence,
              *, lam: float = 0.0) -> float:
    """The CADMorph objective ``D_geometry + lambda * R_structure`` (Eq. 1).

    ``lam == 0`` recovers pure shape-fidelity selection (Eq. 4); a positive
    ``lam`` biases toward candidates that keep more of the original sequence.
    """
    return float(geom_distance) + float(lam) * edit_distance(candidate, original)


# --------------------------------------------------------------------------- #
# Candidate scoring
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScoredCandidate(Generic[C]):
    """A candidate sequence with its shape distance and full objective value."""

    candidate: C
    geom_distance: float
    score: float                 # the objective (lower is better)
    round_index: int = 0
    order: int = 0               # position within its round (tie-break)


def score_candidates(candidates: Sequence[C],
                     distance: Callable[[C], float],
                     original: Sequence,
                     *, lam: float = 0.0,
                     round_index: int = 0) -> List[ScoredCandidate]:
    """Score each candidate by ``distance`` then the composite objective."""
    out: List[ScoredCandidate] = []
    for order, cand in enumerate(candidates):
        d = float(distance(cand))
        s = objective(d, cand, original, lam=lam)
        out.append(ScoredCandidate(cand, d, s, round_index, order))
    return out


# --------------------------------------------------------------------------- #
# Cross-iteration priority queue Q
# --------------------------------------------------------------------------- #
class CandidateQueue(Generic[C]):
    """A bounded priority queue retaining the ``X`` best candidates seen so far.

    "Best" means lowest :attr:`ScoredCandidate.score` (the objective). Ties
    break deterministically by ``(round_index, order)`` so replay is stable.
    Pushing more than ``capacity`` candidates evicts the worst; :meth:`best`
    returns the global minimum across every round pushed (paper Eq. 4).
    """

    def __init__(self, capacity: int = 4) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self._items: List[ScoredCandidate[C]] = []

    @staticmethod
    def _rank(sc: "ScoredCandidate[C]") -> Tuple[float, int, int]:
        return (sc.score, sc.round_index, sc.order)

    def push(self, scored: ScoredCandidate[C]) -> None:
        self._items.append(scored)
        self._items.sort(key=self._rank)
        if len(self._items) > self.capacity:
            self._items = self._items[: self.capacity]

    def push_all(self, scored: Sequence[ScoredCandidate[C]]) -> None:
        for sc in scored:
            self.push(sc)

    def best(self) -> Optional[ScoredCandidate[C]]:
        return self._items[0] if self._items else None

    def items(self) -> Tuple[ScoredCandidate[C], ...]:
        """The retained candidates, best first."""
        return tuple(self._items)

    def __len__(self) -> int:
        return len(self._items)


def select_best(candidates: Sequence[C],
                distance: Callable[[C], float],
                original: Sequence,
                *, lam: float = 0.0,
                queue: Optional[CandidateQueue[C]] = None,
                round_index: int = 0) -> ScoredCandidate[C]:
    """Score ``candidates`` and return the best, honouring a cross-round queue.

    When ``queue`` is provided the candidates are pushed into it and the global
    best (across this and all prior rounds) is returned — the paper's
    queue-backed verification (Eq. 4). Without a queue this reduces to the
    single-round argmin (ablation-B behaviour). Raises ``ValueError`` on an
    empty candidate list with an empty/absent queue.
    """
    scored = score_candidates(candidates, distance, original,
                              lam=lam, round_index=round_index)
    if queue is not None:
        queue.push_all(scored)
        best = queue.best()
        if best is None:
            raise ValueError("no candidates to select from")
        return best
    if not scored:
        raise ValueError("no candidates to select from")
    return min(scored, key=CandidateQueue._rank)
