"""Elo rating + Leaderboard — the scalar-free ranking primitive (sec.12).

The blueprint's design-space exploration ranks *variants* with an
**Elo-tournament** rather than a single scalar objective: "clean when there's no
single scalar objective." Elo turns a stream of noisy pairwise verdicts (variant
A beat variant B) into a stable total order, giving diminishing weight to
already-confident ratings.

This module is pure arithmetic — no CAD, no LLM, no wall clock — so it is trivially
deterministic. `EloRating` is the stateless update math; `Leaderboard` accumulates
ratings for many ids across many pairwise results and ranks them.

    expected(a, b) = 1 / (1 + 10 ** ((b - a) / 400))

is the logistic expectation; `expected(a, b) + expected(b, a) == 1` always.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


class EloRating:
    """Stateless Elo update math.

    ``k`` is the update step (higher = faster-moving, noisier ratings); ``base`` is
    the rating a fresh competitor starts at. All methods take/return plain floats,
    so this object holds no per-competitor state — `Leaderboard` does.
    """

    def __init__(self, k: float = 32.0, base: float = 1200.0) -> None:
        self.k = float(k)
        self.base = float(base)

    def expected(self, a: float, b: float) -> float:
        """Expected score (win probability) of rating ``a`` against rating ``b``."""
        return 1.0 / (1.0 + 10.0 ** ((b - a) / 400.0))

    def update(self, winner: float, loser: float) -> Tuple[float, float]:
        """Apply a decisive result; return ``(new_winner, new_loser)``.

        The winner gains exactly what the loser sheds (``k * (1 - expected)``), so
        total rating is conserved.
        """
        ew = self.expected(winner, loser)
        delta = self.k * (1.0 - ew)
        return winner + delta, loser - delta

    def update_draw(self, a: float, b: float) -> Tuple[float, float]:
        """Apply a draw; return ``(new_a, new_b)``.

        Each competitor moves toward the other by ``k * (0.5 - expected)``; the
        higher-rated one loses a little, the lower-rated one gains a little.
        """
        ea = self.expected(a, b)
        eb = self.expected(b, a)  # == 1 - ea
        return a + self.k * (0.5 - ea), b + self.k * (0.5 - eb)


class Leaderboard:
    """Accumulates Elo ratings for many competitor ids across pairwise results.

    Ratings are created lazily at ``base`` on first sight of an id. ``rank()``
    returns the standings with a deterministic tie-break (id) so equal ratings
    never reorder run-to-run.
    """

    def __init__(self, k: float = 32.0, base: float = 1200.0) -> None:
        self.elo = EloRating(k=k, base=base)
        self._ratings: Dict[str, float] = {}

    # --- state ------------------------------------------------------------
    def rating(self, cid: str) -> float:
        """Current rating of ``cid`` (``base`` if it has no result yet)."""
        return self._ratings.get(cid, self.elo.base)

    def add(self, cid: str) -> None:
        """Register ``cid`` at ``base`` so it appears in ``rank()`` even unplayed."""
        self._ratings.setdefault(cid, self.elo.base)

    # --- results ----------------------------------------------------------
    def record(self, winner_id: str, loser_id: str) -> None:
        """Record a decisive pairwise result and update both ratings."""
        w, l = self.rating(winner_id), self.rating(loser_id)
        nw, nl = self.elo.update(w, l)
        self._ratings[winner_id] = nw
        self._ratings[loser_id] = nl

    def record_draw(self, a_id: str, b_id: str) -> None:
        """Record a drawn pairwise result and update both ratings."""
        a, b = self.rating(a_id), self.rating(b_id)
        na, nb = self.elo.update_draw(a, b)
        self._ratings[a_id] = na
        self._ratings[b_id] = nb

    # --- ranking ----------------------------------------------------------
    def rank(self) -> List[Tuple[str, float]]:
        """Standings as ``[(id, rating), ...]`` sorted best-first.

        Descending rating; ties broken by id ascending for determinism.
        """
        return sorted(self._ratings.items(), key=lambda kv: (-kv[1], kv[0]))

    def ratings(self) -> Dict[str, float]:
        """A copy of the id -> rating map."""
        return dict(self._ratings)
