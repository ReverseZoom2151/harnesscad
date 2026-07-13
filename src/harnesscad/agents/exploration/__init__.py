"""exploration — design-space exploration layer (docs/blueprint.md sec.12, sec.4).

The ranking/evolution tier *above* ``strategies.best_of_n``. Best-of-N draws N
candidates and lets the verifier pick one; this package treats a whole population of
design variants as competitors and runs the Co-Scientist loop —
**generate -> debate -> evolve** with **Elo-tournament** ranking, **clustering** to
avoid redundant search (sec.12) — for the case where there is no single scalar
objective. It composes the harness spine (Planner -> HarnessSession -> verify) via
injected ``generate``/``mutator``/``judge`` seams; it never plans or touches state
itself.

Absolute imports, stdlib only, deterministic (seeded ``random.Random`` for every
tie-break/pairing; no wall clock).
"""

from __future__ import annotations

from harnesscad.agents.exploration.elo import EloRating, Leaderboard
from harnesscad.agents.exploration.tournament import (
    Cluster,
    EloTournament,
    ExplorationResult,
    Generation,
    TournamentResult,
    Variant,
    cluster_representatives,
    cluster_variants,
    compare,
    debate,
    evolve,
    explore,
    jaccard,
    op_signature,
)

__all__ = [
    # elo
    "EloRating",
    "Leaderboard",
    # variants + debate
    "Variant",
    "compare",
    "debate",
    # clustering
    "Cluster",
    "cluster_variants",
    "cluster_representatives",
    "op_signature",
    "jaccard",
    # tournament
    "EloTournament",
    "TournamentResult",
    # evolution / loop
    "evolve",
    "explore",
    "ExplorationResult",
    "Generation",
]
