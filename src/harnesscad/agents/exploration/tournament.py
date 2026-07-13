"""Co-Scientist design-space exploration — generate -> debate -> evolve (sec.12).

The blueprint's exploration tier sits *above* Best-of-N. Where `strategies.best_of_n`
draws N candidates and lets the verifier pick one winner, this layer treats a whole
population of design *variants* as competitors and **ranks** them with an
Elo-tournament of pairwise debates — the right tool when there is no single scalar
objective (sec.12: "Co-Scientist generate -> debate -> evolve with Elo-tournament
ranking of variants; cluster to avoid redundant search").

Pipeline (`explore`):

    generate --> cluster (drop redundant designs) --> EloTournament (debate + Elo)
       ^                                                        |
       |                                                        v
       +---------------- evolve top-k (injected mutator) <------+

Design seams (everything network/kernel-touching is injected, so this file is pure
and testable with fakes):
  - ``generate(n, seed) -> list[Variant]``  — produce an evaluated population. Compose
    ``strategies.best_of_n`` here, or a bespoke generator; this layer never plans.
  - ``mutator(parents, rng) -> Variant``    — produce one evaluated child from parents
    (mutate params / recombine op prefixes). Own the session/apply here.
  - ``judge(a, b) -> int``                  — optional LLM-as-judge; positive => a wins.
    Always applied with **swap-augmentation** (judged both orders, averaged) to defeat
    position bias (sec.6). When absent, a deterministic verifier comparator decides.

Determinism: no wall clock. All tie-breaks/pairings use ``random.Random(seed)`` with a
fixed seed threaded through, or lexicographic id order.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from harnesscad.agents.exploration.elo import Leaderboard

# Type aliases for the injected seams.
Judge = Callable[["Variant", "Variant"], float]
Generate = Callable[[int, int], List["Variant"]]
Mutator = Callable[[List["Variant"], random.Random], "Variant"]


# --- the competitor ---------------------------------------------------------
@dataclass
class Variant:
    """One CAD design variant: a competitor in the tournament.

    ``result`` is an ``ApplyOpsResult`` (or any object exposing ``ok``, ``applied``,
    ``diagnostics``) produced by applying ``ops`` through a session — this layer only
    *reads* it. ``params``/``brief`` describe how it was generated; ``score`` and
    ``cluster`` are filled in by this module.
    """

    id: str
    brief: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    ops: List[Any] = field(default_factory=list)
    result: Any = None
    score: Optional[float] = None
    cluster: Optional[int] = None
    generation: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.result is not None and getattr(self.result, "ok", False))


# --- deterministic verifier comparator --------------------------------------
def _rank_key(v: "Variant") -> Tuple[int, int, int, int]:
    """Sortable quality key for a variant; higher tuple is better.

    Mirrors ``strategies.best_of_n.default_scorer`` and extends it with a simplicity
    tie-break: prefer (1) verified-ok, then (2) fewer diagnostics, then (3) more ops
    applied, then (4) a *simpler* op count. Robust to a missing ``result``.
    """
    res = v.result
    ok = 1 if (res is not None and getattr(res, "ok", False)) else 0
    diags = getattr(res, "diagnostics", []) if res is not None else []
    applied = getattr(res, "applied", 0) if res is not None else 0
    return (ok, -len(diags), applied, -len(v.ops))


def compare(a: "Variant", b: "Variant", judge: Optional[Judge] = None) -> int:
    """Pairwise verdict: ``+1`` a beats b, ``-1`` b beats a, ``0`` draw.

    Default is the deterministic verifier comparator (``_rank_key``). An injected
    ``judge(a, b) -> float`` (LLM-as-judge; positive means a is better) overrides it,
    and is **always swap-augmented**: we ask the judge both ``(a, b)`` and ``(b, a)``
    and average the two signed opinions, so a judge that merely favours whichever
    variant is shown first nets out to a draw (sec.6 position-bias defense).
    """
    if judge is not None:
        s_ab = float(judge(a, b))          # positive => a better
        s_ba = float(judge(b, a))          # positive => b better
        net = (s_ab - s_ba) / 2.0          # a's perspective, position bias cancelled
        if net > 0:
            return 1
        if net < 0:
            return -1
        return 0
    ka, kb = _rank_key(a), _rank_key(b)
    if ka > kb:
        return 1
    if ka < kb:
        return -1
    return 0


def debate(a: "Variant", b: "Variant", judge: Optional[Judge] = None) -> "Variant":
    """Run one pairwise debate and return the winning ``Variant``.

    On a genuine tie the lexicographically-smaller id wins, so ``debate`` is a total,
    deterministic function (never ``None``). For win/draw/loss accounting the
    tournament calls ``compare`` directly.
    """
    verdict = compare(a, b, judge=judge)
    if verdict > 0:
        return a
    if verdict < 0:
        return b
    return a if a.id <= b.id else b


# --- clustering: avoid redundant search -------------------------------------
def op_signature(ops: List[Any], ndigits: int = 3) -> frozenset:
    """A canonical token set for an op sequence (for Jaccard similarity).

    Each op contributes a ``tag(k=v,...)`` token with floats rounded to ``ndigits``,
    so two variants built from the same ops (up to tiny numeric jitter) share tokens
    and cluster together, while structurally different designs do not. Ops are
    expected to expose ``to_dict()`` (CISP ``Op`` does); anything else falls back to
    ``repr``.
    """
    tokens = set()
    for op in ops:
        to_dict = getattr(op, "to_dict", None)
        if callable(to_dict):
            d = dict(to_dict())
            tag = d.pop("op", op.__class__.__name__)
            parts = []
            for k in sorted(d):
                val = d[k]
                if isinstance(val, float):
                    val = round(val, ndigits)
                parts.append(f"{k}={val}")
            tokens.add(f"{tag}({','.join(parts)})")
        else:
            tokens.add(repr(op))
    return frozenset(tokens)


def jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity of two token sets; ``1.0`` identical, ``0.0`` disjoint."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


@dataclass
class Cluster:
    """A group of near-duplicate variants and its chosen representative."""

    index: int
    members: List[Variant] = field(default_factory=list)
    representative: Optional[Variant] = None


def cluster_variants(
    variants: List[Variant],
    threshold: float = 0.9,
    signature: Callable[[List[Any]], frozenset] = op_signature,
) -> List[Cluster]:
    """Group near-duplicate variants so the tournament skips redundant comparisons.

    Greedy single-pass clustering: each variant joins the first existing cluster whose
    representative it is at least ``threshold``-similar to (Jaccard over op tokens),
    else it opens a new cluster. The representative is the highest-quality member
    (``_rank_key``, id tie-break), so the tournament fields each cluster's best design.
    Every variant's ``.cluster`` is stamped with its cluster index.
    """
    clusters: List[Cluster] = []
    sigs: List[frozenset] = []
    for v in variants:
        sig = signature(v.ops)
        placed = False
        for ci, cl in enumerate(clusters):
            if jaccard(sig, sigs[ci]) >= threshold:
                cl.members.append(v)
                v.cluster = cl.index
                placed = True
                break
        if not placed:
            cl = Cluster(index=len(clusters), members=[v])
            v.cluster = cl.index
            clusters.append(cl)
            sigs.append(sig)
    for cl in clusters:
        cl.representative = max(cl.members, key=lambda m: (_rank_key(m), _neg_id(m)))
    return clusters


def _neg_id(v: Variant):
    """Tie-break helper: makes ``max`` prefer the smaller id when rank keys tie."""
    # max() picks the LARGEST; to prefer the smaller id we invert the ordering by
    # comparing negated code-point tuples.
    return tuple(-ord(c) for c in v.id)


def cluster_representatives(clusters: List[Cluster]) -> List[Variant]:
    """The representative variant of each cluster, in cluster order."""
    return [cl.representative for cl in clusters if cl.representative is not None]


# --- the tournament ---------------------------------------------------------
@dataclass
class TournamentResult:
    """Outcome of one Elo tournament over a variant pool."""

    leaderboard: Leaderboard
    ranking: List[Tuple[str, float]]           # (variant id, rating), best-first
    ranked_variants: List[Variant]             # same order, resolved to Variants
    pairings: List[Tuple[str, str, int]] = field(default_factory=list)  # (a, b, verdict)

    @property
    def winner(self) -> Optional[Variant]:
        return self.ranked_variants[0] if self.ranked_variants else None


class EloTournament:
    """Runs pairwise debates across a variant pool and Elo-ranks them.

    ``schedule``:
      - ``"round_robin"`` (default): every unordered pair debates once. O(n^2), the
        most signal; fine for the small pools clustering leaves behind.
      - ``"swiss"``: ``rounds`` rounds, each pairing neighbours in the current
        standings (winners meet winners), cheaper for larger pools.

    Pairing *order* is shuffled with ``random.Random(seed)`` so no variant gets a
    systematic scheduling advantage, but the shuffle is fully seeded => deterministic.
    """

    def __init__(
        self,
        variants: List[Variant],
        judge: Optional[Judge] = None,
        k: float = 32.0,
        base: float = 1200.0,
        seed: int = 0,
        schedule: str = "round_robin",
        rounds: int = 3,
    ) -> None:
        if schedule not in ("round_robin", "swiss"):
            raise ValueError(f"unknown schedule {schedule!r}")
        self.variants = list(variants)
        self.judge = judge
        self.seed = seed
        self.schedule = schedule
        self.rounds = rounds
        self.board = Leaderboard(k=k, base=base)
        self._by_id = {v.id: v for v in self.variants}
        for v in self.variants:
            self.board.add(v.id)

    def _play(self, a: Variant, b: Variant,
              pairings: List[Tuple[str, str, int]]) -> None:
        verdict = compare(a, b, judge=self.judge)
        if verdict > 0:
            self.board.record(a.id, b.id)
        elif verdict < 0:
            self.board.record(b.id, a.id)
        else:
            self.board.record_draw(a.id, b.id)
        pairings.append((a.id, b.id, verdict))

    def run(self) -> TournamentResult:
        rng = random.Random(self.seed)
        pairings: List[Tuple[str, str, int]] = []
        if len(self.variants) < 2:
            # Nothing to compare; ratings stay at base.
            return self._result(pairings)

        if self.schedule == "round_robin":
            pairs = [(self.variants[i], self.variants[j])
                     for i in range(len(self.variants))
                     for j in range(i + 1, len(self.variants))]
            rng.shuffle(pairs)
            for a, b in pairs:
                self._play(a, b, pairings)
        else:  # swiss
            for _ in range(max(1, self.rounds)):
                order = [self._by_id[cid] for cid, _ in self.board.rank()]
                # Small deterministic jitter within the standings-ordered pairing.
                for i in range(0, len(order) - 1, 2):
                    self._play(order[i], order[i + 1], pairings)
        return self._result(pairings)

    def _result(self, pairings: List[Tuple[str, str, int]]) -> TournamentResult:
        ranking = self.board.rank()
        ranked_variants = [self._by_id[cid] for cid, _ in ranking]
        for cid, rating in ranking:
            self._by_id[cid].score = rating
        return TournamentResult(
            leaderboard=self.board,
            ranking=ranking,
            ranked_variants=ranked_variants,
            pairings=pairings,
        )


# --- evolution: next generation from the top ranks --------------------------
def evolve(
    ranked_variants: List[Variant],
    mutator: Mutator,
    top_k: int = 3,
    n_children: int = 4,
    seed: int = 0,
) -> List[Variant]:
    """Breed a next generation from the top-ranked variants via an injected mutator.

    The ``mutator(parents, rng) -> Variant`` owns the actual mutation *and* evaluation
    (mutate params / recombine op prefixes, then apply through a fresh session) — this
    layer just supplies the deterministically-seeded parent pool and rng. ``parents``
    is the top-``top_k`` slice of the (already ranked) population; ``n_children`` new
    variants are produced.
    """
    parents = list(ranked_variants[:max(1, top_k)])
    if not parents:
        return []
    rng = random.Random(seed)
    children: List[Variant] = []
    for _ in range(max(0, n_children)):
        child = mutator(parents, rng)
        children.append(child)
    return children


# --- the whole loop ---------------------------------------------------------
@dataclass
class Generation:
    """A single generate/cluster/tournament round of ``explore``."""

    index: int
    n_variants: int
    n_clusters: int
    result: TournamentResult

    @property
    def winner(self) -> Optional[Variant]:
        return self.result.winner


@dataclass
class ExplorationResult:
    """Final output of ``explore``: the ranked survivors + the overall winner."""

    generations: List[Generation] = field(default_factory=list)
    final_ranked: List[Variant] = field(default_factory=list)
    winner: Optional[Variant] = None

    @property
    def leaderboard(self) -> Optional[Leaderboard]:
        return self.generations[-1].result.leaderboard if self.generations else None


def explore(
    generate: Generate,
    rounds: int = 2,
    n: int = 6,
    seed: int = 0,
    *,
    mutator: Optional[Mutator] = None,
    judge: Optional[Judge] = None,
    top_k: int = 3,
    cluster_threshold: float = 0.9,
    k: float = 32.0,
    base: float = 1200.0,
    schedule: str = "round_robin",
) -> ExplorationResult:
    """Run generate -> cluster -> Elo-tournament -> evolve for ``rounds`` rounds.

    Each round: cluster the population (drop redundant designs), field one
    representative per cluster in an ``EloTournament``, and — unless it's the last
    round or no ``mutator`` was given — ``evolve`` the top-``top_k`` into the next
    generation (carried alongside the surviving parents). Returns the final ranked
    representatives and the overall ``winner`` (the top of the last round).

    Fully deterministic for a fixed ``seed``: the generator, per-round tournament
    shuffles, and mutator rng are all seeded off it with distinct offsets.
    """
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1 (got {rounds})")

    population = list(generate(n, seed))
    generations: List[Generation] = []
    final_ranked: List[Variant] = []

    for r in range(rounds):
        for v in population:
            v.generation = r
        clusters = cluster_variants(population, threshold=cluster_threshold)
        reps = cluster_representatives(clusters)
        tour = EloTournament(reps, judge=judge, k=k, base=base,
                             seed=seed + r, schedule=schedule)
        result = tour.run()
        generations.append(Generation(
            index=r, n_variants=len(population),
            n_clusters=len(clusters), result=result))
        final_ranked = result.ranked_variants

        if r == rounds - 1 or mutator is None:
            break
        survivors = final_ranked[:max(1, top_k)]
        children = evolve(final_ranked, mutator, top_k=top_k,
                          n_children=n, seed=seed + 1000 + r)
        population = survivors + children

    winner = final_ranked[0] if final_ranked else None
    return ExplorationResult(
        generations=generations,
        final_ranked=final_ranked,
        winner=winner,
    )
