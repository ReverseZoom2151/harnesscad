"""The SEARCH surface: run a real design search over a HarnessSession.

``agents/exploration`` carried a whole design-space toolbox -- samplers, an
evolutionary CAD-code loop, a self-adaptive evolution strategy, constrained design
spaces, a greedy refiner, a validity-first guided search, an Elo tournament, seeded
technique trials, an RL block-decomposition MDP -- and nothing dispatched to any of
it. There was no way to say "search this design space".

This module is that dispatcher.

    space(state)                     -> the searchable parameters of a model
    shape_objective(target)          -> a ready-made objective (lower is better)
    search(name, state, objective)   -> SearchResult (best design + its history)

THE SEARCH SPACE
----------------
A design is the op stream's shape-bearing numeric parameters (sketch geometry +
extrude distances -- see :mod:`domain.editing.registry`), each with a multiplicative
box around its current value. A *design vector* is realised back into a real op
stream and scored by the caller's objective, so "search" here means search over
models the harness can actually build, not over an abstract vector.

Objectives are MINIMISED. ``shape_objective`` scores the analytic shape distance to
a target model; any callable ``objective(ops) -> float`` works.

RIVALS ARE SELECTED BY NAME AND NEVER BLENDED
---------------------------------------------
Three families in here look interchangeable and are not:

  evolutionary   ``evolution``            EvoCAD: a GA over CAD *programs*, ranked
                                          by a comparator, with crossover/mutation
                                          on the op tree.
                 ``evolution_strategy``   a self-adaptive (mu, lambda) ES over a
                                          *numeric* genome with per-gene sigmas.
  sampling       ``designspace_sampler``  uniform / stratified / grid draws over a
                                          mixed categorical+continuous space.
                 ``constrained_designspace``  a validity-GATED space: draws are
                                          rejected until they satisfy the
                                          inequality/divisibility constraints.
                 ``latin_hypercube``      LHSnorm: normal-distributed Latin
                                          hypercube draws around the current design.
  local          ``greedy_refine``        finite-difference gradient descent /
                                          multistart local optima.
                 ``guided_contact_search``  step-scheduled, VALIDITY-FIRST neighbour
                                          selection with Pareto evidence.

They optimise different things under different assumptions. :func:`rivals` names the
families; the surface will never average two of them into one number.

Stdlib-only, absolute imports, deterministic given ``seed``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad import registry as capability_registry
from harnesscad.core.cisp.ops import Op
from harnesscad.domain.editing import registry as editing

__all__ = [
    "SearchError",
    "UnknownStrategy",
    "RivalBlend",
    "Unsupported",
    "Dimension",
    "Design",
    "SearchResult",
    "Strategy",
    "strategies",
    "strategy",
    "rivals",
    "unadapted",
    "space",
    "realise",
    "shape_objective",
    "search",
    "add_arguments",
    "run_cli",
    "main",
]

EXPLORATION_PACKAGE = "exploration"
_PKG = "harnesscad.agents.exploration."


class SearchError(ValueError):
    """Base class for every search-surface failure."""


class UnknownStrategy(SearchError):
    """A strategy name outside the discovered table."""


class RivalBlend(SearchError):
    """Rival strategies were asked to be combined. They never are."""


class Unsupported(SearchError):
    """This strategy genuinely cannot run on this state (no fallback)."""


# --------------------------------------------------------------------------- #
# The design space
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Dimension:
    """One searchable parameter: op ``index``.``param``, bounded around its value."""

    index: int
    param: str
    value: float
    low: float
    high: float

    def clamp(self, x: float) -> float:
        return min(self.high, max(self.low, float(x)))

    def to_dict(self) -> dict:
        return {"index": self.index, "param": self.param, "value": self.value,
                "low": self.low, "high": self.high}


def space(state: Any, *, low: float = 0.5, high: float = 2.0,
          only: Optional[Sequence[Tuple[int, str]]] = None) -> Tuple[Dimension, ...]:
    """The searchable design space of a session / op stream.

    One dimension per shape-bearing numeric parameter, bounded to
    ``[low*value, high*value]``. Parameters the shape proxy cannot see (a
    constraint's value, a pattern count) are excluded by default: searching them
    would burn budget on moves the objective cannot measure. Pass ``only=`` to
    choose the dimensions yourself.
    """
    ops = editing.ops_of(state)
    refs = editing._editable(ops, only, shape_only=True)
    dims: List[Dimension] = []
    for (i, param) in refs:
        v = float(getattr(ops[i], param))
        lo, hi = sorted((low * v, high * v))
        dims.append(Dimension(i, param, v, lo, hi))
    if not dims:
        raise Unsupported("this model has no searchable shape parameters")
    return tuple(dims)


def realise(state: Any, dims: Sequence[Dimension], vector: Sequence[float]
            ) -> Tuple[Op, ...]:
    """A design vector -> a real op stream (the model the objective will score)."""
    ops = list(editing.ops_of(state))
    for dim, x in zip(dims, vector):
        op = ops[dim.index]
        current = getattr(op, dim.param)
        ops[dim.index] = dataclasses.replace(
            op, **{dim.param: type(current)(dim.clamp(x))})
    return tuple(ops)


def current_vector(dims: Sequence[Dimension]) -> List[float]:
    return [d.value for d in dims]


@dataclass(frozen=True)
class Design:
    """One evaluated design."""

    vector: Tuple[float, ...]
    ops: Tuple[Op, ...]
    score: float

    def to_dict(self) -> dict:
        return {"vector": list(self.vector), "score": self.score,
                "ops": [op.to_dict() for op in self.ops]}


Objective = Callable[[Sequence[Op]], float]


def shape_objective(target: Any) -> Objective:
    """Minimise the analytic shape distance to ``target`` (a session / op stream)."""
    target_shape = (target if isinstance(target, editing.ShapeSignature)
                    else editing.shape_of(target))

    def objective(ops: Sequence[Op]) -> float:
        return editing.shape_distance(editing.shape_of(list(ops)), target_shape)

    return objective


@dataclass
class SearchResult:
    """The outcome of one search. ``history`` is best-so-far per iteration."""

    name: str
    best: Design
    start_score: float
    evaluations: int = 0
    history: List[float] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def improved(self) -> bool:
        return self.best.score < self.start_score

    def to_dict(self) -> dict:
        return {"name": self.name, "best_score": self.best.score,
                "start_score": self.start_score, "improved": self.improved,
                "evaluations": self.evaluations, "history": list(self.history),
                "vector": list(self.best.vector)}


class _Evaluator:
    """Counts evaluations and records the best-so-far trace. Deterministic."""

    def __init__(self, state: Any, dims: Sequence[Dimension], objective: Objective):
        self.state = state
        self.dims = tuple(dims)
        self.objective = objective
        self.count = 0
        self.best: Optional[Design] = None
        self.history: List[float] = []

    def __call__(self, vector: Sequence[float]) -> float:
        ops = realise(self.state, self.dims, vector)
        score = float(self.objective(ops))
        self.count += 1
        if self.best is None or score < self.best.score:
            self.best = Design(tuple(float(v) for v in vector), ops, score)
        self.history.append(self.best.score)
        return score

    def design(self) -> Design:
        if self.best is None:  # pragma: no cover - every strategy evaluates once
            raise SearchError("the strategy evaluated nothing")
        return self.best


# --------------------------------------------------------------------------- #
# Strategy adapters
# --------------------------------------------------------------------------- #
def _s_evolution(ev: _Evaluator, *, seed: int = 0, population: int = 6,
                 generations: int = 4, **kw: Any) -> Dict[str, Any]:
    """EvoCAD: a GA over CAD PROGRAMS (crossover/mutate the op tree, rank, select).

    The population is a set of ``variation.CadProgram`` views of the op stream, so
    crossover and mutation are the paper's *program* operators, not vector
    arithmetic. The RLM ranker is replaced by the caller's objective (rank = order
    by score), which is exactly the interface ``evolve`` asks for.
    """
    from harnesscad.agents.exploration import evolution as m
    from harnesscad.agents.exploration import variation as va

    dims = ev.dims
    rng = random.Random(int(seed))

    def to_program(vector: Sequence[float]) -> va.CadProgram:
        return va.CadProgram.of(*[
            va.CadOp.make("d%d" % i, value=float(x)) for i, x in enumerate(vector)])

    def to_vector(prog: va.CadProgram) -> List[float]:
        by_name = {op.name: op.param_dict().get("value", 0.0) for op in prog.ops}
        return [float(by_name.get("d%d" % i, d.value)) for i, d in enumerate(dims)]

    initial = [to_program([d.value for d in dims])]
    for _ in range(max(1, int(population)) - 1):
        initial.append(to_program([rng.uniform(d.low, d.high) for d in dims]))

    def ranker(pop: Sequence[Any], _gen: int, _rep: int) -> List[float]:
        scores = [ev(to_vector(p)) for p in pop]
        order = sorted(range(len(pop)), key=lambda i: (scores[i], i))
        return list(m.ordering_to_ranks(order, len(pop)))

    result = m.evolve(initial, ranker, va.crossover, va.mutate,
                      generations=int(generations), seed=int(seed), rank_repeats=1,
                      **kw)
    ev(to_vector(result.best_program))
    return {"generations": len(result.history),
            "best_avg_rank": result.best_avg_rank,
            "signature": va.program_signature(result.best_program)}


def _s_evolution_strategy(ev: _Evaluator, *, seed: int = 0, mu: int = 3,
                          lam: int = 8, max_generations: int = 15, **kw: Any
                          ) -> Dict[str, Any]:
    """A self-adaptive (mu, lambda) evolution strategy over the NUMERIC genome.

    RIVAL of ``evolution``: no program operators at all -- Gaussian mutation with
    self-adapting per-gene step sizes on the design vector. Different algorithm,
    different guarantees; the two are never averaged.
    """
    from harnesscad.agents.exploration import evolution_strategy as m

    x0 = current_vector(ev.dims)
    lo = min(d.low for d in ev.dims)
    hi = max(d.high for d in ev.dims)
    result = m.optimise(lambda v: ev(v), x0, seed=int(seed), mu=int(mu),
                        lam=int(lam), max_generations=int(max_generations),
                        bounds=(lo, hi), minimise=True, **kw)
    return {"generations": result.generations, "converged": result.converged,
            "history_best": list(result.history_best)}


def _dimension_space(dims: Sequence[Dimension]) -> Dict[str, Any]:
    return {"d%d" % i: ("range", d.low, d.high) for i, d in enumerate(dims)}


def _s_designspace_sampler(ev: _Evaluator, *, seed: int = 0, n: int = 24,
                           mode: str = "stratified", **kw: Any) -> Dict[str, Any]:
    """Draw a diverse set of designs and keep the best (uniform/stratified/grid).

    Coverage-first: the point of the stratified sampler is that a small budget still
    spreads across every marginal. The coverage it achieved is reported, not assumed.
    """
    from harnesscad.agents.exploration import designspace_sampler as m

    sp = _dimension_space(ev.dims)
    m.validate_space(sp)
    if mode == "uniform":
        samples = m.uniform_sample(sp, int(n), int(seed))
    elif mode == "stratified":
        samples = m.stratified_sample(sp, int(n), int(seed))
    elif mode == "grid":
        samples = m.grid_sample(sp, max(2, int(n) ** (1.0 / max(1, len(sp)))
                                        // 1 or 2), int(seed))
    else:
        raise SearchError("unknown sampling mode %r (uniform|stratified|grid)" % mode)
    keys = ["d%d" % i for i in range(len(ev.dims))]
    for sample in samples:
        ev([sample[k] for k in keys])
    return {"mode": mode, "samples": len(samples),
            "coverage": m.coverage_of_samples(samples, sp),
            "marginal_coverage": m.marginal_coverage(samples, sp)}


def _s_constrained_designspace(ev: _Evaluator, *, seed: int = 0, n: int = 24,
                               constraints: Sequence[Any] = (), **kw: Any
                               ) -> Dict[str, Any]:
    """Sample a design space with a VALIDITY GATE (rival of ``designspace_sampler``).

    Every draw must satisfy the declared inequality / divisibility constraints; an
    invalid draw is rejected and re-drawn rather than repaired. This is not a
    "better sampler" -- it answers a different question (sample the *feasible*
    region), so it is never merged with the unconstrained samplers.
    """
    from harnesscad.agents.exploration import constrained_designspace as m

    params = [m.ParameterSpec("d%d" % i, "continuous", d.low, d.high)
              for i, d in enumerate(ev.dims)]
    dspace = m.DesignSpace(params, list(constraints))
    samples = dspace.sample_valid(int(n), int(seed))
    keys = ["d%d" % i for i in range(len(ev.dims))]
    for sample in samples:
        ev([sample[k] for k in keys])
    return {"samples": len(samples), "constraints": len(list(constraints)),
            "violations_of_start": dspace.violations(
                {"d%d" % i: d.value for i, d in enumerate(ev.dims)})}


def _s_latin_hypercube(ev: _Evaluator, *, seed: int = 0, n: int = 24,
                       spread: float = 0.25, **kw: Any) -> Dict[str, Any]:
    """LHSnorm: normal-distributed Latin-hypercube draws around the CURRENT design.

    A third, distinct sampler: it is not uniform over the box, it is a stratified
    NORMAL sample centred on the design you already have (the latent-space sampler
    of the text-to-3D optimisation papers), so it explores locally but evenly.
    """
    from harnesscad.agents.exploration import latin_hypercube as m

    means = [d.value for d in ev.dims]
    stds = [abs(d.value) * float(spread) or 1.0 for d in ev.dims]
    samples = m.lhs_normal(int(n), means, stds, int(seed))
    for row in samples:
        ev(list(row))
    return {"samples": len(samples), "column_stats": m.column_stats(samples)}


def _s_greedy_refine(ev: _Evaluator, *, seed: int = 0, step: float = 0.5,
                     iters: int = 60, starts: int = 1, **kw: Any) -> Dict[str, Any]:
    """Finite-difference gradient descent from the current design (local, greedy)."""
    from harnesscad.agents.exploration import greedy_refine as m

    bounds = [(d.low, d.high) for d in ev.dims]
    if int(starts) > 1:
        optima = m.multistart_local_optima(lambda v: ev(list(v)), bounds,
                                           n_starts=int(starts), seed=int(seed),
                                           step=float(step), iters=int(iters))
        return {"local_optima": len(optima)}
    x, fx, used = m.gradient_descent(lambda v: ev(list(v)), current_vector(ev.dims),
                                     step=float(step), iters=int(iters),
                                     bounds=bounds)
    return {"final": list(x), "value": fx, "iterations": used}


def _s_guided_contact_search(ev: _Evaluator, *, seed: int = 0, steps: int = 6,
                             neighbours: int = 8, omega: float = 1.0, **kw: Any
                             ) -> Dict[str, Any]:
    """Step-scheduled, VALIDITY-FIRST neighbour selection with Pareto evidence.

    Rival of ``greedy_refine``: it never steps onto an invalid design, and it trades
    the geometry term against a regularisation term on a schedule (early steps favour
    validity/regularity, late steps favour the geometry objective). The Pareto
    evidence of the visited candidates is reported.
    """
    from harnesscad.agents.exploration import guided_contact_search as m

    rng = random.Random(int(seed))
    current = current_vector(ev.dims)
    schedule = tuple(range(1, int(steps) + 1))
    evidence: List[Any] = []

    def neighbours_of(vector, _step):
        return [[d.clamp(x * rng.uniform(0.6, 1.6))
                 for d, x in zip(ev.dims, vector)]
                for _ in range(int(neighbours))]

    def is_valid(vector) -> bool:
        return all(d.low <= x <= d.high for d, x in zip(ev.dims, vector))

    def geometry(vector) -> float:
        return ev(list(vector))

    def regularization(vector, reference) -> float:
        return sum(abs(x - y) for x, y in zip(vector, reference))

    for step in schedule:
        current, rows = m.guided_step(current, step, guidance_steps=schedule,
                                      neighbors=neighbours_of, is_valid=is_valid,
                                      geometry=geometry,
                                      regularization=regularization,
                                      omega=float(omega))
        current = list(current)
        evidence.extend(rows)
    return {"steps": len(schedule), "candidates": len(evidence),
            "pareto": len(m.pareto_evidence(evidence))}


def _s_tournament(ev: _Evaluator, *, seed: int = 0, rounds: int = 2, n: int = 6,
                  **kw: Any) -> Dict[str, Any]:
    """Co-Scientist: generate -> cluster -> Elo debate -> evolve the winners.

    Redundant designs are clustered away before the tournament, so the Elo ranking
    is over genuinely distinct candidates. The judge is the caller's objective
    (positive = a better), swap-augmented by the module itself.
    """
    from harnesscad.agents.exploration import tournament as m

    dims = ev.dims
    scored: Dict[str, float] = {}

    def make(vector: Sequence[float], vid: str, generation: int) -> m.Variant:
        ops = realise(ev.state, dims, vector)
        score = ev(list(vector))
        scored[vid] = score
        return m.Variant(id=vid, params={"vector": [float(v) for v in vector]},
                         ops=[op.to_dict() for op in ops], score=score,
                         generation=generation)

    def generate(count: int, gen_seed: int) -> List[m.Variant]:
        rng = random.Random(gen_seed)
        out = [make([d.value for d in dims], "v0", 0)]
        for i in range(1, count):
            out.append(make([rng.uniform(d.low, d.high) for d in dims],
                            "v%d" % i, 0))
        return out

    def mutator(parents: List[m.Variant], rng: random.Random) -> m.Variant:
        parent = parents[rng.randrange(len(parents))]
        vector = [d.clamp(float(x) * rng.uniform(0.7, 1.4))
                  for d, x in zip(dims, parent.params["vector"])]
        return make(vector, "%s-m%d" % (parent.id, rng.randrange(1 << 20)),
                    parent.generation + 1)

    def judge(a: m.Variant, b: m.Variant) -> float:
        return float(scored.get(b.id, math.inf) - scored.get(a.id, math.inf))

    result = m.explore(generate, rounds=int(rounds), n=int(n), seed=int(seed),
                       mutator=mutator, judge=judge, **kw)
    return {"generations": len(result.generations),
            "winner": result.winner.id if result.winner else None,
            "ranked": [v.id for v in result.final_ranked[:5]]}


def _s_technique_trials(ev: _Evaluator, *, seed: int = 0, attempts: int = 12,
                        **kw: Any) -> Dict[str, Any]:
    """Bounded multi-start trials with an EXACTLY REPLAYABLE winner.

    Each attempt is generated from a derived child seed; a failing attempt is data,
    not a crash; the winning seed is recorded so the winning design can be replayed
    bit-for-bit later. (The module maximises, so the objective is negated here -- and
    only here.)
    """
    from harnesscad.agents.exploration import technique_trials as m

    dims = ev.dims

    def generator(child_seed: int) -> List[float]:
        rng = random.Random(child_seed)
        return [rng.uniform(d.low, d.high) for d in dims]

    def evaluator(vector: Sequence[float]) -> float:
        return -ev(list(vector))          # run_trials keeps the MAX score

    run = m.run_trials(generator, evaluator, master_seed=int(seed),
                       attempts=int(attempts))
    replayed = (m.replay(generator, evaluator, run.winning_seed)
                if run.winning_seed is not None else None)
    return {"attempts": len(run.attempts), "failures": len(run.failures),
            "winning_seed": run.winning_seed,
            "replay_matches": bool(replayed is not None
                                   and replayed.score == run.winning_score)}


def _s_block_decomp(domain: Any, *, quad_weight: float = 10.0, **kw: Any
                    ) -> Dict[str, Any]:
    """Greedy block decomposition of a polygon domain (the RL MDP, played greedily).

    The MDP state, the legal cut actions and the local observation come from
    ``decomp_state``; the quality signal from ``decomp_reward``. The *policy* here is
    greedy (take the legal cut with the highest immediate reward) -- there is no
    trained policy in this repo, and a greedy rollout is an honest baseline over the
    same MDP, not a stand-in for one.
    """
    from harnesscad.agents.exploration import decomp_reward as dr
    from harnesscad.agents.exploration import decomp_state as ds

    state = ds.DecompositionState.initial(domain)
    trace: List[float] = []
    while not state.is_terminal and state.steps < 32:
        actions = state.legal_actions()
        if not actions:
            break
        best_state, best_reward = None, -math.inf
        for action in actions:
            nxt = state.apply(action)
            parts = nxt.all_blocks() + list(nxt.queue)
            r = dr.reward(parts, quad_weight=float(quad_weight)).total
            if r > best_reward:
                best_state, best_reward = nxt, r
        state = best_state
        trace.append(best_reward)
    blocks = state.all_blocks()
    components = dr.reward(blocks, quad_weight=float(quad_weight))
    return {"steps": state.steps, "blocks": len(blocks),
            "reward": components.total,
            "terminal": state.is_terminal,
            "bonus": dr.terminal_bonus(all(b.is_quad() for b in blocks)),
            "trace": trace,
            "observations": len(ds.observe_all(domain))}


@dataclass(frozen=True)
class Strategy:
    """One named search strategy."""

    name: str
    description: str
    modules: Tuple[str, ...]
    run: Callable[..., Dict[str, Any]]
    family: str = ""
    target: str = "session"

    def to_dict(self) -> dict:
        return {"name": self.name, "family": self.family, "target": self.target,
                "description": self.description, "modules": list(self.modules)}


_TABLE: Tuple[Tuple[str, str, Tuple[str, ...], Callable, str, str], ...] = (
    ("evolution",
     "EvoCAD (Preintner et al. 2025): a genetic algorithm over CAD PROGRAMS -- "
     "rank the population, select weighted parents, cross over and mutate the op "
     "tree, carry the elites.",
     ("evolution", "variation"), _s_evolution, "evolutionary", "session"),
    ("evolution_strategy",
     "A self-adaptive (mu, lambda) evolution strategy over the numeric design "
     "vector, with per-gene step sizes. NOT the CAD-code GA.",
     ("evolution_strategy",), _s_evolution_strategy, "evolutionary", "session"),
    ("designspace_sampler",
     "Coverage-first sampling of the design space (uniform / stratified / grid), "
     "reporting the marginal coverage each budget actually achieved.",
     ("designspace_sampler",), _s_designspace_sampler, "sampling", "session"),
    ("constrained_designspace",
     "Sampling a design space behind a VALIDITY GATE: inequality and divisibility "
     "constraints reject a draw rather than repair it.",
     ("constrained_designspace",), _s_constrained_designspace, "sampling", "session"),
    ("latin_hypercube",
     "LHSnorm: stratified NORMAL Latin-hypercube draws centred on the current "
     "design (local but evenly spread).",
     ("latin_hypercube",), _s_latin_hypercube, "sampling", "session"),
    ("greedy_refine",
     "Finite-difference gradient descent from the current design; optionally "
     "multistart, deduplicating the local optima it lands in.",
     ("greedy_refine",), _s_greedy_refine, "local", "session"),
    ("guided_contact_search",
     "Validity-first neighbour selection on a step schedule (regularity early, "
     "geometry late) with Pareto evidence over the visited candidates.",
     ("guided_contact_search",), _s_guided_contact_search, "local", "session"),
    ("tournament",
     "Co-Scientist exploration: generate -> cluster the redundant away -> Elo "
     "tournament with a swap-augmented judge -> evolve the winners.",
     ("tournament", "elo"), _s_tournament, "", "session"),
    ("technique_trials",
     "Bounded seeded multi-start trials: failures are data, and the winner's seed "
     "is recorded so the winning design replays exactly.",
     ("technique_trials",), _s_technique_trials, "", "session"),
    ("block_decomp",
     "Greedy rollout of the RL block-decomposition MDP (legal cuts + quality "
     "reward) over a polygon domain. A baseline policy, not a trained one.",
     ("decomp_state", "decomp_reward"), _s_block_decomp, "", "polygon"),
)

#: Same question, different algorithms. Selected by name; never averaged.
_RIVALS: Dict[str, Tuple[str, ...]] = {
    "evolutionary": ("evolution", "evolution_strategy"),
    "sampling": ("designspace_sampler", "constrained_designspace", "latin_hypercube"),
    "local": ("greedy_refine", "guided_contact_search"),
}


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
_STRATS: Optional[Dict[str, Strategy]] = None
_UNADAPTED: Tuple[str, ...] = ()


def _build() -> Dict[str, Strategy]:
    global _UNADAPTED
    entries = {e.dotted for e in capability_registry.find(package=EXPLORATION_PACKAGE)}
    adapted = set()
    out: Dict[str, Strategy] = {}
    for name, description, mods, fn, family, target in _TABLE:
        dotted = tuple(_PKG + m for m in mods)
        if any(d not in entries for d in dotted):
            continue
        adapted.update(dotted)
        out[name] = Strategy(name, description, dotted, fn, family, target)
    _UNADAPTED = tuple(sorted(d for d in entries
                              if d not in adapted and not d.endswith(".registry")))
    return out


def _all() -> Dict[str, Strategy]:
    global _STRATS
    if _STRATS is None:
        _STRATS = _build()
    return _STRATS


def strategies(family: Optional[str] = None) -> Tuple[str, ...]:
    """Every search strategy whose modules are actually in the tree."""
    return tuple(sorted(n for n, s in _all().items()
                        if family is None or s.family == family))


def strategy(name: str) -> Strategy:
    try:
        return _all()[name]
    except KeyError:
        raise UnknownStrategy("unknown search strategy %r (one of: %s)"
                              % (name, ", ".join(strategies()))) from None


def rivals() -> Dict[str, Tuple[str, ...]]:
    """Families whose members answer the same question differently."""
    return {k: tuple(v) for k, v in sorted(_RIVALS.items())}


def unadapted() -> Tuple[str, ...]:
    """Exploration modules the index knows but no strategy binds."""
    _all()
    return _UNADAPTED


# --------------------------------------------------------------------------- #
# The surface
# --------------------------------------------------------------------------- #
def search(name: str, state: Any, objective: Optional[Objective] = None, *,
           seed: int = 0, dims: Optional[Sequence[Dimension]] = None,
           **kwargs: Any) -> SearchResult:
    """Run a named search over ``state``. Objectives are MINIMISED.

    A strategy that raises is not fatal: the exception is captured in the result's
    ``detail['error']`` and the best design found before the failure is still
    returned (an empty search reports the starting design, unimproved).
    """
    strat = strategy(name)
    if strat.target != "session":
        result = strat.run(state, seed=seed, **kwargs)
        start = Design((), (), 0.0)
        return SearchResult(name, start, 0.0, detail=dict(result))
    if objective is None:
        raise SearchError("a session search needs an objective(ops) -> float")
    dimensions = tuple(dims) if dims is not None else space(state)
    ev = _Evaluator(state, dimensions, objective)
    start = current_vector(dimensions)
    detail: Dict[str, Any] = {}
    start_score = math.inf
    try:
        start_score = ev(start)
        detail = dict(strat.run(ev, seed=seed, **kwargs) or {})
    except Exception as exc:  # noqa: BLE001 - a failing component is data
        detail = {"error": "%s: %s" % (type(exc).__name__, exc)}
    best = ev.best or Design(tuple(start), realise(state, dimensions, start),
                             start_score)
    return SearchResult(name, best, start_score, ev.count, ev.history, detail)


# --------------------------------------------------------------------------- #
# CLI (wired into core.cli as `harnesscad search`)
# --------------------------------------------------------------------------- #
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--list", action="store_true",
                        help="list the search strategies")
    parser.add_argument("--rivals", action="store_true",
                        help="list the rival families (selected by name, never blended)")
    parser.add_argument("--unadapted", action="store_true",
                        help="list exploration modules with no call site yet")
    parser.add_argument("--strategy", default=None, help="the strategy to run")
    parser.add_argument("--ops", default=None, metavar="OPS.JSON",
                        help="the starting design (default: the built-in demo)")
    parser.add_argument("--target", default=None, metavar="OPS.JSON",
                        help="the target model the objective minimises toward")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--space", action="store_true",
                        help="print the searchable design space and exit")
    parser.add_argument("--json", action="store_true")


def run_cli(args: argparse.Namespace) -> int:
    from harnesscad.domain.editing.registry import _load_ops, _session

    if getattr(args, "rivals", False):
        for family, names in rivals().items():
            print("%s: %s" % (family, ", ".join(names)))
            print("    different algorithms for the same question; NEVER averaged")
        return 0
    if getattr(args, "unadapted", False):
        for dotted in unadapted():
            print(dotted)
        print("-- %d exploration modules without a call site" % len(unadapted()))
        return 0

    try:
        if getattr(args, "space", False) or getattr(args, "strategy", None):
            state = _session(_load_ops(args.ops), "stub")
        if getattr(args, "space", False):
            for d in space(state):
                print("%3d  %-10s %10.3f  [%.3f, %.3f]"
                      % (d.index, d.param, d.value, d.low, d.high))
            return 0
        if getattr(args, "strategy", None):
            if not args.target:
                print("error: --strategy needs --target OPS.JSON", file=sys.stderr)
                return 2
            target = _session(_load_ops(args.target), "stub")
            result = search(args.strategy, state, shape_objective(target),
                            seed=args.seed)
            if args.json:
                print(json.dumps(result.to_dict(), sort_keys=True, indent=2))
                return 0 if result.improved else 1
            print("strategy:    %s" % result.name)
            print("score:       %.6f -> %.6f (%s)"
                  % (result.start_score, result.best.score,
                     "improved" if result.improved else "no improvement"))
            print("evaluations: %d" % result.evaluations)
            print("detail:      %s" % json.dumps(result.detail, sort_keys=True,
                                                 default=str))
            return 0 if result.improved else 1
    except (SearchError, editing.EditError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    except OSError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    for name in strategies():
        s = strategy(name)
        tag = (" (rival family: %s)" % s.family) if s.family else ""
        print("%-24s [%s]%s" % (name, s.target, tag))
        print("    %s" % s.description)
    print()
    print("-- %d strategies / %d exploration modules unbound"
          % (len(strategies()), len(unadapted())))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad search",
        description="design-space search over a HarnessSession (selectable, rival-safe)")
    add_arguments(parser)
    return run_cli(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
