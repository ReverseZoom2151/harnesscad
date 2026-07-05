"""MCTS over CISP op sequences — the AlphaCAD / LATS search tier.

Where :mod:`strategies.best_of_n` spends compute in *parallel breadth* (draw N
independent whole-plans, let the verifier pick) and :mod:`strategies.reflexion`
spends it in *sequential depth* (learn one insight per retry), this module opens
the third axis the blueprint's corpus calls out:

    "AlphaCAD -- AlphaZero/MCTS-style search over op sequences; engineering has
     clear reward functions: pass verification / meet tolerance+performance
     specs; ToT/LATS for the hardest geometry."

CAD is a *verifiable-reward* domain, so it is exactly the setting where tree
search pays off: every partial op-DAG can be applied through a fresh session and
scored by the deterministic verifier (and, optionally, a quantitative
:class:`fitness.Objective`). That turns the design space into a game tree with a
dense, cheap reward — the AlphaZero/LATS regime.

Nodes are *partial op-DAG states* (the list of ops applied so far, i.e. a prefix
of a plan). One MCTS iteration is the classic four phases:

  SELECT   — descend from the root through already-expanded nodes by **UCB1**,
             balancing exploitation (high mean reward) against exploration
             (rarely-visited branches).
  EXPAND   — at the first node with an untried continuation, attach one child.
             Continuations come from an injected ``expansion(state, brief, rng)``
             (default: seeded planner draws; or a fixed op-menu).
  ROLLOUT  — apply the child's full op sequence through a FRESH
             ``session_factory()`` session and score the ``ApplyOpsResult`` with
             ``reward(apply_result, verify_report) -> float`` (default reuses
             :func:`strategies.best_of_n.default_scorer` semantics as a scalar).
  BACKPROP — add the reward to every node on the path and increment their visit
             counts, so mean values climb toward the best reachable sequence.

**Search vs. sample (P(success)).**  Best-of-N lifts the single-shot success
probability via ``P = 1 - (1 - p)^N`` but each of the N draws is independent — no
credit assignment, no reuse of a good prefix. MCTS instead *concentrates* its
budget: UCB1 backs off losing branches and pours visits into the promising one,
so shared-prefix plans are explored jointly and a good sub-assembly is reused
across many continuations. For a design whose success needs a specific *sequence*
of choices (the "hardest geometry" ToT/LATS case), search finds it with far fewer
total rollouts than blind sampling; for near-single-shot tasks Best-of-N is
cheaper. This tier is meant to sit *above* Best-of-N/Reflexion and the Elo
tournament: run MCTS to discover a strong op sequence, then hand its winner
(:attr:`MctsResult.best_ops`) down to a tournament or a Best-of-N confirm.

Determinism: all tie-breaking and any stochastic expansion goes through a single
``random.Random(seed)`` — no wall clock — so the same seed rebuilds the same tree.
Absolute imports; stdlib only. The harness spine (loop.py, agent/) is injected,
never edited.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import random

from cisp.ops import Op
from cisp.protocol import ApplyOpsResult
from verify import Diagnostic, Severity, VerifyReport


# Type aliases (documentation only).
Continuation = List[Op]                      # a segment of ops appended to a state
Expansion = Callable[[List[Op], str, "random.Random"], List[Continuation]]
Reward = Callable[[ApplyOpsResult, VerifyReport], float]

# UCB1 exploration constant. sqrt(2) is the textbook value for rewards scaled to
# roughly [0, 1] (which :func:`default_reward` guarantees).
DEFAULT_C = math.sqrt(2.0)


# --------------------------------------------------------------------------- #
# Default reward — a scalar restatement of best_of_n.default_scorer
# --------------------------------------------------------------------------- #
def default_reward(apply_result: Optional[ApplyOpsResult],
                   verify_report: Optional[VerifyReport] = None) -> float:
    """Higher-is-better reward in ``[0, 1]`` with best-of-N selection semantics.

    Mirrors :func:`strategies.best_of_n.default_scorer` — *prefer verified, then
    fewer diagnostics, then more ops applied* — but collapses the sortable tuple
    into a bounded scalar so it can be **averaged** up a search tree and fed to
    UCB1:

      * ``0.70`` weight on ``ok`` (a verified build always beats an unverified
        one — the dominant term, exactly as the tuple's leading ``ok`` field);
      * ``0.20`` weight on *fewness* of diagnostics (``1/(1+n_diags))``);
      * ``0.10`` weight on *progress* (``1 - 1/(1+applied))`` — more accepted ops.

    A ``None`` result (nothing evaluated) scores ``0.0``. Any callable with the
    ``reward(apply_result, verify_report) -> float`` shape is accepted instead —
    e.g. a :class:`fitness.Objective` closed over the report, so mass/cost/target
    tolerances drive the search rather than mere verifiability.
    """
    if apply_result is None:
        return 0.0
    ok = 1.0 if apply_result.ok else 0.0
    n_diags = len(apply_result.diagnostics)
    applied = max(0, apply_result.applied)
    return (
        0.70 * ok
        + 0.20 * (1.0 / (1.0 + n_diags))
        + 0.10 * (1.0 - 1.0 / (1.0 + applied))
    )


# --------------------------------------------------------------------------- #
# Default expansion — seeded planner continuations
# --------------------------------------------------------------------------- #
def _seeded_brief(brief: str, index: int) -> str:
    """Fold a deterministic per-draw seed into the brief (mirrors best_of_n).

    ``Math.random`` is unavailable and we want reproducible trees, so expansion
    diversity comes from an explicit nonce line the planner can read. Draw 0 is
    the untouched brief; later draws carry a distinct seed.
    """
    if index == 0:
        return brief
    return (
        f"{brief}\n\n"
        f"[MCTS expansion draw #{index}; propose an alternative continuation. "
        f"seed={index}]"
    )


def planner_expansion(planner, k: int = 3,
                      session_factory: Optional[Callable[[], object]] = None
                      ) -> Expansion:
    """Build a default ``expansion`` that asks the planner for continuations.

    Draws ``k`` seeded plans from ``planner`` and, for each, returns the *tail*
    that extends the current prefix ``state`` (``plan[len(state):]``) as one
    candidate continuation. Shared-prefix plans therefore branch the tree exactly
    where they diverge — the search decides which tail to commit to. Duplicate
    and empty tails are dropped so a node never re-expands the same child.

    This is a convenience default: for a menu of hand-authored op segments use
    :func:`menu_expansion`; tests inject their own ``expansion``. The planner is
    called with the (optional) live ``session_factory`` state summary so its
    continuations are state-aware.
    """
    def expansion(state: List[Op], brief: str, rng: "random.Random"
                  ) -> List[Continuation]:
        summary = None
        if session_factory is not None:
            try:
                summary = session_factory().summary()
            except Exception:
                summary = None
        seen: set = set()
        out: List[Continuation] = []
        for i in range(max(1, k)):
            try:
                plan = planner.plan(_seeded_brief(brief, i), state_summary=summary)
            except Exception:
                continue
            plan = list(plan or [])
            if len(plan) <= len(state):
                continue
            tail = plan[len(state):]
            key = tuple(_op_key(op) for op in tail)
            if key in seen:
                continue
            seen.add(key)
            out.append(tail)
        return out

    return expansion


def menu_expansion(menu: Sequence[Continuation],
                   max_depth: Optional[int] = None) -> Expansion:
    """A fixed op-menu expansion: every non-terminal node may append any segment.

    ``menu`` is a list of candidate continuations (each a list of ops). Handy for
    tests and for a bounded, hand-authored action set. ``max_depth`` (in appended
    segments) caps recursion; ``None`` offers the menu at every depth (the search
    still terminates via the ``max_depth`` guard in :func:`mcts_search`).
    """
    frozen = [list(seg) for seg in menu]

    def expansion(state: List[Op], brief: str, rng: "random.Random"
                  ) -> List[Continuation]:
        if max_depth is not None and len(state) >= max_depth:
            return []
        return [list(seg) for seg in frozen]

    return expansion


def _op_key(op: Op):
    """A hashable identity for an op (for de-duping continuations)."""
    to_dict = getattr(op, "to_dict", None)
    if callable(to_dict):
        try:
            d = to_dict()
            return tuple(sorted((k, repr(v)) for k, v in d.items()))
        except Exception:
            pass
    return repr(op)


# --------------------------------------------------------------------------- #
# Tree
# --------------------------------------------------------------------------- #
@dataclass
class MctsNode:
    """One node = a partial op-DAG state (the ops applied from the root)."""

    ops: List[Op]                                  # full op prefix at this node
    depth: int = 0                                 # number of appended segments
    parent: Optional["MctsNode"] = None
    children: List["MctsNode"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    # Lazily filled the first time the node is selected-through.
    untried: Optional[List[Continuation]] = None   # None => not yet expanded
    # Cached rollout (deterministic per state, so evaluated at most once).
    reward: Optional[float] = None
    result: Optional[ApplyOpsResult] = None

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    @property
    def is_expanded(self) -> bool:
        return self.untried is not None

    @property
    def fully_expanded(self) -> bool:
        return self.untried is not None and not self.untried


@dataclass
class MctsResult:
    """The outcome of a search.

    Attributes:
        best_ops: the highest-reward op sequence discovered (a full op prefix).
        best_score: that sequence's reward.
        best_result: its ``ApplyOpsResult`` (verifier output for the winner).
        root: the search tree root (visit counts show where budget concentrated).
        iterations: MCTS iterations actually run.
        tree_size: number of nodes created (root included).
    """

    best_ops: List[Op] = field(default_factory=list)
    best_score: float = float("-inf")
    best_result: Optional[ApplyOpsResult] = None
    root: Optional[MctsNode] = None
    iterations: int = 0
    tree_size: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.best_result and self.best_result.ok)


# --------------------------------------------------------------------------- #
# Rollout / evaluation
# --------------------------------------------------------------------------- #
def _evaluate(ops: List[Op], session_factory: Callable[[], object],
              reward: Reward) -> tuple:
    """Apply ``ops`` through a FRESH session and score the result.

    Returns ``(reward_value, ApplyOpsResult)``. A planner/backend blow-up becomes
    a not-ok result carrying an ``apply-error`` diagnostic (so it simply scores
    low rather than crashing the search) — the same block-and-correct discipline
    Best-of-N and Reflexion use.
    """
    session = session_factory()
    try:
        result = session.apply_ops(list(ops))
    except Exception as exc:  # backend/op blow-up -> lowest-value leaf
        try:
            digest = session.digest()
        except Exception:
            digest = ""
        result = ApplyOpsResult(
            ok=False, applied=0, digest=digest,
            diagnostics=[Diagnostic(
                Severity.ERROR, "apply-error",
                f"rollout failed to apply ops: {type(exc).__name__}: {exc}")],
            rejected=None)
    report = VerifyReport(list(result.diagnostics))
    return reward(result, report), result


# --------------------------------------------------------------------------- #
# Selection (UCB1)
# --------------------------------------------------------------------------- #
def _ucb1(child: MctsNode, parent_visits: int, c: float) -> float:
    """UCB1 score of ``child`` given its parent's visit count."""
    if child.visits == 0:
        return float("inf")
    exploit = child.mean_value
    explore = c * math.sqrt(math.log(parent_visits) / child.visits)
    return exploit + explore


def _select_child(node: MctsNode, rng: "random.Random", c: float) -> MctsNode:
    """Pick the max-UCB1 child, breaking ties deterministically via ``rng``."""
    best_score = float("-inf")
    best: List[MctsNode] = []
    for child in node.children:
        s = _ucb1(child, node.visits, c)
        if s > best_score:
            best_score, best = s, [child]
        elif s == best_score:
            best.append(child)
    if len(best) == 1:
        return best[0]
    return best[rng.randrange(len(best))]


# --------------------------------------------------------------------------- #
# The search
# --------------------------------------------------------------------------- #
def mcts_search(
    planner,
    session_factory: Callable[[], object],
    brief: str,
    *,
    iterations: int = 64,
    expansion: Optional[Expansion] = None,
    reward: Optional[Reward] = None,
    seed: int = 0,
    c: float = DEFAULT_C,
    max_depth: int = 8,
    root_ops: Optional[List[Op]] = None,
) -> MctsResult:
    """Monte-Carlo Tree Search over CISP op sequences (AlphaCAD / LATS tier).

    Args:
        planner: object with ``plan(brief, state_summary=None, diagnostics=None)
            -> [Op]`` (e.g. ``agent.planner.Planner``). Used only by the default
            ``expansion``; ignored if you inject your own.
        session_factory: zero-arg factory returning a FRESH ``HarnessSession``.
            Called once per rollout so evaluations never share state.
        brief: the natural-language design brief (passed to ``expansion``).
        iterations: number of MCTS iterations (rollouts) to run.
        expansion: ``expansion(state, brief, rng) -> list[continuation]`` proposing
            candidate next op-segments for a partial state. Defaults to
            :func:`planner_expansion` (seeded planner draws). Pass
            :func:`menu_expansion` for a fixed action set.
        reward: ``reward(apply_result, verify_report) -> float`` (higher better).
            Defaults to :func:`default_reward`. Any :class:`fitness.Objective`
            score works here too.
        seed: seed for the single ``random.Random`` driving all tie-breaks and any
            stochastic expansion — fixes the tree for a given input.
        c: UCB1 exploration constant (``sqrt(2)`` suits ``[0,1]`` rewards).
        max_depth: cap on appended segments (search depth) — bounds the tree even
            when ``expansion`` never returns ``[]``.
        root_ops: optional starting op prefix (seed the search from partway).

    Returns:
        :class:`MctsResult` with ``best_ops`` (the highest-reward sequence found),
        ``best_score``, the winning ``ApplyOpsResult``, the ``root`` tree, and the
        ``iterations`` / ``tree_size`` actually used.
    """
    if iterations < 0:
        raise ValueError(f"iterations must be >= 0 (got {iterations})")
    rng = random.Random(seed)
    expand = expansion or planner_expansion(planner, session_factory=session_factory)
    score = reward or default_reward

    root = MctsNode(ops=list(root_ops or []), depth=0)
    tree_size = 1

    best_ops: List[Op] = []
    best_score = float("-inf")
    best_result: Optional[ApplyOpsResult] = None

    def rollout(node: MctsNode) -> float:
        """Evaluate ``node`` (cached — states are deterministic) and track best."""
        nonlocal best_ops, best_score, best_result
        if node.reward is None:
            r, res = _evaluate(node.ops, session_factory, score)
            node.reward, node.result = r, res
            # Strictly-better keeps the earliest (shortest-found) winner on ties.
            if r > best_score:
                best_score = r
                best_ops = list(node.ops)
                best_result = res
        return node.reward

    for _ in range(iterations):
        # --- SELECT: descend through fully-expanded nodes by UCB1 ---------- #
        node = root
        path = [root]
        while True:
            if node.depth >= max_depth:
                break  # depth cap -> treat as terminal, re-evaluate leaf
            if node.untried is None:  # first visit: ask expansion for children
                node.untried = list(expand(node.ops, brief, rng))
            if node.untried:
                # --- EXPAND: attach exactly one new child ----------------- #
                cont = node.untried.pop(0)
                child = MctsNode(ops=node.ops + list(cont),
                                 depth=node.depth + 1, parent=node)
                node.children.append(child)
                tree_size += 1
                path.append(child)
                node = child
                break
            if not node.children:
                break  # genuine terminal (expansion offered nothing)
            node = _select_child(node, rng, c)
            path.append(node)

        # --- ROLLOUT + BACKPROP ------------------------------------------- #
        value = rollout(node)
        for n in path:
            n.visits += 1
            n.value_sum += value

    # Handle iterations == 0 (or a root that was never rolled out): still report
    # the root's own sequence as the trivial best so callers get a valid result.
    if best_result is None:
        r, res = _evaluate(root.ops, session_factory, score)
        best_ops, best_score, best_result = list(root.ops), r, res

    return MctsResult(
        best_ops=best_ops,
        best_score=best_score,
        best_result=best_result,
        root=root,
        iterations=iterations,
        tree_size=tree_size,
    )
