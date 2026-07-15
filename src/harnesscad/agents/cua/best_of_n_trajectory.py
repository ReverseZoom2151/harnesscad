"""best_of_n_trajectory — Agent-S's Behavior-Best-of-N, as a deterministic selector.

Agent-S's single largest reported gain (+6.6) is *Behavior Best-of-N*: instead of
committing to the first trajectory the policy produces, it GENERATES N candidate
trajectories for the same task, JUDGES each, and KEEPS the best one. The costly,
non-deterministic parts of that are (a) the generation (N model rollouts) and (b)
the judge (a model scoring each rollout). This module ports the part that is pure
and testable — the *selection* — as a deterministic function over ALREADY-GENERATED
candidates, and it does the one thing Agent-S cannot: judge with a FREE EXACT
ORACLE instead of a fallible model.

Why this maps cleanly onto HarnessCAD
-------------------------------------
Our loop is already multi-attempt (``agents/cua/loop.solve`` runs the harness for
up to ``max_iterations`` repair cycles, and a campaign runs a brief more than once).
Each attempt yields a trajectory whose outcome is GRADED by
:func:`harnesscad.agents.cua.grade.grade_ops` — an exact geometric verdict, not an
opinion. Behavior-Best-of-N over those attempts is therefore *free of the thing
that makes it expensive and noisy elsewhere*: the judge is the oracle. Pick the
attempt whose built part actually satisfies the brief, breaking ties toward the
cheaper (fewer-action, fewer-refusal) trajectory. That is the CAD-native form of
Agent-S's headline trick.

Two judges are provided, both deterministic:

* :func:`oracle_score` — the exact one, reading the verified grade signals. This is
  the one to use in HarnessCAD; it cannot be fooled.
* :func:`behavior_score` — a model-free stand-in for Agent-S's *behavior* judge that
  scores from trajectory shape alone (progress, no loops, terminated cleanly), for
  the case where no oracle verdict is attached yet.

Pure stdlib, import-safe. Duck-typed: a "verdict" is anything exposing the graded
booleans, so this never imports the live grader or the app.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence


@dataclass(frozen=True)
class TrajectoryVerdict:
    """The exact, verified signals for one candidate — the oracle's answer.

    These are exactly the fields :class:`harnesscad.agents.cua.grade.GradeResult`
    carries; kept as a small typed record so the selector never needs the live
    grader in the room. ``action_count`` / ``refusals`` are efficiency signals used
    only to break ties between two EQUALLY-correct trajectories (Agent-S prefers the
    behavior that reached the goal more directly).
    """

    solved: bool = False
    gui_valid: bool = False
    differential_agree: bool = False
    gate_ok: bool = False
    target_ok: bool = False
    action_count: int = 0
    refusals: int = 0

    @classmethod
    def from_grade(cls, grade: Any, action_count: int = 0,
                   refusals: int = 0) -> "TrajectoryVerdict":
        """Read a :class:`GradeResult`-shaped object (duck-typed) into a verdict."""
        diff = getattr(grade, "diff", None)
        return cls(
            solved=bool(getattr(grade, "solved", False)),
            gui_valid=bool(getattr(grade, "gui_valid", False)),
            differential_agree=bool(getattr(diff, "agree", False)) if diff else False,
            gate_ok=bool(getattr(grade, "gate_ok", False)),
            target_ok=bool(getattr(grade, "target_ok", False)),
            action_count=int(action_count),
            refusals=int(refusals),
        )

    def to_dict(self) -> dict:
        return {"solved": self.solved, "gui_valid": self.gui_valid,
                "differential_agree": self.differential_agree,
                "gate_ok": self.gate_ok, "target_ok": self.target_ok,
                "action_count": self.action_count, "refusals": self.refusals}


@dataclass(frozen=True)
class TrajectoryCandidate:
    """One generated trajectory, ready to be judged.

    ``actions`` is the ordered list of steps the policy took (dicts or strings —
    opaque here; only their COUNT matters to the tie-break unless a verdict already
    carries one). ``verdict`` is the exact oracle answer when available;
    ``progressed`` / ``looped`` / ``terminated`` are the behavior-only signals the
    fallback judge reads.
    """

    id: str
    actions: Sequence[Any] = field(default_factory=tuple)
    verdict: Optional[TrajectoryVerdict] = None
    progressed: bool = False
    looped: bool = False
    terminated: bool = False

    @property
    def action_count(self) -> int:
        if self.verdict is not None and self.verdict.action_count:
            return self.verdict.action_count
        return len(self.actions)

    def to_dict(self) -> dict:
        return {"id": self.id, "action_count": self.action_count,
                "verdict": None if self.verdict is None else self.verdict.to_dict(),
                "progressed": self.progressed, "looped": self.looped,
                "terminated": self.terminated}


# A scorer maps a candidate to a scalar (higher is better). Deterministic.
Scorer = Callable[[TrajectoryCandidate], float]


def oracle_score(candidate: TrajectoryCandidate) -> float:
    """Exact judge: score from the VERIFIED grade signals.

    Each satisfied correctness signal is worth a weighted point; a fully solved
    trajectory dominates any unsolved one no matter how it looks. This is the judge
    to use in HarnessCAD because the signals come from the geometric oracle, so a
    trajectory that merely *looks* right cannot outrank one that *is* right.

    A candidate with no verdict scores 0.0 — an unjudged trajectory is never
    selected over a judged one.
    """
    v = candidate.verdict
    if v is None:
        return 0.0
    score = 0.0
    if v.solved:
        score += 100.0     # a solve dominates everything else
    if v.gui_valid:
        score += 10.0
    if v.differential_agree:
        score += 10.0
    if v.gate_ok:
        score += 10.0
    if v.target_ok:
        score += 10.0
    return score


def behavior_score(candidate: TrajectoryCandidate) -> float:
    """Model-free stand-in for Agent-S's BEHAVIOR judge (no oracle needed).

    Scores trajectory *shape*: it made progress, it did not loop, it terminated on
    its own. This is the fallback for candidates that have not been graded yet; it
    is deliberately weaker than :func:`oracle_score` and is never mixed with it.
    """
    score = 0.0
    if candidate.progressed:
        score += 2.0
    if candidate.terminated:
        score += 1.0
    if candidate.looped:
        score -= 2.0
    return score


@dataclass(frozen=True)
class Selection:
    """The outcome of a Best-of-N pick: the winner, the full ranking, the why."""

    best: Optional[TrajectoryCandidate]
    ranked: List[TrajectoryCandidate]
    rationale: str

    def to_dict(self) -> dict:
        return {"best": None if self.best is None else self.best.id,
                "ranked": [c.id for c in self.ranked],
                "rationale": self.rationale}


class BehaviorBestOfN:
    """Deterministic Best-of-N selector over injected candidate trajectories.

    ``select`` ranks candidates by ``scorer`` (default the exact oracle), breaking
    ties FIRST toward fewer actions (the more direct behavior reached the goal),
    THEN toward fewer refusals, THEN by id — so the choice is a pure function of the
    inputs and identical across runs. This is Agent-S's Behavior-Best-of-N with the
    generation and the fallible judge lifted out, leaving the reproducible core.
    """

    def __init__(self, scorer: Optional[Scorer] = None) -> None:
        self.scorer: Scorer = scorer or oracle_score

    def _sort_key(self, candidate: TrajectoryCandidate):
        v = candidate.verdict
        refusals = v.refusals if v is not None else 0
        # Higher score first; then fewer actions; then fewer refusals; then id.
        return (-self.scorer(candidate), candidate.action_count, refusals,
                candidate.id)

    def rank(self, candidates: Sequence[TrajectoryCandidate]) -> List[TrajectoryCandidate]:
        return sorted(candidates, key=self._sort_key)

    def select(self, candidates: Sequence[TrajectoryCandidate]) -> Selection:
        if not candidates:
            return Selection(best=None, ranked=[], rationale="no candidates to select from")
        ranked = self.rank(candidates)
        best = ranked[0]
        best_score = self.scorer(best)
        if best_score <= 0.0:
            rationale = ("no candidate scored above zero; picked %r on tie-break "
                         "(fewest actions)" % best.id)
        else:
            rationale = ("picked %r (score %.1f) over %d others; "
                         "tie-broken toward fewer actions"
                         % (best.id, best_score, len(ranked) - 1))
        return Selection(best=best, ranked=ranked, rationale=rationale)
