"""Empirical task difficulty, measured from observed outcomes.

:mod:`harnesscad.eval.curriculum.complexity` is a STRUCTURAL PROXY: it counts
what a task's reference solution is made of. It cannot know that a four-op brief
is the hardest one in the corpus because its stated dimensions are the trap. The
pressure corpus contains exactly that case -- five ``trap_*`` briefs, all
hand-labelled difficulty 4, all structurally indistinguishable from the
difficulty-2 briefs they are built from.

ReCAD (arXiv:2512.06328) closes that gap with a measurement rather than a
heuristic: it samples N candidate solutions per task and calls a task HARD when
the maximum reward over those N samples falls below a threshold ``h`` (they use
``h = 0.8``); hard tasks then receive extra guidance. The max, not the mean, is
the right statistic -- a task the policy solved once is a task the policy CAN
solve, and averaging that away would mark a solved-but-noisy task as hard.

This module implements that rule against whatever the harness already observes
per attempt: a scalar reward, or a pass/fail verdict (``True`` -> 1.0, ``False``
-> 0.0). It is deliberately agnostic about where the numbers come from -- a
pressure ``results.json``, ``agents.selftrain.ledger`` certifications, a bench
run -- so it takes ids and floats and nothing else.

Determinism: the ledger keeps insertion order per task and every public
accessor sorts by task id, so two ledgers holding the same observations report
the same difficulties and the same hard set, in the same order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from harnesscad.eval.curriculum import complexity as _cx
from harnesscad.eval.curriculum import ordering as _ord

__all__ = [
    "HARD_THRESHOLD",
    "as_reward",
    "empirical_difficulty",
    "TaskOutcome",
    "DifficultyLedger",
    "order_by_measured",
]

#: ReCAD's hard-task threshold: max reward over the sampled solutions < h.
HARD_THRESHOLD = 0.8


def as_reward(value: Any) -> float:
    """Coerce an observation to a reward in [0, 1].

    ``True``/``False`` become 1.0/0.0 so a pass/fail verdict can be recorded
    directly. Numbers are clamped into [0, 1] -- a reward outside the unit
    interval would silently break the threshold comparison, and clamping is the
    behaviour the ReCAD rule assumes.
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    reward = float(value)
    if reward < 0.0:
        return 0.0
    if reward > 1.0:
        return 1.0
    return reward


def empirical_difficulty(rewards: Sequence[float]) -> Optional[float]:
    """Difficulty in [0, 1] from observed rewards, or ``None`` if unobserved.

    ``1 - max(rewards)``: the complement of the best result the task ever
    produced. A task solved perfectly once has difficulty 0; a task never solved
    at all has difficulty 1.
    """
    values = [as_reward(r) for r in rewards]
    if not values:
        return None
    return round(1.0 - max(values), 9)


@dataclass(frozen=True)
class TaskOutcome:
    """Every reward observed for one task, in the order they were recorded."""

    task_id: str
    rewards: Tuple[float, ...] = ()

    @property
    def n(self) -> int:
        return len(self.rewards)

    @property
    def max_reward(self) -> Optional[float]:
        return max(self.rewards) if self.rewards else None

    @property
    def mean_reward(self) -> Optional[float]:
        if not self.rewards:
            return None
        return round(sum(self.rewards) / len(self.rewards), 9)

    @property
    def difficulty(self) -> Optional[float]:
        return empirical_difficulty(self.rewards)

    def pass_rate(self, threshold: float = HARD_THRESHOLD) -> Optional[float]:
        """Fraction of samples that reached ``threshold``."""
        if not self.rewards:
            return None
        hits = sum(1 for r in self.rewards if r >= threshold)
        return round(hits / len(self.rewards), 9)

    def is_hard(self, threshold: float = HARD_THRESHOLD) -> bool:
        """ReCAD's rule: no sampled solution reached ``threshold``.

        An UNOBSERVED task is not hard -- it is unmeasured, and calling it hard
        would let an empty ledger declare the whole corpus hard.
        """
        best = self.max_reward
        return best is not None and best < threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "n": self.n,
            "rewards": list(self.rewards),
            "max_reward": self.max_reward,
            "mean_reward": self.mean_reward,
            "difficulty": self.difficulty,
        }


class DifficultyLedger:
    """Accumulates per-task outcomes and answers the ReCAD difficulty queries."""

    def __init__(self, observations: Optional[Dict[str, Sequence[float]]] = None):
        self._rewards: Dict[str, List[float]] = {}
        for task_id, rewards in sorted((observations or {}).items()):
            self.extend(task_id, rewards)

    # -- recording --------------------------------------------------------- #
    def record(self, task_id: str, reward: Any) -> None:
        """Record one observation (a reward in [0, 1], or a pass/fail bool)."""
        self._rewards.setdefault(str(task_id), []).append(as_reward(reward))

    def extend(self, task_id: str, rewards: Iterable[Any]) -> None:
        for reward in rewards:
            self.record(task_id, reward)

    # -- queries ----------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._rewards)

    def __contains__(self, task_id: object) -> bool:
        return str(task_id) in self._rewards

    def outcome(self, task_id: str) -> TaskOutcome:
        """The outcome for one task (empty, not an error, when unobserved)."""
        return TaskOutcome(str(task_id), tuple(self._rewards.get(str(task_id), ())))

    def outcomes(self) -> Tuple[TaskOutcome, ...]:
        """Every observed task, ordered by id."""
        return tuple(self.outcome(t) for t in sorted(self._rewards))

    def difficulty(self, task_id: str) -> Optional[float]:
        """Measured difficulty of one task, or ``None`` when unobserved."""
        return self.outcome(task_id).difficulty

    def hard_tasks(self, threshold: float = HARD_THRESHOLD) -> Tuple[str, ...]:
        """Ids whose MAX reward stayed below ``threshold`` -- ReCAD's hard set.

        Ordered hardest first (lowest max reward), with the task id as the
        deterministic tie-break.
        """
        hard = [o for o in self.outcomes() if o.is_hard(threshold)]
        hard.sort(key=lambda o: (o.max_reward, o.task_id))
        return tuple(o.task_id for o in hard)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold": HARD_THRESHOLD,
            "outcomes": [o.to_dict() for o in self.outcomes()],
        }


# --------------------------------------------------------------------------- #
# re-ordering a curriculum by what was actually measured
# --------------------------------------------------------------------------- #
def order_by_measured(
    tasks: Sequence[Any],
    ledger: DifficultyLedger,
    mode: str = "flat",
) -> List[Any]:
    """Re-order a curriculum by MEASURED difficulty instead of structure.

    A task the ledger has seen sorts by its empirical difficulty. A task it has
    NOT seen is imputed: its position in the static curriculum (``mode``) is
    converted to ``position / n``, putting it on the same [0, 1] scale as a
    measured difficulty without pretending it was observed. Dividing by ``n``
    rather than ``n - 1`` is deliberate -- the imputed value stays strictly
    below 1.0, so a task MEASURED as never solved always sorts behind a merely
    structurally-complex one. The static key is retained as the tie-break, so
    tasks with equal difficulty stay in structural order and the result is
    total.

    With an empty ledger this reduces exactly to
    :func:`ordering.order_tasks` -- the structural ordering is the prior, and
    measurement only ever refines it.
    """
    static = _ord.order_tasks(tasks, mode=mode)
    span = max(len(static), 1)
    imputed = {
        _cx.task_id(task): position / span for position, task in enumerate(static)
    }
    key_fn = _ord.KEYS[mode]

    def key(task: Any):
        tid = _cx.task_id(task)
        measured = ledger.difficulty(tid)
        value = imputed[tid] if measured is None else measured
        return (value, key_fn(task))

    return sorted(static, key=key)
