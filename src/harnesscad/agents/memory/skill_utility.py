"""skill_utility -- the SKILL track of the dual-track agent memory.

Mined from *Memory-Augmented Reinforcement Learning Agent for CAD Generation*
(arXiv:2605.19748). The paper's second store is a **skill library** whose unit is

    K = (Script, Doc, params, applicability, U, stats)

  * ``Script`` -- an executable fragment (here: an ordered, parameterised tool
    template lifted out of a real trajectory).
  * ``Doc`` -- documentation stating the function, the meaning of each
    parameter, the preconditions, and -- the part usually missing from a skill
    store and the part that actually prevents damage -- the FAILURE MODES.
  * ``U`` -- a utility estimate in [0, 1].
  * ``stats`` -- uses, successes, failures, last reward, frozen flag.

WHY UTILITY AND NOT SIMILARITY
------------------------------
Recall by semantic similarity alone retrieves fragments that read as relevant
and are geometrically infeasible in the current state. The paper's fix is
measured utility from execution feedback. This module implements it exactly and
without a neural network:

  * **Update.** ``U <- U + alpha * (r - U)`` with ``alpha = 0.1`` -- an
    exponential moving estimate of the terminal reward the skill participated in.
  * **Eligibility.** A skill is recallable only while ``U >= 0.5``.
  * **Rerank.** Eligible candidates are ordered by ``0.7 * similarity + 0.3 * U``.
  * **Short-term mask.** A skill whose invocation just failed is excluded from
    the NEXT recall round only -- long enough to break a retry loop, short enough
    that one bad state does not cost a good skill its place.
  * **Disposition.** When ``U < 0.5`` and ``uses >= n_min`` the skill is FROZEN
    by default, not deleted: script, doc and stats are kept and only the
    retrievable set loses it. Deleting would destroy the evidence a human would
    need to decide whether the skill or the states it was tried in were at fault.

AUTO-INTERNALISATION
--------------------
Skills are not authored, they are *internalised*: after a successful task
:func:`internalise_trajectory` post-processes the trajectory, finds reusable
contiguous fragments, parameterises their varying arguments, and registers them.
Only clean runs of successful calls are lifted -- a fragment containing a failure
or a rollback is not reusable knowledge, it is a scar.

Stdlib only, deterministic, no wall clock. Similarity is the injectable
one-method :class:`~harnesscad.agents.memory.store.Similarity` seam (stdlib BM25
by default).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import harnesscad.agents.memory.persistence as persistence
from harnesscad.agents.memory.case_library import Case, ToolCall
from harnesscad.agents.memory.similarity import default_similarity
from harnesscad.agents.memory.store import Similarity

__all__ = [
    "SkillDoc",
    "UtilitySkill",
    "UtilitySkillLibrary",
    "internalise_trajectory",
    "ELIGIBILITY_THRESHOLD",
    "INITIAL_UTILITY",
    "UTILITY_ALPHA",
    "SIM_WEIGHT",
    "UTILITY_WEIGHT",
]

#: The paper's constants, named once so tests and callers cite the same numbers.
UTILITY_ALPHA = 0.1          # EMA rate for U <- U + alpha * (r - U)
ELIGIBILITY_THRESHOLD = 0.5  # a skill is recallable only while U >= this
#: Where a freshly internalised skill starts. Just above the bar, not at the top:
#: it inherits credit from the verified episode it was lifted from, so it is
#: recallable at once, but two failed uses are enough to take it back out of
#: recall while the freeze rule waits for n_min before making that permanent.
INITIAL_UTILITY = 0.6
SIM_WEIGHT = 0.7             # rerank = SIM_WEIGHT * sim + UTILITY_WEIGHT * U
UTILITY_WEIGHT = 0.3
FREEZE_MIN_USES = 3          # n_min: do not judge a skill on one bad episode


@dataclass
class SkillDoc:
    """The documentation half of ``K``.

    ``failure_modes`` is not decoration. A skill recalled into a prompt without
    its failure modes is an instruction to try something that has already broken;
    with them it is an instruction plus the reason to check first.
    """

    function: str = ""
    params: Dict[str, str] = field(default_factory=dict)   # name -> meaning
    preconditions: Tuple[str, ...] = ()
    failure_modes: Tuple[str, ...] = ()

    def render(self) -> str:
        lines = [self.function or "(no description)"]
        for name in sorted(self.params):
            lines.append(f"  param {name}: {self.params[name]}")
        for pre in self.preconditions:
            lines.append(f"  requires: {pre}")
        for fail in self.failure_modes:
            lines.append(f"  fails when: {fail}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "function": self.function,
            "params": dict(self.params),
            "preconditions": list(self.preconditions),
            "failure_modes": list(self.failure_modes),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SkillDoc":
        return cls(
            function=str(d.get("function", "")),
            params=dict(d.get("params", {})),
            preconditions=tuple(d.get("preconditions", ())),
            failure_modes=tuple(d.get("failure_modes", ())),
        )


@dataclass
class UtilitySkill:
    """One skill ``K = (Script, Doc, params, applicability, U, stats)``."""

    name: str
    script: str
    doc: SkillDoc = field(default_factory=SkillDoc)
    params: Dict[str, Any] = field(default_factory=dict)   # name -> default
    applicability: Tuple[str, ...] = ()                    # states/intents it fits
    utility: float = INITIAL_UTILITY
    uses: int = 0
    successes: int = 0
    failures: int = 0
    last_reward: Optional[float] = None
    frozen: bool = False
    order: int = 0

    # --- utility ----------------------------------------------------------
    def update(self, reward: float, alpha: float = UTILITY_ALPHA) -> float:
        """``U <- U + alpha * (r - U)``; also folds the reward into the stats."""
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        r = float(reward)
        if not 0.0 <= r <= 1.0:
            raise ValueError("reward must be in [0, 1]")
        self.utility += alpha * (r - self.utility)
        self.uses += 1
        if r >= 0.5:
            self.successes += 1
        else:
            self.failures += 1
        self.last_reward = r
        return self.utility

    @property
    def eligible(self) -> bool:
        """Recallable: not frozen and utility at or above the threshold."""
        return (not self.frozen) and self.utility >= ELIGIBILITY_THRESHOLD

    def document(self) -> str:
        """The text similarity matches against."""
        parts = [self.name, self.script, self.doc.function]
        parts.extend(self.applicability)
        parts.extend(str(k) for k in self.params)
        return " ".join(str(p) for p in parts if p)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "script": self.script,
            "doc": self.doc.to_dict(),
            "params": dict(self.params),
            "applicability": list(self.applicability),
            "utility": float(self.utility),
            "uses": int(self.uses),
            "successes": int(self.successes),
            "failures": int(self.failures),
            "last_reward": self.last_reward,
            "frozen": bool(self.frozen),
            "order": int(self.order),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UtilitySkill":
        return cls(
            name=str(d["name"]),
            script=str(d.get("script", "")),
            doc=SkillDoc.from_dict(d.get("doc", {})),
            params=dict(d.get("params", {})),
            applicability=tuple(d.get("applicability", ())),
            utility=float(d.get("utility", INITIAL_UTILITY)),
            uses=int(d.get("uses", 0)),
            successes=int(d.get("successes", 0)),
            failures=int(d.get("failures", 0)),
            last_reward=d.get("last_reward"),
            frozen=bool(d.get("frozen", False)),
            order=int(d.get("order", 0)),
        )


class UtilitySkillLibrary:
    """Skill store with utility-reranked recall, masking and freezing."""

    def __init__(
        self,
        similarity: Optional[Similarity] = None,
        *,
        alpha: float = UTILITY_ALPHA,
        threshold: float = ELIGIBILITY_THRESHOLD,
        sim_weight: float = SIM_WEIGHT,
        utility_weight: float = UTILITY_WEIGHT,
        freeze_min_uses: int = FREEZE_MIN_USES,
    ) -> None:
        self.similarity: Similarity = similarity or default_similarity()
        self.alpha = float(alpha)
        self.threshold = float(threshold)
        self.sim_weight = float(sim_weight)
        self.utility_weight = float(utility_weight)
        self.freeze_min_uses = int(freeze_min_uses)
        self._skills: Dict[str, UtilitySkill] = {}
        self._counter = 0
        # Short-term masking: `_mask` is consumed by the next recall,
        # `_pending` collects failures reported since that recall.
        self._mask: Set[str] = set()
        self._pending: Set[str] = set()

    # --- registration -----------------------------------------------------
    def register(self, skill: UtilitySkill) -> UtilitySkill:
        """Add a skill, or merge into the existing one of the same name.

        Re-internalising a fragment that is already known must NOT reset its
        measured utility -- that would let a lucky re-derivation launder a bad
        record. The incumbent's utility and stats survive; only the doc and
        applicability grow.
        """
        existing = self._skills.get(skill.name)
        if existing is not None:
            merged = set(existing.applicability) | set(skill.applicability)
            existing.applicability = tuple(sorted(merged))
            if not existing.doc.function:
                existing.doc = skill.doc
            return existing
        skill.order = self._counter
        self._counter += 1
        self._skills[skill.name] = skill
        return skill

    def __len__(self) -> int:
        return len(self._skills)

    def names(self) -> List[str]:
        return sorted(self._skills)

    def get(self, name: str) -> UtilitySkill:
        return self._skills[name]

    def has(self, name: str) -> bool:
        return name in self._skills

    def eligible_names(self) -> List[str]:
        return sorted(n for n, s in self._skills.items() if s.eligible)

    def frozen_names(self) -> List[str]:
        return sorted(n for n, s in self._skills.items() if s.frozen)

    # --- feedback ---------------------------------------------------------
    def record_reward(self, name: str, reward: float) -> float:
        """Fold a terminal reward into one skill's utility estimate."""
        skill = self._skills.get(name)
        if skill is None:
            raise KeyError(f"no such skill: {name}")
        return skill.update(reward, self.alpha)

    def record_invocation_failure(self, name: str) -> None:
        """Mask ``name`` from the NEXT recall round.

        Registered against the round boundary, not the clock: the failure lands
        in ``_pending`` and becomes the live mask when the next round opens
        (:meth:`begin_round`, called by :meth:`recall`), so exactly one
        subsequent recall skips it and the round after that sees it again.
        """
        if name in self._skills:
            self._pending.add(name)

    def begin_round(self) -> None:
        """Open a round: failures reported since the last one become the mask."""
        self._mask = set(self._pending)
        self._pending = set()

    def masked_names(self) -> List[str]:
        """The mask that the NEXT recall will apply."""
        return sorted(self._pending)

    # --- disposition ------------------------------------------------------
    def freeze_sweep(self) -> List[str]:
        """Freeze every skill with ``U < threshold`` and ``uses >= n_min``.

        Returns the names newly frozen. Nothing is deleted: a frozen skill keeps
        its script, doc and stats and only leaves the retrievable set.
        """
        newly: List[str] = []
        for name in sorted(self._skills):
            skill = self._skills[name]
            if skill.frozen:
                continue
            if skill.utility < self.threshold and skill.uses >= self.freeze_min_uses:
                skill.frozen = True
                newly.append(name)
        return newly

    def unfreeze(self, name: str, utility: Optional[float] = None) -> UtilitySkill:
        """Return a frozen skill to the retrievable set (a review decision)."""
        skill = self._skills[name]
        skill.frozen = False
        if utility is not None:
            skill.utility = float(utility)
        return skill

    # --- recall -----------------------------------------------------------
    def recall(self, query: str, k: int = 3) -> List[Tuple[UtilitySkill, float]]:
        """Top-``k`` eligible, unmasked skills by ``0.7 * sim + 0.3 * U``.

        The round opens here: failures reported since the previous recall become
        the live mask and are then cleared, so a just-failed skill is skipped by
        this recall and is back in the pool for the one after it.
        """
        if k < 0:
            raise ValueError("k must be non-negative")
        self.begin_round()
        pool = [
            self._skills[n]
            for n in sorted(self._skills)
            if self._skills[n].eligible and n not in self._mask
        ]
        if not pool or k == 0:
            return []
        sims = self._similarities(query, [s.document() for s in pool])
        scored: List[Tuple[float, int, UtilitySkill]] = []
        for i, skill in enumerate(pool):
            score = self.sim_weight * sims[i] + self.utility_weight * skill.utility
            scored.append((-score, skill.order, skill))
        scored.sort(key=lambda t: (t[0], t[1]))
        return [(skill, -neg) for neg, _, skill in scored[:k]]

    def _similarities(self, query: str, docs: Sequence[str]) -> List[float]:
        rank = getattr(self.similarity, "rank", None)
        if callable(rank):
            return [float(s) for s in rank(query, list(docs))]
        return [float(self.similarity.score(query, d)) for d in docs]

    # --- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": 1,
            "skills": [self._skills[n].to_dict() for n in self.names()],
        }

    @classmethod
    def from_dict(
        cls, d: dict, similarity: Optional[Similarity] = None
    ) -> "UtilitySkillLibrary":
        lib = cls(similarity=similarity)
        for raw in d.get("skills", []):
            skill = UtilitySkill.from_dict(raw)
            lib._skills[skill.name] = skill
            lib._counter = max(lib._counter, skill.order + 1)
        return lib

    def save(self, path: str) -> None:
        persistence.dump_json(self.to_dict(), path)

    @classmethod
    def load(
        cls, path: str, similarity: Optional[Similarity] = None
    ) -> "UtilitySkillLibrary":
        return cls.from_dict(persistence.load_json(path), similarity=similarity)


# --------------------------------------------------------------------------- #
# auto-internalisation
# --------------------------------------------------------------------------- #
def _fragment_name(calls: Sequence[ToolCall]) -> str:
    """A deterministic name for a fragment: its tool sequence, joined."""
    return "-".join(call.tool for call in calls)


def _parameterise(calls: Sequence[ToolCall]) -> Tuple[str, Dict[str, Any], Dict[str, str]]:
    """Lift a run of calls into a script with named parameters.

    Every argument becomes a parameter named ``<step><index>_<arg>``, defaulting
    to the value observed in the trajectory. Values are not baked in: a skill
    whose dimensions are frozen at the ones that happened to work once is a
    single case wearing a skill's clothes.
    """
    lines: List[str] = []
    params: Dict[str, Any] = {}
    meaning: Dict[str, str] = {}
    for i, call in enumerate(calls):
        args: List[str] = []
        for key in sorted(str(k) for k in call.params):
            pname = f"s{i}_{key}"
            params[pname] = call.params[key]
            meaning[pname] = f"{key} of step {i} ({call.tool})"
            args.append(f"{key}={{{pname}}}")
        lines.append(f"{call.tool}(" + ", ".join(args) + ")")
    return "\n".join(lines), params, meaning


def internalise_trajectory(
    case: Case,
    *,
    min_len: int = 2,
    max_len: int = 4,
) -> List[UtilitySkill]:
    """Post-process a SUCCESSFUL case into reusable, parameterised skills.

    Contiguous runs of ``ok`` calls are the only candidates: a fragment spanning
    a failure or a rollback is not a reusable capability. Within each clean run,
    every window of length ``min_len``..``max_len`` becomes a skill, named by its
    tool sequence so the same fragment discovered in two different tasks lands on
    one entry (and one utility record) rather than two.

    Returns the skills in a deterministic order. A case whose outcome did not
    pass yields nothing -- internalisation happens after verification, never
    before.
    """
    if min_len < 1:
        raise ValueError("min_len must be >= 1")
    if max_len < min_len:
        raise ValueError("max_len must be >= min_len")
    if not case.outcome.passed:
        return []

    runs: List[List[ToolCall]] = []
    current: List[ToolCall] = []
    for call in case.trajectory:
        if call.ok:
            current.append(call)
        else:
            if current:
                runs.append(current)
            current = []
    if current:
        runs.append(current)

    # A repaired failure is a documented failure mode for every skill lifted out
    # of the same episode: the state that broke the run is exactly the state a
    # future recall of these fragments must check for.
    failure_modes = tuple(
        f"{call.tool}: {call.repair or 'failed with no recorded repair'}"
        for call in case.trajectory
        if not call.ok
    )
    failure_modes += tuple(f"repaired cause: {c}" for c in case.outcome.repaired_causes)

    seen: Set[str] = set()
    out: List[UtilitySkill] = []
    for run in runs:
        for width in range(min_len, max_len + 1):
            for start in range(0, len(run) - width + 1):
                window = run[start : start + width]
                name = _fragment_name(window)
                if name in seen:
                    continue
                seen.add(name)
                script, params, meaning = _parameterise(window)
                doc = SkillDoc(
                    function=(
                        f"{' then '.join(c.tool for c in window)}"
                        f" -- internalised from case {case.case_id}"
                    ),
                    params=meaning,
                    preconditions=tuple(
                        sorted({r for c in window for r in c.refs_in})
                    ),
                    failure_modes=failure_modes,
                )
                out.append(
                    UtilitySkill(
                        name=name,
                        script=script,
                        doc=doc,
                        params=params,
                        applicability=(case.intent.text,),
                    )
                )
    return out
