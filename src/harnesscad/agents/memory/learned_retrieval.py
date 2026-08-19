"""learned_retrieval -- utility-learned recall over the dual-track memory.

Mined from *Memory-Augmented Reinforcement Learning Agent for CAD Generation*
(arXiv:2605.19748). Its central claim is a negative result about the obvious
design: retrieving by SEMANTIC SIMILARITY ALONE recalls cases that are
"semantically relevant but geometrically infeasible" -- they look right, get
injected, and then fail. Their ablation puts numbers on it (learned retrieval
SUC 0.9950 / Pass@1 0.8300 against semantic retrieval 0.9650 / 0.7850), and the
memory-configuration ablation puts numbers on the two tracks (no memory 0.9494 /
0.7528; only-case 0.9800 / 0.8200; only-skill 0.9850 / 0.8100; both 0.9950 /
0.8300).

This module is the retrieval half, sitting on
:mod:`harnesscad.agents.memory.case_library` (the case track) and
:mod:`harnesscad.agents.memory.skill_utility` (the skill track).

CASE RETRIEVAL
--------------
1. Semantic recall over SUCCESSFUL cases only -> top ``K0`` candidates (20).
2. A learned value estimate is combined with similarity by LINEAR ANNEALING::

       score = lambda_t * sim_norm + (1 - lambda_t) * val_norm

   with ``lambda`` annealed 0.9 -> 0.35 over ``anneal_episodes`` (400). Early on
   the value model has seen nothing and similarity is the only signal worth
   trusting; late on the measured utility is worth more than the resemblance.
   The schedule is driven by an EPISODE COUNTER, never a wall clock, so a run is
   reproducible.
3. ``k`` cases (5) are sampled WITHOUT REPLACEMENT from a temperature-scaled
   distribution (``c = 0.8``) with a small uniform exploration term
   (``eps = 0.05``). Deterministic top-k would collapse retrieval onto whichever
   handful of cases got there first; sampling keeps the tail reachable, and the
   seeded RNG keeps the run reproducible anyway.

VALUE TRAINING
--------------
The reward is TERMINAL and BINARY: 1 when every execution and verification check
passed, else 0. In this repo that signal already exists -- it is the gate /
verification verdict -- so :func:`terminal_reward` adapts the existing verdict
objects rather than inventing a new score.

  * Successful episode: the cases actually SELECTED become positives.
  * Negatives are drawn from the LOW-scoring unselected candidates (the paper's
    Bottom-20, 5 drawn) -- deliberately NOT every unselected candidate. An
    unselected case may have been a perfectly good alternative that sampling
    simply did not draw; labelling it negative is false supervision, and false
    supervision in a memory is indistinguishable from a lie that compounds.
  * Failed episode: the selected cases become negatives. They are what was
    actually tried, so they are what the evidence is about.

Training is binary cross-entropy plus an ENTROPY REGULARISER (coefficient 0.03)
that pushes the predicted Bernoulli back toward uncertainty, so the value head
does not saturate early and stop exploring.

The value model here is a sparse linear logistic model over lexical features of
the (query, case) pair -- deterministic, stdlib-only, and trainable inside a unit
test. The paper's dense encoder is a drop-in for the same seam: pass a
``Similarity`` backed by an embedder for recall, and the value model keeps
working on whatever features it is given.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import harnesscad.agents.memory.persistence as persistence
from harnesscad.agents.memory.case_library import Case, CaseLibrary
from harnesscad.agents.memory.similarity import default_similarity
from harnesscad.agents.memory.skill_utility import (
    UtilitySkill,
    UtilitySkillLibrary,
    internalise_trajectory,
)
from harnesscad.agents.memory.store import Similarity

__all__ = [
    "terminal_reward",
    "CaseValueModel",
    "EpisodeSelection",
    "DualTrackAgentMemory",
    "LAMBDA_START",
    "LAMBDA_END",
    "ANNEAL_EPISODES",
    "TEMPERATURE",
    "EPSILON",
    "ENTROPY_COEF",
]

#: The paper's retrieval constants.
LAMBDA_START = 0.9
LAMBDA_END = 0.35
ANNEAL_EPISODES = 400
TEMPERATURE = 0.8
EPSILON = 0.05
ENTROPY_COEF = 0.03
K0_CANDIDATES = 20
K_INJECTED = 5
BOTTOM_POOL = 20
NEGATIVES_DRAWN = 5

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _TOKEN.findall(str(text).lower())


# --------------------------------------------------------------------------- #
# reward
# --------------------------------------------------------------------------- #
def terminal_reward(outcome: Any) -> float:
    """Adapt this repo's existing verdicts to the paper's binary reward.

    Accepts a bool, a number, or any object exposing ``ok`` / ``passed`` --
    which covers :class:`~harnesscad.agents.memory.harness_memory.OracleVerdict`,
    the gate report, and the verification/gate results the harness already
    produces. Anything else is read as a failure, because an outcome that cannot
    be measured is not a pass.
    """
    if isinstance(outcome, bool):
        return 1.0 if outcome else 0.0
    if isinstance(outcome, (int, float)):
        return 1.0 if float(outcome) >= 0.5 else 0.0
    for attr in ("ok", "passed"):
        if hasattr(outcome, attr):
            return 1.0 if bool(getattr(outcome, attr)) else 0.0
    return 0.0


# --------------------------------------------------------------------------- #
# the value head
# --------------------------------------------------------------------------- #
class CaseValueModel:
    """Sparse logistic value estimate of "will this case help in this state?".

    Features are lexical and query-conditioned: tokens of the case document
    (``c:``), tokens of the query (``q:``), and their intersection (``x:``). The
    query terms give the head somewhere to put "this case is worth retrieving
    WHEN the request looks like this", which is the whole difference between a
    static case quality score and a state-conditioned value.
    """

    def __init__(
        self,
        lr: float = 0.1,
        entropy_coef: float = ENTROPY_COEF,
        max_tokens: int = 64,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if entropy_coef < 0.0:
            raise ValueError("entropy_coef must be non-negative")
        self.lr = float(lr)
        self.entropy_coef = float(entropy_coef)
        self.max_tokens = int(max_tokens)
        self.weights: Dict[str, float] = {}
        self.updates: int = 0

    # --- features ---------------------------------------------------------
    def features(self, query: str, doc: str) -> Dict[str, float]:
        q = _tokens(query)[: self.max_tokens]
        d = _tokens(doc)[: self.max_tokens]
        names = ["bias"]
        names.extend("c:" + t for t in sorted(set(d)))
        names.extend("q:" + t for t in sorted(set(q)))
        names.extend("x:" + t for t in sorted(set(q) & set(d)))
        # Scale by 1/sqrt(n) so a long document does not get a bigger gradient
        # simply for being long.
        value = 1.0 / math.sqrt(float(len(names)))
        return {name: value for name in names}

    # --- inference --------------------------------------------------------
    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0.0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    def logit(self, query: str, doc: str) -> float:
        return sum(
            self.weights.get(name, 0.0) * val
            for name, val in self.features(query, doc).items()
        )

    def value(self, query: str, doc: str) -> float:
        """The predicted probability in [0, 1] that this case leads to a pass."""
        return self._sigmoid(self.logit(query, doc))

    # --- training ---------------------------------------------------------
    def train(
        self,
        examples: Sequence[Tuple[str, str, int]],
        epochs: int = 1,
    ) -> float:
        """One or more BCE + entropy-regularised gradient passes.

        ``examples`` are ``(query, doc, label)`` with ``label`` in {0, 1}.
        Returns the mean loss over the LAST pass. The examples are consumed in
        the order given -- no shuffling -- so training is reproducible.

        Gradient of the per-example objective w.r.t. the logit ``z``::

            L  = BCE(p, y) - coef * H(p),   p = sigmoid(z)
            dL/dz = (p - y) + coef * z * p * (1 - p)

        The second term is the entropy pull: it is zero at ``z = 0`` (maximum
        uncertainty) and grows with confidence, so a saturating head is dragged
        back toward exploring.
        """
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        if not examples:
            return 0.0
        loss = 0.0
        for _ in range(epochs):
            total = 0.0
            for query, doc, label in examples:
                y = 1.0 if int(label) else 0.0
                feats = self.features(query, doc)
                z = sum(self.weights.get(n, 0.0) * v for n, v in feats.items())
                p = self._sigmoid(z)
                p_clamped = min(max(p, 1e-9), 1.0 - 1e-9)
                bce = -(y * math.log(p_clamped) + (1.0 - y) * math.log(1.0 - p_clamped))
                entropy = -(
                    p_clamped * math.log(p_clamped)
                    + (1.0 - p_clamped) * math.log(1.0 - p_clamped)
                )
                total += bce - self.entropy_coef * entropy
                grad = (p - y) + self.entropy_coef * z * p * (1.0 - p)
                for name, val in feats.items():
                    self.weights[name] = self.weights.get(name, 0.0) - self.lr * grad * val
                self.updates += 1
            loss = total / float(len(examples))
        return loss

    # --- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": 1,
            "lr": self.lr,
            "entropy_coef": self.entropy_coef,
            "max_tokens": self.max_tokens,
            "updates": self.updates,
            "weights": {k: self.weights[k] for k in sorted(self.weights)},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CaseValueModel":
        model = cls(
            lr=float(d.get("lr", 0.1)),
            entropy_coef=float(d.get("entropy_coef", ENTROPY_COEF)),
            max_tokens=int(d.get("max_tokens", 64)),
        )
        model.weights = {str(k): float(v) for k, v in d.get("weights", {}).items()}
        model.updates = int(d.get("updates", 0))
        return model


# --------------------------------------------------------------------------- #
# one episode's retrieval decision
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeSelection:
    """What retrieval did for one episode -- the record training reads back."""

    query: str
    episode: int
    lambda_t: float
    candidates: List[Tuple[str, float]] = field(default_factory=list)  # (id, score)
    selected: List[str] = field(default_factory=list)                  # case ids
    cases: List[Case] = field(default_factory=list)
    skills: List[UtilitySkill] = field(default_factory=list)

    def skill_names(self) -> List[str]:
        return [s.name for s in self.skills]


def _normalise(values: Sequence[float]) -> List[float]:
    """Min-max to [0, 1]; a flat vector maps to 0.5 (no information either way)."""
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-12:
        return [0.5 for _ in values]
    span = hi - lo
    return [(v - lo) / span for v in values]


class DualTrackAgentMemory:
    """Case library + skill library + learned retrieval, as one seam.

    Two verbs, mirroring the harness's existing memory facade:

        ``begin_episode(query)`` -> what to inject (cases and skills)
        ``end_episode(selection, outcome, case=...)`` -> everything learned

    Determinism: one seeded :class:`random.Random`, one integer episode counter.
    Two runs over the same queries and outcomes produce identical selections,
    identical utilities and identical weights.
    """

    def __init__(
        self,
        *,
        cases: Optional[CaseLibrary] = None,
        skills: Optional[UtilitySkillLibrary] = None,
        value_model: Optional[CaseValueModel] = None,
        similarity: Optional[Similarity] = None,
        seed: int = 0,
        k0: int = K0_CANDIDATES,
        k: int = K_INJECTED,
        k_skills: int = 3,
        lambda_start: float = LAMBDA_START,
        lambda_end: float = LAMBDA_END,
        anneal_episodes: int = ANNEAL_EPISODES,
        temperature: float = TEMPERATURE,
        epsilon: float = EPSILON,
        bottom_pool: int = BOTTOM_POOL,
        negatives: int = NEGATIVES_DRAWN,
        internalise: bool = True,
    ) -> None:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if anneal_episodes < 1:
            raise ValueError("anneal_episodes must be >= 1")
        sim = similarity or default_similarity()
        self.cases = cases if cases is not None else CaseLibrary(similarity=sim)
        self.skills = skills if skills is not None else UtilitySkillLibrary(similarity=sim)
        self.value_model = value_model if value_model is not None else CaseValueModel()
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self.k0 = int(k0)
        self.k = int(k)
        self.k_skills = int(k_skills)
        self.lambda_start = float(lambda_start)
        self.lambda_end = float(lambda_end)
        self.anneal_episodes = int(anneal_episodes)
        self.temperature = float(temperature)
        self.epsilon = float(epsilon)
        self.bottom_pool = int(bottom_pool)
        self.negatives = int(negatives)
        self.internalise = bool(internalise)

        self.episode: int = 0
        self.last_examples: List[Tuple[str, int]] = []   # (case_id, label)
        self.stats: Dict[str, int] = {
            "episodes": 0,
            "passed": 0,
            "failed": 0,
            "cases_written": 0,
            "cases_refused": 0,
            "skills_internalised": 0,
            "skills_frozen": 0,
        }

    # --- annealing --------------------------------------------------------
    def lambda_t(self, episode: Optional[int] = None) -> float:
        """The similarity weight at ``episode``: linear ``0.9 -> 0.35``, then flat."""
        ep = self.episode if episode is None else int(episode)
        frac = min(1.0, max(0.0, ep / float(self.anneal_episodes)))
        return self.lambda_start + (self.lambda_end - self.lambda_start) * frac

    def advance_episodes(self, n: int = 1) -> int:
        """Move the annealing clock without running an episode (tests, resumes)."""
        if n < 0:
            raise ValueError("n must be non-negative")
        self.episode += int(n)
        return self.episode

    # --- case ranking (deterministic) -------------------------------------
    def rank_cases(
        self, query: str, k0: Optional[int] = None
    ) -> List[Tuple[Case, float, float, float]]:
        """Candidates as ``(case, score, sim_norm, val_norm)``, best first.

        This is the scoring step alone -- no sampling -- so it is a pure function
        of the store, the value weights and the episode counter. Ranking is what
        the ablation is about; sampling is only how the injected subset is drawn.
        """
        recalled = self.cases.recall(query, k0=self.k0 if k0 is None else int(k0))
        if not recalled:
            return []
        sims = [s for _, s in recalled]
        vals = [self.value_model.value(query, c.document()) for c, _ in recalled]
        sim_n = _normalise(sims)
        val_n = _normalise(vals)
        lam = self.lambda_t()
        rows: List[Tuple[float, int, Case, float, float]] = []
        for i, (case, _) in enumerate(recalled):
            score = lam * sim_n[i] + (1.0 - lam) * val_n[i]
            rows.append((-score, i, case, sim_n[i], val_n[i]))
        rows.sort(key=lambda t: (t[0], t[1]))
        return [(case, -neg, s, v) for neg, _, case, s, v in rows]

    # --- sampling ---------------------------------------------------------
    def _sample(self, scored: Sequence[Tuple[Case, float]], k: int) -> List[Case]:
        """Sample ``k`` cases without replacement, temperature + epsilon-greedy."""
        pool = list(scored)
        out: List[Case] = []
        while pool and len(out) < k:
            if self._rng.random() < self.epsilon:
                idx = self._rng.randrange(len(pool))
            else:
                idx = self._softmax_draw([s for _, s in pool])
            out.append(pool.pop(idx)[0])
        return out

    def _softmax_draw(self, scores: Sequence[float]) -> int:
        top = max(scores)
        weights = [math.exp((s - top) / self.temperature) for s in scores]
        total = sum(weights)
        if total <= 0.0:
            return 0
        draw = self._rng.random() * total
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if draw < acc:
                return i
        return len(weights) - 1

    # --- the two verbs ----------------------------------------------------
    def begin_episode(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        k_skills: Optional[int] = None,
    ) -> EpisodeSelection:
        """Retrieve the cases and skills to inject for ``query``."""
        ranked = self.rank_cases(query)
        scored = [(case, score) for case, score, _, _ in ranked]
        chosen = self._sample(scored, self.k if k is None else int(k))
        skills = [
            s
            for s, _ in self.skills.recall(
                query, k=self.k_skills if k_skills is None else int(k_skills)
            )
        ]
        return EpisodeSelection(
            query=query,
            episode=self.episode,
            lambda_t=self.lambda_t(),
            candidates=[(c.case_id, s) for c, s in scored],
            selected=[c.case_id for c in chosen],
            cases=chosen,
            skills=skills,
        )

    def record_skill_failure(self, name: str) -> None:
        """A skill invocation just failed: mask it from the next recall round."""
        self.skills.record_invocation_failure(name)

    def end_episode(
        self,
        selection: EpisodeSelection,
        outcome: Any,
        *,
        case: Optional[Case] = None,
        skills_used: Optional[Sequence[str]] = None,
        train_epochs: int = 1,
    ) -> float:
        """Close the episode: learn from it, write back, and advance the clock.

        ``outcome`` is anything :func:`terminal_reward` understands -- normally
        the gate / verification verdict the harness already produces. ``case`` is
        the finished ``C = (I, T, O)``; it is written back (and internalised into
        skills) only when the outcome passed.

        Returns the terminal reward.
        """
        reward = terminal_reward(outcome)
        passed = reward >= 0.5

        # 1. value training -- positives/negatives per the paper's labelling.
        self.last_examples = self._train_value(selection, passed, train_epochs)

        # 2. case bookkeeping.
        for case_id in selection.selected:
            self.cases.record_use(case_id, passed)

        # 3. skill utility from the SAME terminal reward.
        names = list(skills_used) if skills_used is not None else selection.skill_names()
        for name in names:
            if self.skills.has(name):
                self.skills.record_reward(name, reward)

        # 4. write-back and auto-internalisation, gated on verification.
        if case is not None:
            if self.cases.write_back(case):
                self.stats["cases_written"] += 1
                if self.internalise:
                    for skill in internalise_trajectory(case):
                        before = len(self.skills)
                        self.skills.register(skill)
                        if len(self.skills) > before:
                            self.stats["skills_internalised"] += 1
            else:
                self.stats["cases_refused"] += 1

        # 5. disposition: freeze the skills the evidence has ruled against.
        self.stats["skills_frozen"] += len(self.skills.freeze_sweep())

        self.episode += 1
        self.stats["episodes"] += 1
        self.stats["passed" if passed else "failed"] += 1
        return reward

    # --- training ---------------------------------------------------------
    def _train_value(
        self, selection: EpisodeSelection, passed: bool, epochs: int
    ) -> List[Tuple[str, int]]:
        by_id = {cid: score for cid, score in selection.candidates}
        labelled: List[Tuple[str, int]] = []

        if passed:
            for cid in selection.selected:
                labelled.append((cid, 1))
            unselected = [
                (cid, score)
                for cid, score in selection.candidates
                if cid not in set(selection.selected)
            ]
            # Bottom-N by score, then a small draw from it. Only the clearly
            # low-scoring unselected candidates are treated as evidence against;
            # a near-miss is not a negative.
            unselected.sort(key=lambda t: (t[1], t[0]))
            bottom = unselected[: self.bottom_pool]
            drawn = self._draw(bottom, self.negatives)
            for cid in drawn:
                labelled.append((cid, 0))
        else:
            # A failed episode indicts what it actually used.
            for cid in selection.selected:
                labelled.append((cid, 0))

        examples: List[Tuple[str, str, int]] = []
        for cid, label in labelled:
            if cid not in by_id:
                continue
            try:
                doc = self.cases.get(cid).document()
            except KeyError:
                continue
            examples.append((selection.query, doc, label))
        if examples:
            self.value_model.train(examples, epochs=epochs)
        return labelled

    def _draw(self, pool: Sequence[Tuple[str, float]], n: int) -> List[str]:
        """Sample up to ``n`` ids from ``pool`` without replacement (seeded)."""
        items = [cid for cid, _ in pool]
        if len(items) <= n:
            return items
        chosen: List[str] = []
        remaining = list(items)
        for _ in range(n):
            idx = self._rng.randrange(len(remaining))
            chosen.append(remaining.pop(idx))
        return chosen

    # --- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": 1,
            "episode": self.episode,
            "seed": self.seed,
            "stats": dict(sorted(self.stats.items())),
            "cases": self.cases.to_dict(),
            "skills": self.skills.to_dict(),
            "value_model": self.value_model.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, d: dict, similarity: Optional[Similarity] = None
    ) -> "DualTrackAgentMemory":
        sim = similarity or default_similarity()
        mem = cls(
            cases=CaseLibrary.from_dict(d.get("cases", {}), similarity=sim),
            skills=UtilitySkillLibrary.from_dict(d.get("skills", {}), similarity=sim),
            value_model=CaseValueModel.from_dict(d.get("value_model", {})),
            similarity=sim,
            seed=int(d.get("seed", 0)),
        )
        mem.episode = int(d.get("episode", 0))
        mem.stats.update({str(k): int(v) for k, v in d.get("stats", {}).items()})
        return mem

    def save(self, path: str) -> None:
        persistence.dump_json(self.to_dict(), path)

    @classmethod
    def load(
        cls, path: str, similarity: Optional[Similarity] = None
    ) -> "DualTrackAgentMemory":
        return cls.from_dict(persistence.load_json(path), similarity=similarity)
