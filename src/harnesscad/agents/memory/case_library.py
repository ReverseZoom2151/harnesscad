"""case_library -- the CASE track of the dual-track agent memory.

Mined from *Memory-Augmented Reinforcement Learning Agent for CAD Generation*
(arXiv:2605.19748). The paper keeps two stores. This module is the first one: a
**case library** whose unit is

    C = (I, T, O)

  * ``I`` -- the user INTENT: the raw request plus the structured parse the
    front end produced from it (shape class, declared dimensions, features).
  * ``T`` -- the tool-invocation TRAJECTORY: the ordered calls with their key
    parameters, the intermediate references they produced and consumed, and the
    rollback / repair records taken when a call failed.
  * ``O`` -- the OUTCOME summary: the final model reference, the verification
    items that passed, the key geometric statistics, and the causes that were
    repaired after a failure.

WRITE GATE
----------
A case is written back **only after the task passes verification**. That is the
same invariant :mod:`harnesscad.agents.memory.harness_memory` already enforces
on episodic memory ("no trajectory enters memory unless the oracle passed it"),
and it is the reason recall over this store can be trusted at all:
:meth:`CaseLibrary.recall` searches successful cases exclusively, so a
"semantically relevant but geometrically infeasible" neighbour cannot even be a
candidate. Filtering the *remaining* traps -- cases that did pass once but do not
transfer to the current state -- is the job of the learned value estimate in
:mod:`harnesscad.agents.memory.learned_retrieval`; this module only supplies the
K0 semantic candidates it reranks.

DETERMINISM AND DEPENDENCIES
----------------------------
Stdlib only. No wall clock: a case records the integer EPISODE index it was
written in, exactly as :class:`~harnesscad.agents.memory.harness_memory.HarnessMemory`
records a tick. Similarity is the injectable one-method
:class:`~harnesscad.agents.memory.store.Similarity` seam, defaulting to the
stdlib Okapi BM25 backend in :mod:`harnesscad.agents.memory.similarity`; the
paper's dense embedder (BAAI/bge-m3) drops into the same seam later via
``EmbeddingSimilarity(encode)`` without becoming a required dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import harnesscad.agents.memory.persistence as persistence
from harnesscad.agents.memory.similarity import default_similarity
from harnesscad.agents.memory.store import Similarity

__all__ = [
    "ToolCall",
    "CaseIntent",
    "CaseOutcome",
    "Case",
    "CaseLibrary",
]


def _clean(value: Any) -> Any:
    """Coerce a parameter value into something JSON-round-trippable."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return str(value)


# --------------------------------------------------------------------------- #
# T -- the trajectory
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    """One tool invocation inside a case trajectory.

    ``status`` is the paper's execution feedback at call granularity:
    ``"ok"``, ``"failed"`` (the call raised or its check failed) or
    ``"rolled_back"`` (the call was undone by a repair). ``repair`` names the
    corrective action taken afterwards, which is what makes a failed call worth
    storing at all -- the repair, not the failure, is the reusable knowledge.
    """

    tool: str
    params: Dict[str, Any] = field(default_factory=dict)
    refs_in: Tuple[str, ...] = ()
    refs_out: Tuple[str, ...] = ()
    status: str = "ok"
    repair: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def signature(self) -> str:
        """Tool name + sorted parameter NAMES -- the shape of the call, not its
        values. Two calls share a signature iff they are the same operation with
        the same knobs, which is what skill internalisation parameterises over.
        """
        return self.tool + "(" + ",".join(sorted(str(k) for k in self.params)) + ")"

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "params": _clean(dict(self.params)),
            "refs_in": list(self.refs_in),
            "refs_out": list(self.refs_out),
            "status": self.status,
            "repair": self.repair,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ToolCall":
        return cls(
            tool=str(d["tool"]),
            params=dict(d.get("params", {})),
            refs_in=tuple(d.get("refs_in", ())),
            refs_out=tuple(d.get("refs_out", ())),
            status=str(d.get("status", "ok")),
            repair=d.get("repair"),
        )


# --------------------------------------------------------------------------- #
# I -- the intent
# --------------------------------------------------------------------------- #
@dataclass
class CaseIntent:
    """The user request plus its structured parse."""

    text: str
    parse: Dict[str, Any] = field(default_factory=dict)

    def document(self) -> str:
        """The retrieval document for this intent: the raw text followed by the
        parse rendered as ``key value`` pairs, so a structured field ("through
        hole", "M6") is searchable by the same lexical backend as the prose.
        """
        parts = [str(self.text)]
        for key in sorted(self.parse, key=str):
            parts.append(f"{key} {_clean(self.parse[key])}")
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {"text": self.text, "parse": _clean(dict(self.parse))}

    @classmethod
    def from_dict(cls, d: dict) -> "CaseIntent":
        return cls(text=str(d.get("text", "")), parse=dict(d.get("parse", {})))


# --------------------------------------------------------------------------- #
# O -- the outcome
# --------------------------------------------------------------------------- #
@dataclass
class CaseOutcome:
    """What the measurement said about the finished model.

    ``passed`` mirrors the harness's terminal verdict (all execution and
    verification checks green). ``checks`` names the verification items that
    passed, ``stats`` carries the key geometric statistics (volume, bbox, face
    count, ...) and ``repaired_causes`` the failure causes that were diagnosed
    and fixed during the episode.
    """

    passed: bool = False
    model: str = ""
    checks: Tuple[str, ...] = ()
    stats: Dict[str, Any] = field(default_factory=dict)
    repaired_causes: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "passed": bool(self.passed),
            "model": self.model,
            "checks": list(self.checks),
            "stats": _clean(dict(self.stats)),
            "repaired_causes": list(self.repaired_causes),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CaseOutcome":
        return cls(
            passed=bool(d.get("passed", False)),
            model=str(d.get("model", "")),
            checks=tuple(d.get("checks", ())),
            stats=dict(d.get("stats", {})),
            repaired_causes=tuple(d.get("repaired_causes", ())),
        )


# --------------------------------------------------------------------------- #
# C = (I, T, O)
# --------------------------------------------------------------------------- #
@dataclass
class Case:
    """One stored case ``C = (I, T, O)``."""

    case_id: str
    intent: CaseIntent
    trajectory: List[ToolCall] = field(default_factory=list)
    outcome: CaseOutcome = field(default_factory=CaseOutcome)
    episode: int = 0          # logical write time (episode index), never a clock
    selections: int = 0       # how often retrieval injected this case
    successes: int = 0        # how often an episode that used it passed

    # --- retrieval surface ------------------------------------------------
    def document(self) -> str:
        """The text retrieval matches against: intent, then the tool sequence,
        then the passed checks. The trajectory is included deliberately -- two
        briefs worded differently but built the same way should be neighbours.
        """
        parts = [self.intent.document()]
        parts.extend(call.tool for call in self.trajectory)
        parts.extend(self.outcome.checks)
        return " ".join(str(p) for p in parts)

    def tool_sequence(self) -> Tuple[str, ...]:
        return tuple(call.tool for call in self.trajectory)

    def summary(self) -> str:
        """A one-line human/prompt-readable digest of the case."""
        tools = " -> ".join(self.tool_sequence()) or "(no calls)"
        checks = ", ".join(self.outcome.checks) or "(no named checks)"
        return f'{self.case_id}: "{self.intent.text}" | {tools} | passed: {checks}'

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "intent": self.intent.to_dict(),
            "trajectory": [c.to_dict() for c in self.trajectory],
            "outcome": self.outcome.to_dict(),
            "episode": int(self.episode),
            "selections": int(self.selections),
            "successes": int(self.successes),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Case":
        return cls(
            case_id=str(d["case_id"]),
            intent=CaseIntent.from_dict(d.get("intent", {})),
            trajectory=[ToolCall.from_dict(c) for c in d.get("trajectory", [])],
            outcome=CaseOutcome.from_dict(d.get("outcome", {})),
            episode=int(d.get("episode", 0)),
            selections=int(d.get("selections", 0)),
            successes=int(d.get("successes", 0)),
        )


class CaseLibrary:
    """A write-gated store of successful cases with semantic recall.

    ``write_back`` refuses anything whose outcome did not pass, so the library is
    monotonically trustworthy in the same sense as
    :class:`~harnesscad.agents.memory.skills.SkillLibrary`. ``recall`` returns the
    top ``k0`` candidates by similarity (the paper uses K0 = 20) together with
    their similarity scores; it does NOT decide what is injected. Selection is
    the learned retriever's job.
    """

    def __init__(self, similarity: Optional[Similarity] = None) -> None:
        self.similarity: Similarity = similarity or default_similarity()
        self._cases: Dict[str, Case] = {}
        self._order: List[str] = []
        self.refused: int = 0

    # --- writes -----------------------------------------------------------
    def write_back(self, case: Case) -> bool:
        """Admit ``case`` iff its outcome passed verification.

        Returns True when stored. A failed episode writes nothing here; its
        lesson goes to the error notebook and to the value model as a negative.
        """
        if not case.outcome.passed:
            self.refused += 1
            return False
        if case.case_id not in self._cases:
            self._order.append(case.case_id)
        self._cases[case.case_id] = case
        return True

    def record_use(self, case_id: str, success: bool) -> None:
        """Note that retrieval injected ``case_id`` and how the episode ended."""
        case = self._cases.get(case_id)
        if case is None:
            return
        case.selections += 1
        if success:
            case.successes += 1

    # --- reads ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._cases)

    def ids(self) -> List[str]:
        return list(self._order)

    def get(self, case_id: str) -> Case:
        return self._cases[case_id]

    def cases(self) -> List[Case]:
        return [self._cases[cid] for cid in self._order]

    def recall(self, query: str, k0: int = 20) -> List[Tuple[Case, float]]:
        """Top-``k0`` successful cases by similarity, highest first.

        Uses the backend's corpus-aware ``rank`` when it offers one (BM25 does),
        because IDF over the candidate set is what stops a word every brief
        contains from dominating. Ties break on insertion order, so the candidate
        list is a deterministic function of the stored cases.
        """
        if k0 < 0:
            raise ValueError("k0 must be non-negative")
        pool = self.cases()
        if not pool or k0 == 0:
            return []
        docs = [c.document() for c in pool]
        sims = self._similarities(query, docs)
        scored = [(-sims[i], i, pool[i]) for i in range(len(pool))]
        scored.sort(key=lambda t: (t[0], t[1]))
        return [(case, -neg) for neg, _, case in scored[:k0]]

    def _similarities(self, query: str, docs: Sequence[str]) -> List[float]:
        rank = getattr(self.similarity, "rank", None)
        if callable(rank):
            return [float(s) for s in rank(query, list(docs))]
        return [float(self.similarity.score(query, d)) for d in docs]

    # --- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {"version": 1, "cases": [c.to_dict() for c in self.cases()]}

    @classmethod
    def from_dict(
        cls, d: dict, similarity: Optional[Similarity] = None
    ) -> "CaseLibrary":
        lib = cls(similarity=similarity)
        for raw in d.get("cases", []):
            case = Case.from_dict(raw)
            lib._order.append(case.case_id)
            lib._cases[case.case_id] = case
        return lib

    def save(self, path: str) -> None:
        persistence.dump_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: str, similarity: Optional[Similarity] = None) -> "CaseLibrary":
        return cls.from_dict(persistence.load_json(path), similarity=similarity)
