"""HarnessMemory — the one memory the agent that builds parts actually has.

The parts were all here and none of them were plugged in. ``MemoryStore``
(four types), ``SkillLibrary`` (Voyager, execution-verified), ``ErrorNotebook``
(corrective few-shot), ``decay`` (Ebbinghaus reinforced forgetting) and
``reflexion.heuristic_reflect`` (diagnostics -> reusable insight) all existed,
were tested, and were unreachable from ``core.harness.AgentHarness``. This module
does not build a fifth thing. It is a **facade** that composes the four that
exist and gives the harness exactly two verbs:

    recall(brief)  -> everything the agent should have been told
    commit(...)    -> everything the agent is allowed to remember

THE ORACLE GATE ON WRITES
-------------------------
Agent-S built an experience-augmented memory as its headline ICLR contribution
and then deleted it in the version that set SOTA ("simpler, better, and faster"):
retrieved experience was net-negative because their store had no way to know
whether a remembered trajectory had ever actually *worked*, so it filled with
plausible garbage and poisoned the loop.

We are the one project that does not have that problem. ``io/gate.py`` can tell
us, without a human, whether the part we built is real. So:

    **No trajectory enters memory unless the oracle passed it.**

``commit`` takes an :class:`OracleVerdict`, not a model's self-report. A failed
run writes NOTHING to episodic memory and contributes NO exemplar. This is the
same lever that lost the pressure experiment: a false diagnostic is an
instruction, and a capable model obeys it precisely. A false memory is the same
instruction with a longer fuse.

The single exception, and it is the point of the whole module: when the oracle
says the part is FINE and the verifier fleet said it was broken, that
disagreement is itself a verified fact, and it is written to the ErrorNotebook
as a **verifier false positive**. The washer (80 mm disc, 8 mm thick, 30 mm
bore) was rejected forty times by one bad rule and nothing in the harness ever
noticed the pattern. Now it does, and the record is exactly the evidence
``eval/selftest/fleet_audit.py`` scores rules on.

DETERMINISM
-----------
No wall clock. Decay is driven by a **logical clock**: ``HarnessMemory.tick`` is
an event count, incremented once per ``commit``, and it is what
``memory.decay`` is handed as its "day" number. Two runs over the same briefs in
the same order produce byte-identical memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad.agents.memory.decay import Salience, decay_sweep, reinforce, retention
from harnesscad.agents.memory.error_notebook import ErrorNotebook, ErrorNotebookEntry
from harnesscad.agents.memory.skills import Skill, SkillLibrary
from harnesscad.agents.memory.store import Episode, MemoryStore, Similarity

__all__ = [
    "OracleVerdict",
    "gate_oracle",
    "Recalled",
    "HarnessMemory",
    "FALSE_POSITIVE_PREFIX",
    "SEMANTIC_INSIGHT_KEY",
]


# Insights live under the SAME semantic key reflexion.py already uses, so the
# reflexion strategy and the harness share one insight list instead of two.
SEMANTIC_INSIGHT_KEY = "reflexion:insights"

# ErrorNotebook entries recording a verifier false positive are tagged in their
# specification-independent `insight` field with this prefix, so the fleet audit
# can pull them out without a second store.
FALSE_POSITIVE_PREFIX = "verifier-false-positive"


# --------------------------------------------------------------------------- #
# the oracle
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OracleVerdict:
    """What the *measurement* said — never what the model said.

    ``ok`` is the only thing that gates a write. ``failures`` are the gate's own
    named measurement failures (empty iff ok). ``source`` names the instrument so
    a stored episode can be traced back to the thing that admitted it.
    """

    ok: bool
    failures: Tuple[str, ...] = ()
    source: str = "gate"

    def to_dict(self) -> dict:
        return {"ok": self.ok, "failures": list(self.failures), "source": self.source}


def gate_oracle(session: Any, ops: Optional[Sequence[Any]] = None) -> OracleVerdict:
    """The default oracle: ``io.gate.check`` over the session's built geometry.

    This is the harness's existing output gate — the one door every artifact
    leaves through — used here as an admission gate on memory instead. It is
    reference-free: it proves the part is closed, manifold, non-degenerate and
    honours the intent its own op stream declared. It cannot know the brief asked
    for four holes and the model cut one. That bound is stated, not hidden: see
    :meth:`HarnessMemory.limits`.

    ``ops`` (when given) supplies the declared-intent source so the gate can also
    check shell/cut/extrude intent. A gate that raises is a failed verdict, never
    an exception into the loop.
    """
    from harnesscad.io import gate as gate_mod

    backend = getattr(session, "backend", session)
    try:
        report = gate_mod.check(backend, None, source=list(ops) if ops else None)
    except Exception as exc:  # noqa: BLE001 - an unmeasurable part is not a pass
        return OracleVerdict(False, (f"gate-error: {type(exc).__name__}: {exc}",))
    failures = tuple(
        f"{getattr(f, 'code', 'failure')}: {getattr(f, 'message', str(f))}"
        for f in getattr(report, "failures", ())
    )
    return OracleVerdict(bool(getattr(report, "ok", False)), failures)


# --------------------------------------------------------------------------- #
# what retrieval hands back
# --------------------------------------------------------------------------- #
@dataclass
class Recalled:
    """Everything memory has to say about one brief, ready for the prompt."""

    episodes: List[Episode] = field(default_factory=list)
    skills: List[Skill] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    false_positives: List[ErrorNotebookEntry] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.episodes or self.skills or self.insights
                    or self.false_positives)

    def to_dict(self) -> dict:
        return {
            "episodes": [e.to_dict() for e in self.episodes],
            "skills": [s.name for s in self.skills],
            "insights": list(self.insights),
            "false_positives": [e.to_dict() for e in self.false_positives],
        }

    # --- prompt composition ------------------------------------------------
    def prompt_block(self, max_ops: int = 24) -> str:
        """Render as ONE prompt section. Empty string when memory has nothing.

        The exemplars are ORACLE-VERIFIED op streams — parts that were actually
        measured and passed — so the model is shown a working answer to a similar
        brief rather than being asked to rediscover it. This is the few-shot the
        system prompt never had (``agents/agent/system_prompt.py`` is pure
        zero-shot, by construction: it is the static contract, and exemplars are
        dynamic, so they belong in the user turn where retrieval put them).
        """
        if not self:
            return ""
        parts: List[str] = []

        if self.episodes:
            lines = [
                "VERIFIED PRIOR SOLUTIONS — these op streams were BUILT and PASSED "
                "the measured output gate. They are examples of the required "
                "output format and of constructions that work. Adapt the "
                "dimensions to the brief; do not copy them blindly."
            ]
            for ep in self.episodes:
                lines.append(f'\nBrief: "{ep.brief}"')
                ops = ep.ops[:max_ops]
                lines.append("Ops: " + json.dumps(ops, sort_keys=True))
                if len(ep.ops) > max_ops:
                    lines.append(f"  (... {len(ep.ops) - max_ops} further ops elided)")
            parts.append("\n".join(lines))

        if self.skills:
            lines = ["VERIFIED SKILLS available as construction patterns:"]
            for sk in self.skills:
                lines.append(f"- {sk.name}: {sk.description}")
            parts.append("\n".join(lines))

        if self.insights:
            lines = ["LESSONS from earlier failed attempts on similar briefs:"]
            lines += [f"- {s}" for s in self.insights]
            parts.append("\n".join(lines))

        if self.false_positives:
            lines = [
                "KNOWN VERIFIER FALSE POSITIVES — on a similar brief the checker "
                "raised these codes and the MEASURED gate then found the part "
                "correct. Treat a repeat of these codes as unreliable; do not "
                "distort a correct part to silence them:"
            ]
            for e in self.false_positives:
                codes = ", ".join(e.known_wrong()) or "(none)"
                lines.append(f'- on "{e.specification}": {codes}')
            parts.append("\n".join(lines))

        return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# the facade
# --------------------------------------------------------------------------- #
class HarnessMemory:
    """MemoryStore + SkillLibrary + ErrorNotebook + Ebbinghaus decay, as one seam.

    Args:
        store: the four-type :class:`MemoryStore`. Created if omitted.
        skills: an execution-verified :class:`SkillLibrary`. Optional.
        notebook: the corrective :class:`ErrorNotebook`. Created if omitted.
        k_episodes / k_skills / k_insights / k_false_positives: retrieval widths.
        min_similarity: an episode below this lexical similarity to the new brief
            is not recalled at all. A far-fetched exemplar is a distractor, and a
            distractor is how Agent-S's memory went net-negative.
        tau / forget_threshold: the Ebbinghaus parameters. Retention
            ``R = exp(-dt / (S*tau))`` where ``dt`` is measured in **ticks**
            (commits), not seconds.
        keep_min: episodes protected from forgetting regardless of retention.
    """

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        *,
        skills: Optional[SkillLibrary] = None,
        notebook: Optional[ErrorNotebook] = None,
        similarity: Optional[Similarity] = None,
        k_episodes: int = 2,
        k_skills: int = 2,
        k_insights: int = 3,
        k_false_positives: int = 2,
        min_similarity: float = 0.25,
        tau: float = 5.0,
        forget_threshold: float = 0.15,
        keep_min: int = 8,
    ) -> None:
        self.store = store if store is not None else MemoryStore(similarity=similarity)
        self.skills = skills
        self.notebook = notebook if notebook is not None else ErrorNotebook(scorer="jaccard")
        self.k_episodes = int(k_episodes)
        self.k_skills = int(k_skills)
        self.k_insights = int(k_insights)
        self.k_false_positives = int(k_false_positives)
        self.min_similarity = float(min_similarity)
        self.tau = float(tau)
        self.forget_threshold = float(forget_threshold)
        self.keep_min = int(keep_min)

        # THE LOGICAL CLOCK. Never a wall clock. One tick per commit.
        self.tick: int = 0
        self._salience: Dict[str, Salience] = {}
        # Counters so the A/B can report what memory actually did.
        self.stats: Dict[str, int] = {
            "recalls": 0, "recall_hits": 0,
            "commits_admitted": 0, "commits_refused": 0,
            "false_positives_recorded": 0, "insights_written": 0,
            "forgotten": 0,
        }

    # --- keys -------------------------------------------------------------
    @staticmethod
    def _episode_key(ep: Episode) -> str:
        return f"{ep.digest or 'nodigest'}:{len(ep.ops)}"

    # --- retrieval --------------------------------------------------------
    def recall(self, brief: str) -> Recalled:
        """Everything memory knows about ``brief``, reinforcing what it returns.

        Episodes are ranked by ``similarity * retention``: an episode that has
        not been recalled in a long time fades, and every recall reinforces it
        (``S += 1``, clock reset). That is the Ebbinghaus curve in ``decay.py``,
        finally attached to the store it was written for — the two half-systems
        the audit found were never connected.
        """
        self.stats["recalls"] += 1

        pool = [e for e in self.store.episodic if e.outcome == "ok"]
        scored: List[Tuple[float, int, Episode]] = []
        for i, ep in enumerate(pool):
            sim = self.store.similarity.score(brief, ep.brief)
            if sim < self.min_similarity:
                continue
            sal = self._salience_for(ep)
            ret = retention(sal.strength, self.tick - sal.last_recall_day, self.tau)
            scored.append((sim * ret, i, ep))
        scored.sort(key=lambda t: (-t[0], t[1]))
        episodes = [ep for _, _, ep in scored[: self.k_episodes]]

        # Reinforce exactly what we returned (a recall hit).
        for ep in episodes:
            key = self._episode_key(ep)
            self._salience[key] = reinforce(self._salience_for(ep), float(self.tick))

        skills: List[Skill] = []
        if self.skills is not None and self.skills.names():
            skills = [s for s in self.skills.find(brief, k=self.k_skills) if s.verified]

        insights = list(self.store.get_semantic(SEMANTIC_INSIGHT_KEY, []) or [])
        insights = insights[-self.k_insights:] if self.k_insights else []

        fps = [
            e for e, sim in self.notebook.retrieve(
                brief, n=self.k_false_positives, exclude_spec_exact=False)
            if sim >= self.min_similarity
            and str(e.insight).startswith(FALSE_POSITIVE_PREFIX)
        ]

        out = Recalled(episodes=episodes, skills=skills, insights=insights,
                       false_positives=fps)
        if out:
            self.stats["recall_hits"] += 1
        return out

    def _salience_for(self, ep: Episode) -> Salience:
        key = self._episode_key(ep)
        sal = self._salience.get(key)
        if sal is None:
            sal = Salience(node_id=key, strength=1.0,
                           last_recall_day=float(ep.meta.get("tick", 0)))
            self._salience[key] = sal
        return sal

    # --- writes (ALL oracle-gated) ----------------------------------------
    def commit(
        self,
        brief: str,
        ops: Sequence[Any],
        verdict: OracleVerdict,
        *,
        digest: Optional[str] = None,
        fleet_diagnostics: Optional[Sequence[Any]] = None,
        summary: str = "",
    ) -> Dict[str, Any]:
        """Write one trajectory to memory — **only if the oracle passed it**.

        Returns a record of what was written. On a refused verdict the episodic
        store is untouched: a memory of a wrong answer is worse than no memory.
        A failed run may still write a *lesson* (an insight synthesised from its
        diagnostics), because a lesson is a statement about a failure, not an
        exemplar of a success — but it may never contribute an exemplar.

        When the oracle PASSES and the verifier fleet nevertheless raised ERRORs,
        that disagreement is recorded as a verifier false positive. The gate is
        the ground truth there, and the fleet is the thing on trial.
        """
        self.tick += 1
        written: Dict[str, Any] = {
            "tick": self.tick, "admitted": False,
            "false_positive": None, "insight": None, "forgotten": [],
        }

        codes = _error_codes(fleet_diagnostics or [])

        if not verdict.ok:
            self.stats["commits_refused"] += 1
            insight = self._reflect(fleet_diagnostics or [], verdict, brief)
            if insight:
                self._write_insight(insight)
                written["insight"] = insight
            return written

        # --- the oracle passed. This trajectory is real. ---
        ep = self.store.add_episodic(
            brief=brief,
            ops=list(ops),
            outcome="ok",
            digest=digest,
            summary=summary or "oracle-verified",
            tick=self.tick,
            oracle=verdict.source,
        )
        key = self._episode_key(ep)
        self._salience.setdefault(
            key, Salience(node_id=key, last_recall_day=float(self.tick)))
        self.stats["commits_admitted"] += 1
        written["admitted"] = True

        # THE MEMORY THAT PAYS FOR ITSELF: the fleet said broken, the gate
        # measured it and said fine. Record the disagreement.
        if codes:
            entry = self.notebook.record_mistake(
                specification=brief,
                wrong_answer=codes,               # what the fleet claimed
                ground_truth=[],                  # the gate found nothing wrong
                insight=(f"{FALSE_POSITIVE_PREFIX}: fleet raised "
                         f"{', '.join(codes)}; the measured gate "
                         f"({verdict.source}) found the part correct"),
            )
            self.stats["false_positives_recorded"] += 1
            written["false_positive"] = entry.to_dict()

        written["forgotten"] = self.sweep()
        return written

    def _reflect(self, diagnostics: Sequence[Any], verdict: OracleVerdict,
                 brief: str) -> str:
        """Synthesise a lesson. Reuses ``reflexion.heuristic_reflect`` — the
        reflection that already exists — rather than writing a second one."""
        from harnesscad.eval.reliability.strategies.reflexion import heuristic_reflect

        if diagnostics:
            return heuristic_reflect(list(diagnostics), brief)
        if verdict.failures:
            return ("the measured output gate refused the part: "
                    + "; ".join(verdict.failures[:3]))
        return ""

    def _write_insight(self, insight: str) -> None:
        stored = list(self.store.get_semantic(SEMANTIC_INSIGHT_KEY, []) or [])
        if insight not in stored:
            stored.append(insight)
            self.stats["insights_written"] += 1
        self.store.set_semantic(SEMANTIC_INSIGHT_KEY, stored)

    # --- forgetting -------------------------------------------------------
    def sweep(self) -> List[str]:
        """Ebbinghaus sweep at the current logical tick; drop faded episodes.

        Deterministic: retention is a pure function of (strength, ticks-elapsed).
        ``keep_min`` protects the strongest N so the store never empties itself.
        """
        if not self.store.episodic:
            return []
        sals = [self._salience_for(ep) for ep in self.store.episodic]
        result = decay_sweep(sals, float(self.tick), tau=self.tau,
                             forget_threshold=self.forget_threshold,
                             keep_min=self.keep_min)
        drop = {nid for nid, _ in result.forgotten}
        if not drop:
            return []
        self.store.episodic = [
            ep for ep in self.store.episodic
            if self._episode_key(ep) not in drop
        ]
        for nid in drop:
            self._salience.pop(nid, None)
        self.stats["forgotten"] += len(drop)
        return sorted(drop)

    # --- cold start -------------------------------------------------------
    def seed_from_skills(
        self,
        library: SkillLibrary,
        session_factory: Callable[[], Any],
        oracle: Callable[[Any, Sequence[Any]], OracleVerdict] = gate_oracle,
    ) -> List[str]:
        """Attach a SkillLibrary and promote each skill to an oracle-verified
        exemplar — a cold-start memory that is still measured, not asserted.

        The SkillLibrary's own gate is ``apply_ops(...).ok`` (Voyager). That is a
        weaker gate than ours: it proves the ops applied, not that the resulting
        SOLID is real. So each skill is expanded, built on a fresh session, and
        run through the SAME oracle a live run must pass. A skill whose geometry
        does not survive the gate seeds nothing.
        """
        self.skills = library
        admitted: List[str] = []
        for name in sorted(library.names()):
            sk = library.get(name)
            try:
                ops = sk.expand()
                session = session_factory()
                result = session.apply_ops(ops)
            except Exception:  # noqa: BLE001
                continue
            if not getattr(result, "ok", False):
                continue
            verdict = oracle(session, ops)
            if not verdict.ok:
                continue
            self.commit(sk.description, ops, verdict,
                        digest=getattr(result, "digest", None),
                        summary=f"skill:{name}")
            admitted.append(name)
        return admitted

    # --- the fleet-audit signal -------------------------------------------
    def false_positive_records(self) -> List[dict]:
        """Every recorded verifier false positive, as flat rows.

        ``{"brief": ..., "codes": [...], "tick": ...}``. This is the data
        ``eval/selftest/fleet_audit.py`` needs and never had: an in-the-wild
        count of how often each rule fired on a part the gate then measured as
        correct. A rule with a high count here is a rule that costs briefs.
        """
        rows: List[dict] = []
        for e in self.notebook.entries:
            if not str(e.insight).startswith(FALSE_POSITIVE_PREFIX):
                continue
            rows.append({"brief": e.specification,
                         "codes": list(e.known_wrong()),
                         "entry_id": e.entry_id})
        return rows

    def false_positive_counts(self) -> Dict[str, int]:
        """verifier code -> number of parts it wrongly rejected. The scoreboard."""
        counts: Dict[str, int] = {}
        for row in self.false_positive_records():
            for c in row["codes"]:
                counts[c] = counts.get(c, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    # --- honesty ----------------------------------------------------------
    @staticmethod
    def limits() -> List[str]:
        """What this memory still cannot do. Kept in code so it cannot rot."""
        return [
            "The write gate is REFERENCE-FREE. io/gate.py proves the part is "
            "closed, manifold and honours its own declared intent. It has never "
            "read the brief. A part that is beautifully built and answers the "
            "wrong question is admitted to memory as a success.",
            "Retrieval is lexical (token overlap + difflib). 'washer' and "
            "'annular spacer' do not retrieve each other. The Similarity seam "
            "exists for an embedder; no embedder is implemented.",
            "Skills are not learned. SkillLibrary only ever contains what a human "
            "registered; nothing in the loop proposes a new skill from a solved "
            "brief, so the procedural memory does not grow.",
            "The false-positive notebook records CODES, not the verifier that "
            "raised them, when a diagnostic carries no verifier attribution.",
            "Decay is per-episode; nothing decays a stale INSIGHT, so a lesson "
            "learned from a fixed bug persists forever.",
        ]

    # --- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": 1,
            "tick": self.tick,
            "store": self.store.to_dict(),
            "notebook": json.loads(self.notebook.to_json()),
            "salience": {k: {"strength": s.strength,
                             "last_recall_day": s.last_recall_day,
                             "recall_count": s.recall_count}
                         for k, s in sorted(self._salience.items())},
            "stats": dict(self.stats),
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict, *, skills: Optional[SkillLibrary] = None,
                  similarity: Optional[Similarity] = None) -> "HarnessMemory":
        hm = cls(MemoryStore.from_dict(d.get("store", {}), similarity=similarity),
                 skills=skills,
                 notebook=ErrorNotebook.from_json(
                     json.dumps(d.get("notebook", {"entries": []}))))
        hm.tick = int(d.get("tick", 0))
        for key, s in (d.get("salience") or {}).items():
            hm._salience[key] = Salience(
                node_id=key,
                strength=float(s.get("strength", 1.0)),
                last_recall_day=float(s.get("last_recall_day", 0.0)),
                recall_count=int(s.get("recall_count", 0)))
        hm.stats.update(d.get("stats") or {})
        return hm

    @classmethod
    def load(cls, path: str, *, skills: Optional[SkillLibrary] = None,
             similarity: Optional[Similarity] = None) -> "HarnessMemory":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh), skills=skills, similarity=similarity)


# --------------------------------------------------------------------------- #
def _error_codes(diagnostics: Sequence[Any]) -> List[str]:
    """The distinct ERROR-severity codes in a diagnostic list, sorted."""
    out = set()
    for d in diagnostics:
        if hasattr(d, "to_dict"):
            d = d.to_dict()
        if isinstance(d, dict):
            sev = d.get("severity", "error")
            sev = getattr(sev, "value", sev)
            if str(sev) != "error":
                continue
            code = d.get("code")
            if code:
                out.add(str(code))
    return sorted(out)
