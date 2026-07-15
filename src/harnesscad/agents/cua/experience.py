"""experience — Agent-S's narrative/episodic knowledge, as a deterministic store.

Agent-S's ``KnowledgeBase`` (``gui_agents/s2/core/knowledge.py``) is the part of
that system that makes it *learn across tasks*. It keeps two knowledge types, both
distilled from past trajectories:

* **narrative** memory — task-level: "to do THIS kind of task, here is the shape of
  the plan that worked", keyed by the task and retrieved by similarity.
* **episodic** memory — subtask-level: "this specific subtask was accomplished by
  this specific sequence", saved when a subtask completes and retrieved to seed the
  next attempt at a similar one.

Both are written by a *summarisation agent* (an LLM) and read by *embedding*
similarity. This module ports the ARCHITECTURE — the two stores, the save-on-
completion lifecycle, the retrieve-to-seed flow — as a deterministic data
structure: the "summary" is a template computed from the graded outcome, not an
LLM's prose, and retrieval is the repo's embedding-free
:class:`~harnesscad.agents.memory.store.TokenOverlapSimilarity`.

Why this is a DISTINCT thing from :mod:`harnesscad.agents.memory.store`
---------------------------------------------------------------------
``MemoryStore`` already has generic ``episodic`` (brief -> ops -> digest) and
``procedural`` slots. This is not those. This is *CUA-specific* and its central
object is one no generic memory has: :class:`DialogFeatureMemory`, the
**"which dialog builds which feature"** table — a map from a CISP op (the feature
the agent wants) to the verified GUI *recipe* that built it (the tier, the dialog,
the field entries), with success statistics. That is the single most useful thing a
CAD computer-use agent can remember, because the hard part of driving a CAD GUI is
not *what* to build but *which dialog* builds it and *which fields* to fill — and
that knowledge is stable across parts (the Pad dialog is the Pad dialog whether the
block is 30mm or 300mm). Agent-S learns "click sequences" for web apps; the CAD
analogue is "the dialog recipe for a feature", and it is worth storing as its own
typed thing rather than as opaque prose.

Pure stdlib, import-safe, JSON-persistable. No LLM, no embedder, no app.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad.agents.memory.store import Similarity, TokenOverlapSimilarity


@dataclass
class DialogRecipe:
    """The verified GUI recipe that builds one feature (one CISP op tag).

    ``op_tag`` is the CISP op this recipe produces (``"pad"``, ``"box"``,
    ``"fillet"``); ``tier`` is which action tier drove it (``"semantic_gui"`` /
    ``"viewport_pick"`` — see :class:`harnesscad.agents.cua.loop.ActionTier`);
    ``dialog`` names the panel/command; ``fields`` are the dialog leaves that were
    written (e.g. ``{"boxLength": "length", ...}`` — control-id -> op-param), which
    is exactly the binding :mod:`harnesscad.io.cua.bindings_freecad` needs and which
    :mod:`harnesscad.eval.grounding.cadspot` harvests for free. ``successes`` /
    ``attempts`` accumulate across trajectories so a recipe earns confidence.
    """

    op_tag: str
    tier: str = "semantic_gui"
    dialog: str = ""
    command: str = ""
    fields: Dict[str, str] = field(default_factory=dict)
    successes: int = 0
    attempts: int = 0
    notes: str = ""

    @property
    def confidence(self) -> float:
        """Success rate; 0.0 with no attempts (an unproven recipe is not trusted)."""
        return (self.successes / self.attempts) if self.attempts else 0.0

    def record(self, ok: bool) -> None:
        self.attempts += 1
        if ok:
            self.successes += 1

    def to_dict(self) -> dict:
        return {"op_tag": self.op_tag, "tier": self.tier, "dialog": self.dialog,
                "command": self.command, "fields": dict(self.fields),
                "successes": self.successes, "attempts": self.attempts,
                "notes": self.notes}

    @classmethod
    def from_dict(cls, d: dict) -> "DialogRecipe":
        return cls(op_tag=d["op_tag"], tier=d.get("tier", "semantic_gui"),
                   dialog=d.get("dialog", ""), command=d.get("command", ""),
                   fields=dict(d.get("fields", {})),
                   successes=int(d.get("successes", 0)),
                   attempts=int(d.get("attempts", 0)), notes=d.get("notes", ""))


class DialogFeatureMemory:
    """The "which dialog builds which feature" table. The CAD-specific core.

    Maps a CISP op tag to the recipes known to build it, best (highest confidence,
    then most attempts) first. This is what an agent consults BEFORE it opens a
    dialog: "I need to pad a sketch — what recipe has worked?" — turning a
    from-scratch GUI search into a recall.
    """

    def __init__(self) -> None:
        self._by_op: Dict[str, List[DialogRecipe]] = {}

    def learn(self, recipe: DialogRecipe, ok: bool) -> DialogRecipe:
        """Record an OUTCOME for a recipe, merging into an identical known one.

        Two recipes are "the same" iff their (op_tag, tier, dialog, command,
        fields) match — so repeated successes accumulate on one row rather than
        spawning duplicates, and confidence is a real frequency.
        """
        bucket = self._by_op.setdefault(recipe.op_tag, [])
        key = self._key(recipe)
        for existing in bucket:
            if self._key(existing) == key:
                existing.record(ok)
                return existing
        fresh = DialogRecipe(op_tag=recipe.op_tag, tier=recipe.tier,
                             dialog=recipe.dialog, command=recipe.command,
                             fields=dict(recipe.fields), notes=recipe.notes)
        fresh.record(ok)
        bucket.append(fresh)
        return fresh

    @staticmethod
    def _key(r: DialogRecipe) -> Tuple[Any, ...]:
        return (r.op_tag, r.tier, r.dialog, r.command,
                tuple(sorted(r.fields.items())))

    def recall(self, op_tag: str) -> Optional[DialogRecipe]:
        """The best proven recipe for ``op_tag``, or ``None`` if we know none."""
        bucket = self._by_op.get(op_tag) or []
        proven = [r for r in bucket if r.attempts > 0]
        if not proven:
            return None
        proven.sort(key=lambda r: (-r.confidence, -r.attempts, r.dialog))
        return proven[0]

    def recipes(self, op_tag: str) -> List[DialogRecipe]:
        return list(self._by_op.get(op_tag) or [])

    def known_ops(self) -> List[str]:
        return sorted(self._by_op)

    def to_dict(self) -> dict:
        return {tag: [r.to_dict() for r in recipes]
                for tag, recipes in self._by_op.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "DialogFeatureMemory":
        m = cls()
        for tag, recipes in d.items():
            m._by_op[tag] = [DialogRecipe.from_dict(r) for r in recipes]
        return m


@dataclass
class NarrativeEntry:
    """Task-level experience: the plan shape that worked for a class of brief."""

    brief: str
    op_tags: List[str] = field(default_factory=list)
    outcome: str = "unknown"              # "solved" | "failed"
    summary: str = ""

    def to_dict(self) -> dict:
        return {"brief": self.brief, "op_tags": list(self.op_tags),
                "outcome": self.outcome, "summary": self.summary}

    @classmethod
    def from_dict(cls, d: dict) -> "NarrativeEntry":
        return cls(brief=d["brief"], op_tags=list(d.get("op_tags", [])),
                   outcome=d.get("outcome", "unknown"), summary=d.get("summary", ""))


def summarize_trajectory(brief: str, op_tags: Sequence[str], solved: bool,
                         tier_counts: Optional[Dict[str, int]] = None,
                         misses: Optional[Sequence[str]] = None) -> str:
    """A DETERMINISTIC narrative summary — the template that replaces the LLM.

    Agent-S calls a summarisation model here; we compute a factual, replayable
    sentence from the graded outcome. It reads like the LLM's would ("Built X via N
    semantic-GUI actions; solved"), but it is a pure function of the trajectory, so
    the same run always yields the same memory — which is what lets the store be
    tested and diffed.
    """
    verb = "solved" if solved else "did NOT solve"
    feats = ", ".join(op_tags) if op_tags else "no features"
    parts = ["Brief %r %s." % (brief[:60], verb),
             "Features built: %s." % feats]
    if tier_counts:
        used = ", ".join("%s=%d" % (k, v) for k, v in sorted(tier_counts.items()) if v)
        if used:
            parts.append("Action tiers: %s." % used)
    if not solved and misses:
        parts.append("Unmet: %s." % "; ".join(misses))
    return " ".join(parts)


class ExperienceStore:
    """Agent-S's two-store knowledge, deterministic: narrative + dialog-feature.

    The lifecycle mirrors ``KnowledgeBase``: :meth:`ingest` is called once per
    graded trajectory (the analogue of ``finalize_task``) and writes BOTH a
    narrative entry (task shape) AND, for each executed feature, a dialog-feature
    recipe outcome. :meth:`retrieve` is called before a new attempt (the analogue
    of ``retrieve_narrative_experience``) to seed it with the most similar past
    plan. There is no subtask embedding call and no web-search fusion — those are
    the LLM/network parts of Agent-S, deliberately dropped.
    """

    def __init__(self, similarity: Optional[Similarity] = None) -> None:
        self.similarity: Similarity = similarity or TokenOverlapSimilarity()
        self.narrative: List[NarrativeEntry] = []
        self.dialogs = DialogFeatureMemory()

    def ingest(self, brief: str, op_tags: Sequence[str], solved: bool,
               recipes: Optional[Sequence[DialogRecipe]] = None,
               tier_counts: Optional[Dict[str, int]] = None,
               misses: Optional[Sequence[str]] = None) -> NarrativeEntry:
        """Fold one graded trajectory into both stores. Returns the narrative entry."""
        summary = summarize_trajectory(brief, op_tags, solved, tier_counts, misses)
        entry = NarrativeEntry(brief=brief, op_tags=list(op_tags),
                               outcome=("solved" if solved else "failed"),
                               summary=summary)
        self.narrative.append(entry)
        for recipe in (recipes or []):
            # A recipe's outcome is the trajectory's outcome: the feature was built
            # by that dialog and the part graded out (or did not).
            self.dialogs.learn(recipe, ok=solved)
        return entry

    def retrieve(self, brief: str, k: int = 1,
                 solved_only: bool = True) -> List[NarrativeEntry]:
        """The ``k`` most similar past narratives to ``brief`` (Agent-S's seed step).

        ``solved_only`` returns only plans that WORKED — you seed from success, not
        from a past failure — but a caller can ask for all to learn what to avoid.
        """
        pool = [e for e in self.narrative
                if (not solved_only or e.outcome == "solved")]
        scored = [(self.similarity.score(brief, e.brief), i, e)
                  for i, e in enumerate(pool)]
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [e for _s, _i, e in scored[:k]]

    def recipe_for(self, op_tag: str) -> Optional[DialogRecipe]:
        """Shortcut: the best known dialog recipe for a feature."""
        return self.dialogs.recall(op_tag)

    def to_dict(self) -> dict:
        return {"version": 1,
                "narrative": [e.to_dict() for e in self.narrative],
                "dialogs": self.dialogs.to_dict()}

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict, similarity: Optional[Similarity] = None) -> "ExperienceStore":
        s = cls(similarity=similarity)
        s.narrative = [NarrativeEntry.from_dict(e) for e in d.get("narrative", [])]
        s.dialogs = DialogFeatureMemory.from_dict(d.get("dialogs", {}))
        return s

    @classmethod
    def load(cls, path: str, similarity: Optional[Similarity] = None) -> "ExperienceStore":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh), similarity=similarity)
