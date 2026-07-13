"""MemoryStore — the four memory types from the blueprint (sec.8).

This is the grounding layer that sits beside the harness spine:

  - **working**   : the live scratchpad for the current task. The event-sourced
                    session state (OpDAG in state/opdag.py) is the authoritative
                    working memory; this store keeps a small companion dict for
                    per-task notes the agent wants to carry across turns.
  - **episodic**  : past design attempts, each keyed by its natural-language
                    brief -> the ops it produced -> the outcome/digest. Retrieved
                    by lightweight text similarity to a NEW brief so a similar
                    past attempt can seed generation (blueprint: "past successful
                    models keyed by NL description, retrieved by similarity").
  - **semantic**  : facts / preferences as key -> value (material properties,
                    standard dimensions, user preferences).
  - **procedural**: generation rules / prompt fragments (rewritten by a
                    reflection node after failures).

Everything is JSON-persistable (save/load) and dependency-free. Retrieval uses a
pluggable ``Similarity`` — the default is embedding-free (difflib + token
overlap); a real embedder is a documented future upgrade (swap the object in via
``MemoryStore(similarity=MyEmbedder())`` without touching call sites).
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class Similarity(Protocol):
    """Pluggable text-similarity strategy.

    ``score(query, doc) -> float`` in [0, 1], higher = more similar. The default
    implementation is embedding-free; a real embedder (sentence-transformers,
    an API embedder, ...) implements this same one-method interface later.
    """

    def score(self, query: str, doc: str) -> float: ...


class TokenOverlapSimilarity:
    """Embedding-free similarity: blend of token Jaccard + difflib ratio.

    Cheap, deterministic, no external deps. Good enough to retrieve "the right
    past attempt" for a similar brief; a semantic embedder is the future upgrade.
    """

    def score(self, query: str, doc: str) -> float:
        qa, da = _tokens(query), _tokens(doc)
        if not qa or not da:
            return 0.0
        qs, ds = set(qa), set(da)
        jaccard = len(qs & ds) / len(qs | ds)
        ratio = difflib.SequenceMatcher(None, query.lower(), doc.lower()).ratio()
        return 0.5 * jaccard + 0.5 * ratio


def _ops_to_json(ops: Sequence[Any]) -> List[dict]:
    """Accept ops as CISP Op objects OR already-serialised dicts."""
    out: List[dict] = []
    for op in ops:
        if hasattr(op, "to_dict"):
            out.append(op.to_dict())
        elif isinstance(op, dict):
            out.append(dict(op))
        else:
            raise TypeError(f"cannot serialise op of type {type(op).__name__}")
    return out


@dataclass
class Episode:
    """One past design attempt."""

    brief: str
    ops: List[dict] = field(default_factory=list)
    outcome: str = "unknown"          # e.g. "ok" | "failed"
    digest: Optional[str] = None      # backend state digest (replay invariant)
    summary: str = ""                 # short human/agent-readable digest note
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "brief": self.brief,
            "ops": self.ops,
            "outcome": self.outcome,
            "digest": self.digest,
            "summary": self.summary,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Episode":
        return cls(
            brief=d["brief"],
            ops=d.get("ops", []),
            outcome=d.get("outcome", "unknown"),
            digest=d.get("digest"),
            summary=d.get("summary", ""),
            meta=d.get("meta", {}),
        )


class MemoryStore:
    def __init__(self, similarity: Optional[Similarity] = None) -> None:
        self.similarity: Similarity = similarity or TokenOverlapSimilarity()
        self.working: Dict[str, Any] = {}      # per-task scratchpad
        self.episodic: List[Episode] = []
        self.semantic: Dict[str, Any] = {}
        self.procedural: Dict[str, str] = {}

    # --- working (scratchpad) --------------------------------------------
    def note(self, key: str, value: Any) -> None:
        self.working[key] = value

    def get_note(self, key: str, default: Any = None) -> Any:
        return self.working.get(key, default)

    def clear_working(self) -> None:
        self.working = {}

    # --- episodic ---------------------------------------------------------
    def add_episodic(self, brief: str, ops: Sequence[Any], outcome: str = "ok",
                     digest: Optional[str] = None, summary: str = "",
                     **meta: Any) -> Episode:
        ep = Episode(brief=brief, ops=_ops_to_json(ops), outcome=outcome,
                     digest=digest, summary=summary, meta=dict(meta))
        self.episodic.append(ep)
        return ep

    def recall_episodic(self, brief: str, k: int = 3,
                        outcome: Optional[str] = None) -> List[Episode]:
        """Return the k past attempts whose brief is most similar to `brief`.

        Optionally filter by `outcome` (e.g. only "ok" attempts to seed from).
        """
        pool = self.episodic
        if outcome is not None:
            pool = [e for e in pool if e.outcome == outcome]
        scored: List[Tuple[float, int, Episode]] = [
            (self.similarity.score(brief, e.brief), i, e)
            for i, e in enumerate(pool)
        ]
        # Sort by score desc, then insertion order for a stable tie-break.
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [e for _, _, e in scored[:k]]

    # --- semantic ---------------------------------------------------------
    def set_semantic(self, key: str, value: Any) -> None:
        self.semantic[key] = value

    def get_semantic(self, key: str, default: Any = None) -> Any:
        return self.semantic.get(key, default)

    # --- procedural -------------------------------------------------------
    def set_procedural(self, name: str, text: str) -> None:
        self.procedural[name] = text

    def get_procedural(self, name: str, default: Any = None) -> Any:
        return self.procedural.get(name, default)

    # --- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": 1,
            "working": self.working,
            "episodic": [e.to_dict() for e in self.episodic],
            "semantic": self.semantic,
            "procedural": self.procedural,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict, similarity: Optional[Similarity] = None) -> "MemoryStore":
        store = cls(similarity=similarity)
        store.working = d.get("working", {})
        store.episodic = [Episode.from_dict(e) for e in d.get("episodic", [])]
        store.semantic = d.get("semantic", {})
        store.procedural = d.get("procedural", {})
        return store

    @classmethod
    def load(cls, path: str, similarity: Optional[Similarity] = None) -> "MemoryStore":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh), similarity=similarity)
