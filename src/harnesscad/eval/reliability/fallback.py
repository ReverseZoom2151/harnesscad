"""reliability.fallback — the retrieval fallback tier of the recovery ladder.

The blueprint's error-recovery ladder (sec.10) ends in *graceful degradation*:
when the main generation loop fails or times out on a brief and the
block-and-correct *repair* path (``reliability.repair`` — which fixes the very op
that failed) has nothing left to fix, the harness should still hand back
*something buildable* rather than an empty result. This module is that last,
always-succeeds tier.

:class:`RetrievalFallback` answers a failed brief with the *closest known-good
precedent* drawn from three graceful, independently-optional sources:

  * ``library.catalog.PartCatalog.find``  — execution-verified standard parts
    (the strongest signal: these provably build);
  * ``memory.store.recall_episodic``      — a similar *past* design attempt and
    the ops it produced;
  * ``rag.retriever.HybridRetriever``     — a retrieved precedent chunk (text).

Every answer is flagged ``approximate=True`` with its ``source`` and a
``confidence`` in [0, 1]. Distinct from repair: repair edits the failing attempt;
fallback *substitutes* a different, known-good design. When no source matches (or
none is configured), a generic, always-valid prismatic block is returned at
confidence 0.0 so the response is guaranteed non-empty and buildable.

Absolute imports, stdlib + in-repo similarity only. Deterministic; no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from harnesscad.core.cisp.ops import NewSketch, AddRectangle, Extrude
from harnesscad.agents.memory.store import TokenOverlapSimilarity


# Source ranking for tie-breaks: a verified catalog part beats a remembered
# design beats a retrieved text precedent beats the generic default.
_SOURCE_RANK = {"catalog": 3, "memory": 2, "rag": 1, "default": 0}


@dataclass
class FallbackResult:
    """A single fallback answer to a failed brief.

    - ``ops_or_part`` : the substitute design — a list of CISP ops (catalog part /
                        remembered attempt), or a precedent descriptor (rag text).
    - ``source``      : provenance tag, e.g. ``"catalog:flange"`` / ``"memory"`` /
                        ``"rag:doc.md"`` / ``"default"``.
    - ``confidence``  : [0, 1] closeness of the precedent to the brief.
    - ``approximate`` : always True — a fallback is never the exact requested part.
    - ``note``        : human-readable status (records the failure ``reason``).
    - ``kind``        : "ops" | "part" | "precedent" — what ``ops_or_part`` holds.
    """

    ops_or_part: Any
    source: str
    confidence: float
    approximate: bool = True
    note: str = ""
    kind: str = "ops"

    @property
    def ok(self) -> bool:
        return self.ops_or_part is not None

    def _serialise_payload(self) -> Any:
        payload = self.ops_or_part
        if isinstance(payload, list):
            return [op.to_dict() if hasattr(op, "to_dict") else op for op in payload]
        return payload

    def to_dict(self) -> dict:
        return {
            "ops_or_part": self._serialise_payload(),
            "source": self.source,
            "confidence": self.confidence,
            "approximate": self.approximate,
            "note": self.note,
            "kind": self.kind,
        }


class RetrievalFallback:
    """Nearest-known-good retrieval over an optional catalog / retriever / memory.

    Any of ``catalog`` / ``retriever`` / ``memory`` may be None; the fallback uses
    whichever are present and always returns a valid, non-empty
    :class:`FallbackResult` (the generic default when nothing matches).
    """

    def __init__(self, catalog=None, retriever=None, memory=None,
                 similarity=None, min_confidence: float = 0.0) -> None:
        self.catalog = catalog
        self.retriever = retriever
        self.memory = memory
        self.similarity = similarity or TokenOverlapSimilarity()
        # A candidate must clear this to beat the generic default.
        self.min_confidence = min_confidence

    # --- brief normalisation ---------------------------------------------
    def _query_text(self, brief_or_features) -> str:
        """Flatten a brief string or a feature mapping/sequence into query text."""
        if brief_or_features is None:
            return ""
        if isinstance(brief_or_features, str):
            return brief_or_features.strip()
        if isinstance(brief_or_features, dict):
            parts: List[str] = []
            for k, v in brief_or_features.items():
                if isinstance(v, (list, tuple)):
                    v = " ".join(str(x) for x in v)
                parts.append(f"{k} {v}")
            return " ".join(parts).strip()
        if isinstance(brief_or_features, (list, tuple)):
            return " ".join(str(x) for x in brief_or_features).strip()
        return str(brief_or_features).strip()

    # --- per-source candidates (each guarded; a broken source is skipped) --
    def _from_catalog(self, query: str) -> Optional[FallbackResult]:
        if self.catalog is None or not query:
            return None
        try:
            cards = self.catalog.find(query, k=1)
        except Exception:  # noqa: BLE001
            return None
        if not cards:
            return None
        card = cards[0]
        doc = f"{card.name} {' '.join(getattr(card, 'function_tags', []))} " \
              f"{getattr(card, 'description', '')}"
        conf = self.similarity.score(query, doc)
        ops = self._instantiate(card)
        if ops is None:
            return None
        return FallbackResult(
            ops_or_part=ops, source=f"catalog:{card.name}", confidence=conf,
            note=f"nearest verified catalog part '{card.name}'", kind="ops")

    def _instantiate(self, card) -> Optional[list]:
        """Realise a card's ops from its defaults (catalog or bare card)."""
        try:
            if self.catalog is not None and hasattr(self.catalog, "instantiate"):
                return self.catalog.instantiate(card.name)
        except Exception:  # noqa: BLE001
            pass
        try:
            return card.build(**card.defaults())
        except Exception:  # noqa: BLE001
            return None

    def _from_memory(self, query: str) -> Optional[FallbackResult]:
        if self.memory is None or not query:
            return None
        try:
            episodes = self.memory.recall_episodic(query, k=1)
        except Exception:  # noqa: BLE001
            return None
        if not episodes:
            return None
        ep = episodes[0]
        conf = self.similarity.score(query, getattr(ep, "brief", ""))
        ops = list(getattr(ep, "ops", []) or [])
        if not ops:
            return None
        return FallbackResult(
            ops_or_part=ops, source="memory",
            confidence=conf,
            note=f"nearest prior design (episode: {getattr(ep, 'brief', '')!r}, "
                 f"outcome={getattr(ep, 'outcome', 'unknown')})", kind="ops")

    def _from_rag(self, query: str) -> Optional[FallbackResult]:
        if self.retriever is None or not query:
            return None
        try:
            hits = self.retriever.retrieve(query, k=1)
        except Exception:  # noqa: BLE001
            return None
        if not hits:
            return None
        hit = hits[0]
        text = getattr(hit, "text", "")
        conf = self.similarity.score(query, text)
        return FallbackResult(
            ops_or_part=text, source=f"rag:{getattr(hit, 'source', 'doc')}",
            confidence=conf,
            note="retrieved precedent chunk (text; not directly buildable ops)",
            kind="precedent")

    # --- generic always-valid default -------------------------------------
    @staticmethod
    def _default_ops() -> list:
        """A minimal, always-buildable prismatic block (unit-ish plate)."""
        return [
            NewSketch(plane="XY"),
            AddRectangle(sketch="sk1", x=-5.0, y=-5.0, w=10.0, h=10.0),
            Extrude(sketch="sk1", distance=5.0),
        ]

    def _default(self, reason: str) -> FallbackResult:
        return FallbackResult(
            ops_or_part=self._default_ops(), source="default", confidence=0.0,
            note=f"no matching precedent ({reason}); returning a generic "
                 "buildable prismatic block", kind="ops")

    # --- main entry point -------------------------------------------------
    def fallback(self, brief_or_features, reason: str = "") -> FallbackResult:
        """Return the closest known-good precedent for a failed/timed-out brief.

        Consults every configured source, picks the highest-confidence candidate
        (ties broken by source trust: catalog > memory > rag), and always returns
        a valid, non-empty result — the generic default when nothing clears
        ``min_confidence``. ``reason`` (e.g. "timeout" / "regen-fail") is recorded
        in the returned note for traceability.
        """
        query = self._query_text(brief_or_features)
        candidates: List[FallbackResult] = []
        for cand in (self._from_catalog(query), self._from_memory(query),
                     self._from_rag(query)):
            if cand is not None and cand.confidence >= self.min_confidence:
                candidates.append(cand)

        if not candidates:
            return self._decorate(self._default(reason or "no source matched"),
                                  reason)

        def sort_key(c: FallbackResult):
            tag = c.source.split(":", 1)[0]
            return (c.confidence, _SOURCE_RANK.get(tag, 0))

        best = max(candidates, key=sort_key)
        return self._decorate(best, reason)

    @staticmethod
    def _decorate(result: FallbackResult, reason: str) -> FallbackResult:
        if reason:
            result.note = f"{result.note} [fallback reason: {reason}]"
        result.approximate = True
        return result
