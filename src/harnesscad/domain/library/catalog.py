"""PartCatalog — a functionally-indexed, execution-verified parts library.

This is the retrieval front-end for :mod:`library.parts`: register Model Cards,
:meth:`find` them by function tag / text similarity, and :meth:`instantiate` one
into a validated CISP op stream. Admission goes through the Voyager gate
(:meth:`add_verified`): a card is only admitted if its ops actually *build* on a
fresh ``HarnessSession`` (``result.ok``) — so, exactly like
:class:`memory.skills.SkillLibrary`, the catalog is monotonic and only ever
contains parts whose geometry regenerates cleanly.

The verification gate is *reused*, not re-implemented: the catalog keeps an
internal ``SkillLibrary`` and delegates to its ``add_verified`` via
:meth:`library.parts.ModelCard.to_skill`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from harnesscad.core.cisp.ops import Op
from harnesscad.agents.memory.skills import SkillLibrary
from harnesscad.agents.memory.store import Similarity, TokenOverlapSimilarity

from harnesscad.domain.library.parts import ModelCard, default_cards


class PartCatalog:
    def __init__(self, similarity: Optional[Similarity] = None) -> None:
        self.similarity: Similarity = similarity or TokenOverlapSimilarity()
        self._cards: Dict[str, ModelCard] = {}
        # Reuse the Voyager execution gate from memory.skills verbatim.
        self._lib = SkillLibrary(similarity=self.similarity)

    # --- registration -----------------------------------------------------
    def register(self, card: ModelCard) -> ModelCard:
        """Add a card unconditionally (no execution gate)."""
        self._cards[card.name] = card
        return card

    def add_verified(self, card: ModelCard,
                     session_factory: Callable[[], Any]) -> bool:
        """Voyager gate: build the card's ops on a fresh session and admit it
        ONLY if they verify (``ok == True``). Returns True if admitted, False if
        the ops fail (catalog unchanged — the monotonic-trust invariant).

        Delegates the actual execution check to ``SkillLibrary.add_verified``
        through the card's ``to_skill()`` bridge, so both libraries share one
        gate implementation.
        """
        admitted = self._lib.add_verified(card.to_skill(), session_factory)
        if not admitted:
            return False
        card.verified = True
        self.register(card)
        return True

    # --- lookup -----------------------------------------------------------
    def __contains__(self, name: str) -> bool:
        return name in self._cards

    def get(self, name: str) -> ModelCard:
        return self._cards[name]

    def names(self) -> List[str]:
        return list(self._cards)

    def cards(self) -> List[ModelCard]:
        return list(self._cards.values())

    def find(self, function_or_query: str, k: int = 5) -> List[ModelCard]:
        """Retrieve the k best-matching cards for a function tag or free query.

        Scoring: an exact/substring function-tag hit dominates (so ``find('flange')``
        and ``find('mounting')`` both surface the flange), with lightweight text
        similarity over name + tags + description as the tie-break / fallback.
        """
        q = function_or_query.lower().strip()
        scored: List[Tuple[float, int, ModelCard]] = []
        for i, card in enumerate(self._cards.values()):
            tags = [t.lower() for t in card.function_tags]
            if q in tags or card.name.lower() == q:
                tag_score = 2.0
            elif any(q in t or t in q for t in tags) or q in card.name.lower():
                tag_score = 1.0
            else:
                tag_score = 0.0
            doc = f"{card.name} {' '.join(card.function_tags)} {card.description}"
            sim = self.similarity.score(function_or_query, doc)
            scored.append((tag_score * 10.0 + sim, i, card))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [c for score, _, c in scored[:k] if score > 0.0]

    def instantiate(self, name: str, **params: Any) -> List[Op]:
        """Range-validate ``params`` against the card's schema and return the ops.

        Raises ``KeyError`` if the part is unknown, ``ValueError`` if any param is
        unknown or out of its declared valid range."""
        if name not in self._cards:
            raise KeyError(f"no part '{name}' in catalog (have: {self.names()})")
        return self._cards[name].instantiate(**params)


def build_default_catalog(session_factory: Callable[[], Any],
                          similarity: Optional[Similarity] = None) -> PartCatalog:
    """A PartCatalog seeded with the standard parts, each execution-verified on a
    fresh session from ``session_factory`` (e.g.
    ``lambda: HarnessSession(StubBackend())``). A part that fails to build is
    silently skipped — the catalog only ships parts that regenerate cleanly."""
    cat = PartCatalog(similarity=similarity)
    for card in default_cards():
        cat.add_verified(card, session_factory)
    return cat
