"""Design-specification tiling: decompose a spec into sub-specification tiles.

The DST paper (Sec. 3) measures knowledge sufficiency by tiling a query's
*components*. This module operationalises the complementary, generation-facing
half the paper leaves to the LLM: splitting one natural-language design
specification into **independent sub-specification tiles** that can be reasoned
about (and later generated) semi-independently, plus resolving the ordering
imposed by spatial/topological dependencies between tiles.

A *tile* is a contiguous clause describing one design feature (a primitive plus
its modifiers), e.g. "a cylinder with a central hole" or "mounted on top of the
base plate". Decomposition is purely syntactic and deterministic:

  1. Segment the spec into sentences (on ``. ; \n`` boundaries).
  2. Sub-segment each sentence on coordinating cues (", and ", ", then ",
     " and then ") so each feature becomes its own tile.
  3. Attach each tile's multi-granular ComponentSet (from spectiling_components).

Dependency resolution: a tile that contains a *relational* cue (on/above/below/
adjacent/concentric/union/...) and shares a content noun with an earlier tile is
taken to depend on that earlier tile. The resulting DAG is topologically sorted
(stable, ties by original order) to produce a build order; cycles are broken
deterministically by keeping the earliest-declared tile first.

stdlib-only; deterministic (no wall clock, no RNG).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Sequence, Set, Tuple

from harnesscad.agents.context.spec_components import (
    ComponentSet,
    DEFAULT_GRANULARITIES,
    tokenize,
)

# Relational / topological cues that make a tile reference another feature.
RELATION_CUES: Tuple[str, ...] = (
    "on top of", "on", "above", "below", "under", "beneath", "beside",
    "adjacent", "next to", "centered", "centred", "concentric", "aligned",
    "attached", "mounted", "connected", "union", "intersect", "intersection",
    "subtract", "cut", "through", "inside", "within", "around",
)

# Very small stop-word set so shared-noun detection keys on content words.
_STOP: Set[str] = {
    "a", "an", "the", "of", "with", "and", "then", "to", "is", "are", "that",
    "this", "it", "its", "at", "in", "on", "by", "for", "as", "has", "have",
    "which", "into", "from", "be", "there", "each", "two", "one", "both",
}

_SENT_SPLIT = re.compile(r"[.;\n]+")
_CLAUSE_SPLIT = re.compile(r",\s+and\s+then\s+|,\s+then\s+|,\s+and\s+|\s+then\s+")


@dataclass(frozen=True)
class SpecTile:
    """One independent sub-specification tile."""

    id: int
    text: str
    components: ComponentSet
    content_words: FrozenSet[str] = frozenset()

    def has_relation_cue(self) -> bool:
        low = self.text.lower()
        return any(cue in low for cue in RELATION_CUES)


def _content_words(text: str) -> Set[str]:
    return {t for t in tokenize(text) if t not in _STOP and len(t) > 2}


def decompose_spec(
    text: str,
    granularities: Sequence[int] = DEFAULT_GRANULARITIES,
) -> List[SpecTile]:
    """Split ``text`` into ordered :class:`SpecTile` sub-specifications.

    Empty / whitespace-only fragments are dropped. Ordering follows reading
    order. Each tile carries its own multi-granular component set so downstream
    per-tile exemplar selection (see :mod:`context.spectiling_prompt`) can run
    against a focused query.
    """
    tiles: List[SpecTile] = []
    tid = 0
    for sentence in _SENT_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        for clause in _CLAUSE_SPLIT.split(sentence):
            clause = clause.strip().strip(",")
            if not clause:
                continue
            tiles.append(
                SpecTile(
                    id=tid,
                    text=clause,
                    components=ComponentSet.from_text(clause, granularities),
                    content_words=frozenset(_content_words(clause)),
                )
            )
            tid += 1
    return tiles


def build_dependencies(tiles: Sequence[SpecTile]) -> Dict[int, Set[int]]:
    """Dependency map ``{tile_id: {ids it depends on}}``.

    A tile that carries a relational cue (on/attached/mounted/union/...) depends
    on any *other* tile -- declared before or after it -- with which it shares a
    content word, interpreted as the referenced prerequisite feature. Forward
    references are honoured (a "gusset attached to the flange" declared before
    the flange still depends on it). Mutual relational references can in
    principle form a cycle; :func:`resolve_order` breaks any such cycle
    deterministically, so the build order is always well-defined.
    """
    deps: Dict[int, Set[int]] = {t.id: set() for t in tiles}
    for tile in tiles:
        if not tile.has_relation_cue():
            continue
        for other in tiles:
            if other.id == tile.id:
                continue
            if tile.content_words & other.content_words:
                deps[tile.id].add(other.id)
    return deps


def resolve_order(tiles: Sequence[SpecTile]) -> List[int]:
    """Deterministic topological build order over the dependency DAG.

    Kahn's algorithm with ties broken by ascending tile id (== reading order),
    so the output is a pure function of the input. Any residual cycle (should
    not occur given backward-only edges) is broken by emitting the lowest
    remaining id.
    """
    deps = build_dependencies(tiles)
    remaining: Set[int] = set(deps.keys())
    order: List[int] = []
    # Work on a mutable copy of indegree sources.
    pending: Dict[int, Set[int]] = {k: set(v) for k, v in deps.items()}
    while remaining:
        ready = sorted(
            tid for tid in remaining if not (pending[tid] & remaining)
        )
        if not ready:
            # Cycle fallback: force the lowest id.
            ready = [min(remaining)]
        for tid in ready:
            order.append(tid)
            remaining.discard(tid)
    return order


def ordered_tiles(tiles: Sequence[SpecTile]) -> List[SpecTile]:
    """Return ``tiles`` reordered by :func:`resolve_order`."""
    by_id = {t.id: t for t in tiles}
    return [by_id[tid] for tid in resolve_order(tiles)]
