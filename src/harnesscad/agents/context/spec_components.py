"""Multi-granular design-specification components + weighted tiling ratio.

Implements the core representation from *Design-Specification Tiling for
ICL-based CAD Code Generation* (Du, Xi, Sun, Li -- Nanjing University), Sec. 3.1
and Sec. 3.2.

A natural-language design specification is decomposed into *multi-granular
components* by extracting word n-grams at exponentially spaced window sizes
``N = {2, 4, 8, 16, 32}`` (Eq. 3-4). Longer n-grams capture compositional
phrases ("cylinder with holes"), short ones capture atomic primitives.

The *weighted tiling* of a component set is ``w(C) = sum_{n in N} n * |C^(n)|``
(Eq. 13): longer components receive proportionally higher weight, encoding the
disparate impact of multi-granular specifications. The *tiling ratio* of an
exemplar-component union ``C(S)`` against a query set ``C_query`` is::

    f_suff(S; q) = w( C(S) & C_query ) / w( C_query )               (Eq. 6)

which ranges from 0 (nothing tiled) to 1.0 (fully tiled) and is a computable
surrogate for knowledge sufficiency.

Everything here is deterministic and stdlib-only. Tokenisation is a simple,
Unicode-lowercasing word split so results are reproducible across runs.
"""

from __future__ import annotations

import re
from typing import Dict, FrozenSet, Iterable, List, Sequence, Tuple

# Exponentially spaced n-gram sizes (paper Sec. 3.1).
DEFAULT_GRANULARITIES: Tuple[int, ...] = (2, 4, 8, 16, 32)

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Deterministic lowercase word tokenisation.

    Splits on any non-alphanumeric run. Unicode is folded to lowercase first so
    that "Cylinder" and "cylinder" produce the same components.
    """
    return _WORD_RE.findall(text.lower())


def ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    """All consecutive n-grams (sliding window of size ``n``) -- Eq. 3.

    Returns an empty list when the token stream is shorter than ``n``.
    """
    if n <= 0:
        raise ValueError("n-gram size must be positive")
    L = len(tokens)
    if L < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(L - n + 1)]


class ComponentSet:
    """Multi-granular component set ``C(X)`` for one specification (Eq. 4).

    Components are stored per-granularity as ``frozenset`` of token tuples so
    that set union / intersection are exact and order-independent. A component
    is tagged with its size ``n`` (the granularity that produced it), which is
    what drives the weighting in :func:`weighted_size`.
    """

    __slots__ = ("granularities", "_by_n")

    def __init__(
        self,
        by_n: Dict[int, FrozenSet[Tuple[str, ...]]],
        granularities: Sequence[int],
    ) -> None:
        self.granularities: Tuple[int, ...] = tuple(sorted(set(granularities)))
        # Ensure every declared granularity has an entry (possibly empty).
        self._by_n: Dict[int, FrozenSet[Tuple[str, ...]]] = {
            n: frozenset(by_n.get(n, frozenset())) for n in self.granularities
        }

    # -- construction ------------------------------------------------------
    @classmethod
    def from_text(
        cls,
        text: str,
        granularities: Sequence[int] = DEFAULT_GRANULARITIES,
    ) -> "ComponentSet":
        toks = tokenize(text)
        by_n = {n: frozenset(ngrams(toks, n)) for n in granularities}
        return cls(by_n, granularities)

    @classmethod
    def empty(
        cls, granularities: Sequence[int] = DEFAULT_GRANULARITIES
    ) -> "ComponentSet":
        return cls({}, granularities)

    # -- accessors ---------------------------------------------------------
    def at(self, n: int) -> FrozenSet[Tuple[str, ...]]:
        return self._by_n.get(n, frozenset())

    def all_components(self) -> FrozenSet[Tuple[int, Tuple[str, ...]]]:
        """Flat set of ``(n, component)`` pairs across every granularity."""
        out = set()
        for n, comps in self._by_n.items():
            for c in comps:
                out.add((n, c))
        return frozenset(out)

    def is_empty(self) -> bool:
        return all(len(v) == 0 for v in self._by_n.values())

    # -- algebra -----------------------------------------------------------
    def union(self, other: "ComponentSet") -> "ComponentSet":
        gran = sorted(set(self.granularities) | set(other.granularities))
        by_n = {n: self.at(n) | other.at(n) for n in gran}
        return ComponentSet(by_n, gran)

    def intersection(self, other: "ComponentSet") -> "ComponentSet":
        gran = sorted(set(self.granularities) | set(other.granularities))
        by_n = {n: self.at(n) & other.at(n) for n in gran}
        return ComponentSet(by_n, gran)

    def __or__(self, other: "ComponentSet") -> "ComponentSet":
        return self.union(other)

    def __and__(self, other: "ComponentSet") -> "ComponentSet":
        return self.intersection(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ComponentSet):
            return NotImplemented
        return self._by_n == other._by_n

    def __repr__(self) -> str:
        counts = {n: len(v) for n, v in sorted(self._by_n.items())}
        return f"ComponentSet(counts={counts})"


def weighted_size(cs: ComponentSet) -> int:
    """``w(C) = sum_{n} n * |C^(n)|`` -- Eq. 13.

    Longer components (larger ``n``) contribute proportionally more weight.
    """
    return sum(n * len(cs.at(n)) for n in cs.granularities)


def union_components(sets: Iterable[ComponentSet]) -> ComponentSet:
    """``C(S) = union_i C_i`` -- Eq. 5. Empty iterable yields an empty set."""
    acc = None
    for cs in sets:
        acc = cs if acc is None else acc.union(cs)
    return acc if acc is not None else ComponentSet.empty()


def tiling_ratio(covered: ComponentSet, query: ComponentSet) -> float:
    """``f_suff = w(covered & query) / w(query)`` -- Eq. 6.

    ``covered`` is the union of components already provided by selected
    exemplars (or the raw exemplar union -- this function intersects with the
    query itself). Returns 0.0 for a query with zero weighted size.
    """
    denom = weighted_size(query)
    if denom == 0:
        return 0.0
    num = weighted_size(covered.intersection(query))
    return num / denom
