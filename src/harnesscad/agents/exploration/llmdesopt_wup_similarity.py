"""Wu & Palmer (WUP) taxonomy similarity for prompt-word encoding.

Paper: T. Rios, S. Menzel, B. Sendhoff, "Large Language and Text-to-3D Models
for Engineering Design Optimization" (Honda Research Institute Europe).

Section III-A encodes the bag-of-words prompt design variables using the Wu &
Palmer similarity between a sampled word and a reference word ("fast" for the
adjective, "wing" for the noun), computed over the WordNet taxonomy.  WordNet
itself is external data, but the WUP *metric* is a small, deterministic graph
computation over a rooted "is-a" taxonomy:

    wup(a, b) = 2 * depth(LCS(a, b)) / (depth(a) + depth(b))

where LCS is the least (deepest) common subsumer of the two concepts and depth
is the number of edges from the taxonomy root.  The paper notes (Sec. IV-B,
Fig. 8) that words in the same semantic class -- e.g. "snake" and "frog" -- get
nearly identical WUP values with respect to "car" even though their geometry is
very different; this module lets that behaviour be reproduced on a toy taxonomy.

Deterministic: pure graph arithmetic, no randomness, no wall clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


class Taxonomy:
    """A rooted is-a taxonomy (each node has at most one parent = a tree).

    Depth of the root is 1 (WordNet convention: root synset has depth 1), so
    that WUP is well defined and non-zero for the root.
    """

    def __init__(self, root: str) -> None:
        self.root = root
        self._parent: Dict[str, Optional[str]] = {root: None}

    def add(self, child: str, parent: str) -> "Taxonomy":
        if parent not in self._parent:
            raise KeyError(f"unknown parent: {parent!r}")
        if child in self._parent:
            raise ValueError(f"node already present: {child!r}")
        self._parent[child] = parent
        return self

    def __contains__(self, node: str) -> bool:
        return node in self._parent

    def ancestors(self, node: str) -> List[str]:
        """Path from ``node`` up to and including the root."""
        if node not in self._parent:
            raise KeyError(f"unknown node: {node!r}")
        path: List[str] = []
        cur: Optional[str] = node
        while cur is not None:
            path.append(cur)
            cur = self._parent[cur]
        return path

    def depth(self, node: str) -> int:
        """Edges-from-root + 1 (root depth == 1)."""
        return len(self.ancestors(node))

    def lcs(self, a: str, b: str) -> str:
        """Least (deepest) common subsumer of two nodes."""
        anc_a = self.ancestors(a)
        set_a = set(anc_a)
        for node in self.ancestors(b):
            if node in set_a:
                return node
        # Both share the root by construction; unreachable.
        raise ValueError("nodes share no common ancestor")


def wup_similarity(taxonomy: Taxonomy, a: str, b: str) -> float:
    """Wu & Palmer similarity in [0, 1]; 1.0 iff a == b."""
    subsumer = taxonomy.lcs(a, b)
    d_lcs = taxonomy.depth(subsumer)
    return 2.0 * d_lcs / (taxonomy.depth(a) + taxonomy.depth(b))


def wup_distance(taxonomy: Taxonomy, a: str, b: str) -> float:
    """Convenience: 1 - similarity, a dissimilarity in [0, 1)."""
    return 1.0 - wup_similarity(taxonomy, a, b)


def rank_by_similarity(taxonomy: Taxonomy, reference: str,
                       words: List[str]) -> List[tuple]:
    """Return ``[(word, wup)]`` sorted by descending similarity to ``reference``.

    Ties are broken by word to keep the order deterministic.
    """
    scored = [(w, wup_similarity(taxonomy, reference, w)) for w in words]
    scored.sort(key=lambda t: (-t[1], t[0]))
    return scored
