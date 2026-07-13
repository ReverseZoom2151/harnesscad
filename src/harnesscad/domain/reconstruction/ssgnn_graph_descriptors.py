"""Deterministic structural graph descriptors for heuristic CAD retrieval.

Quan et al., *Self-supervised GNN for Mechanical CAD Retrieval* (GC-CAD), review
the retrieval landscape (Section 2.1): before learned embeddings, CAD parts were
compared with *heuristic* descriptors -- **histogram features** and **graph
features** -- and similarity was computed by graph matching, which is
"NP-hard" and "very time-consuming". This module provides the fast, closed-form
graph descriptors that serve as a hand-crafted baseline against GC-CAD's learned
embeddings, operating on the face-node / curve-edge :class:`CADGraph` from
:mod:`reconstruction.ssgnn_graph_augment`.

Three deterministic structural signatures turn a graph into a fixed-length,
permutation-invariant vector suitable for vector retrieval:

* :func:`degree_histogram` -- distribution of node degrees (topological
  connectivity fingerprint).
* :func:`wl_label_histogram` -- Weisfeiler-Lehman colour-refinement histogram: a
  polynomial-time structural signature that approximates graph isomorphism (a
  positive substitute for the NP-hard exact graph match the paper flags).
* :func:`descriptor_vector` -- degree + WL histograms concatenated with global
  scalars (node/edge counts, density) into one retrieval embedding.

Pairwise similarity uses the Weisfeiler-Lehman subtree kernel
(:func:`wl_kernel`), the standard polynomial-time graph-similarity replacement
for exact isomorphism testing.

All functions are pure, deterministic, and stdlib-only. WL colours are hashed
through a canonical, order-independent string so the signature is byte
reproducible across runs.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence, Tuple

from harnesscad.domain.reconstruction.ssgnn_graph_augment import CADGraph, degrees, neighbours


def _bucket(value: int, edges: Sequence[int]) -> int:
    """Index of the histogram bucket ``value`` falls in given ascending edges.

    ``edges`` are the inclusive upper bounds of each bucket except the last, which
    is an overflow bucket capturing everything larger.
    """
    for i, hi in enumerate(edges):
        if value <= hi:
            return i
    return len(edges)


def degree_histogram(graph: CADGraph,
                     edges: Sequence[int] = (0, 1, 2, 3, 4, 6, 8, 12)
                     ) -> List[float]:
    """L1-normalised histogram of node degrees over fixed buckets.

    Buckets are ``[<=e0], (e0, e1], ..., (e_{-1}, inf)`` so the descriptor has
    ``len(edges) + 1`` bins and is directly comparable between graphs of different
    size. Returns all zeros for an empty graph.
    """
    counts = [0.0] * (len(edges) + 1)
    deg = degrees(graph)
    if not deg:
        return counts
    for d in deg:
        counts[_bucket(d, edges)] += 1.0
    total = sum(counts)
    return [c / total for c in counts]


# --- Weisfeiler-Lehman colour refinement -------------------------------------
def _hash_label(text: str) -> str:
    """Stable short hash of a colour string (order-independent, cross-run)."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


def _initial_colours(graph: CADGraph) -> List[str]:
    """Degree-based initial colours (structure only; feature-agnostic)."""
    deg = degrees(graph)
    return [_hash_label(f"d{d}") for d in deg]


def wl_refine(graph: CADGraph, iterations: int = 3) -> List[List[str]]:
    """Run WL colour refinement, returning the colour list after each iteration.

    Iteration 0 is the initial (degree) colouring. Each subsequent colour of a
    node is the hash of its own colour plus the *sorted* multiset of neighbour
    colours -- the canonical Weisfeiler-Lehman update, permutation-invariant by
    construction.
    """
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    adj = neighbours(graph)
    colours = _initial_colours(graph)
    history = [list(colours)]
    for _ in range(iterations):
        new_colours: List[str] = []
        for node, colour in enumerate(colours):
            neigh = sorted(colours[j] for j in adj[node])
            new_colours.append(_hash_label(colour + "|" + ",".join(neigh)))
        colours = new_colours
        history.append(list(colours))
    return history


def wl_label_histogram(graph: CADGraph, iterations: int = 3,
                       dims: int = 32) -> List[float]:
    """Hashed WL colour histogram: a fixed-length structural signature.

    Colours from every refinement iteration are folded into ``dims`` buckets by
    hashing, then L1-normalised. Two isomorphic graphs produce identical
    histograms; near-isomorphic graphs produce close ones -- a polynomial-time
    surrogate for the NP-hard exact graph match the paper avoids.
    """
    if dims <= 0:
        raise ValueError("dims must be positive")
    counts = [0.0] * dims
    if graph.n_nodes == 0:
        return counts
    for colours in wl_refine(graph, iterations):
        for colour in colours:
            idx = int(colour, 16) % dims
            counts[idx] += 1.0
    total = sum(counts)
    return [c / total for c in counts] if total else counts


def _colour_multiset(graph: CADGraph, iterations: int) -> Dict[str, int]:
    """Multiset (dict colour -> count) of all WL colours across all iterations."""
    bag: Dict[str, int] = {}
    for colours in wl_refine(graph, iterations):
        for colour in colours:
            bag[colour] = bag.get(colour, 0) + 1
    return bag


def wl_kernel(g1: CADGraph, g2: CADGraph, iterations: int = 3) -> int:
    """Weisfeiler-Lehman subtree kernel: dot product of WL colour multisets.

    The classic Shervashidze et al. graph kernel -- counts colours the two graphs
    share across all refinement iterations. Higher means more structurally
    similar. Deterministic and polynomial-time.
    """
    b1 = _colour_multiset(g1, iterations)
    b2 = _colour_multiset(g2, iterations)
    small, large = (b1, b2) if len(b1) <= len(b2) else (b2, b1)
    return sum(count * large.get(colour, 0) for colour, count in small.items())


def wl_similarity(g1: CADGraph, g2: CADGraph, iterations: int = 3) -> float:
    """Normalised WL kernel in ``[0, 1]`` (cosine-normalised subtree kernel).

    ``k(g1, g2) / sqrt(k(g1, g1) * k(g2, g2))``. Returns 0.0 if either self-kernel
    is 0 (an empty graph). Equals 1.0 for isomorphic graphs.
    """
    k12 = wl_kernel(g1, g2, iterations)
    k11 = wl_kernel(g1, g1, iterations)
    k22 = wl_kernel(g2, g2, iterations)
    denom = (k11 * k22) ** 0.5
    if denom == 0.0:
        return 0.0
    return k12 / denom


# --- combined retrieval descriptor -------------------------------------------
def descriptor_vector(graph: CADGraph, *, wl_iterations: int = 3,
                      wl_dims: int = 32,
                      degree_edges: Sequence[int] = (0, 1, 2, 3, 4, 6, 8, 12)
                      ) -> List[float]:
    """Concatenated structural descriptor for graph-based CAD retrieval.

    Stacks the degree histogram, the hashed WL colour histogram, and three global
    scalars (log-scaled node count, log-scaled edge count, and graph density) into
    one fixed-length, permutation-invariant vector usable as a hand-crafted
    baseline embedding for the retrieval eval in
    :mod:`bench.ssgnn_retrieval_eval`.
    """
    import math

    deg_h = degree_histogram(graph, degree_edges)
    wl_h = wl_label_histogram(graph, wl_iterations, wl_dims)
    n, m = graph.n_nodes, graph.n_edges
    max_edges = n * (n - 1) / 2 if n > 1 else 0
    density = (m / max_edges) if max_edges > 0 else 0.0
    scalars = [math.log1p(n), math.log1p(m), density]
    return deg_h + wl_h + scalars
