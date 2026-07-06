"""Canonicalisation of GraphBrep surface-adjacency graphs.

GraphBrep learns *permutation-invariant* adjacency matrices (following DiGress /
Graph-GDP): the surface ordering is arbitrary, so two B-Rep graphs that differ
only by a relabelling of surfaces represent the same topology. To compare,
deduplicate, or hash generated graphs deterministically we need a canonical
form that is invariant to node permutation.

This module provides:

  * ``serialize`` -- a deterministic string for a *fixed* labelling;
  * ``wl_colors`` / ``wl_signature`` -- a Weisfeiler-Lehman colour refinement and
    its permutation-invariant fingerprint (a fast necessary condition for
    isomorphism);
  * ``canonical_labelling`` / ``canonical_key`` -- an exact canonical node order
    (lexicographically minimal serialisation) found by searching only the
    permutations consistent with the WL colours, so isomorphic graphs collapse
    to an identical key;
  * ``are_isomorphic`` -- exact weighted-graph isomorphism via canonical keys.

The exact search is factorial in the size of each WL colour class, so a
``permutation guard`` caps the work and raises for pathologically symmetric large
graphs; typical seam-split B-Rep surface graphs refine to small classes.

All functions operate on square symmetric non-negative integer matrices
(``graphbrep_surface_graph`` adjacency); they are stdlib-only and deterministic.
"""

from __future__ import annotations

from itertools import permutations, product

Matrix = tuple[tuple[int, ...], ...]

DEFAULT_PERMUTATION_GUARD = 40320  # 8!


def serialize(matrix: Matrix) -> str:
    """Deterministic string of the upper triangle for a fixed labelling."""
    n = len(matrix)
    return f"{n}|" + ",".join(
        str(matrix[i][j]) for i in range(n) for j in range(i + 1, n)
    )


def _permute(matrix: Matrix, order: tuple[int, ...]) -> Matrix:
    return tuple(tuple(matrix[order[i]][order[j]] for j in range(len(order)))
                 for i in range(len(order)))


# --- Weisfeiler-Lehman colour refinement -------------------------------------
def wl_colors(matrix: Matrix, rounds: int | None = None) -> tuple[int, ...]:
    """Stable per-node WL colours (integers), invariant up to relabelling.

    Initial colour is the node's sorted multiset of incident edge weights; each
    round refines by hashing a node's colour together with the sorted multiset of
    ``(neighbour colour, edge weight)`` pairs, then compressing to dense ints.
    Runs until the colour partition stabilises (or ``rounds`` iterations).
    """
    n = len(matrix)
    if n == 0:
        return ()

    def compress(signatures: list) -> tuple[int, ...]:
        ranking = {sig: rank for rank, sig in enumerate(sorted(set(signatures)))}
        return tuple(ranking[sig] for sig in signatures)

    colors = compress([
        tuple(sorted(w for w in matrix[i] if w > 0)) for i in range(n)
    ])
    limit = n if rounds is None else rounds
    for _ in range(limit):
        signatures = []
        for i in range(n):
            neighbours = tuple(sorted(
                (colors[j], matrix[i][j]) for j in range(n) if matrix[i][j] > 0
            ))
            signatures.append((colors[i], neighbours))
        new_colors = compress(signatures)
        if new_colors == colors:
            break
        colors = new_colors
    return colors


def wl_signature(matrix: Matrix, rounds: int | None = None) -> tuple[tuple[int, int], ...]:
    """Permutation-invariant fingerprint: sorted ``(colour, count)`` histogram."""
    colors = wl_colors(matrix, rounds)
    counts: dict[int, int] = {}
    for c in colors:
        counts[c] = counts.get(c, 0) + 1
    return tuple(sorted(counts.items()))


# --- exact canonical labelling ----------------------------------------------
def _class_permutation_count(colors: tuple[int, ...]) -> int:
    counts: dict[int, int] = {}
    for c in colors:
        counts[c] = counts.get(c, 0) + 1
    total = 1
    for size in counts.values():
        factorial = 1
        for k in range(2, size + 1):
            factorial *= k
        total *= factorial
    return total


def canonical_labelling(matrix: Matrix,
                        permutation_guard: int = DEFAULT_PERMUTATION_GUARD) -> tuple[int, ...]:
    """Node order giving the lexicographically minimal serialisation.

    Only orderings consistent with the WL colour partition are searched: nodes
    are grouped by colour (sorted by colour), and every within-group permutation
    is tried. Raises ``ValueError`` if the number of candidate orderings exceeds
    ``permutation_guard``.
    """
    n = len(matrix)
    if n == 0:
        return ()
    colors = wl_colors(matrix)
    count = _class_permutation_count(colors)
    if count > permutation_guard:
        raise ValueError(
            f"canonicalisation search space {count} exceeds guard {permutation_guard}")

    groups: dict[int, list[int]] = {}
    for node, color in enumerate(colors):
        groups.setdefault(color, []).append(node)
    ordered_colors = sorted(groups)
    class_perms = [list(permutations(groups[c])) for c in ordered_colors]

    best_order: tuple[int, ...] | None = None
    best_key: str | None = None
    for combo in product(*class_perms):
        order = tuple(node for block in combo for node in block)
        key = serialize(_permute(matrix, order))
        if best_key is None or key < best_key:
            best_key = key
            best_order = order
    assert best_order is not None
    return best_order


def canonical_matrix(matrix: Matrix,
                     permutation_guard: int = DEFAULT_PERMUTATION_GUARD) -> Matrix:
    """The adjacency matrix relabelled into canonical node order."""
    return _permute(matrix, canonical_labelling(matrix, permutation_guard))


def canonical_key(matrix: Matrix,
                  permutation_guard: int = DEFAULT_PERMUTATION_GUARD) -> str:
    """Permutation-invariant canonical string; equal iff graphs are isomorphic."""
    return serialize(canonical_matrix(matrix, permutation_guard))


def are_isomorphic(a: Matrix, b: Matrix,
                   permutation_guard: int = DEFAULT_PERMUTATION_GUARD) -> bool:
    """Exact weighted-graph isomorphism test via canonical keys."""
    if len(a) != len(b):
        return False
    if wl_signature(a) != wl_signature(b):
        return False
    return canonical_key(a, permutation_guard) == canonical_key(b, permutation_guard)
