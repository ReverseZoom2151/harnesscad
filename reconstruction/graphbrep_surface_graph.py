"""GraphBrep surface-adjacency graph ``G = (S, E, A)`` (Lai et al., 2025).

GraphBrep's distinctive B-Rep representation departs from both the hierarchical
face/edge/coedge graph used by CADParser (``cadparser_brep_graph``) and the
bipartite edge-surface matrix ``R`` used by CMT (``cmt_topology_predictor``).
Here the graph nodes are the **surfaces only**, and topology is an *undirected
weighted* adjacency matrix

    A in Z^{n_S x n_S},   A[i, j] = number of edges shared by surface i and j,

with ``A[i, j] in [0, e_max]``. Closed faces (cylinders, ...) are split along
their seams first, so the B-Rep "would contain no loops and the adjacency matrix
can be simplified as an undirected graph"; the diagonal is masked (no
self-connection). This module builds ``A`` from a face->edge incidence, provides
the paper's inference-time post-processing (symmetrise the predicted matrix,
clip+round to non-negative integers, dense<->sparse), recovers the concrete edge
list (each with its two endpoint surfaces) that conditions edge generation, and
validates the graph.

Everything is deterministic and stdlib-only; the learned graph-diffusion
denoiser itself is external and out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

Matrix = tuple[tuple[int, ...], ...]


# --- construction ------------------------------------------------------------
def build_surface_adjacency(face_edges: Sequence[Iterable[Hashable]]) -> Matrix:
    """Weighted surface adjacency ``A`` from per-face edge-id sets.

    ``face_edges[i]`` is the collection of edge identifiers bounding surface
    ``i``. Two surfaces are adjacent with weight equal to the number of edge ids
    they share. The diagonal is masked to zero (self-connections excluded).
    """
    sets = [frozenset(edges) for edges in face_edges]
    n = len(sets)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            shared = len(sets[i] & sets[j])
            matrix[i][j] = shared
            matrix[j][i] = shared
    return tuple(tuple(row) for row in matrix)


def build_from_edge_faces(edge_faces: Sequence[tuple[int, int]], n_surfaces: int) -> Matrix:
    """Weighted surface adjacency from a list of ``(surface_i, surface_j)`` edges.

    Each B-Rep edge is shared by exactly two surfaces; this accumulates one unit
    of weight per edge onto ``A[i, j]``. Self-edges (``i == j``) are rejected
    because seam-splitting removes loops.
    """
    matrix = [[0] * n_surfaces for _ in range(n_surfaces)]
    for a, b in edge_faces:
        if a == b:
            raise ValueError("self-edge encountered; closed faces must be seam-split")
        if not (0 <= a < n_surfaces and 0 <= b < n_surfaces):
            raise ValueError(f"edge references surface out of range: {(a, b)!r}")
        matrix[a][b] += 1
        matrix[b][a] += 1
    return tuple(tuple(row) for row in matrix)


# --- predicted-matrix post-processing (paper Sec. 4.1.3) ---------------------
def symmetrise(matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    """Average a raw (predicted) matrix with its transpose: ``(A + A^T) / 2``.

    The diffusion denoiser adds symmetric noise, but numerically the sampled
    matrix is symmetrised by adding it to its own transpose before rounding.
    """
    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be square")
    return tuple(
        tuple((matrix[i][j] + matrix[j][i]) / 2.0 for j in range(n))
        for i in range(n)
    )


def finalise_predicted(matrix: Sequence[Sequence[float]], e_max: int) -> Matrix:
    """Symmetrise, clip to ``[0, e_max]``, round to ints, zero the diagonal.

    Reproduces the inference rule: "the adjacency matrix is added to its
    transpose to ensure symmetry ... each value ... is a non-negative integer,
    the final matrix is clipped and rounded to integers".
    """
    if e_max < 0:
        raise ValueError("e_max must be non-negative")
    sym = symmetrise(matrix)
    n = len(sym)
    out = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            value = int(round(sym[i][j]))
            if value < 0:
                value = 0
            elif value > e_max:
                value = e_max
            out[i][j] = value
    return tuple(tuple(row) for row in out)


# --- dense <-> sparse --------------------------------------------------------
def dense_to_sparse(matrix: Matrix) -> tuple[tuple[int, int, int], ...]:
    """Upper-triangle non-zero entries as ``(i, j, weight)`` with ``i < j``."""
    n = len(matrix)
    return tuple(
        (i, j, matrix[i][j])
        for i in range(n)
        for j in range(i + 1, n)
        if matrix[i][j] > 0
    )


def sparse_to_dense(entries: Iterable[tuple[int, int, int]], n_surfaces: int) -> Matrix:
    """Rebuild a dense symmetric matrix from ``(i, j, weight)`` triples."""
    matrix = [[0] * n_surfaces for _ in range(n_surfaces)]
    for i, j, w in entries:
        if i == j:
            raise ValueError("sparse entry on diagonal")
        matrix[i][j] = w
        matrix[j][i] = w
    return tuple(tuple(row) for row in matrix)


# --- edge recovery (conditions edge generation) ------------------------------
def recover_edges(matrix: Matrix) -> tuple[tuple[int, int], ...]:
    """Concrete edge list ``(surface_i, surface_j)`` implied by ``A``.

    Every unit of weight ``A[i, j] = k`` (``i < j``) expands to ``k`` edges, each
    tagged with its two endpoint surfaces -- exactly the "two surfaces sharing
    each edge" that GraphBrep injects as conditions for edge generation. Avoids
    BrepGen's fixed maximum-edge padding: the length is the *actual* edge count.
    """
    edges: list[tuple[int, int]] = []
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            edges.extend([(i, j)] * matrix[i][j])
    return tuple(edges)


def total_edges(matrix: Matrix) -> int:
    """Number of geometric edges encoded (sum over the upper triangle)."""
    n = len(matrix)
    return sum(matrix[i][j] for i in range(n) for j in range(i + 1, n))


def surface_degrees(matrix: Matrix) -> tuple[int, ...]:
    """Per-surface number of incident edges (weighted row sums)."""
    return tuple(sum(row) for row in matrix)


# --- validity ----------------------------------------------------------------
@dataclass(frozen=True)
class GraphDiagnostic:
    code: str
    detail: str
    context: tuple = ()


def check_graph(matrix: Matrix, e_max: int | None = None,
                require_connected: bool = False
                ) -> tuple[bool, tuple[GraphDiagnostic, ...]]:
    """Validate an undirected weighted surface adjacency matrix.

    Flags: non-square shape, asymmetry, negative weights, non-zero diagonal
    (masked self-connection), weights above ``e_max``, isolated surfaces
    (degree 0), and optionally a disconnected graph.
    """
    diagnostics: list[GraphDiagnostic] = []
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("adjacency matrix must be square")

    for i in range(n):
        if matrix[i][i] != 0:
            diagnostics.append(GraphDiagnostic(
                "non-zero-diagonal", "self-connection is masked to zero", (i,)))
        for j in range(n):
            if matrix[i][j] != matrix[j][i]:
                diagnostics.append(GraphDiagnostic(
                    "asymmetric", "A[i,j] != A[j,i]", (i, j)))
            if matrix[i][j] < 0:
                diagnostics.append(GraphDiagnostic(
                    "negative-weight", "shared-edge count is negative", (i, j)))
            if e_max is not None and matrix[i][j] > e_max:
                diagnostics.append(GraphDiagnostic(
                    "over-e-max", f"weight {matrix[i][j]} exceeds e_max {e_max}", (i, j)))

    for i, degree in enumerate(surface_degrees(matrix)):
        if degree == 0:
            diagnostics.append(GraphDiagnostic(
                "isolated-surface", "surface shares no edge with any other", (i,)))

    if require_connected and n > 0 and not is_connected(matrix):
        diagnostics.append(GraphDiagnostic(
            "disconnected", "surface graph has more than one component", ()))

    return not diagnostics, tuple(diagnostics)


def is_connected(matrix: Matrix) -> bool:
    """Whether the (non-zero-weight) surface graph is a single component."""
    n = len(matrix)
    if n <= 1:
        return True
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for other in range(n):
            if other not in seen and matrix[node][other] > 0:
                seen.add(other)
                stack.append(other)
    return len(seen) == n


def is_valid(matrix: Matrix, e_max: int | None = None,
             require_connected: bool = False) -> bool:
    ok, _ = check_graph(matrix, e_max, require_connected)
    return ok
