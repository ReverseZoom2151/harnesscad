"""Hierarchical mesh coarsening / upsampling and template positional encoding.

Deterministic re-encoding of two fixed, weight-free geometry operations that
sit around a mesh-regression transformer:

* **Coarse <-> fine mesh sampling.** Rather than regressing a full mesh
  directly, such a model predicts a *coarse* vertex set and lifts it back to
  the full mesh through a pair of precomputed linear operators: a downsample
  step (``N -> M``) and an upsample step (``M -> N``).  Those operators are
  sparse matrices fixed by the mesh topology (a quadric-decimation hierarchy
  in the original).  The reusable idea is the *pair of operators*: a downsample
  matrix whose rows average a cluster of fine vertices, and an upsample matrix
  whose rows blend the nearest coarse vertices.  This module builds both
  deterministically from a mesh via shortest-edge clustering -- no learned
  weights -- so a caller can push any per-vertex signal up or down the
  hierarchy.

* **Template positional encoding.** Each transformer token can be given a
  position by concatenating the *canonical template coordinate* of that
  vertex/joint to the image feature.  The position of vertex *i* is therefore
  just its coordinate in a fixed template pose.  This module provides that raw
  encoding and a sinusoidal frequency expansion of it (a frequency-based
  positional embedding with frequencies ``2**k``).

This is DISTINCT from the marching-cubes / dual-contouring meshers in
``geometry.volumes`` and from ``geometry.mesh.segmentation``: here the mesh is
fixed and we build resolution-transfer operators over it.

Stdlib only, deterministic. No learned weights, no randomness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "MeshHierarchy",
    "coarsen",
    "upsample_matrix",
    "apply_operator",
    "template_positional_encoding",
    "sinusoidal_positional_encoding",
]

Vec3 = Sequence[float]


def _edges_of(faces: Sequence[Sequence[int]]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for f in faces:
        n = len(f)
        for k in range(n):
            a, b = int(f[k]), int(f[(k + 1) % n])
            if a == b:
                continue
            edges.add((a, b) if a < b else (b, a))
    return edges


def _length(p: Vec3, q: Vec3) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(p, q)))


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.count = n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # deterministic: attach larger-root under smaller-root
        lo, hi = (ra, rb) if ra < rb else (rb, ra)
        self.parent[hi] = lo
        self.count -= 1
        return True


@dataclass(frozen=True)
class MeshHierarchy:
    """A coarsening of a mesh plus the linear operators that transfer signals.

    ``vertex_map[i]`` is the coarse index for fine vertex ``i``. ``down`` and
    ``up`` are sparse operators as lists of rows, each row a list of
    ``(source_index, weight)`` pairs whose weights sum to 1.
    """

    coarse_vertices: tuple[tuple[float, float, float], ...]
    vertex_map: tuple[int, ...]
    clusters: tuple[tuple[int, ...], ...]
    down: tuple[tuple[tuple[int, float], ...], ...]
    up: tuple[tuple[tuple[int, float], ...], ...]

    @property
    def num_fine(self) -> int:
        return len(self.vertex_map)

    @property
    def num_coarse(self) -> int:
        return len(self.coarse_vertices)


def coarsen(
    vertices: Sequence[Vec3],
    faces: Sequence[Sequence[int]],
    target: int,
    up_neighbours: int = 3,
) -> MeshHierarchy:
    """Cluster a mesh to ``target`` coarse vertices by greedy shortest-edge collapse.

    Edges are collapsed in ascending ``(length, min_idx, max_idx)`` order, merging
    the two endpoints' clusters, until ``target`` clusters remain. The downsample
    operator averages each cluster; the upsample operator blends each fine vertex's
    ``up_neighbours`` nearest coarse vertices by inverse distance.
    """
    n = len(vertices)
    if target < 1:
        raise ValueError("target must be >= 1")
    if target > n:
        raise ValueError(f"target {target} exceeds vertex count {n}")

    dsu = _DSU(n)
    edges = sorted(
        _edges_of(faces),
        key=lambda e: (_length(vertices[e[0]], vertices[e[1]]), e[0], e[1]),
    )
    for a, b in edges:
        if dsu.count <= target:
            break
        dsu.union(a, b)
    # If the mesh had too few edges to reach target, that's fine: whatever
    # clusters exist are returned (fewer merges than requested).

    # Canonicalise cluster ids: order clusters by their smallest member index.
    roots: dict[int, list[int]] = {}
    for i in range(n):
        roots.setdefault(dsu.find(i), []).append(i)
    ordered_roots = sorted(roots, key=lambda r: min(roots[r]))
    root_to_coarse = {r: c for c, r in enumerate(ordered_roots)}

    vertex_map = tuple(root_to_coarse[dsu.find(i)] for i in range(n))
    clusters = tuple(tuple(roots[r]) for r in ordered_roots)

    coarse_vertices = []
    down_rows = []
    for members in clusters:
        w = 1.0 / len(members)
        down_rows.append(tuple((m, w) for m in members))
        cx = sum(float(vertices[m][0]) for m in members) / len(members)
        cy = sum(float(vertices[m][1]) for m in members) / len(members)
        cz = sum(float(vertices[m][2]) for m in members) / len(members)
        coarse_vertices.append((cx, cy, cz))

    up_rows = upsample_matrix(vertices, tuple(coarse_vertices), vertex_map, up_neighbours)

    return MeshHierarchy(
        coarse_vertices=tuple(coarse_vertices),
        vertex_map=vertex_map,
        clusters=clusters,
        down=tuple(down_rows),
        up=up_rows,
    )


def upsample_matrix(
    vertices: Sequence[Vec3],
    coarse_vertices: Sequence[Vec3],
    vertex_map: Sequence[int],
    k: int = 3,
) -> tuple[tuple[tuple[int, float], ...], ...]:
    """Per-fine-vertex blend weights over the ``k`` nearest coarse vertices.

    The fine vertex's own cluster is always included; ties in distance are broken
    by ascending coarse index. Zero-distance (coincident) coarse vertices collapse
    to a single unit weight.
    """
    m = len(coarse_vertices)
    k = max(1, min(k, m))
    rows = []
    for i, p in enumerate(vertices):
        dists = sorted(
            ((_length(p, coarse_vertices[j]), j) for j in range(m)),
            key=lambda t: (t[0], t[1]),
        )
        # ensure the home cluster is represented
        home = int(vertex_map[i])
        chosen = [(d, j) for d, j in dists[:k]]
        if home not in [j for _, j in chosen]:
            chosen[-1] = (next(d for d, j in dists if j == home), home)
            chosen.sort(key=lambda t: (t[0], t[1]))
        if chosen[0][0] == 0.0:
            rows.append(((chosen[0][1], 1.0),))
            continue
        inv = [(j, 1.0 / d) for d, j in chosen]
        s = sum(w for _, w in inv)
        rows.append(tuple((j, w / s) for j, w in inv))
    return tuple(rows)


def apply_operator(
    operator: Sequence[Sequence[tuple[int, float]]],
    values: Sequence[Sequence[float]],
) -> list[list[float]]:
    """Apply a sparse row operator to per-vertex feature vectors.

    ``values[i]`` is the feature vector of source vertex ``i``. Returns one output
    vector per operator row: ``out[r] = sum_i weight[r,i] * values[i]``.
    """
    if not values:
        raise ValueError("no input values")
    dim = len(values[0])
    out: list[list[float]] = []
    for row in operator:
        acc = [0.0] * dim
        for idx, w in row:
            v = values[idx]
            for d in range(dim):
                acc[d] += w * float(v[d])
        out.append(acc)
    return out


def template_positional_encoding(vertices: Sequence[Vec3]) -> list[tuple[float, ...]]:
    """The raw positional encoding: each vertex's canonical template coordinate."""
    return [tuple(float(c) for c in v) for v in vertices]


def sinusoidal_positional_encoding(
    vertices: Sequence[Vec3],
    num_freqs: int = 6,
    include_input: bool = True,
) -> list[tuple[float, ...]]:
    """Frequency positional embedding of each vertex coordinate (freqs ``2**k``).

    For each coordinate ``x`` and frequency ``f = 2**k`` (``k`` in ``0..num_freqs-1``)
    emit ``sin(f*x)`` and ``cos(f*x)``; optionally prepend the raw coordinate. This
    is the standard frequency-based positional-embedding construction.
    """
    if num_freqs < 1:
        raise ValueError("num_freqs must be >= 1")
    freqs = [2.0 ** k for k in range(num_freqs)]
    out = []
    for v in vertices:
        enc: list[float] = []
        for c in v:
            x = float(c)
            if include_input:
                enc.append(x)
            for f in freqs:
                enc.append(math.sin(f * x))
                enc.append(math.cos(f * x))
        out.append(tuple(enc))
    return out
