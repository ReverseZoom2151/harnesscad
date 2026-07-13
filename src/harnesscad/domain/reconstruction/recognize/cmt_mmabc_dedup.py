"""mmABC dataset curation primitives from CMT (Sec. 3).

To build mmABC from ABC, CMT applies two deterministic cleaning steps that do not
need any trained model:

  * **duplicate removal by quantized-point hash** -- "we filter data with same
    hash value of their 6-bit quantized point coordinates sampled from surfaces
    of CAD models". Two models that quantize to the same (order-independent) set
    of surface points are treated as identical and one is dropped.
  * **multi-body decomposition** -- "we decompose complex multi-body models into
    multiple basic single models to augment the dataset". Given the surfaces of
    a model and which surfaces share edges, each connected component is one body.

Both are implemented here with stdlib only and deterministic hashing.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from harnesscad.domain.reconstruction.tokens.cmt_tokenization import quantize

Point = tuple[float, float, float]


def quantize_point(point: Point, bits: int = 6,
                   lo: float = 0.0, hi: float = 1.0) -> tuple[int, int, int]:
    """Quantize a point's coordinates to ``bits`` (paper uses 6-bit)."""
    return tuple(quantize(v, bits, lo, hi) for v in point)


def model_hash(points: tuple[Point, ...], bits: int = 6,
               lo: float = 0.0, hi: float = 1.0) -> str:
    """Order-independent hash of a model's quantized surface point cloud."""
    quantized = sorted(quantize_point(p, bits, lo, hi) for p in points)
    payload = repr(quantized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class DedupResult:
    kept: tuple[str, ...]
    dropped: tuple[str, ...]
    groups: tuple[tuple[str, ...], ...]  # each group of ids sharing one hash


def deduplicate(models: tuple[tuple[str, tuple[Point, ...]], ...],
                bits: int = 6, lo: float = 0.0, hi: float = 1.0) -> DedupResult:
    """Drop models whose 6-bit quantized surface points collide, keep the first.

    ``models`` is a sequence of ``(id, sampled_points)``. Order is preserved:
    within each colliding group the earliest id is kept.
    """
    by_hash: dict = {}
    for model_id, points in models:
        h = model_hash(points, bits, lo, hi)
        by_hash.setdefault(h, []).append(model_id)
    kept: list[str] = []
    dropped: list[str] = []
    groups: list[tuple[str, ...]] = []
    for model_id, points in models:
        h = model_hash(points, bits, lo, hi)
        group = by_hash[h]
        if group and model_id == group[0]:
            kept.append(model_id)
            if len(group) > 1:
                groups.append(tuple(group))
        else:
            dropped.append(model_id)
    return DedupResult(tuple(kept), tuple(dropped), tuple(groups))


def _find(parent: list[int], x: int) -> int:
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


def connected_bodies(n_surfaces: int,
                     shared_edges: tuple[tuple[int, int], ...]) -> tuple[tuple[int, ...], ...]:
    """Split surfaces into connected components (single bodies) via union-find.

    ``shared_edges`` lists pairs of surface indices that share at least one edge.
    Each returned tuple is one body's sorted surface indices; bodies are ordered
    by their smallest surface index for determinism.
    """
    if n_surfaces < 0:
        raise ValueError("n_surfaces must be non-negative")
    parent = list(range(n_surfaces))
    for a, b in shared_edges:
        if not (0 <= a < n_surfaces and 0 <= b < n_surfaces):
            raise ValueError("surface index out of range")
        parent[_find(parent, a)] = _find(parent, b)
    components: dict = {}
    for s in range(n_surfaces):
        components.setdefault(_find(parent, s), []).append(s)
    bodies = [tuple(sorted(members)) for members in components.values()]
    return tuple(sorted(bodies, key=lambda body: body[0]))
