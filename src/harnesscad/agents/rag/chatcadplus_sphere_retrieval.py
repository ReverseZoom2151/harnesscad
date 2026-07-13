"""Cosine k-NN retrieval via spherical projection + KD-tree (ChatCAD+ sec. III-A.2).

ChatCAD+'s hierarchical in-context learning retrieves the top-k semantically
similar prior reports to use as few-shot exemplars. The retrieval trick it uses
is general and worth isolating:

* A KD-tree gives ``O(log n)`` average nearest-neighbour queries, but it ranks
  by **Euclidean (L2) distance**, whereas semantic similarity is **cosine**.
* The paper's fix (Eq. 4) is to *project every embedding onto the unit
  hypersphere* first. On the unit sphere the L2 distance between two vectors is
  ``2 * sin(theta / 2)`` where ``theta`` is the angle between them -- a strictly
  monotonic function of the angle. So **L2 ordering on the sphere is identical
  to cosine ordering**, and a plain L2 KD-tree returns exactly the cosine
  top-k.

This is a domain-agnostic retrieval primitive: any harness component that keeps
a bank of vectors (retrieval-augmented exemplars for a CAD-code generator, a
cache of prior design embeddings, nearest-template lookup) can do fast cosine
k-NN this way instead of brute-force scanning every vector. The harness's
existing ``rag`` retriever fuses BM25 with brute-force hashed-vector cosine;
this module adds the sphere-projected KD-tree that ChatCAD+ relies on, which
neither ``rag/index.py`` nor ``rag/retriever.py`` implements.

Pure stdlib, deterministic (stable tie-breaking by insertion index). No numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

__all__ = ["normalise", "SphereKDTree"]

Vector = Sequence[float]


def normalise(vec: Vector) -> Tuple[float, ...]:
    """Project ``vec`` onto the unit hypersphere (L2-normalise).

    A zero vector has no direction; we map it to the origin, which simply means
    it is maximally distant from every real unit vector and never wins a query.
    """
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return tuple(0.0 for _ in vec)
    return tuple(x / norm for x in vec)


def _l2_sq(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) * (x - y) for x, y in zip(a, b))


@dataclass
class _Node:
    point: Tuple[float, ...]
    index: int  # original insertion index (payload id + stable tie-break)
    axis: int
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None


class SphereKDTree:
    """KD-tree over unit-normalised vectors for cosine top-k retrieval.

    Build with :meth:`from_vectors` (or pass raw vectors to the constructor).
    Every inserted vector is normalised on the way in, and queries are
    normalised too, so :meth:`query` returns the same ranking as an exhaustive
    cosine-similarity scan -- but in ``O(log n)`` average time.

    Results are returned as ``(index, cosine_similarity)`` pairs, highest
    similarity first, with ties broken by ascending original index for full
    determinism.
    """

    def __init__(self, vectors: Optional[Sequence[Vector]] = None) -> None:
        self._root: Optional[_Node] = None
        self._dim: Optional[int] = None
        self._size = 0
        self._points: List[Tuple[float, ...]] = []
        if vectors:
            pts = [(normalise(v), i) for i, v in enumerate(vectors)]
            self._dim = len(vectors[0])
            for v in vectors:
                if len(v) != self._dim:
                    raise ValueError("all vectors must share one dimension")
            self._points = [p for p, _ in pts]
            self._size = len(pts)
            self._root = self._build(pts, depth=0)

    # --- construction --------------------------------------------------------

    @classmethod
    def from_vectors(cls, vectors: Sequence[Vector]) -> "SphereKDTree":
        return cls(vectors)

    def _build(self, pts: List[Tuple[Tuple[float, ...], int]], depth: int):
        if not pts:
            return None
        axis = depth % self._dim  # type: ignore[operator]
        # Sort by axis coord, tie-break by original index for stable trees.
        pts.sort(key=lambda pi: (pi[0][axis], pi[1]))
        mid = len(pts) // 2
        point, index = pts[mid]
        node = _Node(point=point, index=index, axis=axis)
        node.left = self._build(pts[:mid], depth + 1)
        node.right = self._build(pts[mid + 1:], depth + 1)
        return node

    def __len__(self) -> int:
        return self._size

    # --- query ---------------------------------------------------------------

    def query(self, vec: Vector, k: int = 3) -> List[Tuple[int, float]]:
        """Return the ``k`` nearest vectors to ``vec`` by cosine similarity.

        Output: ``[(index, cosine_similarity), ...]`` sorted by descending
        similarity, ascending index on ties.
        """
        if self._root is None or k <= 0:
            return []
        if self._dim is not None and len(vec) != self._dim:
            raise ValueError(f"query dim {len(vec)} != tree dim {self._dim}")
        q = normalise(vec)
        # Max-heap emulation via a list kept sorted; small k so this is cheap.
        # Store (l2_sq, index, point). We keep the k smallest l2_sq.
        best: List[Tuple[float, int, Tuple[float, ...]]] = []

        def consider(node: _Node) -> None:
            d = _l2_sq(q, node.point)
            if len(best) < k:
                best.append((d, node.index, node.point))
                best.sort(key=lambda t: (t[0], t[1]))
            elif (d, node.index) < (best[-1][0], best[-1][1]):
                best[-1] = (d, node.index, node.point)
                best.sort(key=lambda t: (t[0], t[1]))

        def recurse(node: Optional[_Node]) -> None:
            if node is None:
                return
            axis = node.axis
            diff = q[axis] - node.point[axis]
            near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
            recurse(near)
            consider(node)
            # Only descend the far side if the splitting plane could hold a
            # closer point than our current worst.
            if len(best) < k or diff * diff < best[-1][0]:
                recurse(far)

        recurse(self._root)
        # Convert L2^2 on the unit sphere back to cosine: |a-b|^2 = 2 - 2cos.
        out: List[Tuple[int, float]] = []
        for d, index, _ in best:
            cos = 1.0 - d / 2.0
            # Clamp tiny FP overshoot into [-1, 1].
            cos = max(-1.0, min(1.0, cos))
            out.append((index, cos))
        out.sort(key=lambda ic: (-ic[1], ic[0]))
        return out

    # --- reference implementation for verification --------------------------

    def brute_force(self, vec: Vector, k: int = 3) -> List[Tuple[int, float]]:
        """Exhaustive cosine top-k -- the ground truth the KD-tree must match."""
        q = normalise(vec)
        scored = []
        for i, p in enumerate(self._points):
            cos = max(-1.0, min(1.0, sum(a * b for a, b in zip(q, p))))
            scored.append((i, cos))
        scored.sort(key=lambda ic: (-ic[1], ic[0]))
        return scored[: max(0, k)]
