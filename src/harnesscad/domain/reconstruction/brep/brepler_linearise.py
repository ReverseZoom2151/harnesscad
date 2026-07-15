"""Brepler B-rep face-sequence linearisation (area-descending BFS from a corner seed).

Deterministic re-encoding of the ordering Brepler imposes on the faces of a solid
before serialising it (``data/prepare_data.py``). A B-rep is an unordered graph of
faces; an autoregressive / diffusion model needs a *canonical linear order*.
Brepler produces one with a graph traversal that is fully deterministic and does
not depend on any learned weights:

1.  **Corner seed.** Pick the seed face whose representative point (Brepler uses
    the centre sample of each face's UV grid) is closest to a fixed spatial
    corner -- ``(-1, -1, -1)`` in Brepler's normalised cube::

        length = ||face_centre - (-1,-1,-1)||
        id_min = argmin(length)

2.  **Area-descending breadth-first search.** BFS out from the seed over the
    face-adjacency graph. When a face is dequeued its not-yet-seen neighbours are
    collected, sorted by *descending face area*, and appended to the queue::

        to_visit = [seed]
        while to_visit:
            cur = to_visit.pop(0); order.append(cur)
            nbrs = [n for n in neighbours(cur) if n not in seen]
            nbrs.sort(key=lambda n: -area[n])          # largest neighbour first
            seen.update(nbrs); to_visit.extend(nbrs)

    (Brepler marks neighbours seen at enqueue time; equal-area neighbours keep
    ascending-id order, matching Brepler's stable sort over ``np.where`` output.)

The result is a permutation of the face ids: the order in which faces are written
to the sequence. This module owns that permutation. It is *distinct* from the
existing :mod:`harnesscad.domain.reconstruction.brep.coedge_walk` (a local
half-edge traversal) and from the token quantisers: this is the global
face-visitation order, the backbone the tokens hang off.

Beyond Brepler's connected-solid assumption this module is total: if the graph is
disconnected, once a component is exhausted the next component is seeded by the
same corner rule over the remaining faces (documented extension; on a connected
graph the output is identical to Brepler).

Stdlib only, deterministic. No wall clock, no randomness.
"""

from __future__ import annotations

import math
from typing import Hashable, Iterable, Mapping, Sequence

__all__ = [
    "DEFAULT_CORNER",
    "adjacency_from_edges",
    "corner_seed",
    "area_descending_bfs",
    "linearise_faces",
]

DEFAULT_CORNER: tuple[float, float, float] = (-1.0, -1.0, -1.0)


def adjacency_from_edges(
    face_ids: Iterable[Hashable],
    shared_edges: Iterable[tuple[Hashable, Hashable]],
) -> dict:
    """Build a symmetric adjacency map from a face id set and a list of face pairs.

    Each pair ``(a, b)`` marks two faces meeting at a shared edge. Neighbour lists
    are returned sorted (by natural ordering) for deterministic traversal.
    """
    adj: dict = {f: set() for f in face_ids}
    for a, b in shared_edges:
        if a not in adj or b not in adj:
            raise ValueError(f"edge references unknown face: ({a!r}, {b!r})")
        if a == b:
            continue
        adj[a].add(b)
        adj[b].add(a)
    return {f: sorted(ns) for f, ns in adj.items()}


def _distance(p: Sequence[float], q: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(p, q)))


def corner_seed(
    points: Mapping[Hashable, Sequence[float]],
    corner: Sequence[float] = DEFAULT_CORNER,
) -> Hashable:
    """Face whose representative point is closest to ``corner`` (Brepler ``id_min``).

    Ties in distance are broken by natural ordering of the face id, keeping the
    seed deterministic.
    """
    if not points:
        raise ValueError("no faces to seed from")
    best = None
    best_key = None
    for fid in sorted(points, key=_sort_key):
        d = _distance(points[fid], corner)
        key = (d, _sort_key(fid))
        if best_key is None or key < best_key:
            best_key = key
            best = fid
    return best


def _sort_key(x: Hashable):
    # Order ints/floats naturally; fall back to repr for mixed/opaque ids so the
    # traversal is always deterministic regardless of id type.
    if isinstance(x, (int, float)):
        return (0, x)
    return (1, repr(x))


def area_descending_bfs(
    adjacency: Mapping[Hashable, Iterable[Hashable]],
    areas: Mapping[Hashable, float],
    seed: Hashable,
) -> list:
    """BFS from ``seed``, expanding each face's unseen neighbours largest-area-first.

    Visits only the connected component containing ``seed``.
    """
    if seed not in adjacency:
        raise ValueError(f"seed {seed!r} not in adjacency")
    seen = {seed}
    to_visit = [seed]
    order: list = []
    while to_visit:
        cur = to_visit.pop(0)
        order.append(cur)
        fresh = [n for n in adjacency.get(cur, ()) if n not in seen]
        # descending area; stable over ascending-id neighbour order (tie-break).
        fresh.sort(key=lambda n: _sort_key(n))
        fresh.sort(key=lambda n: -float(areas[n]))
        for n in fresh:
            seen.add(n)
        to_visit.extend(fresh)
    return order


def linearise_faces(
    adjacency: Mapping[Hashable, Iterable[Hashable]],
    areas: Mapping[Hashable, float],
    points: Mapping[Hashable, Sequence[float]] | None = None,
    corner: Sequence[float] = DEFAULT_CORNER,
    seed: Hashable | None = None,
) -> list:
    """Return the Brepler linear face order (a permutation of the adjacency keys).

    Parameters
    ----------
    adjacency : face id -> neighbouring face ids (shared-edge graph).
    areas     : face id -> face area (drives the BFS neighbour ordering).
    points    : face id -> representative point; used to pick the corner seed
                when ``seed`` is not given. Required unless ``seed`` is supplied.
    corner    : the spatial corner the seed is pulled toward (default (-1,-1,-1)).
    seed      : explicit seed face id, overriding the corner rule.

    Handles disconnected graphs by re-seeding each remaining component with the
    same corner rule; on a connected graph this reproduces Brepler exactly.
    """
    faces = list(adjacency.keys())
    if not faces:
        return []

    remaining = set(faces)
    order: list = []

    def pick_seed(candidates: set) -> Hashable:
        if seed is not None and seed in candidates:
            return seed
        if points is not None:
            sub = {f: points[f] for f in candidates if f in points}
            if sub:
                return corner_seed(sub, corner)
        # No geometry: fall back to the largest-area face, then id order.
        return sorted(candidates, key=lambda f: (-float(areas.get(f, 0.0)), _sort_key(f)))[0]

    while remaining:
        s = pick_seed(remaining)
        component = area_descending_bfs(adjacency, areas, s)
        for f in component:
            if f in remaining:
                order.append(f)
                remaining.discard(f)
    return order
