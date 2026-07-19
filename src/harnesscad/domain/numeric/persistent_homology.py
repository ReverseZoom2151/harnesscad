"""Sublevel-set persistent homology (0-dim) on a scalar/SDF grid.

Filtration process: given
the signed distance field ``S`` sampled on a grid, define the nested filtration
``K_i = { sigma in K | S(sigma) <= s_i }`` by sweeping a threshold ``s_i``
upward.  Topological features are *born* and *die* along the sweep, producing a
set of birth-death pairs ``(b, d)`` -- the persistence diagram (PD).  The
persistence ``|d - b|`` measures how long a feature survives; points close to the
diagonal (short persistence) are noise (Sec. 3, last paragraph).

This module computes the **0-dimensional** persistence diagram -- the birth and
death of connected components along the sublevel filtration -- exactly and
deterministically with a union-find and the *elder rule* (Edelsbrunner & Harer):
when two components merge, the one with the *later* birth dies at the current
threshold.  This is the deterministic core of the paper's PH analysis; higher
dimensions require boundary-matrix reduction and are out of scope here.

Design notes (distinct from anything in the repo -- no PH exists yet):

  * ``persistence_pairs`` -- 0-dim PD as ``(birth, death)`` pairs; the single
    globally-surviving component gets ``death = +inf`` (an *essential* class).
  * ``persistence_points`` -- the paper's ``g_i = (b_i, d_i - b_i)`` birth /
    persistence representation used as diffusion conditioning features (Sec. 4.2,
    "Persistence points").
  * ``top_k_persistent`` -- keep the ``k`` longest-lived points (the paper keeps
    the 16 longest, discarding near-diagonal noise; Sec. 5.2).
  * ``betti_curve`` -- ``beta_0`` of ``K_i`` as a function of the threshold.

The grid is given as a mapping ``coord -> value`` (``coord`` an int tuple) plus
6-connectivity, or as a nested ``grid[x][y][z]`` list via ``field_from_grid``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

Coord = Tuple[int, ...]

_NEIGHBORS_6 = (
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
)


def field_from_grid(grid: Sequence[Sequence[Sequence[float]]]) -> Dict[Coord, float]:
    """Scalar field ``{(x, y, z): value}`` from a nested ``grid[x][y][z]``."""
    field: Dict[Coord, float] = {}
    for x, plane in enumerate(grid):
        for y, row in enumerate(plane):
            for z, val in enumerate(row):
                field[(x, y, z)] = float(val)
    return field


class _UnionFind:
    """Union-find whose root carries the *earliest* birth value (elder rule)."""

    def __init__(self) -> None:
        self.parent: Dict[Coord, Coord] = {}
        self.birth: Dict[Coord, float] = {}

    def add(self, c: Coord, value: float) -> None:
        self.parent[c] = c
        self.birth[c] = value

    def find(self, c: Coord) -> Coord:
        root = c
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[c] != root:  # path compression
            self.parent[c], c = root, self.parent[c]
        return root

    def union(self, a: Coord, b: Coord) -> Optional[Tuple[float, float]]:
        """Merge; return ``(birth_of_dead, birth_of_survivor)`` if a real merge."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return None
        # Survivor = earlier birth; tie broken by coordinate for determinism.
        if (self.birth[ra], ra) <= (self.birth[rb], rb):
            survivor, dead = ra, rb
        else:
            survivor, dead = rb, ra
        self.parent[dead] = survivor
        return self.birth[dead], self.birth[survivor]


def _neighbors(c: Coord) -> List[Coord]:
    n = len(c)
    if n == 3:
        return [(c[0] + dx, c[1] + dy, c[2] + dz) for dx, dy, dz in _NEIGHBORS_6]
    out: List[Coord] = []
    for axis in range(n):
        for step in (1, -1):
            nb = list(c)
            nb[axis] += step
            out.append(tuple(nb))
    return out


def persistence_pairs(
    field: Dict[Coord, float],
    include_essential: bool = True,
) -> List[Tuple[float, float]]:
    """0-dim sublevel-set persistence diagram of ``field``.

    Vertices are added in increasing value order (ties broken by coordinate for
    determinism).  A new vertex forms a component (born at its value) or joins
    existing ones; on a merge the younger component dies at the current value
    (elder rule).  The one component that never dies is *essential* with
    ``death = +inf`` when ``include_essential`` is set.
    """
    if not field:
        return []
    order = sorted(field.items(), key=lambda kv: (kv[1], kv[0]))
    uf = _UnionFind()
    present: Dict[Coord, bool] = {}
    pairs: List[Tuple[float, float]] = []
    for coord, value in order:
        uf.add(coord, value)
        present[coord] = True
        for nb in _neighbors(coord):
            if present.get(nb):
                merged = uf.union(coord, nb)
                if merged is not None:
                    dead_birth, _survivor_birth = merged
                    if dead_birth != value:  # zero-persistence pairs skipped
                        pairs.append((dead_birth, value))
    if include_essential:
        global_birth = order[0][1]
        pairs.append((global_birth, math.inf))
    pairs.sort(key=lambda bd: (bd[0], bd[1]))
    return pairs


def persistence_points(
    pairs: Sequence[Tuple[float, float]],
    finite_only: bool = True,
) -> List[Tuple[float, float]]:
    """Birth / persistence points ``g_i = (b_i, d_i - b_i)`` (Sec. 4.2).

    Essential (``inf``) classes are dropped when ``finite_only`` is set.
    """
    out: List[Tuple[float, float]] = []
    for b, d in pairs:
        if math.isinf(d):
            if finite_only:
                continue
            out.append((b, math.inf))
        else:
            out.append((b, d - b))
    return out


def persistence_values(pairs: Sequence[Tuple[float, float]]) -> List[float]:
    """Persistence ``d - b`` of each finite pair (``inf`` for essential)."""
    return [math.inf if math.isinf(d) else d - b for b, d in pairs]


def top_k_persistent(
    pairs: Sequence[Tuple[float, float]],
    k: int,
    keep_essential: bool = True,
) -> List[Tuple[float, float]]:
    """Keep the ``k`` longest-persistence pairs (Sec. 5.2 keeps the 16 longest).

    Essential classes (infinite persistence) are always kept first when
    ``keep_essential`` is set, then the longest finite pairs fill the remainder.
    Ties broken by birth value for determinism.
    """
    if k <= 0:
        return []
    essential = [p for p in pairs if math.isinf(p[1])]
    finite = [p for p in pairs if not math.isinf(p[1])]
    finite_sorted = sorted(finite, key=lambda bd: (-(bd[1] - bd[0]), bd[0], bd[1]))
    selected: List[Tuple[float, float]] = []
    if keep_essential:
        selected.extend(essential[:k])
    remaining = k - len(selected)
    if remaining > 0:
        selected.extend(finite_sorted[:remaining])
    return selected


@dataclass(frozen=True)
class BettiCurvePoint:
    threshold: float
    beta0: int


def betti_curve(field: Dict[Coord, float]) -> List[BettiCurvePoint]:
    """``beta_0`` of the sublevel complex ``K_i`` at each distinct threshold.

    Returns one point per distinct field value (sorted ascending), giving the
    component count once all vertices with value ``<= threshold`` are present.
    """
    if not field:
        return []
    order = sorted(field.items(), key=lambda kv: (kv[1], kv[0]))
    uf = _UnionFind()
    present: Dict[Coord, bool] = {}
    alive = 0
    curve: List[BettiCurvePoint] = []
    i = 0
    n = len(order)
    while i < n:
        value = order[i][1]
        # Process all vertices at this exact threshold before recording beta_0.
        while i < n and order[i][1] == value:
            coord = order[i][0]
            uf.add(coord, value)
            present[coord] = True
            alive += 1
            for nb in _neighbors(coord):
                if present.get(nb):
                    if uf.union(coord, nb) is not None:
                        alive -= 1
            i += 1
        curve.append(BettiCurvePoint(value, alive))
    return curve
