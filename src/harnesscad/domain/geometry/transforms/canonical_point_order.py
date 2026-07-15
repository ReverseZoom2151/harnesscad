"""CanonicalVAE canonical point ordering (template-matched nearest-neighbour order).

Deterministic re-encoding of the ordering rule at the heart of CanonicalVAE. That
model learns to *unfold* every shape onto one shared canonical template (a fixed
2D grid / sphere ``batch_p_2d``) and then reads off a canonical order by matching
each point to the template via nearest neighbour
(``get_order_by_chamfer_index`` -> chamfer ``idx1/idx2``,
``src/canonicalvq/loss.py``). The learned unfolding is a model artifact, but the
*ordering rule it induces* is a weight-free geometric operation: given a shape's
points and a fixed template, assign each point to its nearest template slot and
sort by that slot. Two shapes ordered against the *same* template become directly
comparable index-by-index -- exactly what CanonicalVAE exploits and what makes a
canonical order useful for reconstruction and differential comparison.

This module provides:

* a deterministic default template (a Fibonacci-sphere lattice, golden-angle
  spaced -- fixed for a given ``n``);
* ``nearest_template_index`` -- CanonicalVAE's ``idx1``: each point's nearest
  template slot;
* ``canonical_order`` / ``reorder`` -- the induced non-bijective sort;
* ``bijective_assignment`` -- a greedy one-to-one point<->slot matching (for
  equal-size sets), giving a true canonical permutation;
* ``canonical_distance`` -- order two point sets against one template and sum the
  aligned per-slot distances: a cheap, permutation-invariant differential
  comparison.

This is DISTINCT from the sketch canonical orderings under
``reconstruction.sketch`` (gencad/skexgen token orders) and from
``geometry.transforms.principal_axes`` (pose canonicalisation): here we canonicalise
the *ordering of a point set* against a shared template.

Stdlib only, deterministic. No learned weights, no randomness.
"""

from __future__ import annotations

import math
from typing import Sequence

__all__ = [
    "fibonacci_sphere",
    "grid_template",
    "nearest_template_index",
    "canonical_order",
    "reorder",
    "bijective_assignment",
    "canonical_distance",
]

Point = Sequence[float]


def fibonacci_sphere(n: int, radius: float = 1.0) -> list[tuple[float, float, float]]:
    """Deterministic near-uniform points on a sphere (golden-angle spiral).

    Fixed for a given ``n``; serves as CanonicalVAE's shared unfolding template.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return [(0.0, 0.0, radius)]
    ga = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
    pts = []
    for i in range(n):
        z = 1.0 - 2.0 * i / (n - 1)  # z in [1, -1]
        r = math.sqrt(max(0.0, 1.0 - z * z))
        theta = ga * i
        pts.append((radius * r * math.cos(theta), radius * r * math.sin(theta), radius * z))
    return pts


def grid_template(rows: int, cols: int, lo: float = -1.0, hi: float = 1.0) -> list[tuple[float, float, float]]:
    """A deterministic 2D lattice template (z=0), CanonicalVAE's flat ``p_2d`` variant."""
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be >= 1")
    def lin(k, count):
        return lo if count == 1 else lo + (hi - lo) * k / (count - 1)
    return [(lin(c, cols), lin(r, rows), 0.0) for r in range(rows) for c in range(cols)]


def _sqdist(p: Point, q: Point) -> float:
    return sum((float(a) - float(b)) ** 2 for a, b in zip(p, q))


def nearest_template_index(points: Sequence[Point], template: Sequence[Point]) -> list[int]:
    """For each point, the index of its nearest template slot (CanonicalVAE ``idx1``).

    Ties are broken by ascending template index for determinism.
    """
    if not template:
        raise ValueError("empty template")
    out = []
    for p in points:
        best_j = 0
        best_d = _sqdist(p, template[0])
        for j in range(1, len(template)):
            d = _sqdist(p, template[j])
            if d < best_d:
                best_d = d
                best_j = j
        out.append(best_j)
    return out


def canonical_order(points: Sequence[Point], template: Sequence[Point]) -> list[int]:
    """Permutation of point indices sorted by nearest template slot.

    Sort key: ``(nearest_template_index, distance_to_that_slot, original_index)`` --
    fully deterministic. ``points[perm[k]]`` walks the shape in template order.
    """
    idx = nearest_template_index(points, template)
    keyed = []
    for i, p in enumerate(points):
        j = idx[i]
        keyed.append((j, _sqdist(p, template[j]), i))
    keyed.sort()
    return [k[-1] for k in keyed]


def reorder(points: Sequence[Point], perm: Sequence[int]) -> list:
    """Apply a permutation to a point set."""
    return [points[i] for i in perm]


def bijective_assignment(points: Sequence[Point], template: Sequence[Point]) -> list[int]:
    """Greedy one-to-one matching of points to template slots (equal sizes).

    Returns ``assign`` where ``assign[i]`` is the distinct template slot given to
    point ``i``. Pairs are committed in ascending distance order (ties broken by
    point then slot index), each point and slot used once -- a deterministic
    approximation to optimal assignment, yielding a true canonical permutation.
    """
    if len(points) != len(template):
        raise ValueError("bijective assignment requires equal sizes")
    n = len(points)
    pairs = []
    for i in range(n):
        for j in range(n):
            pairs.append((_sqdist(points[i], template[j]), i, j))
    pairs.sort()
    assign = [-1] * n
    used_slot = [False] * n
    remaining = n
    for _d, i, j in pairs:
        if remaining == 0:
            break
        if assign[i] == -1 and not used_slot[j]:
            assign[i] = j
            used_slot[j] = True
            remaining -= 1
    return assign


def canonical_distance(
    a: Sequence[Point],
    b: Sequence[Point],
    template: Sequence[Point],
) -> float:
    """Differential comparison: order both sets against ``template`` and sum aligned distances.

    Both point sets are reordered into template order (via :func:`canonical_order`)
    and then compared slot-by-slot up to the shorter length. Returns the mean
    Euclidean distance between aligned points -- a cheap, order-invariant shape
    difference in the spirit of CanonicalVAE's canonical-space comparison.
    """
    pa = reorder(a, canonical_order(a, template))
    pb = reorder(b, canonical_order(b, template))
    m = min(len(pa), len(pb))
    if m == 0:
        return 0.0
    total = 0.0
    for k in range(m):
        total += math.sqrt(_sqdist(pa[k], pb[k]))
    return total / m
