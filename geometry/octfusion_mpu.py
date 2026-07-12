"""Multi-level Partition-of-Unity (MPU) blending for OctFusion's implicit field.

From "OctFusion: Octree-based Diffusion Models for 3D Shape Generation"
(Xiong et al., 2024), Section 3.1, Eq. (1). Each octree leaf carries a latent
feature decoded into a *local* signed-distance field; those local fields are
fused into a single continuous global field by a multi-level partition-of-unity
blend:

    F_sdf(p) = ( sum_i w_i(x) * Phi(x, f_i) ) / ( sum_i w_i(x) ),
    with  x = (p - o_i) / r_i ,

where ``o_i`` is a leaf's cell center, ``r_i`` its cell size, ``w_i`` a
"locally-supported linear B-Spline function", and ``Phi`` the (learned) shared
MLP mapping local coordinates + feature to an SDF value. Because ``w_i`` and
``Phi`` are continuous and the weights are normalised, ``F_sdf`` "are guaranteed
to be continuous" (Ohtake et al. 2003; Wang et al. 2022).

Everything *except* the learned ``Phi`` is deterministic: the local-coordinate
transform, the linear B-spline (tri-linear tent) weight, and the normalised
blend. This module implements those and takes ``Phi`` as a caller-supplied
callable ``phi(local_x, payload) -> float``. The blend is a genuine partition of
unity: with a caller ``phi`` returning a constant it reproduces that constant
exactly wherever any weight is positive, and linear B-splines reproduce linear
fields exactly on a regular leaf grid.

Stdlib-only, deterministic. Interoperates with
``geometry.octfusion_octree.Octree`` but does not require it.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float, float]

# A leaf record for blending: (center o_i, cell size r_i, payload/feature f_i).
LeafRecord = Tuple[Point, float, object]


def local_coords(point: Point, center: Point, r: float) -> Point:
    """Local coordinates ``x = (p - o_i) / r_i`` of ``point`` about a leaf."""
    if r <= 0.0:
        raise ValueError("cell size r must be positive")
    return (
        (point[0] - center[0]) / r,
        (point[1] - center[1]) / r,
        (point[2] - center[2]) / r,
    )


def _tent(t: float) -> float:
    """Linear B-spline (tent) basis: ``max(0, 1 - |t|)``, support ``(-1, 1)``."""
    a = t if t >= 0.0 else -t
    return 1.0 - a if a < 1.0 else 0.0


def bspline_weight(local_x: Point) -> float:
    """Separable tri-linear B-spline weight ``w_i(x)`` of local coordinates.

    Product of per-axis tents; locally supported (zero once any ``|x_a| >= 1``).
    On a regular grid of cell size ``r`` with ``r_i = r`` these tents form a
    partition of unity (neighbouring weights sum to 1 everywhere).
    """
    return _tent(local_x[0]) * _tent(local_x[1]) * _tent(local_x[2])


def mpu_blend(
    point: Point,
    leaves: Iterable[LeafRecord],
    phi: Callable[[Point, object], float],
    default: float = 0.0,
) -> float:
    """Evaluate the MPU-blended global field at ``point`` (Eq. 1).

    ``phi(local_x, payload)`` supplies each leaf's local field value. Leaves
    whose B-spline support does not cover ``point`` contribute nothing. When no
    leaf covers ``point`` (total weight 0) ``default`` is returned.
    """
    num = 0.0
    den = 0.0
    for (center, r, payload) in leaves:
        x = local_coords(point, center, r)
        w = bspline_weight(x)
        if w > 0.0:
            num += w * phi(x, payload)
            den += w
    if den <= 0.0:
        return default
    return num / den


def mpu_weights(point: Point, leaves: Sequence[LeafRecord]) -> List[float]:
    """Normalised partition-of-unity weights of each leaf at ``point``.

    Returns one weight per leaf (same order as ``leaves``); the non-zero entries
    sum to 1 when any leaf covers ``point``, else all entries are 0.
    """
    raw = [bspline_weight(local_coords(point, c, r)) for (c, r, _p) in leaves]
    total = sum(raw)
    if total <= 0.0:
        return [0.0 for _ in raw]
    return [w / total for w in raw]


def leaf_records_from_octree(
    tree,
    payload_of: Callable[[object], object],
    occupied_only: bool = True,
) -> List[LeafRecord]:
    """Build MPU leaf records from an :class:`Octree`.

    ``payload_of(leaf_node)`` returns the feature/payload passed to ``phi``.
    Uses ``tree.origin``/``tree.size`` for centers and cell sizes. When
    ``occupied_only`` (default) only surface leaves are included, matching
    OctFusion keeping features on non-empty leaves.
    """
    src = tree.occupied_leaves() if occupied_only else tree.leaves()
    records: List[LeafRecord] = []
    for leaf in src:
        center = leaf.center(tree.origin, tree.size)
        r = leaf.cell_size(tree.size)
        records.append((center, r, payload_of(leaf)))
    return records
