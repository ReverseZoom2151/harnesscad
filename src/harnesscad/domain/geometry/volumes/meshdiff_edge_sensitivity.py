"""Marching-tetrahedra edge-crossing noise sensitivity (MeshDiffusion Sec. 4.3).

MeshDiffusion identifies a specific, deterministic pathology of marching
tetrahedra: the extracted surface vertex on a mesh-generating edge ``(a, b)`` is
``vp = (va*sb - vb*sa)/(sb - sa)``, so a small SDF noise ``eps`` on ``sa`` and
``sb`` perturbs it by

    vp_noisy - vp = eps * (va - vb) / (sb - sa),

i.e. inversely proportional to ``|sb - sa|``.  Edges whose endpoints have a tiny
SDF gap therefore *amplify* noise, which is why the paper normalises SDFs to
``+/-1`` (forcing ``|sb - sa| = 2`` on every mesh-generating edge) before training.

This module turns that analysis into a deterministic mesh-quality metric -- the
per-edge amplification factor ``|va - vb| / |sb - sa|`` -- and its max/mean over a
mesh, letting one *quantify* how noise-sensitive a fitted tetrahedral grid is and
verify that sign-normalisation reduces it.  This metric is distinct from the
existing Chamfer / 1-NNA / EMD / watertightness / Euler metrics in the codebase.
"""

from __future__ import annotations

import math


def mesh_generating_edges(tets, sdf):
    """Sorted unique edges ``(a, b)`` whose endpoints straddle the zero level-set.

    An edge is mesh-generating when exactly one endpoint has ``sdf > 0`` (outside)
    -- these are the edges marching tetrahedra places a surface vertex on.
    """
    edges = set()
    for tet in tets:
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = tet[i], tet[j]
                if (sdf[a] > 0) != (sdf[b] > 0):
                    edges.add((a, b) if a < b else (b, a))
    return sorted(edges)


def _distance(pa, pb):
    return math.sqrt(sum((pa[r] - pb[r]) ** 2 for r in range(3)))


def edge_crossing_sensitivity(vertices, tets, sdf):
    """Per-edge and aggregate marching-tets noise-amplification factors.

    For each mesh-generating edge the amplification factor is
    ``|va - vb| / |sb - sa|`` (the coefficient of the SDF noise ``eps``).

    Returns a dict with keys:
        ``per_edge``: ``{(a, b): factor}`` for every mesh-generating edge,
        ``count``: number of mesh-generating edges,
        ``max``: largest factor (0.0 if no crossing edges),
        ``mean``: mean factor (0.0 if no crossing edges).
    """
    per_edge = {}
    for (a, b) in mesh_generating_edges(tets, sdf):
        gap = abs(sdf[b] - sdf[a])
        length = _distance(vertices[a], vertices[b])
        per_edge[(a, b)] = length / gap
    if per_edge:
        vals = list(per_edge.values())
        mx = max(vals)
        mean = sum(vals) / len(vals)
    else:
        mx = 0.0
        mean = 0.0
    return {
        "per_edge": per_edge,
        "count": len(per_edge),
        "max": mx,
        "mean": mean,
    }


def max_crossing_sensitivity(vertices, tets, sdf):
    """Convenience: the worst-case (maximum) amplification factor."""
    return edge_crossing_sensitivity(vertices, tets, sdf)["max"]
