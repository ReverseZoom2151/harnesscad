"""Deterministic self-intersection detection and repair for triangle meshes.

Ported (as a deterministic, stdlib-only reduction) from *Instant Self-Intersection
Repair for 3D Meshes* (Jang, Jung, Lee, Lee -- ACM TOG 2025,
``instant-mesh-intersection-repair``).  That method runs a loop:

    detect self-intersecting triangle pairs (a BVH triangle-triangle collider)
    -> take a gradient step of a *penetration energy* that pushes intersecting
       geometry apart, regularised by shape-preserving constraints (volume,
       area, curvature) in a Laplacian-preconditioned parameterisation
    -> re-detect, stopping as soon as the collision count reaches zero.

The reference implementation is CUDA + autodiff + a Cholesky Laplacian solve, so
it is not portable.  The *transferable, deterministic* substrate is the loop
skeleton itself and its stopping rule, which the harness can run on top of the
machinery it already owns:

* broad phase -- :class:`geometry.mesh.bvh.BVH` overlap pairs;
* narrow phase -- :func:`geometry.mesh.triangle_intersect.triangles_intersect`
  (exact-predicate triangle-triangle test);
* the repair step -- move the vertices of each colliding triangle pair apart
  along the direction separating their centroids (a discrete, deterministic
  stand-in for the penetration-energy gradient), then apply light umbrella
  (uniform Laplacian) smoothing so the displacement stays local and the surface
  is not torn (the deterministic analogue of the paper's curvature constraint).

Why this matters for the harness: the measured output gate refuses any part
whose surface mesh self-intersects (``io/gate.py`` -> ``self_intersections``).
Before this module the only response to such a part was refusal.  A deterministic
repair pass can turn a subset of those refusals into valid, non-self-intersecting
meshes -- e.g. two overlapping solids pushed apart, or a mildly interpenetrating
extrusion relaxed -- without any kernel, autodiff, or trained model.

This is *not* a boolean resolver: it does not retriangulate intersections, so it
cannot merge two solids into one.  It resolves *penetration* by displacement,
which is exactly what the gate's self-intersection check keys on.

Pure stdlib, deterministic (fixed iteration order; averaged displacements).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from harnesscad.domain.geometry.mesh.bvh import AABB, BVH, boxes_of_triangles
from harnesscad.domain.geometry.mesh.triangle_intersect import triangles_intersect

__all__ = [
    "find_self_intersections",
    "RepairResult",
    "repair_self_intersections",
]

Vec3 = Tuple[float, float, float]


# --------------------------------------------------------------------------
# small vector helpers (kept local so this module is self-contained)
# --------------------------------------------------------------------------

def _sub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: Sequence[float], s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _cross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _normalize(a: Sequence[float]) -> Optional[Vec3]:
    n = _norm(a)
    if n <= 1e-15:
        return None
    return (a[0] / n, a[1] / n, a[2] / n)


def _centroid(tri: Sequence[Sequence[float]]) -> Vec3:
    return ((tri[0][0] + tri[1][0] + tri[2][0]) / 3.0,
            (tri[0][1] + tri[1][1] + tri[2][1]) / 3.0,
            (tri[0][2] + tri[1][2] + tri[2][2]) / 3.0)


def _tri_normal(tri: Sequence[Sequence[float]]) -> Optional[Vec3]:
    return _normalize(_cross(_sub(tri[1], tri[0]), _sub(tri[2], tri[0])))


def _tri(vertices: Sequence[Sequence[float]], face: Sequence[int]):
    return (tuple(vertices[face[0]]), tuple(vertices[face[1]]), tuple(vertices[face[2]]))


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

def find_self_intersections(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    *,
    ignore_shared_vertices: bool = True,
) -> List[Tuple[int, int]]:
    """Return sorted ``(i, j)`` pairs of triangles that genuinely intersect.

    Uses the harness BVH for broad phase and the exact-predicate
    triangle-triangle test for narrow phase.  Triangle pairs that share a mesh
    vertex are, by default, *not* reported: adjacent faces meet along a shared
    vertex/edge by construction and that incidence is not a self-intersection
    (this mirrors what the output gate treats as an intersection).
    """
    faces = [tuple(f) for f in faces]
    if not faces:
        return []
    boxes = boxes_of_triangles(vertices, faces)
    bvh = BVH(boxes)
    out: List[Tuple[int, int]] = []
    for (i, j) in bvh.self_collisions():
        fi, fj = faces[i], faces[j]
        if ignore_shared_vertices and (set(fi) & set(fj)):
            continue
        if triangles_intersect(_tri(vertices, fi), _tri(vertices, fj)):
            out.append((i, j))
    out.sort()
    return out


# --------------------------------------------------------------------------
# repair
# --------------------------------------------------------------------------

@dataclass
class RepairResult:
    """Outcome of a repair run.

    ``vertices`` is the repaired vertex list (a new list; input is not mutated).
    ``resolved`` is True iff the final mesh has zero self-intersections.
    """
    vertices: List[Vec3]
    iterations: int
    initial_intersections: int
    final_intersections: int
    history: List[int]

    @property
    def resolved(self) -> bool:
        return self.final_intersections == 0


def _vertex_neighbours(faces: Sequence[Sequence[int]], n_verts: int) -> List[List[int]]:
    adj: List[Set[int]] = [set() for _ in range(n_verts)]
    for f in faces:
        a, b, c = f[0], f[1], f[2]
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    # deterministic order
    return [sorted(s) for s in adj]


def repair_self_intersections(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    *,
    max_iters: int = 200,
    step: float = 0.5,
    smooth: float = 0.0,
    ignore_shared_vertices: bool = True,
) -> RepairResult:
    """Iteratively push apart self-intersecting triangles until none remain.

    The loop mirrors the reference method's ``detect -> step -> re-detect ->
    stop-at-zero`` structure:

    1. Detect intersecting triangle pairs (:func:`find_self_intersections`).
    2. For every pair, compute the unit direction separating the two triangle
       centroids and accumulate, on each triangle's three vertices, a
       displacement of ``step * penetration_scale`` along that direction (each
       triangle moves away from the other).  ``penetration_scale`` is the local
       triangle size, so the step is scale-invariant.
    3. Average the accumulated displacement per vertex and apply it.
    4. Optionally apply ``smooth`` rounds-worth of uniform Laplacian smoothing
       (umbrella operator) to keep the displacement from tearing the surface --
       the deterministic analogue of the paper's curvature constraint.
    5. Repeat until zero intersections or ``max_iters``.

    Determinism: pairs are visited in sorted order, displacements are summed
    then averaged, and smoothing uses a fixed neighbour ordering, so the result
    is a pure function of the inputs.
    """
    verts: List[Vec3] = [tuple(float(c) for c in v) for v in vertices]  # type: ignore
    faces = [tuple(int(i) for i in f) for f in faces]
    n = len(verts)
    neighbours = _vertex_neighbours(faces, n) if smooth > 0.0 else None

    initial = find_self_intersections(verts, faces,
                                      ignore_shared_vertices=ignore_shared_vertices)
    history = [len(initial)]
    if not initial:
        return RepairResult(list(verts), 0, 0, 0, history)

    pairs = initial
    it = 0
    # Keep the BEST iterate, not the last one. The relaxation is not monotone:
    # pushing apart one intersecting pair can drive a different pair together,
    # so a history like [12, 1, 7] is ordinary -- and returning the final
    # iterate hands back the 7-collision mesh when a 1-collision mesh was in
    # hand. Upstream (instant-mesh-intersection-repair) tracks the best; this
    # port dropped that and so could return a result strictly worse than one it
    # had already computed.
    best_verts = list(verts)
    best_count = len(initial)
    best_iter = 0
    while pairs and it < max_iters:
        it += 1
        disp: Dict[int, List[float]] = {}
        count: Dict[int, int] = {}

        for (i, j) in pairs:
            fi, fj = faces[i], faces[j]
            ti, tj = _tri(verts, fi), _tri(verts, fj)
            ci, cj = _centroid(ti), _centroid(tj)
            d = _normalize(_sub(ci, cj))
            if d is None:
                # coincident centroids: separate along i's face normal
                d = _tri_normal(ti) or (1.0, 0.0, 0.0)
            # scale-invariant magnitude: mean edge length of the two triangles
            size = 0.5 * (_norm(_sub(ti[1], ti[0])) + _norm(_sub(tj[1], tj[0])))
            if size <= 1e-15:
                size = 1.0
            mag = step * size
            push_i = _scale(d, mag)
            push_j = _scale(d, -mag)
            for vidx in fi:
                acc = disp.setdefault(vidx, [0.0, 0.0, 0.0])
                acc[0] += push_i[0]; acc[1] += push_i[1]; acc[2] += push_i[2]
                count[vidx] = count.get(vidx, 0) + 1
            for vidx in fj:
                acc = disp.setdefault(vidx, [0.0, 0.0, 0.0])
                acc[0] += push_j[0]; acc[1] += push_j[1]; acc[2] += push_j[2]
                count[vidx] = count.get(vidx, 0) + 1

        for vidx, acc in disp.items():
            c = count[vidx]
            verts[vidx] = (verts[vidx][0] + acc[0] / c,
                           verts[vidx][1] + acc[1] / c,
                           verts[vidx][2] + acc[2] / c)

        if smooth > 0.0 and neighbours is not None:
            verts = _laplacian_smooth(verts, neighbours, smooth)

        pairs = find_self_intersections(verts, faces,
                                        ignore_shared_vertices=ignore_shared_vertices)
        history.append(len(pairs))
        if len(pairs) < best_count:
            best_verts = list(verts)
            best_count = len(pairs)
            best_iter = it

    # `iterations` reports where the returned mesh came from, so the number and
    # the vertices always describe the same moment.
    return RepairResult(best_verts, best_iter, len(initial), best_count, history)


def _laplacian_smooth(
    verts: Sequence[Vec3],
    neighbours: Sequence[Sequence[int]],
    weight: float,
) -> List[Vec3]:
    """One pass of uniform (umbrella) Laplacian smoothing with the given weight."""
    out: List[Vec3] = []
    for i, v in enumerate(verts):
        nb = neighbours[i]
        if not nb:
            out.append(tuple(v))  # type: ignore
            continue
        sx = sy = sz = 0.0
        for k in nb:
            sx += verts[k][0]; sy += verts[k][1]; sz += verts[k][2]
        m = len(nb)
        cx, cy, cz = sx / m, sy / m, sz / m
        out.append((v[0] + weight * (cx - v[0]),
                    v[1] + weight * (cy - v[1]),
                    v[2] + weight * (cz - v[2])))
    return out
