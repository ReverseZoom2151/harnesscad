"""Triangle-mesh repair toolkit: weld, unify normals, fill holes, decimate.

Derived from kerf (MIT, Copyright (c) 2026 Imran Paruk).

The toolkit operates on indexed triangle meshes with a never-raise contract
and the harness mesh convention used by :mod:`harnesscad.domain.geometry.mesh.polyhedron` and
:mod:`harnesscad.domain.geometry.mesh.halfedge`: vertices are a
``List[Tuple[float, float, float]]`` and triangles are a
``List[Tuple[int, int, int]]`` (0-based indices, CCW winding = outward
normal).  All operations are plain functions on ``(vertices, triangles)``
pairs -- no mesh class is introduced.

Operations
----------
* :func:`weld_vertices` -- spatial-hash vertex welding at a tolerance;
  collapsed (degenerate) triangles are dropped during remapping.
* :func:`unify_normals` -- BFS over the face-adjacency dual graph; each
  connected component is re-wound so every interior edge is traversed in
  opposite directions by its two faces.
* :func:`fill_holes` -- boundary-loop extraction (directed half-edges whose
  twin is absent) followed by fan triangulation of each loop.
* :func:`remove_degenerate` -- removes repeated-index, zero-area and
  duplicate triangles; reports non-manifold edges.
* :func:`decimate` -- quadric-error-metric (QEM) edge
  collapse to a target triangle count or error bound, with the optimal
  collapse point solved from the summed 4x4 quadric (midpoint fallback).
* :func:`is_closed` / :func:`is_manifold` -- edge-use and vertex-fan
  diagnostics.
* :func:`repair_pipeline` -- weld -> unify_normals -> fill_holes ->
  remove_degenerate convenience chain with per-step summaries.

Out of scope
------------
Mesh boolean operations (triangle-triangle intersection plus ray-parity
inside/outside classification) and mesh offsetting (vertex displacement along
averaged normals) are not handled here -- they are large enough to warrant
their own module.  Laplacian / Taubin mesh smoothing is likewise left out of
scope.

Contract: every public entry point never raises.  Success
returns ``{"ok": True, "vertices": [...], "triangles": [...], ...stats...}``;
failure returns ``{"ok": False, "reason": "..."}``.  All iteration over
hash-ordered containers is sorted so results are deterministic.

Pure stdlib, deterministic.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict, deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

__all__ = [
    "weld_vertices",
    "unify_normals",
    "fill_holes",
    "remove_degenerate",
    "decimate",
    "is_closed",
    "is_manifold",
    "repair_pipeline",
]

Point = Tuple[float, float, float]
Tri = Tuple[int, int, int]


# --------------------------------------------------------------------------
# small vector helpers
# --------------------------------------------------------------------------


def _sub(a: Sequence[float], b: Sequence[float]) -> Point:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Sequence[float], b: Sequence[float]) -> Point:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length(v: Sequence[float]) -> float:
    return math.sqrt(_dot(v, v))


def _tri_area(vertices: List[Point], t: Tri) -> float:
    ab = _sub(vertices[t[1]], vertices[t[0]])
    ac = _sub(vertices[t[2]], vertices[t[0]])
    return _length(_cross(ab, ac)) * 0.5


# --------------------------------------------------------------------------
# validation and copying
# --------------------------------------------------------------------------


def _validate_mesh(vertices: object, triangles: object) -> Optional[str]:
    """Return an error string, or None when the inputs look valid."""
    if not isinstance(vertices, (list, tuple)):
        return "vertices must be a list"
    if not isinstance(triangles, (list, tuple)):
        return "triangles must be a list"
    nv = len(vertices)
    for i, v in enumerate(vertices):
        if not (isinstance(v, (list, tuple)) and len(v) >= 3):
            return "vertices[%d] must be (x, y, z)" % i
        try:
            float(v[0])
            float(v[1])
            float(v[2])
        except (TypeError, ValueError):
            return "vertices[%d] must contain numbers" % i
    for i, t in enumerate(triangles):
        if not (isinstance(t, (list, tuple)) and len(t) >= 3):
            return "triangles[%d] must be (i, j, k)" % i
        try:
            a, b, c = int(t[0]), int(t[1]), int(t[2])
        except (TypeError, ValueError):
            return "triangles[%d] must contain integers" % i
        if not (0 <= a < nv and 0 <= b < nv and 0 <= c < nv):
            return "triangles[%d] index out of range (nv=%d)" % (i, nv)
    return None


def _copy_mesh(
    vertices: Sequence, triangles: Sequence
) -> Tuple[List[Point], List[Tri]]:
    vs = [(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
    ts = [(int(t[0]), int(t[1]), int(t[2])) for t in triangles]
    return vs, ts


def _edge_tri_map(triangles: List[Tri]) -> Dict[Tuple[int, int], List[int]]:
    """Undirected edge -> list of triangle indices using it (insertion order)."""
    et: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for ti, t in enumerate(triangles):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            key = (a, b) if a < b else (b, a)
            et[key].append(ti)
    return et


# ==========================================================================
# weld_vertices
# ==========================================================================


def weld_vertices(
    vertices: Sequence,
    triangles: Sequence,
    tol: float = 1e-6,
) -> dict:
    """Merge vertices within *tol* of each other using a spatial grid bucket.

    Returns ``{"ok", "vertices", "triangles", "merged_count"}``; triangles
    whose corners collapse together during remapping are dropped.
    """
    try:
        err = _validate_mesh(vertices, triangles)
        if err:
            return {"ok": False, "reason": err}
        if not isinstance(tol, (int, float)) or tol < 0:
            return {"ok": False, "reason": "tol must be a non-negative number"}

        vs, ts = _copy_mesh(vertices, triangles)
        if not vs:
            return {"ok": True, "vertices": [], "triangles": [],
                    "merged_count": 0}

        # Spatial bucket: cell size = tol (minimum 1e-9 to avoid div-zero).
        cell = max(tol, 1e-9)

        def _cell_key(v: Point) -> Tuple[int, int, int]:
            return (int(math.floor(v[0] / cell)),
                    int(math.floor(v[1] / cell)),
                    int(math.floor(v[2] / cell)))

        buckets: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
        for idx, v in enumerate(vs):
            buckets[_cell_key(v)].append(idx)

        # Build old -> representative mapping.
        mapping: List[int] = list(range(len(vs)))
        merged_count = 0
        processed: Set[int] = set()

        for idx in range(len(vs)):
            if idx in processed:
                continue
            v = vs[idx]
            cx, cy, cz = _cell_key(v)
            for nx in range(cx - 1, cx + 2):
                for ny in range(cy - 1, cy + 2):
                    for nz in range(cz - 1, cz + 2):
                        for other in buckets.get((nx, ny, nz), []):
                            if other <= idx:
                                continue
                            w = vs[other]
                            d = math.sqrt((v[0] - w[0]) ** 2 +
                                          (v[1] - w[1]) ** 2 +
                                          (v[2] - w[2]) ** 2)
                            if d <= tol:
                                if mapping[other] == other:
                                    merged_count += 1
                                mapping[other] = mapping[idx]
                                processed.add(other)

        # Compact the vertex list, first-representative order.
        old_to_new: Dict[int, int] = {}
        new_vertices: List[Point] = []
        for old_idx in range(len(vs)):
            rep = mapping[old_idx]
            if rep not in old_to_new:
                old_to_new[rep] = len(new_vertices)
                new_vertices.append(vs[rep])

        # Remap triangles; drop collapsed (degenerate) ones.
        new_triangles: List[Tri] = []
        for t in ts:
            ni = old_to_new[mapping[t[0]]]
            nj = old_to_new[mapping[t[1]]]
            nk = old_to_new[mapping[t[2]]]
            if ni != nj and nj != nk and ni != nk:
                new_triangles.append((ni, nj, nk))

        return {
            "ok": True,
            "vertices": new_vertices,
            "triangles": new_triangles,
            "merged_count": merged_count,
        }
    except Exception as exc:
        return {"ok": False, "reason": "weld_vertices failed: %s" % exc}


# ==========================================================================
# unify_normals
# ==========================================================================


def unify_normals(vertices: Sequence, triangles: Sequence) -> dict:
    """BFS over the face-adjacency dual graph to make winding consistent.

    Two triangles sharing an edge are consistently oriented when they traverse
    the shared edge in opposite directions.  Each connected component is
    seeded by its lowest-index triangle (its existing winding is treated as
    correct).  Returns ``{"ok", "vertices", "triangles", "flipped_count"}``.
    """
    try:
        err = _validate_mesh(vertices, triangles)
        if err:
            return {"ok": False, "reason": err}

        vs, ts = _copy_mesh(vertices, triangles)
        if not ts:
            return {"ok": True, "vertices": vs, "triangles": ts,
                    "flipped_count": 0}

        et = _edge_tri_map(ts)

        # Dual-graph adjacency: triangle ti -> list of (tj, a, b) where (a, b)
        # is the shared undirected edge.  Only manifold (2-use) edges connect.
        adj: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
        for (a, b) in sorted(et):
            tri_list = et[(a, b)]
            if len(tri_list) == 2:
                ti, tj = tri_list
                adj[ti].append((tj, a, b))
                adj[tj].append((ti, a, b))

        visited = [False] * len(ts)
        flipped = 0

        for seed in range(len(ts)):
            if visited[seed]:
                continue
            visited[seed] = True
            queue: deque = deque([seed])
            while queue:
                ti = queue.popleft()
                t = ts[ti]
                directed = {(t[0], t[1]), (t[1], t[2]), (t[2], t[0])}
                for tj, a, b in adj[ti]:
                    if visited[tj]:
                        continue
                    visited[tj] = True
                    g = ts[tj]
                    g_directed = {(g[0], g[1]), (g[1], g[2]), (g[2], g[0])}
                    if (a, b) in directed:
                        # ti traverses a->b; tj must traverse b->a.
                        if (b, a) not in g_directed:
                            ts[tj] = (g[0], g[2], g[1])
                            flipped += 1
                    else:
                        # ti traverses b->a; tj must traverse a->b.
                        if (a, b) not in g_directed:
                            ts[tj] = (g[0], g[2], g[1])
                            flipped += 1
                    queue.append(tj)

        return {
            "ok": True,
            "vertices": vs,
            "triangles": ts,
            "flipped_count": flipped,
        }
    except Exception as exc:
        return {"ok": False, "reason": "unify_normals failed: %s" % exc}


# ==========================================================================
# fill_holes
# ==========================================================================


def fill_holes(vertices: Sequence, triangles: Sequence) -> dict:
    """Detect boundary loops and fill each with a fan triangulation.

    A boundary half-edge is a directed edge a->b present in exactly one
    triangle whose twin b->a appears in none.  Chaining these half-edges
    yields the hole perimeters, each of which is fan-triangulated from its
    first vertex.  Returns ``{"ok", "vertices", "triangles", "holes_filled"}``.
    """
    try:
        err = _validate_mesh(vertices, triangles)
        if err:
            return {"ok": False, "reason": err}

        vs, ts = _copy_mesh(vertices, triangles)
        if not ts:
            return {"ok": True, "vertices": vs, "triangles": ts,
                    "holes_filled": 0}

        directed: Set[Tuple[int, int]] = set()
        for t in ts:
            directed.add((t[0], t[1]))
            directed.add((t[1], t[2]))
            directed.add((t[2], t[0]))

        # boundary_next[a] = b for every boundary half-edge a->b.
        boundary_next: Dict[int, int] = {}
        for (a, b) in sorted(directed):
            if (b, a) not in directed:
                boundary_next[a] = b

        if not boundary_next:
            return {"ok": True, "vertices": vs, "triangles": ts,
                    "holes_filled": 0}

        # Extract loops by chaining boundary_next (sorted starts: stable).
        visited: Set[int] = set()
        loops: List[List[int]] = []
        for start in sorted(boundary_next):
            if start in visited:
                continue
            loop: List[int] = []
            cur = start
            for _ in range(len(boundary_next) + 2):
                if cur in visited:
                    break
                visited.add(cur)
                loop.append(cur)
                nxt = boundary_next.get(cur, -1)
                if nxt == -1 or nxt == start:
                    break
                cur = nxt
            if len(loop) >= 3:
                loops.append(loop)

        holes_filled = 0
        for loop in loops:
            anchor = loop[0]
            for k in range(1, len(loop) - 1):
                ts.append((anchor, loop[k], loop[k + 1]))
            holes_filled += 1

        return {
            "ok": True,
            "vertices": vs,
            "triangles": ts,
            "holes_filled": holes_filled,
        }
    except Exception as exc:
        return {"ok": False, "reason": "fill_holes failed: %s" % exc}


# ==========================================================================
# remove_degenerate
# ==========================================================================


def remove_degenerate(vertices: Sequence, triangles: Sequence) -> dict:
    """Remove repeated-index, zero-area and duplicate triangles.

    Duplicates compare on the sorted index tuple, so a triangle and its
    mirror count as one.  Also reports edges still used by more than two
    triangles.  Returns ``{"ok", "vertices", "triangles", "removed_count",
    "non_manifold_edges"}``.
    """
    try:
        err = _validate_mesh(vertices, triangles)
        if err:
            return {"ok": False, "reason": err}

        vs, ts = _copy_mesh(vertices, triangles)
        if not ts:
            return {"ok": True, "vertices": vs, "triangles": [],
                    "removed_count": 0, "non_manifold_edges": []}

        kept: List[Tri] = []
        removed = 0
        seen_canonical: Set[Tuple[int, int, int]] = set()

        for t in ts:
            a, b, c = t
            if a == b or b == c or a == c:
                removed += 1
                continue
            if _tri_area(vs, t) < 1e-15:
                removed += 1
                continue
            canon = tuple(sorted((a, b, c)))
            if canon in seen_canonical:
                removed += 1
                continue
            seen_canonical.add(canon)
            kept.append(t)

        et = _edge_tri_map(kept)
        non_manifold = sorted(e for e, tl in et.items() if len(tl) > 2)

        return {
            "ok": True,
            "vertices": vs,
            "triangles": kept,
            "removed_count": removed,
            "non_manifold_edges": non_manifold,
        }
    except Exception as exc:
        return {"ok": False, "reason": "remove_degenerate failed: %s" % exc}


# ==========================================================================
# decimate (quadric-error-metric edge collapse)
# ==========================================================================


def _make_quadrics(vs: List[Point], ts: List[Tri]) -> List[List[float]]:
    """Per-vertex 4x4 quadric error matrices (row-major, 16 floats each)."""
    Q: List[List[float]] = [[0.0] * 16 for _ in range(len(vs))]
    for t in ts:
        a, b, c = vs[t[0]], vs[t[1]], vs[t[2]]
        n = _cross(_sub(b, a), _sub(c, a))
        ln = _length(n)
        if ln < 1e-15:
            continue
        nx, ny, nz = n[0] / ln, n[1] / ln, n[2] / ln
        d = -(nx * a[0] + ny * a[1] + nz * a[2])
        p = (nx, ny, nz, d)
        for vi in (t[0], t[1], t[2]):
            qv = Q[vi]
            for r in range(4):
                for cc in range(4):
                    qv[r * 4 + cc] += p[r] * p[cc]
    return Q


def _q_add(a: List[float], b: List[float]) -> List[float]:
    return [a[i] + b[i] for i in range(16)]


def _q_error(Q: List[float], v: Point) -> float:
    """v^T Q v where v = (x, y, z, 1)."""
    vv = (v[0], v[1], v[2], 1.0)
    s = 0.0
    for r in range(4):
        for c in range(4):
            s += vv[r] * Q[r * 4 + c] * vv[c]
    return s


def _optimal_collapse_point(Qc: List[float], va: Point, vb: Point) -> Point:
    """Solve for the optimal collapse vertex; fall back to the midpoint."""
    a00, a01, a02 = Qc[0], Qc[1], Qc[2]
    a10, a11, a12 = Qc[4], Qc[5], Qc[6]
    a20, a21, a22 = Qc[8], Qc[9], Qc[10]
    b0, b1, b2 = -Qc[3], -Qc[7], -Qc[11]

    det = (a00 * (a11 * a22 - a12 * a21)
           - a01 * (a10 * a22 - a12 * a20)
           + a02 * (a10 * a21 - a11 * a20))

    if abs(det) < 1e-12:
        return ((va[0] + vb[0]) * 0.5,
                (va[1] + vb[1]) * 0.5,
                (va[2] + vb[2]) * 0.5)

    inv_det = 1.0 / det
    x = inv_det * (b0 * (a11 * a22 - a12 * a21)
                   - a01 * (b1 * a22 - a12 * b2)
                   + a02 * (b1 * a21 - a11 * b2))
    y = inv_det * (a00 * (b1 * a22 - a12 * b2)
                   - b0 * (a10 * a22 - a12 * a20)
                   + a02 * (a10 * b2 - b1 * a20))
    z = inv_det * (a00 * (a11 * b2 - b1 * a21)
                   - a01 * (a10 * b2 - b1 * a20)
                   + b0 * (a10 * a21 - a11 * a20))
    return (x, y, z)


def decimate(
    vertices: Sequence,
    triangles: Sequence,
    target_faces: Optional[int] = None,
    max_error: Optional[float] = None,
) -> dict:
    """QEM edge-collapse decimation.

    Repeatedly collapses the lowest-error edge -- the error being v^T Q v of
    the summed vertex quadrics at the optimal collapse point -- until the
    active triangle count reaches *target_faces* or the cheapest collapse
    exceeds *max_error*.  At least one of the two bounds must be given.
    Candidates are rebuilt each iteration (a full sorted sweep rather than a
    heap), which is fine for the small meshes the harness handles.  Returns
    ``{"ok", "vertices", "triangles", "original_faces", "final_faces"}``.
    """
    try:
        err = _validate_mesh(vertices, triangles)
        if err:
            return {"ok": False, "reason": err}
        if target_faces is None and max_error is None:
            return {"ok": False,
                    "reason": "provide target_faces or max_error"}
        if target_faces is not None:
            try:
                target_faces = int(target_faces)
            except (TypeError, ValueError):
                return {"ok": False,
                        "reason": "target_faces must be an integer"}
            if target_faces < 1:
                return {"ok": False, "reason": "target_faces must be >= 1"}
        if max_error is not None:
            try:
                max_error = float(max_error)
            except (TypeError, ValueError):
                return {"ok": False, "reason": "max_error must be a number"}
            if max_error <= 0:
                return {"ok": False, "reason": "max_error must be positive"}

        vs, ts = _copy_mesh(vertices, triangles)
        original_count = len(ts)

        if not ts or len(ts) <= (target_faces or 1):
            return {"ok": True, "vertices": vs, "triangles": ts,
                    "original_faces": original_count,
                    "final_faces": len(ts)}

        Q = _make_quadrics(vs, ts)
        active_verts: List[bool] = [True] * len(vs)
        active_tris: List[bool] = [True] * len(ts)

        vert_tris: List[Set[int]] = [set() for _ in range(len(vs))]
        for ti, t in enumerate(ts):
            for vi in t:
                vert_tris[vi].add(ti)

        def _build_candidates() -> List[Tuple[float, int, int]]:
            """(error, va, vb) for every active edge, sorted ascending."""
            seen_edges: Set[Tuple[int, int]] = set()
            cands: List[Tuple[float, int, int]] = []
            for ti, t in enumerate(ts):
                if not active_tris[ti]:
                    continue
                for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                    key = (a, b) if a < b else (b, a)
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                    Qc = _q_add(Q[a], Q[b])
                    vopt = _optimal_collapse_point(Qc, vs[a], vs[b])
                    cands.append((_q_error(Qc, vopt), a, b))
            cands.sort()
            return cands

        max_iterations = max(1, original_count - (target_faces or 1)) * 4

        for _ in range(max_iterations):
            nactive = sum(1 for x in active_tris if x)
            if target_faces is not None and nactive <= target_faces:
                break

            cands = _build_candidates()
            if not cands:
                break

            error, va, vb = cands[0]
            if max_error is not None and error > max_error:
                break

            # Collapse vb into va at the optimal point.
            Qc = _q_add(Q[va], Q[vb])
            vs[va] = _optimal_collapse_point(Qc, vs[va], vs[vb])
            Q[va] = Qc
            active_verts[vb] = False

            for ti in sorted(vert_tris[vb]):
                if not active_tris[ti]:
                    continue
                t = ts[ti]
                new_t = tuple(va if x == vb else x for x in t)
                if len(set(new_t)) < 3:
                    # Triangle collapsed to an edge or point: deactivate.
                    active_tris[ti] = False
                    for vi in t:
                        vert_tris[vi].discard(ti)
                else:
                    ts[ti] = (new_t[0], new_t[1], new_t[2])
                    vert_tris[vb].discard(ti)
                    for vi in new_t:
                        vert_tris[vi].add(ti)

        # Compact the surviving vertices and triangles.
        new_vs: List[Point] = []
        remap: Dict[int, int] = {}
        for i, v in enumerate(vs):
            if active_verts[i]:
                remap[i] = len(new_vs)
                new_vs.append(v)

        new_ts: List[Tri] = []
        for ti, t in enumerate(ts):
            if active_tris[ti]:
                nt = [remap[vi] for vi in t if vi in remap]
                if len(nt) == 3 and len(set(nt)) == 3:
                    new_ts.append((nt[0], nt[1], nt[2]))

        return {
            "ok": True,
            "vertices": new_vs,
            "triangles": new_ts,
            "original_faces": original_count,
            "final_faces": len(new_ts),
        }
    except Exception as exc:
        return {"ok": False, "reason": "decimate failed: %s" % exc}


# ==========================================================================
# diagnostics
# ==========================================================================


def is_closed(vertices: Sequence, triangles: Sequence) -> dict:
    """True when every undirected edge is shared by exactly 2 triangles.

    Returns ``{"ok", "closed"}``.
    """
    try:
        err = _validate_mesh(vertices, triangles)
        if err:
            return {"ok": False, "reason": err}
        _, ts = _copy_mesh(vertices, triangles)
        et = _edge_tri_map(ts)
        closed = bool(et) and all(len(tl) == 2 for tl in et.values())
        return {"ok": True, "closed": closed}
    except Exception as exc:
        return {"ok": False, "reason": "is_closed failed: %s" % exc}


def is_manifold(vertices: Sequence, triangles: Sequence) -> dict:
    """Check edge- and vertex-manifold conditions.

    Edge-manifold: no undirected edge used by more than 2 triangles.
    Vertex-manifold: the triangles incident to each vertex form a single
    edge-connected fan.  Returns ``{"ok", "manifold", "non_manifold_edges",
    "non_manifold_vertices"}``.
    """
    try:
        err = _validate_mesh(vertices, triangles)
        if err:
            return {"ok": False, "reason": err}
        vs, ts = _copy_mesh(vertices, triangles)
        et = _edge_tri_map(ts)

        bad_edges = sorted(e for e, tl in et.items() if len(tl) > 2)

        vert_tri_list: Dict[int, List[int]] = defaultdict(list)
        for ti, t in enumerate(ts):
            for vi in t:
                vert_tri_list[vi].append(ti)

        bad_verts: List[int] = []
        for vi in range(len(vs)):
            tlist = vert_tri_list.get(vi, [])
            if len(tlist) < 2:
                continue
            # Local adjacency: two incident triangles are adjacent when they
            # share an edge (>= 2 common vertices).
            local_adj: Dict[int, Set[int]] = defaultdict(set)
            for i, ti in enumerate(tlist):
                ti_verts = set(ts[ti])
                for tj in tlist[i + 1:]:
                    if len(ti_verts & set(ts[tj])) >= 2:
                        local_adj[ti].add(tj)
                        local_adj[tj].add(ti)
            # The fan is manifold when the local graph is connected.
            visited_local: Set[int] = set()
            dq: deque = deque([tlist[0]])
            while dq:
                cur = dq.popleft()
                if cur in visited_local:
                    continue
                visited_local.add(cur)
                for nb in sorted(local_adj[cur]):
                    if nb not in visited_local:
                        dq.append(nb)
            if len(visited_local) != len(tlist):
                bad_verts.append(vi)

        manifold = not bad_edges and not bad_verts
        return {
            "ok": True,
            "manifold": manifold,
            "non_manifold_edges": bad_edges,
            "non_manifold_vertices": bad_verts,
        }
    except Exception as exc:
        return {"ok": False, "reason": "is_manifold failed: %s" % exc}


# ==========================================================================
# repair_pipeline
# ==========================================================================


def repair_pipeline(
    vertices: Sequence,
    triangles: Sequence,
    tol: float = 1e-6,
) -> dict:
    """Convenience chain: weld -> unify_normals -> fill_holes -> remove_degenerate.

    Returns ``{"ok", "vertices", "triangles", "steps"}`` where *steps* is a
    list of ``{"step", "ok", "detail"}`` summaries.  Never raises.
    """
    try:
        steps: List[dict] = []

        r = weld_vertices(vertices, triangles, tol=tol)
        steps.append({
            "step": "weld_vertices",
            "ok": r["ok"],
            "detail": r.get(
                "reason",
                "merged %d vertices" % r.get("merged_count", 0)),
        })
        if not r["ok"]:
            return {"ok": False, "vertices": list(vertices),
                    "triangles": list(triangles), "steps": steps}
        vertices, triangles = r["vertices"], r["triangles"]

        r = unify_normals(vertices, triangles)
        steps.append({
            "step": "unify_normals",
            "ok": r["ok"],
            "detail": r.get(
                "reason",
                "flipped %d triangles" % r.get("flipped_count", 0)),
        })
        if not r["ok"]:
            return {"ok": False, "vertices": vertices,
                    "triangles": triangles, "steps": steps}
        vertices, triangles = r["vertices"], r["triangles"]

        r = fill_holes(vertices, triangles)
        steps.append({
            "step": "fill_holes",
            "ok": r["ok"],
            "detail": r.get(
                "reason",
                "filled %d holes" % r.get("holes_filled", 0)),
        })
        if not r["ok"]:
            return {"ok": False, "vertices": vertices,
                    "triangles": triangles, "steps": steps}
        vertices, triangles = r["vertices"], r["triangles"]

        r = remove_degenerate(vertices, triangles)
        steps.append({
            "step": "remove_degenerate",
            "ok": r["ok"],
            "detail": r.get(
                "reason",
                "removed %d degenerate triangles" % r.get("removed_count", 0)),
        })
        if not r["ok"]:
            return {"ok": False, "vertices": vertices,
                    "triangles": triangles, "steps": steps}
        vertices, triangles = r["vertices"], r["triangles"]

        return {"ok": True, "vertices": vertices, "triangles": triangles,
                "steps": steps}
    except Exception as exc:
        return {
            "ok": False,
            "vertices": [],
            "triangles": [],
            "steps": [{"step": "repair_pipeline", "ok": False,
                       "detail": str(exc)}],
        }


# --------------------------------------------------------------------------
# synthetic geometry for the selfcheck
# --------------------------------------------------------------------------


def _unit_cube_mesh() -> Tuple[List[Point], List[Tri]]:
    """Unit cube as 8 vertices and 12 CCW-outward triangles."""
    v: List[Point] = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]
    quads = [
        (0, 3, 2, 1),  # bottom, -z
        (4, 5, 6, 7),  # top, +z
        (0, 1, 5, 4),  # -y
        (1, 2, 6, 5),  # +x
        (2, 3, 7, 6),  # +y
        (3, 0, 4, 7),  # -x
    ]
    t: List[Tri] = []
    for a, b, c, d in quads:
        t.append((a, b, c))
        t.append((a, c, d))
    return v, t


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.geometry.mesh.repair_toolkit",
        description="Triangle-mesh repair toolkit: weld, unify normals, "
                    "fill holes, remove degenerates, QEM decimation "
                    "with a never-raise contract.",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="build synthetic cube meshes and prove weld / "
                             "unify / fill / degenerate-removal / decimate "
                             "behaviors.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    # --- weld merges duplicated vertices ---------------------------------
    cube_v, cube_t = _unit_cube_mesh()
    dup_v = list(cube_v) + [cube_v[0], cube_v[6]]  # exact duplicates
    # Rewire the first two triangles to use the duplicates (0->8, 6->9).
    dup_t = [tuple(8 if x == 0 else (9 if x == 6 else x) for x in t)
             for t in cube_t[:2]] + list(cube_t[2:])
    r = weld_vertices(dup_v, dup_t, tol=1e-9)
    assert r["ok"], r
    assert r["merged_count"] == 2, r["merged_count"]
    assert len(r["vertices"]) == 8, len(r["vertices"])
    assert is_closed(r["vertices"], r["triangles"])["closed"]
    print("[selfcheck] weld_vertices merged %d duplicated vertices -> "
          "%d vertices, closed mesh" % (r["merged_count"],
                                        len(r["vertices"])))

    # --- is_closed False on holed cube; True after fill_holes ------------
    holed_t = cube_t[:2] + cube_t[4:]  # punch out the top face (2 triangles)
    assert is_closed(cube_v, cube_t)["closed"]
    rc = is_closed(cube_v, holed_t)
    assert rc["ok"] and not rc["closed"], rc
    rf = fill_holes(cube_v, holed_t)
    assert rf["ok"] and rf["holes_filled"] == 1, rf
    rc2 = is_closed(rf["vertices"], rf["triangles"])
    assert rc2["ok"] and rc2["closed"], rc2
    print("[selfcheck] holed cube closed=%s; after fill_holes "
          "(holes_filled=%d) closed=%s"
          % (rc["closed"], rf["holes_filled"], rc2["closed"]))

    # --- unify_normals fixes a deliberately flipped triangle -------------
    flipped_t = list(cube_t)
    t0 = flipped_t[3]
    flipped_t[3] = (t0[0], t0[2], t0[1])
    ru = unify_normals(cube_v, flipped_t)
    assert ru["ok"] and ru["flipped_count"] == 1, ru
    directed = set()
    for t in ru["triangles"]:
        directed.update(((t[0], t[1]), (t[1], t[2]), (t[2], t[0])))
    assert all((b, a) in directed for (a, b) in directed), \
        "winding still inconsistent"
    print("[selfcheck] unify_normals re-flipped %d triangle(s); every "
          "directed edge now has its twin" % ru["flipped_count"])

    # --- remove_degenerate drops a zero-area triangle --------------------
    degen_t = list(cube_t) + [(0, 1, 1), (0, 1, 0)]
    zero_v = list(cube_v) + [cube_v[0]]  # vertex 8 == vertex 0
    zero_t = list(cube_t) + [(0, 8, 1)]  # zero-area sliver
    rd = remove_degenerate(cube_v, degen_t)
    assert rd["ok"] and rd["removed_count"] == 2, rd
    assert len(rd["triangles"]) == 12, len(rd["triangles"])
    rz = remove_degenerate(zero_v, zero_t)
    assert rz["ok"] and rz["removed_count"] == 1, rz
    print("[selfcheck] remove_degenerate dropped %d repeated-index and "
          "%d zero-area triangle(s)" % (rd["removed_count"],
                                        rz["removed_count"]))

    # --- decimate reduces triangle count while staying manifold ----------
    rdec = decimate(cube_v, cube_t, target_faces=10)
    assert rdec["ok"], rdec
    assert rdec["final_faces"] < 12, rdec["final_faces"]
    rm = is_manifold(rdec["vertices"], rdec["triangles"])
    assert rm["ok"] and rm["manifold"], rm
    assert is_closed(rdec["vertices"], rdec["triangles"])["closed"]
    print("[selfcheck] decimate: %d -> %d triangles, result closed and "
          "manifold" % (rdec["original_faces"], rdec["final_faces"]))

    # --- never-raise contract on garbage input ---------------------------
    assert weld_vertices("nope", [])["ok"] is False
    assert decimate([], [], target_faces=0)["ok"] is False
    assert fill_holes([(0, 0, 0)], [(0, 1, 2)])["ok"] is False
    assert repair_pipeline(None, None)["ok"] is False
    print("[selfcheck] bad inputs return ok=False dicts, never raise")

    # --- full pipeline on a welded, holed, dirty cube ---------------------
    rp = repair_pipeline(dup_v, dup_t[:-1] + [(0, 1, 1)])
    assert rp["ok"], rp
    assert all(s["ok"] for s in rp["steps"]), rp["steps"]
    print("[selfcheck] repair_pipeline: %s"
          % "; ".join("%s (%s)" % (s["step"], s["detail"])
                      for s in rp["steps"]))

    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
