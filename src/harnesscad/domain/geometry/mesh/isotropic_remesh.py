"""Isotropic remeshing: split / collapse / flip / smooth.

The pure-Python algorithm uses these deterministic passes:

1. split every edge longer than ``4/3 * L`` at its midpoint, processed
   longest-first (kerf splits one edge per pass and rebuilds the edge map,
   bounded at 20 passes per iteration);
2. collapse every interior edge shorter than ``4/5 * L`` into its midpoint,
   processed shortest-first (one collapse per pass, bounded at 20 passes);
3. flip interior edges when doing so reduces the total deviation of the four
   incident vertices from the target valence (6 for interior vertices, 4 for
   boundary vertices);
4. tangentially smooth interior vertices: move each vertex toward the
   centroid of its one-ring (uniform Laplacian, strength 0.5) with the update
   projected onto the tangent plane of the area-weighted vertex normal.

The four steps run for ``iterations`` cycles (kerf default 5) and degenerate
triangles are dropped at the end.  Boundary contract (kerf's documented
behaviour): boundary edges are never split or collapsed, and boundary
vertices are never moved -- an open patch keeps its boundary polygon exactly.

Fidelity notes relative to the kerf source:

* kerf's split pass does not actually skip boundary edges even though its
  module docstring promises it never splits them; this port enforces the
  documented contract and skips boundary edges in the split pass.
* kerf's collapse always moves the surviving vertex to the edge midpoint,
  which would drag a boundary vertex off the boundary; this port collapses
  an interior/boundary edge pair into the boundary vertex (position kept)
  and skips edges whose endpoints are both on the boundary.  A link
  condition (the one-ring intersection of the endpoints must be exactly the
  set of opposite vertices) is checked so a collapse cannot create a
  non-manifold edge.
* kerf's flip picks the two opposite vertices without tracking the directed
  orientation of the shared edge, which can invert winding; this port keeps
  the flip orientation-aware, and refuses a flip that would duplicate an
  existing edge.
* kerf targets valence 6 everywhere; this port uses the Botsch-Kobbelt
  targets of 6 for interior and 4 for boundary vertices, with valence
  counted as the number of distinct one-ring neighbours.

Pure stdlib, deterministic: every priority queue is sorted with the vertex
indices as tie-breakers.  Returns a result dict exactly as kerf does.
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

__all__ = ["isotropic_remesh"]

Point = Tuple[float, float, float]
Tri = Tuple[int, int, int]
Edge = Tuple[int, int]

_MAX_SPLIT_PASSES = 20
_MAX_COLLAPSE_PASSES = 20
_MAX_FLIP_SWEEPS = 10
_SMOOTH_STRENGTH = 0.5
_INTERIOR_TARGET_VALENCE = 6
_BOUNDARY_TARGET_VALENCE = 4


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def isotropic_remesh(
    vertices: Sequence[Sequence[float]],
    triangles: Sequence[Sequence[int]],
    target_edge_length: float,
    iterations: int = 5,
) -> Dict[str, list]:
    """Remesh toward uniform edge length *target_edge_length*.

    Parameters
    ----------
    vertices : sequence of (x, y, z)
        Vertex positions.
    triangles : sequence of vertex-index sequences
        Faces; triangles are used as-is, larger polygons are fan-triangulated
        (kerf accepts quads the same way).
    target_edge_length : float
        Desired average edge length after remeshing.
    iterations : int
        Number of split -> collapse -> flip -> smooth cycles (default 5).

    Returns
    -------
    dict
        ``{"vertices": List[Point], "faces": List[Tri]}`` -- all faces are
        triangles.  Vertices left unreferenced by a collapse are kept (as in
        kerf); consumers that need compaction must reindex.
    """
    if target_edge_length <= 0:
        raise ValueError("target_edge_length must be positive")

    verts: List[List[float]] = [
        [float(p[0]), float(p[1]), float(p[2])] for p in vertices
    ]
    faces: List[List[int]] = _triangulate(triangles)

    if not verts or not faces:
        return {"vertices": [tuple(v) for v in verts], "faces": []}

    length = float(target_edge_length)
    split_thresh = (4.0 / 3.0) * length
    collapse_thresh = (4.0 / 5.0) * length

    for _iteration in range(int(iterations)):
        faces = _split_long_edges(verts, faces, split_thresh)
        faces = _collapse_short_edges(verts, faces, collapse_thresh)
        faces = _flip_edges(verts, faces)
        _smooth_vertices(verts, faces)

    # Clean up any degenerate triangles introduced.
    faces = [f for f in faces if len(set(f)) == 3]

    return {
        "vertices": [(v[0], v[1], v[2]) for v in verts],
        "faces": [(f[0], f[1], f[2]) for f in faces],
    }


# ---------------------------------------------------------------------------
# triangulation (handle quads and higher n-gons by fanning, as kerf does)
# ---------------------------------------------------------------------------


def _triangulate(faces: Sequence[Sequence[int]]) -> List[List[int]]:
    tris: List[List[int]] = []
    for face in faces:
        indices = [int(v) for v in face]
        n = len(indices)
        if n < 3:
            continue
        if n == 3:
            tris.append(indices)
        else:
            for i in range(1, n - 1):
                tris.append([indices[0], indices[i], indices[i + 1]])
    return tris


# ---------------------------------------------------------------------------
# helper geometry / topology
# ---------------------------------------------------------------------------


def _edge_length(verts: List[List[float]], a: int, b: int) -> float:
    va, vb = verts[a], verts[b]
    return math.sqrt(
        (va[0] - vb[0]) ** 2 + (va[1] - vb[1]) ** 2 + (va[2] - vb[2]) ** 2
    )


def _midpoint(verts: List[List[float]], a: int, b: int) -> List[float]:
    va, vb = verts[a], verts[b]
    return [0.5 * (va[i] + vb[i]) for i in range(3)]


def _ekey(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def _build_edge_map(faces: List[List[int]]) -> Dict[Edge, List[int]]:
    """Map undirected edge (min, max) -> list of incident face indices."""
    edge_map: Dict[Edge, List[int]] = {}
    for fi, f in enumerate(faces):
        n = len(f)
        for k in range(n):
            edge_map.setdefault(_ekey(f[k], f[(k + 1) % n]), []).append(fi)
    return edge_map


def _boundary_edges(edge_map: Dict[Edge, List[int]]) -> Set[Edge]:
    return {e for e, fs in edge_map.items() if len(fs) == 1}


def _boundary_vertices(edge_map: Dict[Edge, List[int]]) -> Set[int]:
    verts: Set[int] = set()
    for a, b in _boundary_edges(edge_map):
        verts.add(a)
        verts.add(b)
    return verts


def _vertex_adjacency(faces: List[List[int]]) -> Dict[int, Set[int]]:
    adj: Dict[int, Set[int]] = {}
    for f in faces:
        n = len(f)
        for k in range(n):
            a, b = f[k], f[(k + 1) % n]
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    return adj


# ---------------------------------------------------------------------------
# (1) split long edges (longest-first, boundary edges never split)
# ---------------------------------------------------------------------------


def _split_long_edges(
    verts: List[List[float]],
    faces: List[List[int]],
    threshold: float,
) -> List[List[int]]:
    """Split interior edges longer than *threshold* at their midpoints.

    Kerf structure: one split per pass, edge map rebuilt each pass, bounded
    at ``_MAX_SPLIT_PASSES`` passes.  Boundary edges are never split (kerf's
    documented contract; see module docstring fidelity notes).
    """
    changed = True
    passes_left = _MAX_SPLIT_PASSES
    while changed and passes_left > 0:
        passes_left -= 1
        changed = False
        edge_map = _build_edge_map(faces)
        boundary = _boundary_edges(edge_map)
        long_edges = [
            (e, _edge_length(verts, e[0], e[1]))
            for e in edge_map
            if e not in boundary and _edge_length(verts, e[0], e[1]) > threshold
        ]
        if not long_edges:
            break
        # Longest first; vertex indices break ties deterministically.
        long_edges.sort(key=lambda item: (-item[1], item[0][0], item[0][1]))
        for (a, b), _len in long_edges:
            if a >= len(verts) or b >= len(verts):
                continue
            if _edge_length(verts, a, b) <= threshold:
                continue
            mid_vi = len(verts)
            verts.append(_midpoint(verts, a, b))
            key = _ekey(a, b)
            incident = set(_build_edge_map(faces).get(key, []))
            new_faces: List[List[int]] = []
            for fi, f in enumerate(faces):
                if fi not in incident:
                    new_faces.append(f)
                    continue
                n = len(f)
                inserted = False
                for k in range(n):
                    if _ekey(f[k], f[(k + 1) % n]) == key:
                        p0, p1 = f[k], f[(k + 1) % n]
                        opp = f[(k + 2) % n]  # valid only for a triangle
                        new_faces.append([p0, mid_vi, opp])
                        new_faces.append([mid_vi, p1, opp])
                        inserted = True
                        break
                if not inserted:
                    new_faces.append(f)
            faces = new_faces
            changed = True
            break  # restart the outer while after one split (kerf structure)
    return faces


# ---------------------------------------------------------------------------
# (2) collapse short edges (shortest-first, boundary preserved)
# ---------------------------------------------------------------------------


def _collapse_legal(
    faces: List[List[int]],
    edge_map: Dict[Edge, List[int]],
    a: int,
    b: int,
) -> bool:
    """Link condition: the shared one-ring of *a* and *b* must be exactly the
    opposite vertices of the faces incident on edge (a, b).  Otherwise the
    collapse would fuse two triangles onto one edge (non-manifold)."""
    adj = _vertex_adjacency(faces)
    opposite: Set[int] = set()
    for fi in edge_map.get(_ekey(a, b), []):
        for v in faces[fi]:
            if v != a and v != b:
                opposite.add(v)
    shared = adj.get(a, set()) & adj.get(b, set())
    return shared == opposite


def _collapse_short_edges(
    verts: List[List[float]],
    faces: List[List[int]],
    threshold: float,
) -> List[List[int]]:
    """Collapse interior edges shorter than *threshold*.

    Kerf structure: one collapse per pass (shortest first), bounded at
    ``_MAX_COLLAPSE_PASSES`` passes.  Boundary edges are never collapsed.
    An edge with exactly one boundary endpoint collapses into the boundary
    vertex without moving it; an interior edge between two boundary vertices
    is skipped (it would pinch the boundary).
    """
    for _pass in range(_MAX_COLLAPSE_PASSES):
        edge_map = _build_edge_map(faces)
        boundary = _boundary_edges(edge_map)
        boundary_verts = _boundary_vertices(edge_map)
        short_edges = [
            (e, _edge_length(verts, e[0], e[1]))
            for e in edge_map
            if e not in boundary and _edge_length(verts, e[0], e[1]) < threshold
        ]
        if not short_edges:
            break
        # Shortest first; vertex indices break ties deterministically.
        short_edges.sort(key=lambda item: (item[1], item[0][0], item[0][1]))

        collapsed = False
        for (a, b), _len in short_edges:
            a_bnd = a in boundary_verts
            b_bnd = b in boundary_verts
            if a_bnd and b_bnd:
                continue  # would move or pinch the boundary
            if not _collapse_legal(faces, edge_map, a, b):
                continue
            if a_bnd:
                keep, gone = a, b  # keep the boundary vertex where it is
            elif b_bnd:
                keep, gone = b, a
            else:
                keep, gone = a, b
                verts[keep] = _midpoint(verts, a, b)
            new_faces: List[List[int]] = []
            for f in faces:
                new_f = [keep if vi == gone else vi for vi in f]
                if len(set(new_f)) == 3:  # drop degenerates
                    new_faces.append(new_f)
            faces = new_faces
            collapsed = True
            break  # rebuild the edge map after one collapse (kerf structure)
        if not collapsed:
            break
    return faces


# ---------------------------------------------------------------------------
# (3) valence-optimising edge flips
# ---------------------------------------------------------------------------


def _flip_edges(
    verts: List[List[float]],
    faces: List[List[int]],
) -> List[List[int]]:
    """Flip interior edges to reduce deviation from the target valence.

    Target valence: 6 for interior vertices, 4 for boundary vertices
    (Botsch-Kobbelt).  For an interior edge (a, b) shared by triangles with
    opposite vertices c and d, a flip replaces the edge (a, b) with (c, d)
    when the summed valence deviation of a, b, c, d decreases.  The flip is
    orientation-aware so the winding stays consistent, and is refused when
    the edge (c, d) already exists.
    """
    edge_map = _build_edge_map(faces)
    boundary_verts = _boundary_vertices(edge_map)

    # Valence = number of distinct one-ring neighbours.
    valence: Dict[int, int] = {
        v: len(nbrs) for v, nbrs in _vertex_adjacency(faces).items()
    }

    def _target(v: int) -> int:
        if v in boundary_verts:
            return _BOUNDARY_TARGET_VALENCE
        return _INTERIOR_TARGET_VALENCE

    changed = True
    sweeps_left = _MAX_FLIP_SWEEPS
    while changed and sweeps_left > 0:
        sweeps_left -= 1
        changed = False
        edge_map = _build_edge_map(faces)
        boundary = _boundary_edges(edge_map)
        existing_edges = set(edge_map)
        for e in sorted(edge_map):
            if e in boundary:
                continue
            fi_list = edge_map[e]
            if len(fi_list) != 2:
                continue
            fi0, fi1 = fi_list
            f0, f1 = faces[fi0], faces[fi1]
            a, b = e
            c_list = [v for v in f0 if v != a and v != b]
            d_list = [v for v in f1 if v != a and v != b]
            if len(c_list) != 1 or len(d_list) != 1:
                continue
            c, d = c_list[0], d_list[0]
            if c == d or _ekey(c, d) in existing_edges:
                continue

            def _dev(v: int, delta: int = 0) -> int:
                return abs(valence.get(v, 0) + delta - _target(v))

            before = _dev(a) + _dev(b) + _dev(c) + _dev(d)
            after = _dev(a, -1) + _dev(b, -1) + _dev(c, 1) + _dev(d, 1)
            if after >= before:
                continue

            # Orientation-aware rewrite: let f_ab be the face holding the
            # directed edge a->b (opposite vertex c'), f_ba the other.
            # (a, b, c') + (b, a, d') -> (a, d', c') + (d', b, c').
            if _has_directed_edge(f0, a, b):
                f_ab, f_ba, ia, ib = fi0, fi1, c, d
            elif _has_directed_edge(f1, a, b):
                f_ab, f_ba, ia, ib = fi1, fi0, d, c
            else:
                continue  # inconsistent winding; leave the edge alone
            faces[f_ab] = [a, ib, ia]
            faces[f_ba] = [ib, b, ia]
            valence[a] = valence.get(a, 0) - 1
            valence[b] = valence.get(b, 0) - 1
            valence[c] = valence.get(c, 0) + 1
            valence[d] = valence.get(d, 0) + 1
            existing_edges.discard(_ekey(a, b))
            existing_edges.add(_ekey(c, d))
            changed = True
            # Face lists changed under edge_map; rebuild before continuing.
            break
    return faces


def _has_directed_edge(face: List[int], a: int, b: int) -> bool:
    n = len(face)
    for k in range(n):
        if face[k] == a and face[(k + 1) % n] == b:
            return True
    return False


# ---------------------------------------------------------------------------
# (4) tangential Laplacian smoothing
# ---------------------------------------------------------------------------


def _smooth_vertices(
    verts: List[List[float]],
    faces: List[List[int]],
    strength: float = _SMOOTH_STRENGTH,
) -> None:
    """Move each interior vertex toward the centroid of its one-ring,
    projected onto the tangent plane of the area-weighted vertex normal.
    Boundary vertices are not moved (kerf skips them entirely)."""
    edge_map = _build_edge_map(faces)
    boundary_verts = _boundary_vertices(edge_map)
    adj = _vertex_adjacency(faces)

    # Per-vertex normal: sum of (area-weighted, i.e. unnormalised cross
    # product) incident face normals.
    normals: List[List[float]] = [[0.0, 0.0, 0.0] for _ in verts]
    for f in faces:
        pa, pb, pc = verts[f[0]], verts[f[1]], verts[f[2]]
        ux, uy, uz = pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]
        vx, vy, vz = pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        for vi in f:
            normals[vi][0] += nx
            normals[vi][1] += ny
            normals[vi][2] += nz
    for n in normals:
        mag = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
        if mag > 1e-12:
            n[0] /= mag
            n[1] /= mag
            n[2] /= mag

    new_positions: Dict[int, List[float]] = {}
    for vi in range(len(verts)):
        if vi in boundary_verts:
            continue
        neighbours = adj.get(vi)
        if not neighbours:
            continue
        cx = cy = cz = 0.0
        for nb in neighbours:
            cx += verts[nb][0]
            cy += verts[nb][1]
            cz += verts[nb][2]
        inv = 1.0 / len(neighbours)
        dx = cx * inv - verts[vi][0]
        dy = cy * inv - verts[vi][1]
        dz = cz * inv - verts[vi][2]
        n = normals[vi]
        mag = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
        if mag > 1e-12:
            dot = dx * n[0] + dy * n[1] + dz * n[2]
            dx -= dot * n[0]
            dy -= dot * n[1]
            dz -= dot * n[2]
        new_positions[vi] = [
            verts[vi][0] + strength * dx,
            verts[vi][1] + strength * dy,
            verts[vi][2] + strength * dz,
        ]

    for vi, pos in new_positions.items():
        verts[vi] = pos


# ---------------------------------------------------------------------------
# selfcheck fixtures and validators
# ---------------------------------------------------------------------------


def _unit_cube() -> Tuple[List[Point], List[Tri]]:
    v: List[Point] = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]
    quads = [
        (0, 3, 2, 1),  # bottom -z
        (4, 5, 6, 7),  # top +z
        (0, 1, 5, 4),  # -y
        (1, 2, 6, 5),  # +x
        (2, 3, 7, 6),  # +y
        (3, 0, 4, 7),  # -x
    ]
    tris: List[Tri] = []
    for a, b, c, d in quads:
        tris.append((a, b, c))
        tris.append((a, c, d))
    return v, tris


def _grid_patch(n: int, spacing: float) -> Tuple[List[Point], List[Tri]]:
    """Open flat (n+1) x (n+1) grid patch in the z = 0 plane."""
    verts: List[Point] = []
    for j in range(n + 1):
        for i in range(n + 1):
            verts.append((i * spacing, j * spacing, 0.0))

    def vid(i: int, j: int) -> int:
        return j * (n + 1) + i

    tris: List[Tri] = []
    for j in range(n):
        for i in range(n):
            v00, v10 = vid(i, j), vid(i + 1, j)
            v01, v11 = vid(i, j + 1), vid(i + 1, j + 1)
            tris.append((v00, v10, v11))
            tris.append((v00, v11, v01))
    return verts, tris


def _directed_edge_check(faces: Sequence[Tri]) -> Tuple[bool, bool]:
    """Return (unique, closed): every directed edge appears at most once, and
    every directed edge has its reverse present (closed oriented manifold)."""
    counts: Dict[Edge, int] = {}
    for f in faces:
        for k in range(3):
            de = (f[k], f[(k + 1) % 3])
            counts[de] = counts.get(de, 0) + 1
    unique = all(c == 1 for c in counts.values())
    closed = all((b, a) in counts for (a, b) in counts)
    return unique, closed


def _all_edge_lengths(verts: Sequence[Point], faces: Sequence[Tri]) -> List[float]:
    seen: Set[Edge] = set()
    lengths: List[float] = []
    for f in faces:
        for k in range(3):
            e = _ekey(f[k], f[(k + 1) % 3])
            if e in seen:
                continue
            seen.add(e)
            pa, pb = verts[e[0]], verts[e[1]]
            lengths.append(math.sqrt(
                (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2 + (pa[2] - pb[2]) ** 2
            ))
    return lengths


def _boundary_polygon(
    verts: Sequence[Point], faces: Sequence[Tri]
) -> Tuple[Set[Point], float]:
    """Boundary vertex positions and total boundary length of an open mesh."""
    edge_map = _build_edge_map([list(f) for f in faces])
    positions: Set[Point] = set()
    total = 0.0
    for a, b in _boundary_edges(edge_map):
        positions.add(verts[a])
        positions.add(verts[b])
        pa, pb = verts[a], verts[b]
        total += math.sqrt(
            (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2 + (pa[2] - pb[2]) ** 2
        )
    return positions, total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.geometry.mesh.isotropic_remesh",
        description="Isotropic remeshing "
                    "(split / collapse / flip / tangential smooth), "
                    "with deterministic passes.",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="remesh a coarse unit cube and an open grid "
                             "patch and verify edge lengths, manifoldness "
                             "and boundary preservation.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    # (a) closed coarse unit cube at target edge length 0.25.
    target = 0.25
    cube_verts, cube_tris = _unit_cube()
    result = isotropic_remesh(cube_verts, cube_tris, target, iterations=12)
    out_verts, out_faces = result["vertices"], result["faces"]
    assert len(out_faces) > len(cube_tris), "triangle count must increase"
    unique, closed = _directed_edge_check(out_faces)
    assert unique, "cube result has a duplicated directed edge (non-manifold)"
    assert closed, "cube result has an unpaired directed edge (not closed)"
    lengths = _all_edge_lengths(out_verts, out_faces)
    lo, hi = 0.5 * target, 2.0 * target
    within = sum(1 for l in lengths if lo <= l <= hi)
    fraction = within / float(len(lengths))
    print("[selfcheck] cube: %d tris (from %d), %d edges, %.1f%% of edge "
          "lengths in [%.3f, %.3f]" % (len(out_faces), len(cube_tris),
                                       len(lengths), 100.0 * fraction, lo, hi))
    assert fraction >= 0.9, (
        "expected >= 90%% of edges within [0.5L, 2L], got %.1f%%"
        % (100.0 * fraction))

    # Determinism: an identical run must produce an identical mesh.
    repeat = isotropic_remesh(cube_verts, cube_tris, target, iterations=12)
    assert repeat == result, "remesh must be deterministic"
    print("[selfcheck] cube: repeated run is bit-identical")

    # (b) open flat grid patch: boundary polygon must be preserved exactly
    # (boundary edges never split/collapsed, boundary vertices never moved).
    spacing = 0.25
    grid_verts, grid_tris = _grid_patch(4, spacing)
    before_positions, before_length = _boundary_polygon(grid_verts, grid_tris)
    patch = isotropic_remesh(grid_verts, grid_tris, target, iterations=5)
    p_verts, p_faces = patch["vertices"], patch["faces"]
    unique, _ = _directed_edge_check(p_faces)
    assert unique, "patch result has a duplicated directed edge"
    after_positions, after_length = _boundary_polygon(p_verts, p_faces)
    corners = {(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)}
    tol = 1e-9

    def _present(target_pt: Point, pool: Set[Point]) -> bool:
        return any(
            abs(p[0] - target_pt[0]) <= tol
            and abs(p[1] - target_pt[1]) <= tol
            and abs(p[2] - target_pt[2]) <= tol
            for p in pool
        )

    for corner in sorted(corners):
        assert _present(corner, after_positions), (
            "boundary corner %r lost" % (corner,))
    assert abs(after_length - before_length) <= tol, (
        "boundary length changed: %.12f -> %.12f"
        % (before_length, after_length))
    for pos in after_positions:
        assert _present(pos, before_positions), (
            "boundary vertex %r is not an original boundary vertex" % (pos,))
    print("[selfcheck] patch: boundary length %.6f preserved, "
          "4 corners intact, %d boundary vertices all original"
          % (after_length, len(after_positions)))
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
