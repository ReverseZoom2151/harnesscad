"""Isotropic remeshing: split / collapse / flip / smooth.

The goal is a triangulation whose edges all sit near a requested length *L*
and whose vertices have near-regular valence.  Each iteration applies four
passes in order:

1. **Refine.**  Every interior edge longer than ``4/3 * L`` is marked, a
   midpoint is inserted on each marked edge, and every face is re-triangulated
   in one batch according to how many of its three edges were marked (one,
   two or three).  Marking the whole set before rebuilding keeps the pass
   order-independent and lets a face split symmetrically instead of one edge
   at a time.
2. **Coarsen.**  Interior edges shorter than ``4/5 * L`` are collapsed,
   shortest first.  Within a round the one-rings of an accepted collapse are
   frozen, so several disjoint collapses commit together.
3. **Equalise.**  Interior edges are flipped when the flip lowers the total
   deviation of the four incident vertices from their target valence (6 in
   the interior, 4 on the boundary).  Flips are likewise batched over
   disjoint vertex neighbourhoods.
4. **Relax.**  Interior vertices step toward the centroid of their one-ring
   (uniform Laplacian, strength 0.5) with the step projected out of the
   area-weighted vertex normal, so the surface is retiled without being
   inflated or shrunk.

Boundary contract
-----------------
A boundary edge is one carried by a single face.  Boundary edges are never
split and never collapsed, boundary vertices are never repositioned, and an
interior edge joining two boundary vertices is left alone because collapsing
it would pinch the rim.  An open patch therefore keeps its boundary polygon
vertex-for-vertex and length-for-length.

Manifold contract
-----------------
A collapse is admitted only when the endpoints' one-rings intersect in
exactly the apexes of the two faces on the edge (the link condition), and a
flip is refused when the new diagonal already exists.  Both guards keep a
closed mesh closed and prevent duplicated directed edges.

Determinism
-----------
Candidate lists are sorted on (metric, lower index, upper index) and every
traversal of a set is sorted before use -- including the neighbour sums in
the relaxation pass, since floating-point addition is not associative.  Two
runs on the same input produce bit-identical output.

Pure stdlib.
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

__all__ = ["isotropic_remesh"]

Point = Tuple[float, float, float]
Tri = Tuple[int, int, int]
Edge = Tuple[int, int]

_MAX_REFINE_ROUNDS = 6
_MAX_COARSEN_ROUNDS = 8
_MAX_EQUALISE_SWEEPS = 8
_RELAX_STRENGTH = 0.5
_INTERIOR_TARGET_VALENCE = 6
_BOUNDARY_TARGET_VALENCE = 4
_TINY = 1e-12


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
        (quads are accepted the same way).
    target_edge_length : float
        Desired average edge length after remeshing.
    iterations : int
        Number of refine -> coarsen -> equalise -> relax cycles (default 5).

    Returns
    -------
    dict
        ``{"vertices": List[Point], "faces": List[Tri]}`` -- all faces are
        triangles.  Vertices left unreferenced by a collapse are kept;
        consumers that need compaction must reindex.
    """
    if target_edge_length <= 0:
        raise ValueError("target_edge_length must be positive")

    points: List[List[float]] = [
        [float(p[0]), float(p[1]), float(p[2])] for p in vertices
    ]
    faces: List[Tri] = _as_triangles(triangles)

    if not points or not faces:
        return {"vertices": [(p[0], p[1], p[2]) for p in points], "faces": []}

    length = float(target_edge_length)
    upper = (4.0 / 3.0) * length
    lower = (4.0 / 5.0) * length

    for _cycle in range(int(iterations)):
        faces = _refine(points, faces, upper)
        faces = _coarsen(points, faces, lower)
        faces = _equalise(faces)
        _relax(points, faces)

    faces = [f for f in faces if len(set(f)) == 3]

    return {
        "vertices": [(p[0], p[1], p[2]) for p in points],
        "faces": [(f[0], f[1], f[2]) for f in faces],
    }


# ---------------------------------------------------------------------------
# input conditioning
# ---------------------------------------------------------------------------


def _as_triangles(faces: Sequence[Sequence[int]]) -> List[Tri]:
    """Pass triangles through; fan any larger polygon from its first corner."""
    out: List[Tri] = []
    for face in faces:
        ring = [int(v) for v in face]
        for k in range(1, len(ring) - 1):
            out.append((ring[0], ring[k], ring[k + 1]))
    return out


# ---------------------------------------------------------------------------
# geometry and topology helpers
# ---------------------------------------------------------------------------


def _key(i: int, j: int) -> Edge:
    return (i, j) if i < j else (j, i)


def _span(points: Sequence[Sequence[float]], i: int, j: int) -> float:
    a, b = points[i], points[j]
    dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _halfway(points: Sequence[Sequence[float]], i: int, j: int) -> List[float]:
    a, b = points[i], points[j]
    return [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])]


def _incidence(faces: Sequence[Tri]) -> Dict[Edge, List[int]]:
    """Undirected edge -> indices of the faces carrying it."""
    table: Dict[Edge, List[int]] = {}
    for index, face in enumerate(faces):
        table.setdefault(_key(face[0], face[1]), []).append(index)
        table.setdefault(_key(face[1], face[2]), []).append(index)
        table.setdefault(_key(face[2], face[0]), []).append(index)
    return table


def _rim_vertices(table: Dict[Edge, List[int]]) -> Set[int]:
    """Vertices touched by an edge that carries only one face."""
    rim: Set[int] = set()
    for (i, j), carriers in table.items():
        if len(carriers) == 1:
            rim.add(i)
            rim.add(j)
    return rim


def _one_rings(faces: Sequence[Tri]) -> Dict[int, Set[int]]:
    rings: Dict[int, Set[int]] = {}
    for face in faces:
        for k in range(3):
            a, b = face[k], face[(k + 1) % 3]
            rings.setdefault(a, set()).add(b)
            rings.setdefault(b, set()).add(a)
    return rings


def _apex(face: Tri, i: int, j: int) -> Optional[int]:
    """The corner of *face* that is neither *i* nor *j*."""
    found = [v for v in face if v != i and v != j]
    return found[0] if len(found) == 1 else None


# ---------------------------------------------------------------------------
# pass 1 -- refine: batched midpoint insertion on over-long interior edges
# ---------------------------------------------------------------------------


def _refine(points: List[List[float]],
            faces: List[Tri],
            upper: float) -> List[Tri]:
    """Insert midpoints on interior edges longer than *upper*.

    Every over-long edge in the current mesh is marked first, then the face
    list is rebuilt in a single sweep.  A face is re-triangulated according
    to which of its three edges carry a midpoint; when two do, the leftover
    quadrilateral is split along its shorter diagonal so the pass does not
    manufacture slivers.
    """
    for _round in range(_MAX_REFINE_ROUNDS):
        table = _incidence(faces)
        marked = sorted(
            edge for edge, carriers in table.items()
            if len(carriers) == 2 and _span(points, edge[0], edge[1]) > upper
        )
        if not marked:
            break

        midpoint: Dict[Edge, int] = {}
        for edge in marked:
            midpoint[edge] = len(points)
            points.append(_halfway(points, edge[0], edge[1]))

        rebuilt: List[Tri] = []
        for a, b, c in faces:
            mab = midpoint.get(_key(a, b))
            mbc = midpoint.get(_key(b, c))
            mca = midpoint.get(_key(c, a))
            rebuilt.extend(_split_face(points, (a, b, c), mab, mbc, mca))
        faces = rebuilt
    return faces


def _quad(points: Sequence[Sequence[float]],
          p: int, q: int, r: int, s: int) -> List[Tri]:
    """Triangulate the CCW quadrilateral p-q-r-s along its shorter diagonal."""
    if _span(points, p, r) <= _span(points, q, s):
        return [(p, q, r), (p, r, s)]
    return [(p, q, s), (q, r, s)]


def _split_face(points: Sequence[Sequence[float]],
                face: Tri,
                mab: Optional[int],
                mbc: Optional[int],
                mca: Optional[int]) -> List[Tri]:
    """Re-triangulate one face given the midpoints marked on its edges."""
    a, b, c = face
    present = (mab is not None, mbc is not None, mca is not None)

    if present == (False, False, False):
        return [face]
    if present == (True, True, True):
        return [(a, mab, mca), (mab, b, mbc), (mca, mbc, c),
                (mab, mbc, mca)]

    # Exactly one midpoint: bisect the face from the opposite corner.
    if present == (True, False, False):
        return [(a, mab, c), (mab, b, c)]
    if present == (False, True, False):
        return [(a, b, mbc), (a, mbc, c)]
    if present == (False, False, True):
        return [(b, c, mca), (b, mca, a)]

    # Exactly two midpoints: one corner triangle plus a quadrilateral.
    if present == (True, True, False):
        return [(mab, b, mbc)] + _quad(points, a, mab, mbc, c)
    if present == (False, True, True):
        return [(mbc, c, mca)] + _quad(points, a, b, mbc, mca)
    return [(mca, a, mab)] + _quad(points, mab, b, c, mca)


# ---------------------------------------------------------------------------
# pass 2 -- coarsen: collapse under-length interior edges
# ---------------------------------------------------------------------------


def _link_condition(edge: Edge,
                    faces: Sequence[Tri],
                    table: Dict[Edge, List[int]],
                    rings: Dict[int, Set[int]]) -> bool:
    """True when merging the endpoints cannot create a duplicated edge.

    The endpoints may share only the apexes of the two faces on the edge; any
    further common neighbour would end up joined to the survivor twice, which
    is the classic non-manifold outcome of a careless collapse.
    """
    carriers = table.get(edge, [])
    if len(carriers) != 2:
        return False
    apexes: Set[int] = set()
    for index in carriers:
        corner = _apex(faces[index], edge[0], edge[1])
        if corner is None:
            return False
        apexes.add(corner)
    shared = rings.get(edge[0], set()) & rings.get(edge[1], set())
    return shared == apexes


def _coarsen(points: List[List[float]],
             faces: List[Tri],
             lower: float) -> List[Tri]:
    """Collapse interior edges shorter than *lower*, shortest first.

    A round gathers every candidate, then commits as many as have disjoint
    one-rings; the mesh is rebuilt once and the next round re-measures.  An
    edge with one boundary endpoint collapses onto that endpoint so the rim
    does not move, and an edge with two boundary endpoints is skipped.
    """
    for _round in range(_MAX_COARSEN_ROUNDS):
        table = _incidence(faces)
        rim = _rim_vertices(table)
        rings = _one_rings(faces)

        candidates = sorted(
            (_span(points, edge[0], edge[1]), edge[0], edge[1])
            for edge, carriers in table.items()
            if len(carriers) == 2
            and _span(points, edge[0], edge[1]) < lower
        )
        if not candidates:
            break

        frozen: Set[int] = set()
        substitute: Dict[int, int] = {}
        moves: List[Tuple[int, List[float]]] = []

        for _distance, first, second in candidates:
            if first in frozen or second in frozen:
                continue
            first_on_rim, second_on_rim = first in rim, second in rim
            if first_on_rim and second_on_rim:
                continue
            if not _link_condition((first, second), faces, table, rings):
                continue

            if first_on_rim:
                survivor, absorbed = first, second
            elif second_on_rim:
                survivor, absorbed = second, first
            else:
                survivor, absorbed = first, second
                moves.append((survivor, _halfway(points, first, second)))

            substitute[absorbed] = survivor
            frozen.add(first)
            frozen.add(second)
            frozen |= rings.get(first, set())
            frozen |= rings.get(second, set())

        if not substitute:
            break

        for vertex, position in moves:
            points[vertex] = position

        rebuilt: List[Tri] = []
        for face in faces:
            a = substitute.get(face[0], face[0])
            b = substitute.get(face[1], face[1])
            c = substitute.get(face[2], face[2])
            if a != b and b != c and a != c:
                rebuilt.append((a, b, c))
        faces = rebuilt
    return faces


# ---------------------------------------------------------------------------
# pass 3 -- equalise: valence-improving edge flips
# ---------------------------------------------------------------------------


def _equalise(faces: List[Tri]) -> List[Tri]:
    """Flip interior edges that bring valences closer to their targets.

    Flipping the edge (a, b) of the two faces with apexes c and d replaces it
    with (c, d): a and b each lose a neighbour while c and d each gain one.
    The flip is taken when the summed absolute deviation from the target
    valence strictly drops.  Rewrites are orientation-aware, so the winding
    survives, and are batched over disjoint vertex neighbourhoods.
    """
    faces = list(faces)

    for _sweep in range(_MAX_EQUALISE_SWEEPS):
        table = _incidence(faces)
        rim = _rim_vertices(table)
        rings = _one_rings(faces)
        valence = {vertex: len(ring) for vertex, ring in rings.items()}
        existing = set(table)

        def target(vertex: int) -> int:
            if vertex in rim:
                return _BOUNDARY_TARGET_VALENCE
            return _INTERIOR_TARGET_VALENCE

        def deviation(vertex: int, shift: int = 0) -> int:
            return abs(valence.get(vertex, 0) + shift - target(vertex))

        frozen: Set[int] = set()
        applied = False

        for edge in sorted(table):
            carriers = table[edge]
            if len(carriers) != 2:
                continue
            a, b = edge
            left, right = carriers
            c = _apex(faces[left], a, b)
            d = _apex(faces[right], a, b)
            if c is None or d is None or c == d:
                continue
            if _key(c, d) in existing:
                continue
            if frozen & {a, b, c, d}:
                continue

            before = deviation(a) + deviation(b) + deviation(c) + deviation(d)
            after = (deviation(a, -1) + deviation(b, -1)
                     + deviation(c, 1) + deviation(d, 1))
            if after >= before:
                continue

            # Identify which face runs a->b so the two replacements keep the
            # original winding: (a, b, x) + (b, a, y) -> (a, y, x) + (y, b, x).
            if _runs(faces[left], a, b):
                ccw_face, cw_face, near, far = left, right, c, d
            elif _runs(faces[right], a, b):
                ccw_face, cw_face, near, far = right, left, d, c
            else:
                continue  # inconsistent winding: leave this edge alone

            faces[ccw_face] = (a, far, near)
            faces[cw_face] = (far, b, near)

            existing.discard(edge)
            existing.add(_key(c, d))
            frozen |= {a, b, c, d}
            frozen |= rings.get(a, set())
            frozen |= rings.get(b, set())
            applied = True

        if not applied:
            break
    return faces


def _runs(face: Tri, a: int, b: int) -> bool:
    """True when *face* traverses the directed edge a -> b."""
    return ((face[0] == a and face[1] == b)
            or (face[1] == a and face[2] == b)
            or (face[2] == a and face[0] == b))


# ---------------------------------------------------------------------------
# pass 4 -- relax: tangential Laplacian smoothing
# ---------------------------------------------------------------------------


def _relax(points: List[List[float]],
           faces: Sequence[Tri],
           strength: float = _RELAX_STRENGTH) -> None:
    """Slide interior vertices toward their one-ring centroid, tangentially.

    The Laplacian offset is projected out of the vertex normal so the pass
    redistributes vertices across the surface without displacing it.  The
    normal is the normalised sum of the incident faces' cross products, which
    weights each face by twice its area.  Boundary vertices stay put.
    """
    table = _incidence(faces)
    rim = _rim_vertices(table)
    rings = _one_rings(faces)

    normals: List[List[float]] = [[0.0, 0.0, 0.0] for _ in points]
    for a, b, c in faces:
        pa, pb, pc = points[a], points[b], points[c]
        ux, uy, uz = pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]
        vx, vy, vz = pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]
        nx, ny, nz = (uy * vz - uz * vy,
                      uz * vx - ux * vz,
                      ux * vy - uy * vx)
        for corner in (a, b, c):
            slot = normals[corner]
            slot[0] += nx
            slot[1] += ny
            slot[2] += nz

    updated: List[Tuple[int, List[float]]] = []
    for vertex in range(len(points)):
        if vertex in rim:
            continue
        neighbours = rings.get(vertex)
        if not neighbours:
            continue

        # Sorted so the accumulation order -- and hence the rounding -- is
        # fixed regardless of set iteration order.
        sx = sy = sz = 0.0
        ordered = sorted(neighbours)
        for other in ordered:
            sx += points[other][0]
            sy += points[other][1]
            sz += points[other][2]
        scale = 1.0 / len(ordered)
        here = points[vertex]
        dx = sx * scale - here[0]
        dy = sy * scale - here[1]
        dz = sz * scale - here[2]

        nx, ny, nz = normals[vertex]
        magnitude = math.sqrt(nx * nx + ny * ny + nz * nz)
        if magnitude > _TINY:
            nx, ny, nz = nx / magnitude, ny / magnitude, nz / magnitude
            along = dx * nx + dy * ny + dz * nz
            dx -= along * nx
            dy -= along * ny
            dz -= along * nz

        updated.append((vertex, [here[0] + strength * dx,
                                 here[1] + strength * dy,
                                 here[2] + strength * dz]))

    for vertex, position in updated:
        points[vertex] = position


# ---------------------------------------------------------------------------
# selfcheck fixtures and validators
# ---------------------------------------------------------------------------


def _build_edge_map(faces: Sequence[Sequence[int]]) -> Dict[Edge, List[int]]:
    """Undirected edge -> carrying face indices (accepts list-form faces)."""
    table: Dict[Edge, List[int]] = {}
    for index, face in enumerate(faces):
        n = len(face)
        for k in range(n):
            table.setdefault(_key(face[k], face[(k + 1) % n]), []).append(index)
    return table


def _boundary_edges(table: Dict[Edge, List[int]]) -> Set[Edge]:
    return {edge for edge, carriers in table.items() if len(carriers) == 1}


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


def _all_edge_lengths(verts: Sequence[Point],
                      faces: Sequence[Tri]) -> List[float]:
    seen: Set[Edge] = set()
    lengths: List[float] = []
    for f in faces:
        for k in range(3):
            e = _key(f[k], f[(k + 1) % 3])
            if e in seen:
                continue
            seen.add(e)
            lengths.append(_span(verts, e[0], e[1]))
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
        total += _span(verts, a, b)
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
    corners = {(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
               (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)}
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
