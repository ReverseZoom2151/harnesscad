"""Triangle-mesh repair toolkit: weld, unify normals, fill holes, decimate.

The toolkit operates on indexed triangle meshes with a never-raise contract
and the harness mesh convention used by
:mod:`harnesscad.domain.geometry.mesh.polyhedron` and
:mod:`harnesscad.domain.geometry.mesh.halfedge`: vertices are a
``List[Tuple[float, float, float]]`` and triangles are a
``List[Tuple[int, int, int]]`` (0-based indices, CCW winding = outward
normal).  All operations are plain functions on ``(vertices, triangles)``
pairs -- no mesh class is introduced.

Operations
----------
* :func:`weld_vertices` -- union-find clustering of coincident vertices,
  driven by a uniform spatial grid at the merge tolerance; triangles whose
  corners land in a single cluster are dropped.
* :func:`unify_normals` -- orientation propagation across the face-adjacency
  dual graph; two faces sharing an edge agree when they do *not* share a
  directed half-edge.
* :func:`fill_holes` -- unpaired directed half-edges are chained into
  boundary cycles, each of which is capped by a triangle fan.
* :func:`remove_degenerate` -- drops repeated-index, zero-area and duplicate
  triangles; reports edges left with more than two uses.
* :func:`decimate` -- quadric-error-metric edge collapse driven by a lazy
  priority queue, with a link-condition guard so a collapse cannot break
  manifoldness.  The collapse target is the minimiser of the summed quadric
  (midpoint when the quadric is singular).
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

Contract: every public entry point never raises.  Success returns
``{"ok": True, "vertices": [...], "triangles": [...], ...stats...}``; failure
returns ``{"ok": False, "reason": "..."}``.  Every traversal over a hash-based
container is sorted, so results are reproducible run to run.

Pure stdlib, deterministic.
"""

from __future__ import annotations

import argparse
import heapq
import math
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

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
Edge = Tuple[int, int]

# Areas below this are treated as exactly zero.
_AREA_EPS = 1e-15
# Determinant magnitude below which the quadric 3x3 block is called singular.
_SINGULAR_EPS = 1e-12


# --------------------------------------------------------------------------
# tiny linear-algebra helpers
# --------------------------------------------------------------------------


def _delta(p: Sequence[float], q: Sequence[float]) -> Point:
    """p - q."""
    return (p[0] - q[0], p[1] - q[1], p[2] - q[2])


def _cross(u: Sequence[float], v: Sequence[float]) -> Point:
    return (u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0])


def _norm(u: Sequence[float]) -> float:
    return math.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])


def _distance(p: Sequence[float], q: Sequence[float]) -> float:
    return _norm(_delta(p, q))


def _face_area(points: Sequence[Point], face: Tri) -> float:
    p, q, r = points[face[0]], points[face[1]], points[face[2]]
    return 0.5 * _norm(_cross(_delta(q, p), _delta(r, p)))


def _undirected(i: int, j: int) -> Edge:
    return (i, j) if i < j else (j, i)


def _corners(face: Sequence[int]) -> Tuple[Edge, Edge, Edge]:
    """The three undirected edges of a triangle, in corner order."""
    return (_undirected(face[0], face[1]),
            _undirected(face[1], face[2]),
            _undirected(face[2], face[0]))


def _directed_edges(face: Sequence[int]) -> Tuple[Edge, Edge, Edge]:
    return ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))


# --------------------------------------------------------------------------
# input checking / normalisation
# --------------------------------------------------------------------------


def _check_inputs(vertices: object, triangles: object) -> Optional[str]:
    """Return a human-readable complaint, or ``None`` when the pair is sane."""
    if not isinstance(vertices, (list, tuple)):
        return "vertices must be a sequence of (x, y, z) points"
    if not isinstance(triangles, (list, tuple)):
        return "triangles must be a sequence of (i, j, k) index triples"

    count = len(vertices)
    for position, point in enumerate(vertices):
        if not isinstance(point, (list, tuple)) or len(point) < 3:
            return "vertex %d is not a 3-component point" % position
        try:
            float(point[0]), float(point[1]), float(point[2])
        except (TypeError, ValueError):
            return "vertex %d has a non-numeric component" % position

    for position, face in enumerate(triangles):
        if not isinstance(face, (list, tuple)) or len(face) < 3:
            return "triangle %d is not a 3-component index triple" % position
        try:
            corners = (int(face[0]), int(face[1]), int(face[2]))
        except (TypeError, ValueError):
            return "triangle %d has a non-integer index" % position
        for index in corners:
            if index < 0 or index >= count:
                return ("triangle %d references vertex %d, outside 0..%d"
                        % (position, index, count - 1))
    return None


def _normalise(
    vertices: Sequence, triangles: Sequence
) -> Tuple[List[Point], List[Tri]]:
    """Copy the mesh into canonical float-tuple / int-tuple form."""
    points = [(float(p[0]), float(p[1]), float(p[2])) for p in vertices]
    faces = [(int(f[0]), int(f[1]), int(f[2])) for f in triangles]
    return points, faces


def _edge_uses(faces: Sequence[Sequence[int]]) -> Dict[Edge, List[int]]:
    """Undirected edge -> face indices touching it, in face order."""
    uses: Dict[Edge, List[int]] = {}
    for index, face in enumerate(faces):
        for edge in _corners(face):
            uses.setdefault(edge, []).append(index)
    return uses


# --------------------------------------------------------------------------
# disjoint-set forest (used by welding and by the vertex-fan check)
# --------------------------------------------------------------------------


class _Forest:
    """Union-find with path halving; unions keep the smaller root."""

    __slots__ = ("parent",)

    def __init__(self, size: int) -> None:
        self.parent: List[int] = list(range(size))

    def root(self, item: int) -> int:
        parent = self.parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def join(self, left: int, right: int) -> bool:
        """Merge two sets; return True when they were distinct."""
        a, b = self.root(left), self.root(right)
        if a == b:
            return False
        if a > b:
            a, b = b, a
        self.parent[b] = a
        return True


# ==========================================================================
# weld_vertices
# ==========================================================================


def _grid_cell(point: Point, size: float) -> Tuple[int, int, int]:
    return (int(math.floor(point[0] / size)),
            int(math.floor(point[1] / size)),
            int(math.floor(point[2] / size)))


def _neighbourhood(cell: Tuple[int, int, int]) -> Iterable[Tuple[int, int, int]]:
    """The 27 cells forming the closed neighbourhood of *cell*."""
    x, y, z = cell
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield (x + dx, y + dy, z + dz)


def weld_vertices(
    vertices: Sequence,
    triangles: Sequence,
    tol: float = 1e-6,
) -> dict:
    """Merge vertices lying within *tol* of one another.

    Coincidence is resolved with a union-find forest fed by a uniform spatial
    grid whose cell size equals the tolerance, so only the 27 cells around a
    point need to be tested.  Each cluster survives at its lowest-index
    member's position; triangles whose three corners do not land in three
    distinct clusters are dropped.

    Returns ``{"ok", "vertices", "triangles", "merged_count"}`` where
    *merged_count* is the number of vertices absorbed into another.
    """
    try:
        complaint = _check_inputs(vertices, triangles)
        if complaint:
            return {"ok": False, "reason": complaint}
        if isinstance(tol, bool) or not isinstance(tol, (int, float)):
            return {"ok": False, "reason": "tol must be a non-negative number"}
        if tol < 0:
            return {"ok": False, "reason": "tol must be a non-negative number"}

        points, faces = _normalise(vertices, triangles)
        if not points:
            return {"ok": True, "vertices": [], "triangles": [],
                    "merged_count": 0}

        size = tol if tol > 1e-9 else 1e-9

        occupants: Dict[Tuple[int, int, int], List[int]] = {}
        for index, point in enumerate(points):
            occupants.setdefault(_grid_cell(point, size), []).append(index)

        forest = _Forest(len(points))
        absorbed = 0
        for index, point in enumerate(points):
            for cell in _neighbourhood(_grid_cell(point, size)):
                for other in occupants.get(cell, ()):
                    if other <= index:
                        continue
                    if _distance(point, points[other]) <= tol:
                        if forest.join(index, other):
                            absorbed += 1

        # Emit one vertex per cluster, ordered by first appearance.
        relabel: Dict[int, int] = {}
        kept_points: List[Point] = []
        for index in range(len(points)):
            leader = forest.root(index)
            if leader not in relabel:
                relabel[leader] = len(kept_points)
                kept_points.append(points[leader])

        kept_faces: List[Tri] = []
        for face in faces:
            a = relabel[forest.root(face[0])]
            b = relabel[forest.root(face[1])]
            c = relabel[forest.root(face[2])]
            if a != b and b != c and a != c:
                kept_faces.append((a, b, c))

        return {
            "ok": True,
            "vertices": kept_points,
            "triangles": kept_faces,
            "merged_count": absorbed,
        }
    except Exception as exc:  # pragma: no cover - never-raise safety net
        return {"ok": False, "reason": "weld_vertices failed: %s" % exc}


# ==========================================================================
# unify_normals
# ==========================================================================


def unify_normals(vertices: Sequence, triangles: Sequence) -> dict:
    """Make the winding of neighbouring triangles agree.

    Two triangles sharing an edge are consistently wound exactly when they
    traverse that edge in opposite directions -- equivalently, when they have
    no directed half-edge in common.  Orientation is propagated by a
    depth-first walk of the dual graph; each connected component is seeded by
    its lowest-index face, whose existing winding is taken as authoritative.

    Returns ``{"ok", "vertices", "triangles", "flipped_count"}``.
    """
    try:
        complaint = _check_inputs(vertices, triangles)
        if complaint:
            return {"ok": False, "reason": complaint}

        points, faces = _normalise(vertices, triangles)
        if not faces:
            return {"ok": True, "vertices": points, "triangles": faces,
                    "flipped_count": 0}

        # Neighbour lists over manifold (exactly two-use) edges only.
        neighbours: List[List[int]] = [[] for _ in faces]
        uses = _edge_uses(faces)
        for edge in sorted(uses):
            touching = uses[edge]
            if len(touching) == 2:
                left, right = touching
                neighbours[left].append(right)
                neighbours[right].append(left)

        settled = [False] * len(faces)
        flipped = 0

        for seed in range(len(faces)):
            if settled[seed]:
                continue
            settled[seed] = True
            stack = [seed]
            while stack:
                current = stack.pop()
                oriented = set(_directed_edges(faces[current]))
                for other in neighbours[current]:
                    if settled[other]:
                        continue
                    settled[other] = True
                    candidate = faces[other]
                    # A shared directed half-edge means the two faces are
                    # wound the same way around the shared edge: reverse one.
                    if oriented & set(_directed_edges(candidate)):
                        faces[other] = (candidate[0], candidate[2],
                                        candidate[1])
                        flipped += 1
                    stack.append(other)

        return {
            "ok": True,
            "vertices": points,
            "triangles": faces,
            "flipped_count": flipped,
        }
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "reason": "unify_normals failed: %s" % exc}


# ==========================================================================
# fill_holes
# ==========================================================================


def _boundary_cycles(faces: Sequence[Tri]) -> List[List[int]]:
    """Chain unpaired directed half-edges into vertex cycles."""
    oriented: Set[Edge] = set()
    for face in faces:
        oriented.update(_directed_edges(face))

    successor: Dict[int, int] = {}
    for tail, head in sorted(oriented):
        if (head, tail) not in oriented:
            successor.setdefault(tail, head)

    cycles: List[List[int]] = []
    consumed: Set[int] = set()
    for origin in sorted(successor):
        if origin in consumed:
            continue
        walk: List[int] = []
        cursor = origin
        while cursor in successor and cursor not in consumed:
            consumed.add(cursor)
            walk.append(cursor)
            cursor = successor[cursor]
        if len(walk) >= 3 and cursor == origin:
            cycles.append(walk)
    return cycles


def fill_holes(vertices: Sequence, triangles: Sequence) -> dict:
    """Cap every boundary cycle with a triangle fan.

    A directed edge a->b carried by exactly one triangle whose reverse b->a
    is carried by none marks the hole rim.  Chaining those half-edges yields
    each hole's perimeter; the cap is fanned from the perimeter's first
    vertex and wound opposite to the rim so it agrees with the surrounding
    surface.

    Returns ``{"ok", "vertices", "triangles", "holes_filled"}``.
    """
    try:
        complaint = _check_inputs(vertices, triangles)
        if complaint:
            return {"ok": False, "reason": complaint}

        points, faces = _normalise(vertices, triangles)
        if not faces:
            return {"ok": True, "vertices": points, "triangles": faces,
                    "holes_filled": 0}

        cycles = _boundary_cycles(faces)
        for rim in cycles:
            hub = rim[0]
            for step in range(1, len(rim) - 1):
                faces.append((hub, rim[step + 1], rim[step]))

        return {
            "ok": True,
            "vertices": points,
            "triangles": faces,
            "holes_filled": len(cycles),
        }
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "reason": "fill_holes failed: %s" % exc}


# ==========================================================================
# remove_degenerate
# ==========================================================================


def remove_degenerate(vertices: Sequence, triangles: Sequence) -> dict:
    """Drop repeated-index, zero-area and duplicated triangles.

    Duplication is judged on the sorted index triple, so a face and its
    mirror image count as the same face and only the first is kept.  Edges
    still carrying more than two faces after the sweep are reported.

    Returns ``{"ok", "vertices", "triangles", "removed_count",
    "non_manifold_edges"}``.
    """
    try:
        complaint = _check_inputs(vertices, triangles)
        if complaint:
            return {"ok": False, "reason": complaint}

        points, faces = _normalise(vertices, triangles)
        if not faces:
            return {"ok": True, "vertices": points, "triangles": [],
                    "removed_count": 0, "non_manifold_edges": []}

        survivors: List[Tri] = []
        already: Set[Tri] = set()
        dropped = 0

        for face in faces:
            a, b, c = face
            repeated = (a == b) or (b == c) or (a == c)
            if repeated or _face_area(points, face) < _AREA_EPS:
                dropped += 1
                continue
            signature = (a, b, c) if a <= b <= c else tuple(sorted(face))
            if signature in already:
                dropped += 1
                continue
            already.add(signature)  # type: ignore[arg-type]
            survivors.append(face)

        overused = sorted(edge for edge, uses in _edge_uses(survivors).items()
                          if len(uses) > 2)

        return {
            "ok": True,
            "vertices": points,
            "triangles": survivors,
            "removed_count": dropped,
            "non_manifold_edges": overused,
        }
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "reason": "remove_degenerate failed: %s" % exc}


# ==========================================================================
# decimate -- quadric error metric edge collapse
# ==========================================================================
#
# A plane (a, b, c, d) with a^2 + b^2 + c^2 = 1 contributes the rank-1
# quadric p p^T; the squared distance of a point v = (x, y, z, 1) to that
# plane is v^T (p p^T) v.  Summing the quadrics of the faces around a vertex
# gives a form whose value at any candidate position is the total squared
# plane distance -- the Garland-Heckbert error.  Because the form is
# symmetric only its ten distinct entries are stored, in the order
#
#     (aa, ab, ac, ad, bb, bc, bd, cc, cd, dd).


_Quadric = Tuple[float, float, float, float, float, float, float, float,
                 float, float]

_ZERO_QUADRIC: _Quadric = (0.0,) * 10


def _plane_quadric(plane: Tuple[float, float, float, float]) -> _Quadric:
    a, b, c, d = plane
    return (a * a, a * b, a * c, a * d,
            b * b, b * c, b * d,
            c * c, c * d,
            d * d)


def _quadric_sum(left: _Quadric, right: _Quadric) -> _Quadric:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2],
            left[3] + right[3], left[4] + right[4], left[5] + right[5],
            left[6] + right[6], left[7] + right[7], left[8] + right[8],
            left[9] + right[9])


def _quadric_value(q: _Quadric, at: Point) -> float:
    x, y, z = at
    aa, ab, ac, ad, bb, bc, bd, cc, cd, dd = q
    return (aa * x * x + bb * y * y + cc * z * z
            + 2.0 * (ab * x * y + ac * x * z + bc * y * z)
            + 2.0 * (ad * x + bd * y + cd * z)
            + dd)


def _vertex_quadrics(points: Sequence[Point],
                     faces: Sequence[Tri]) -> List[_Quadric]:
    """Accumulate the plane quadric of every face onto its three corners."""
    table: List[_Quadric] = [_ZERO_QUADRIC] * len(points)
    for face in faces:
        p, q, r = points[face[0]], points[face[1]], points[face[2]]
        normal = _cross(_delta(q, p), _delta(r, p))
        scale = _norm(normal)
        if scale < _AREA_EPS:
            continue
        a, b, c = normal[0] / scale, normal[1] / scale, normal[2] / scale
        contribution = _plane_quadric(
            (a, b, c, -(a * p[0] + b * p[1] + c * p[2])))
        for corner in face:
            table[corner] = _quadric_sum(table[corner], contribution)
    return table


def _quadric_minimiser(q: _Quadric, first: Point, second: Point) -> Point:
    """Position minimising the quadric; the edge midpoint if it is singular.

    The stationary point solves the 3x3 system formed by the leading block of
    the quadric against the negated linear column, which is solved here by
    Cramer's rule.
    """
    aa, ab, ac, ad, bb, bc, bd, cc, cd, _dd = q
    rhs = (-ad, -bd, -cd)

    # Cofactors of the symmetric leading block.
    c00 = bb * cc - bc * bc
    c01 = ac * bc - ab * cc
    c02 = ab * bc - ac * bb
    determinant = aa * c00 + ab * c01 + ac * c02

    if abs(determinant) < _SINGULAR_EPS:
        return (0.5 * (first[0] + second[0]),
                0.5 * (first[1] + second[1]),
                0.5 * (first[2] + second[2]))

    c11 = aa * cc - ac * ac
    c12 = ab * ac - aa * bc
    c22 = aa * bb - ab * ab

    # The inverse of a symmetric matrix is symmetric, so the cofactor matrix
    # doubles as the adjugate.
    inverse = 1.0 / determinant
    return (inverse * (c00 * rhs[0] + c01 * rhs[1] + c02 * rhs[2]),
            inverse * (c01 * rhs[0] + c11 * rhs[1] + c12 * rhs[2]),
            inverse * (c02 * rhs[0] + c12 * rhs[1] + c22 * rhs[2]))


def _collapse_is_safe(edge: Edge,
                      faces: List[Optional[Tri]],
                      incident: List[Set[int]],
                      uses: Dict[Edge, List[int]]) -> bool:
    """Link condition: the collapse must not fuse unrelated surface sheets.

    Two endpoints may be merged only when the vertices they share are exactly
    the apexes of the faces straddling the edge.  Any extra shared neighbour
    would become a duplicated edge, i.e. a non-manifold result.
    """
    first, second = edge
    straddling = uses.get(edge, [])
    if len(straddling) != 2:
        return False

    apexes: Set[int] = set()
    for index in straddling:
        face = faces[index]
        if face is None:
            return False
        for corner in face:
            if corner != first and corner != second:
                apexes.add(corner)

    def ring(vertex: int) -> Set[int]:
        found: Set[int] = set()
        for index in incident[vertex]:
            face = faces[index]
            if face is None:
                continue
            for corner in face:
                if corner != vertex:
                    found.add(corner)
        return found

    return (ring(first) & ring(second)) == apexes


def decimate(
    vertices: Sequence,
    triangles: Sequence,
    target_faces: Optional[int] = None,
    max_error: Optional[float] = None,
) -> dict:
    """Reduce triangle count by quadric-error-metric edge collapse.

    Edges are ranked by the value of their summed endpoint quadric at that
    quadric's minimiser, and the cheapest is collapsed repeatedly until the
    live face count reaches *target_faces* or the cheapest available collapse
    would cost more than *max_error*.  At least one bound must be supplied.
    Ranking uses a lazy binary heap: entries are stamped with the revision of
    their endpoints and discarded on pop when stale, so no full re-sort is
    needed.  A link condition rejects collapses that would tear manifoldness.

    Returns ``{"ok", "vertices", "triangles", "original_faces",
    "final_faces"}``.
    """
    try:
        complaint = _check_inputs(vertices, triangles)
        if complaint:
            return {"ok": False, "reason": complaint}
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

        points, source_faces = _normalise(vertices, triangles)
        started_with = len(source_faces)

        floor = target_faces if target_faces is not None else 1
        if not source_faces or started_with <= floor:
            return {"ok": True, "vertices": points,
                    "triangles": list(source_faces),
                    "original_faces": started_with,
                    "final_faces": started_with}

        quadrics = _vertex_quadrics(points, source_faces)
        positions: List[Point] = list(points)
        faces: List[Optional[Tri]] = list(source_faces)
        alive_faces = started_with

        incident: List[Set[int]] = [set() for _ in positions]
        for index, face in enumerate(source_faces):
            for corner in face:
                incident[corner].add(index)

        # Revision counters make stale heap entries recognisable.
        revision = [0] * len(positions)
        retired = [False] * len(positions)

        def rank(edge: Edge) -> Tuple[float, Point]:
            merged = _quadric_sum(quadrics[edge[0]], quadrics[edge[1]])
            target = _quadric_minimiser(merged, positions[edge[0]],
                                        positions[edge[1]])
            return _quadric_value(merged, target), target

        queue: List[Tuple[float, int, int, int, int]] = []

        def offer(edge: Edge) -> None:
            cost, _ = rank(edge)
            heapq.heappush(queue, (cost, edge[0], edge[1],
                                   revision[edge[0]], revision[edge[1]]))

        live_edges = sorted(_edge_uses(source_faces))
        for edge in live_edges:
            offer(edge)

        while queue:
            if target_faces is not None and alive_faces <= target_faces:
                break
            cost, first, second, stamp_a, stamp_b = heapq.heappop(queue)
            if retired[first] or retired[second]:
                continue
            if stamp_a != revision[first] or stamp_b != revision[second]:
                continue
            if max_error is not None and cost > max_error:
                break

            edge = (first, second)
            # Edge-use map keyed by live face index, for the link-condition.
            live_uses: Dict[Edge, List[int]] = {}
            for index, face in enumerate(faces):
                if face is None:
                    continue
                for key in _corners(face):
                    live_uses.setdefault(key, []).append(index)

            if not _collapse_is_safe(edge, faces, incident, live_uses):
                continue

            merged = _quadric_sum(quadrics[first], quadrics[second])
            _, landing = rank(edge)

            positions[first] = landing
            quadrics[first] = merged
            retired[second] = True
            revision[first] += 1

            for index in sorted(incident[second]):
                face = faces[index]
                if face is None:
                    continue
                rewritten = tuple(first if c == second else c for c in face)
                if len(set(rewritten)) < 3:
                    faces[index] = None
                    alive_faces -= 1
                    for corner in set(face):
                        incident[corner].discard(index)
                else:
                    faces[index] = (rewritten[0], rewritten[1], rewritten[2])
                    incident[first].add(index)
            incident[second] = set()

            # Re-price every edge now touching the survivor.
            neighbours: Set[int] = set()
            for index in incident[first]:
                face = faces[index]
                if face is None:
                    continue
                for corner in face:
                    if corner != first and not retired[corner]:
                        neighbours.add(corner)
            for other in sorted(neighbours):
                revision[other] += 1
            for other in sorted(neighbours):
                offer(_undirected(first, other))

        # Compact.
        relabel: Dict[int, int] = {}
        kept_points: List[Point] = []
        for index, point in enumerate(positions):
            if not retired[index]:
                relabel[index] = len(kept_points)
                kept_points.append(point)

        kept_faces: List[Tri] = []
        for face in faces:
            if face is None:
                continue
            if any(c not in relabel for c in face):
                continue
            a, b, c = relabel[face[0]], relabel[face[1]], relabel[face[2]]
            if a != b and b != c and a != c:
                kept_faces.append((a, b, c))

        return {
            "ok": True,
            "vertices": kept_points,
            "triangles": kept_faces,
            "original_faces": started_with,
            "final_faces": len(kept_faces),
        }
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "reason": "decimate failed: %s" % exc}


# ==========================================================================
# diagnostics
# ==========================================================================


def is_closed(vertices: Sequence, triangles: Sequence) -> dict:
    """A mesh is closed when every undirected edge carries exactly two faces.

    Returns ``{"ok", "closed"}``.
    """
    try:
        complaint = _check_inputs(vertices, triangles)
        if complaint:
            return {"ok": False, "reason": complaint}
        _, faces = _normalise(vertices, triangles)
        uses = _edge_uses(faces)
        watertight = len(uses) > 0 and all(len(u) == 2 for u in uses.values())
        return {"ok": True, "closed": watertight}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "reason": "is_closed failed: %s" % exc}


def is_manifold(vertices: Sequence, triangles: Sequence) -> dict:
    """Check the edge and vertex manifold conditions.

    Edge condition: no undirected edge carries more than two faces.
    Vertex condition: the faces meeting at a vertex form one edge-connected
    fan, which is tested by unioning incident faces that share an edge and
    checking a single component remains.

    Returns ``{"ok", "manifold", "non_manifold_edges",
    "non_manifold_vertices"}``.
    """
    try:
        complaint = _check_inputs(vertices, triangles)
        if complaint:
            return {"ok": False, "reason": complaint}
        points, faces = _normalise(vertices, triangles)

        uses = _edge_uses(faces)
        bad_edges = sorted(edge for edge, u in uses.items() if len(u) > 2)

        fan: Dict[int, List[int]] = {}
        for index, face in enumerate(faces):
            for corner in face:
                fan.setdefault(corner, []).append(index)

        bad_vertices: List[int] = []
        for vertex in range(len(points)):
            members = fan.get(vertex, [])
            if len(members) < 2:
                continue
            slot = {face_index: n for n, face_index in enumerate(members)}
            forest = _Forest(len(members))
            # Two incident faces belong to the same fan when they share an
            # edge; walk the global edge-use lists restricted to this fan.
            for face_index in members:
                for edge in _corners(faces[face_index]):
                    for partner in uses[edge]:
                        if partner in slot:
                            forest.join(slot[face_index], slot[partner])
            components = len({forest.root(n) for n in range(len(members))})
            if components != 1:
                bad_vertices.append(vertex)

        return {
            "ok": True,
            "manifold": not bad_edges and not bad_vertices,
            "non_manifold_edges": bad_edges,
            "non_manifold_vertices": bad_vertices,
        }
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "reason": "is_manifold failed: %s" % exc}


# ==========================================================================
# repair_pipeline
# ==========================================================================


_PIPELINE = (
    ("weld_vertices", "merged_count", "merged %d vertices"),
    ("unify_normals", "flipped_count", "flipped %d triangles"),
    ("fill_holes", "holes_filled", "filled %d holes"),
    ("remove_degenerate", "removed_count", "removed %d degenerate triangles"),
)


def repair_pipeline(
    vertices: Sequence,
    triangles: Sequence,
    tol: float = 1e-6,
) -> dict:
    """Run weld -> unify_normals -> fill_holes -> remove_degenerate in turn.

    Each stage feeds the next; the first failure stops the chain and the mesh
    is returned as it stood at that point.  Returns ``{"ok", "vertices",
    "triangles", "steps"}`` where *steps* holds one
    ``{"step", "ok", "detail"}`` record per stage attempted.  Never raises.
    """
    steps: List[dict] = []
    try:
        stage_calls = {
            "weld_vertices": lambda v, t: weld_vertices(v, t, tol=tol),
            "unify_normals": unify_normals,
            "fill_holes": fill_holes,
            "remove_degenerate": remove_degenerate,
        }

        current_v: object = vertices
        current_t: object = triangles

        for name, stat, template in _PIPELINE:
            outcome = stage_calls[name](current_v, current_t)
            steps.append({
                "step": name,
                "ok": outcome["ok"],
                "detail": (outcome["reason"] if not outcome["ok"]
                           else template % outcome.get(stat, 0)),
            })
            if not outcome["ok"]:
                return {
                    "ok": False,
                    "vertices": list(current_v)
                    if isinstance(current_v, (list, tuple)) else [],
                    "triangles": list(current_t)
                    if isinstance(current_t, (list, tuple)) else [],
                    "steps": steps,
                }
            current_v = outcome["vertices"]
            current_t = outcome["triangles"]

        return {"ok": True, "vertices": current_v, "triangles": current_t,
                "steps": steps}
    except Exception as exc:  # pragma: no cover
        steps.append({"step": "repair_pipeline", "ok": False,
                      "detail": str(exc)})
        return {"ok": False, "vertices": [], "triangles": [], "steps": steps}


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
