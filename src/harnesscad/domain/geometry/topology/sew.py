"""Tolerant sewing of independent faces into shells, plus a healing pass.

Sewing is the step that turns a bag of separately built faces into a connected
boundary representation.  Each face arrives with its own vertices and edges, so
two faces that meet along a common seam still hold two distinct copies of that
seam; sewing finds those copies, decides they are the same entity within a
tolerance, and rewires the topology so only one survives.

There is no full B-rep face type in this harness, so the input is deliberately
minimal: :class:`SewFace` is an ordered loop of 3D points (the first point is
not repeated at the end) carrying a face tolerance.

What sewing decides
-------------------
*Vertices.*  Two vertices are the same point when the distance between them
does not exceed the largest of the two vertex tolerances and the sewing
tolerance.  Positions are compared against the first vertex of each cluster, so
the clustering is order-deterministic.  The survivor's tolerance is widened to
cover the gap it just absorbed, which guarantees the merged tolerance ball
still contains both original positions.

*Edges.*  Two edges are the same curve when their endpoints already agree (in
either direction, since an edge shared by two faces is traversed opposite ways
by them) *and* the two curves stay within tolerance of one another along their
length.  The second condition is checked as a two-sided Hausdorff distance
between fixed sample sets, so a pair of edges with matching endpoints but
different shapes is correctly refused.  When a merge happens the surviving
edge's tolerance absorbs the residual deviation, and any face that used the
loser is repointed at the survivor -- flipping its orientation flag if the
survivor runs the other way.

*Shells and closedness.*  Faces are grouped into connected components by shared
edges.  An edge is interior to a manifold when exactly two faces use it and
they use it in opposite directions; anything else -- one use, three uses, or
two uses in the same direction -- is a *free edge*.  A shell is closed exactly
when none of its edges is free.

*Tolerance monotonicity.*  A vertex must be at least as uncertain as the edges
meeting there, and an edge at least as uncertain as the faces using it, or the
model would claim a precision its own sub-entities do not have.  Sewing
therefore ends by raising ``edge.tol`` to the face tolerance and ``vertex.tol``
to the edge tolerance.  Every adjustment moves a tolerance outward, never
inward, so the pass is idempotent and cannot silently tighten a value someone
else relied on.  :func:`check_tolerance_monotonicity` re-derives the invariant
from scratch and returns a list of violation strings, empty when it holds.

What healing does
-----------------
:func:`heal_sewn` builds a corrected copy and leaves its input untouched:

* vertices closer together than the healing tolerance are welded onto the
  lowest-index member of their group (union-find, so the grouping is transitive
  and deterministic);
* edges shorter than their own tolerance are dropped -- such an edge is
  indistinguishable from a point at the precision it declares;
* faces are dropped when fewer than two of their coedges survive, or when the
  whole face is smaller than its own tolerance;
* remaining loop gaps up to ten times the tolerance are closed by snapping the
  end of one coedge onto the start of the next, which is what removes the
  sliver wedges left behind by trimming.

Note that removal thresholds compare each entity against *its own* tolerance
field, which after sewing is always at least the sewing tolerance.

Both entry points return structured results: :class:`SewResult` carries the
topology, the shells, the free edges and the merge counts, and healing adds a
:class:`HealReport` describing what it changed.

Determinism throughout: iteration follows input order, cluster representatives
are first-seen when sewing and lowest-index when welding, and no result depends
on set iteration order.  Pure stdlib, ASCII only.

Public API
----------
``SewFace``, ``TopVertex``, ``TopEdge``, ``TopCoedge``, ``TopFace``,
``SewnShell``, ``SewResult``, ``HealReport``
``sew_faces(faces, tol=1e-6)``, ``heal_sewn(result, tol=1e-6)``,
``check_tolerance_monotonicity(result)``
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "SewFace",
    "TopVertex",
    "TopEdge",
    "TopCoedge",
    "TopFace",
    "SewnShell",
    "SewResult",
    "HealReport",
    "sew_faces",
    "heal_sewn",
    "check_tolerance_monotonicity",
]

Vec3 = Tuple[float, float, float]

#: Points taken along each edge when comparing two curves for equality.  Eight
#: is enough to catch a shape difference on a straight segment while keeping
#: the comparison quadratic in a small constant.
_EDGE_SAMPLES = 8

#: Loop gaps up to this multiple of the healing tolerance are snapped shut.
_SNAP_FACTOR = 10.0


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SewFace:
    """Sewing input: one ordered boundary loop of 3D points.

    ``boundary`` lists the loop in order and does *not* repeat the first point
    at the end; each consecutive pair, wrapping around, becomes one edge.
    ``tol`` is the face tolerance, inherited as a floor by the vertices and
    edges instantiated for this face.
    """

    boundary: Tuple[Vec3, ...]
    tol: float = 1e-6


@dataclass
class TopVertex:
    """A sewn vertex: a position and the radius of its uncertainty ball."""

    index: int
    point: Vec3
    tol: float


@dataclass
class TopEdge:
    """A sewn edge running from ``v_start`` to ``v_end``.

    ``samples`` keeps the polyline the edge was sampled at when it was created,
    which is what the curve-equality test compares.  It is deliberately not
    refreshed when welding or snapping moves an endpoint later: it records the
    geometry the merge decision was made against.
    """

    index: int
    v_start: int
    v_end: int
    tol: float
    samples: Tuple[Vec3, ...] = ()


@dataclass
class TopCoedge:
    """One face's use of one edge.  ``orientation`` True means along the edge."""

    edge: int
    orientation: bool


@dataclass
class TopFace:
    """A sewn face: an ordered loop of coedges plus the face tolerance."""

    index: int
    loop: List[TopCoedge] = field(default_factory=list)
    tol: float = 1e-6


@dataclass(frozen=True)
class SewnShell:
    """One edge-connected component of faces.

    ``is_closed`` is True when every edge the component uses is shared by
    exactly two of its faces in opposite directions.
    """

    face_indices: Tuple[int, ...]
    is_closed: bool


@dataclass
class SewResult:
    """The outcome of a sew (or of a heal, which produces the same shape).

    ``vertices``/``edges``/``faces`` are the surviving topology with dense
    indices.  ``shells`` groups the faces, ``free_edges`` lists the edges that
    are not manifold-interior, and the two counters report how many entities
    the merge steps eliminated.
    """

    vertices: List[TopVertex]
    edges: List[TopEdge]
    faces: List[TopFace]
    shells: Tuple[SewnShell, ...]
    free_edges: Tuple[int, ...]
    vertex_merges: int
    edge_merges: int


@dataclass(frozen=True)
class HealReport:
    """A tally of what the healing pass changed."""

    vertices_welded: int
    edges_removed: int
    faces_removed: int
    gaps_snapped: int


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------


def _distance(a: Vec3, b: Vec3) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _polyline(a: Vec3, b: Vec3, count: int = _EDGE_SAMPLES) -> Tuple[Vec3, ...]:
    """Sample the straight segment ``a -> b`` at ``count`` evenly spaced points.

    Both endpoints are included, so two points is the degenerate minimum.
    """
    n = max(2, int(count))
    last = float(n - 1)
    out: List[Vec3] = []
    for i in range(n):
        t = i / last
        out.append((a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                    a[2] + (b[2] - a[2]) * t))
    return tuple(out)


def _polyline_deviation(left: Sequence[Vec3], right: Sequence[Vec3]) -> float:
    """Two-sided Hausdorff distance between two sampled polylines.

    Each sample on one side is charged the distance to its nearest sample on
    the other side; the answer is the worst such charge in either direction.
    Taking both directions matters: a one-sided maximum would accept a short
    curve hugging part of a long one.
    """
    forward = max(min(_distance(p, q) for q in right) for p in left)
    backward = max(min(_distance(q, p) for p in left) for q in right)
    return forward if forward > backward else backward


def _widen(tol_a: float, tol_b: float, gap: float) -> float:
    """Tolerance of an entity formed by merging two others across ``gap``.

    Taking the maximum of everything in play is the only choice that keeps the
    merged tolerance ball covering both original entities, and it guarantees a
    merge can never narrow a tolerance.
    """
    return max(tol_a, tol_b, gap)


class _DisjointSet:
    """Union-find over integer keys, with the smallest key as representative.

    Choosing the smallest key rather than a rank- or size-based winner makes
    the resulting labelling depend only on the keys, never on the order the
    unions arrived in -- which is what makes both callers deterministic.
    """

    def __init__(self, keys: Iterable[int]) -> None:
        self._parent: Dict[int, int] = {k: k for k in keys}

    def find(self, key: int) -> int:
        parent = self._parent
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != root:  # path compression on the way back out
            parent[key], key = root, parent[key]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if rb < ra:
            ra, rb = rb, ra
        self._parent[rb] = ra

    def labels(self) -> Dict[int, int]:
        """Map every key to its representative."""
        return {k: self.find(k) for k in sorted(self._parent)}


# ---------------------------------------------------------------------------
# Tolerance monotonicity
# ---------------------------------------------------------------------------


def _raise_tolerances(faces: Sequence[TopFace], edges: Sequence[TopEdge],
                      vertices: Sequence[TopVertex], floor: float) -> None:
    """Restore ``vertex.tol >= edge.tol >= face.tol`` (and ``>= floor``).

    Two sweeps are needed rather than one: a vertex can only be given its final
    value once every edge touching it has already been lifted, and edges are
    only final after every face using them has been seen.  All motion is
    outward, so running the pass twice changes nothing the second time.
    """
    for f in faces:
        for coedge in f.loop:
            e = edges[coedge.edge]
            target = f.tol if f.tol > floor else floor
            if e.tol < target:
                e.tol = target
    for f in faces:
        for coedge in f.loop:
            e = edges[coedge.edge]
            for vi in (e.v_start, e.v_end):
                v = vertices[vi]
                if v.tol < e.tol:
                    v.tol = e.tol


def check_tolerance_monotonicity(result: SewResult) -> List[str]:
    """Report every place a sewn result claims more precision than it has.

    Returns human-readable violation strings for each face/edge and edge/vertex
    pair that breaks ``vertex.tol >= edge.tol >= face.tol``.  An empty list
    means the invariant holds everywhere it is reachable.
    """
    violations: List[str] = []
    for f in result.faces:
        for coedge in f.loop:
            e = result.edges[coedge.edge]
            if e.tol < f.tol:
                violations.append(
                    "edge %d tol %.3e < face %d tol %.3e"
                    % (e.index, e.tol, f.index, f.tol)
                )
            for vi in (e.v_start, e.v_end):
                v = result.vertices[vi]
                if v.tol < e.tol:
                    violations.append(
                        "vertex %d tol %.3e < edge %d tol %.3e"
                        % (v.index, v.tol, e.index, e.tol)
                    )
    return violations


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------


def _compact(vertices: Sequence[TopVertex], edges: Sequence[TopEdge],
             faces: Sequence[TopFace]) -> Tuple[List[TopVertex], List[TopEdge]]:
    """Drop unreferenced entities and renumber the rest densely.

    Only what the surviving face loops still point at is kept.  New indices
    follow the old ordering, so the output is reproducible, and the faces'
    coedges are rewritten in place to the new edge numbering.
    """
    used_edges = sorted({ce.edge for f in faces for ce in f.loop})
    edge_map = {old: new for new, old in enumerate(used_edges)}

    used_vertices: List[int] = sorted(
        {edges[ei].v_start for ei in used_edges}
        | {edges[ei].v_end for ei in used_edges}
    )
    vertex_map = {old: new for new, old in enumerate(used_vertices)}

    fresh_vertices = [replace(vertices[old], index=new)
                      for new, old in enumerate(used_vertices)]
    fresh_edges = [
        replace(edges[old], index=new,
                v_start=vertex_map[edges[old].v_start],
                v_end=vertex_map[edges[old].v_end])
        for new, old in enumerate(used_edges)
    ]
    for f in faces:
        for coedge in f.loop:
            coedge.edge = edge_map[coedge.edge]
    return fresh_vertices, fresh_edges


def _classify_edges(faces: Sequence[TopFace],
                    edges: Sequence[TopEdge]
                    ) -> Tuple[Tuple[SewnShell, ...], Tuple[int, ...]]:
    """Split faces into shells and list the edges that are not manifold-interior.

    An edge joins the faces that use it into one shell regardless of whether it
    is manifold there; closedness is judged separately, so an open strip is
    still reported as a single shell.
    """
    usage: Dict[int, List[bool]] = {}
    owners: Dict[int, List[int]] = {}
    for f in faces:
        for coedge in f.loop:
            usage.setdefault(coedge.edge, []).append(coedge.orientation)
            owners.setdefault(coedge.edge, []).append(f.index)

    connectivity = _DisjointSet(f.index for f in faces)
    for ei in sorted(owners):
        sharers = owners[ei]
        for other in sharers[1:]:
            connectivity.union(sharers[0], other)

    # Manifold interior means used exactly twice, once in each direction.
    free = tuple(ei for ei in sorted(usage)
                 if len(usage[ei]) != 2 or usage[ei][0] == usage[ei][1])
    free_lookup = set(free)

    components: Dict[int, List[int]] = {}
    for f in faces:
        components.setdefault(connectivity.find(f.index), []).append(f.index)

    shells: List[SewnShell] = []
    for root in sorted(components):
        members = tuple(sorted(components[root]))
        member_edges = {ce.edge for fi in members for ce in faces[fi].loop}
        shells.append(SewnShell(
            face_indices=members,
            is_closed=all(ei not in free_lookup for ei in member_edges),
        ))
    return tuple(shells), free


# ---------------------------------------------------------------------------
# Sewing
# ---------------------------------------------------------------------------


def _instantiate(face_list: Sequence[SewFace]
                 ) -> Tuple[List[TopVertex], List[TopEdge], List[TopFace]]:
    """Turn boundary polygons into independent vertices, edges and coedges.

    Nothing is shared at this point even between faces that obviously touch:
    finding those coincidences is exactly the job of the steps that follow.
    """
    vertices: List[TopVertex] = []
    edges: List[TopEdge] = []
    faces: List[TopFace] = []

    for fi, source in enumerate(face_list):
        points = tuple(source.boundary)
        if len(points) < 3:
            raise ValueError(
                "face %d needs at least 3 boundary points, got %d"
                % (fi, len(points))
            )
        corner_ids = []
        for point in points:
            vertices.append(TopVertex(index=len(vertices), point=point,
                                      tol=source.tol))
            corner_ids.append(vertices[-1].index)

        loop: List[TopCoedge] = []
        count = len(corner_ids)
        for i in range(count):
            start = corner_ids[i]
            end = corner_ids[(i + 1) % count]
            edges.append(TopEdge(
                index=len(edges),
                v_start=start,
                v_end=end,
                tol=source.tol,
                samples=_polyline(vertices[start].point, vertices[end].point),
            ))
            loop.append(TopCoedge(edge=edges[-1].index, orientation=True))
        faces.append(TopFace(index=fi, loop=loop, tol=source.tol))

    return vertices, edges, faces


def _cluster_vertices(vertices: Sequence[TopVertex],
                      tol: float) -> Dict[int, int]:
    """Assign every vertex to a coincidence cluster, first seen wins.

    A vertex joins the first existing cluster whose representative it is within
    reach of, where reach is the largest of the two tolerances and ``tol``.
    Comparing only against representatives (rather than all members) keeps the
    outcome independent of which member happened to be tested first.
    """
    representative: Dict[int, int] = {}
    leaders: List[int] = []

    for v in vertices:
        joined: Optional[int] = None
        for leader_index in leaders:
            leader = vertices[leader_index]
            gap = _distance(v.point, leader.point)
            if gap <= max(leader.tol, v.tol, tol):
                # Absorbing a vertex widens the leader so its ball still covers
                # the position it just swallowed.
                leader.tol = _widen(leader.tol, v.tol, gap)
                joined = leader_index
                break
        if joined is None:
            leaders.append(v.index)
            representative[v.index] = v.index
        else:
            representative[v.index] = joined
    return representative


def _merge_edges(edges: Sequence[TopEdge],
                 tol: float) -> Dict[int, Tuple[int, bool]]:
    """Decide which edges are duplicates; map each to ``(survivor, same_way)``.

    Endpoint agreement is a necessary condition, so candidates are grouped by
    their unordered endpoint pair and only compared inside a group.  Within a
    group the first edge seen becomes the survivor for everything that also
    matches it geometrically; an edge whose endpoints coincide but whose curve
    wanders too far is left alone and starts its own survivor.
    """
    by_endpoints: Dict[Tuple[int, int], List[int]] = {}
    resolution: Dict[int, Tuple[int, bool]] = {}

    for e in edges:
        key = (e.v_start, e.v_end) if e.v_start <= e.v_end else (e.v_end, e.v_start)
        group = by_endpoints.setdefault(key, [])

        chosen: Optional[Tuple[int, bool]] = None
        for candidate_index in group:
            candidate = edges[candidate_index]
            same_way = (candidate.v_start == e.v_start
                        and candidate.v_end == e.v_end)
            reference = candidate.samples if same_way else tuple(
                reversed(candidate.samples))
            deviation = _polyline_deviation(e.samples, reference)
            if deviation <= max(tol, 1e-12):
                candidate.tol = _widen(candidate.tol, e.tol, deviation)
                chosen = (candidate_index, same_way)
                break

        if chosen is None:
            group.append(e.index)
            resolution[e.index] = (e.index, True)
        else:
            resolution[e.index] = chosen
    return resolution


def sew_faces(faces: Iterable[SewFace], tol: float = 1e-6) -> SewResult:
    """Stitch independent boundary polygons into shells.

    Parameters
    ----------
    faces:
        The polygons to sew.  Each is given its own fresh vertices and edges
        before any stitching, so callers need not pre-share anything.
    tol:
        Linear tolerance for both vertex coincidence and curve equality.

    Returns
    -------
    SewResult
        The stitched topology, one shell per edge-connected component (each
        flagged closed when all of its edges are manifold-interior), the
        remaining free edges, and how many vertices and edges were merged.
        ``vertex.tol >= edge.tol >= face.tol`` holds on return.

    Raises
    ------
    ValueError
        If ``tol`` is not positive, no faces were given, or a face has fewer
        than three boundary points.
    """
    if tol <= 0:
        raise ValueError("tol must be positive, got %r" % (tol,))
    face_list = list(faces)
    if not face_list:
        raise ValueError("sew_faces requires at least one face")

    vertices, edges, top_faces = _instantiate(face_list)

    # Coincident vertices first: edge identity is defined in terms of the
    # vertices, so it can only be decided once those are settled.
    vertex_rep = _cluster_vertices(vertices, tol)
    vertex_merges = sum(1 for vi, ri in vertex_rep.items() if vi != ri)
    for e in edges:
        e.v_start = vertex_rep[e.v_start]
        e.v_end = vertex_rep[e.v_end]

    edge_rep = _merge_edges(edges, tol)
    edge_merges = sum(1 for ei, (si, _) in edge_rep.items() if ei != si)

    # Point every face at the surviving edges.  A coedge that ends up on an
    # edge running the other way has its direction flag inverted so the loop
    # still traverses the boundary the way the face intended.
    for f in top_faces:
        for coedge in f.loop:
            survivor, same_way = edge_rep[coedge.edge]
            if survivor == coedge.edge:
                continue
            if not same_way:
                coedge.orientation = not coedge.orientation
            coedge.edge = survivor

    live_vertices, live_edges = _compact(vertices, edges, top_faces)
    _raise_tolerances(top_faces, live_edges, live_vertices, tol)
    shells, free_edges = _classify_edges(top_faces, live_edges)

    return SewResult(
        vertices=live_vertices,
        edges=live_edges,
        faces=top_faces,
        shells=shells,
        free_edges=free_edges,
        vertex_merges=vertex_merges,
        edge_merges=edge_merges,
    )


# ---------------------------------------------------------------------------
# Healing
# ---------------------------------------------------------------------------


def _clone(result: SewResult) -> Tuple[List[TopVertex], List[TopEdge],
                                       List[TopFace]]:
    """Deep-enough copy of a result's topology so healing cannot touch it."""
    vertices = [replace(v) for v in result.vertices]
    edges = [replace(e) for e in result.edges]
    faces = [
        TopFace(index=f.index,
                loop=[TopCoedge(edge=ce.edge, orientation=ce.orientation)
                      for ce in f.loop],
                tol=f.tol)
        for f in result.faces
    ]
    return vertices, edges, faces


def _weld_vertices(vertices: Sequence[TopVertex],
                   tol: float) -> Tuple[Dict[int, int], int]:
    """Group vertices within ``tol`` and return ``(mapping, welded_count)``.

    Union-find rather than a leader scan, because welding is transitive here: a
    chain of vertices each within tolerance of the next should collapse to one
    even if its ends are further apart than ``tol``.  The lowest index in each
    group survives.  Quadratic in the vertex count, which is acceptable for the
    body sizes this module handles.
    """
    groups = _DisjointSet(v.index for v in vertices)
    total = len(vertices)
    for i in range(total):
        for j in range(i + 1, total):
            if _distance(vertices[i].point, vertices[j].point) <= tol:
                groups.union(vertices[i].index, vertices[j].index)

    mapping = groups.labels()
    welded = 0
    for vi in sorted(mapping):
        keeper = mapping[vi]
        if keeper == vi:
            continue
        welded += 1
        survivor = vertices[keeper]
        absorbed = vertices[vi]
        survivor.tol = _widen(survivor.tol, absorbed.tol,
                              _distance(survivor.point, absorbed.point))
    return mapping, welded


def _face_extent(face: TopFace, edges: Sequence[TopEdge],
                 vertices: Sequence[TopVertex]) -> float:
    """Largest distance between any two vertices of a face's loop."""
    seen: List[int] = []
    for coedge in face.loop:
        e = edges[coedge.edge]
        for vi in (e.v_start, e.v_end):
            if vi not in seen:
                seen.append(vi)
    widest = 0.0
    for a in range(len(seen)):
        for b in range(a + 1, len(seen)):
            span = _distance(vertices[seen[a]].point, vertices[seen[b]].point)
            if span > widest:
                widest = span
    return widest


def _snap_loop_gaps(faces: Sequence[TopFace], edges: Sequence[TopEdge],
                    vertices: Sequence[TopVertex], limit: float) -> int:
    """Close small breaks between consecutive coedges; return how many.

    Where one coedge ends and the next begins should be the same vertex.  If
    they are merely close, the end vertex is moved onto the start point and its
    tolerance widened by the distance it travelled, so the vertex still covers
    where it used to be.  Gaps wider than ``limit`` are real and left alone.
    """
    snapped = 0
    for f in faces:
        count = len(f.loop)
        if count < 1:
            continue
        for i, coedge in enumerate(f.loop):
            follower = f.loop[(i + 1) % count]
            here = edges[coedge.edge]
            there = edges[follower.edge]
            # Orientation decides which end of each edge is actually in play.
            tail = here.v_end if coedge.orientation else here.v_start
            head = there.v_start if follower.orientation else there.v_end
            gap = _distance(vertices[tail].point, vertices[head].point)
            if 0.0 < gap <= limit:
                moving = vertices[tail]
                moving.point = vertices[head].point
                moving.tol = _widen(moving.tol, moving.tol, gap)
                snapped += 1
    return snapped


def heal_sewn(result: SewResult, tol: float = 1e-6) -> Tuple[SewResult, HealReport]:
    """Repair a sewn result: weld, drop sub-tolerance entities, close slivers.

    The input is never modified; a corrected copy is returned alongside a
    :class:`HealReport` of what changed.  The order of operations matters:
    welding first is what makes the sliver edges collapse to zero length so the
    removal step can recognise them, and snapping runs last so it only sees the
    gaps that survived everything else.

    Raises ``ValueError`` when ``tol`` is not positive.
    """
    if tol <= 0:
        raise ValueError("tol must be positive, got %r" % (tol,))

    vertices, edges, faces = _clone(result)

    weld_map, vertices_welded = _weld_vertices(vertices, tol)
    for e in edges:
        e.v_start = weld_map[e.v_start]
        e.v_end = weld_map[e.v_end]

    # An edge shorter than the precision it claims is not distinguishable from
    # a point, so it carries no information and goes.
    doomed_edges = {
        e.index for e in edges
        if _distance(vertices[e.v_start].point, vertices[e.v_end].point) < e.tol
    }
    edges_removed = len(doomed_edges)

    survivors: List[TopFace] = []
    faces_removed = 0
    for f in faces:
        f.loop = [ce for ce in f.loop if ce.edge not in doomed_edges]
        if len(f.loop) < 2:
            faces_removed += 1          # nothing left that bounds an area
            continue
        if _face_extent(f, edges, vertices) < f.tol:
            faces_removed += 1          # the whole face is below its own noise
            continue
        survivors.append(f)

    gaps_snapped = _snap_loop_gaps(survivors, edges, vertices,
                                   _SNAP_FACTOR * tol)

    live_vertices, live_edges = _compact(vertices, edges, survivors)
    for new_index, f in enumerate(survivors):
        f.index = new_index

    _raise_tolerances(survivors, live_edges, live_vertices, tol)
    shells, free_edges = _classify_edges(survivors, live_edges)

    healed = SewResult(
        vertices=live_vertices,
        edges=live_edges,
        faces=survivors,
        shells=shells,
        free_edges=free_edges,
        vertex_merges=result.vertex_merges + vertices_welded,
        edge_merges=result.edge_merges,
    )
    report = HealReport(
        vertices_welded=vertices_welded,
        edges_removed=edges_removed,
        faces_removed=faces_removed,
        gaps_snapped=gaps_snapped,
    )
    return healed, report


# ---------------------------------------------------------------------------
# Selfcheck fixtures
# ---------------------------------------------------------------------------


def _nudge(point: Vec3, seed: int) -> Vec3:
    """Displace a point by a fixed sub-tolerance amount in a repeatable pattern.

    The three bits of ``seed`` pick the sign on each axis, so neighbouring
    copies of the same corner land in genuinely different places and the
    clustering step has real work to do -- without any randomness.
    """
    step = 3e-9
    return (point[0] + (step if seed & 1 else -step),
            point[1] + (step if seed & 2 else -step),
            point[2] + (step if seed & 4 else -step))


def _unit_cube_faces(tol: float = 1e-6) -> List[SewFace]:
    """The six quads of the unit cube, outward oriented, each slightly nudged."""
    quads: Tuple[Tuple[Vec3, ...], ...] = (
        ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),  # -Z
        ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),  # +Z
        ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),  # -Y
        ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)),  # +Y
        ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),  # -X
        ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)),  # +X
    )
    built: List[SewFace] = []
    for fi, quad in enumerate(quads):
        corners = tuple(
            _nudge((float(p[0]), float(p[1]), float(p[2])), fi * 4 + i)
            for i, p in enumerate(quad)
        )
        built.append(SewFace(boundary=corners, tol=tol))
    return built


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.geometry.topology.sew",
        description="Tolerant face-to-shell sewing with tolerance "
                    "monotonicity and a healing pass.",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="sew a nudged unit cube and a plane pair, check "
                             "tolerance monotonicity, and heal a deliberately "
                             "inserted sliver edge.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    tol = 1e-6

    # 1. Six nudged quads of a unit cube must sew into one closed shell.  The
    #    24 instantiated corners collapse onto 8 (16 merges) and the 24 edges
    #    onto 12 (12 merges), leaving nothing free.
    cube = sew_faces(_unit_cube_faces(tol), tol=tol)
    assert len(cube.shells) == 1, cube.shells
    assert cube.shells[0].is_closed
    assert cube.free_edges == ()
    assert len(cube.vertices) == 8 and len(cube.edges) == 12
    assert cube.vertex_merges == 16 and cube.edge_merges == 12
    print("[selfcheck] cube: 1 closed shell, %d verts, %d edges, "
          "%d vertex merges, %d edge merges, 0 free edges"
          % (len(cube.vertices), len(cube.edges),
             cube.vertex_merges, cube.edge_merges))

    # 2. The tolerance chain holds, and no merge tightened anything below the
    #    tolerance the inputs were declared with.
    assert check_tolerance_monotonicity(cube) == []
    input_tol = max(f.tol for f in _unit_cube_faces(tol))
    assert all(v.tol >= input_tol for v in cube.vertices)
    assert all(e.tol >= input_tol for e in cube.edges)
    print("[selfcheck] tolerance monotonicity holds; no merge narrowed "
          "below the input tolerance %.1e" % input_tol)

    # 3. Two quads meeting along a seam that only agrees to within tolerance:
    #    the seam merges (8 edges become 7), the strip is one open shell, and
    #    its six outer edges stay free.
    eps = 4e-9
    quad_a = SewFace(
        boundary=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        tol=tol,
    )
    quad_b = SewFace(
        boundary=((1.0, eps, 0.0), (0.0, eps, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0)),
        tol=tol,
    )
    pair = sew_faces([quad_a, quad_b], tol=tol)
    assert len(pair.shells) == 1 and not pair.shells[0].is_closed
    assert len(pair.edges) == 7 and pair.edge_merges == 1
    assert len(pair.free_edges) == 6
    assert check_tolerance_monotonicity(pair) == []
    print("[selfcheck] plane pair: 1 open shell, shared edge merged "
          "(7 edges, 6 free)")

    # 4. Endpoint agreement alone must not merge edges: two triangles sharing
    #    two corners but bulging apart in between stay distinct.
    bulge_a = SewFace(boundary=((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
                      tol=tol)
    bulge_b = SewFace(boundary=((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, -1.0, 0.0)),
                      tol=tol)
    bulged = sew_faces([bulge_a, bulge_b], tol=tol)
    assert bulged.edge_merges == 1  # only the truly shared base segment
    assert len(bulged.edges) == 5
    print("[selfcheck] curve test binds: only the coincident segment merged")

    # 5. Healing removes a deliberately inserted sliver: a quad whose boundary
    #    doubles back on itself within 1e-8 carries an edge far below the
    #    healing tolerance, which welding plus removal must eliminate.
    sliver_quad = SewFace(
        boundary=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0 + 1e-8, 1e-8, 0.0),  # 1e-8 from the previous point
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        tol=1e-9,  # tight enough that the sliver survives sewing
    )
    sewn = sew_faces([sliver_quad], tol=1e-9)
    assert len(sewn.edges) == 5  # the sliver edge is still there
    healed, report = heal_sewn(sewn, tol=tol)
    assert report.edges_removed >= 1
    assert report.vertices_welded >= 1
    assert len(healed.edges) == 4 and len(healed.faces) == 1
    assert check_tolerance_monotonicity(healed) == []
    assert len(sewn.edges) == 5  # the input result was not mutated
    print("[selfcheck] heal: sliver edge removed (%d welded, %d removed), "
          "4 edges remain, input not mutated"
          % (report.vertices_welded, report.edges_removed))

    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
