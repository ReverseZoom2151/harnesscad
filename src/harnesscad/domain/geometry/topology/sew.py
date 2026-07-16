"""Tolerant face-to-shell sewing with tolerance monotonicity and healing.

Ported from kerf-main geom/sew.py and geom/body_heal.py (kerf-cad-core).

The harness has no full B-rep face type, so the sewing input is a minimal
:class:`SewFace`: an ordered boundary polygon of 3D points with a per-face
tolerance.  Sewing instantiates fresh vertices and edges for every face,
then stitches them (BREP_CONTRACT-compliant semantics, kept from kerf):

* **Shared vertices**: two vertices ``V1``, ``V2`` are merged when
  ``||V1.point - V2.point|| <= max(V1.tol, V2.tol, tol)``.  All edges
  referencing the loser are repointed at the survivor; the survivor's
  tolerance is bumped to ``max(V1.tol, V2.tol, gap)`` so a merge never
  *narrows* a tolerance and the merged vertex always encloses both input
  positions (kerf BREP_CONTRACT section 4.5).
* **Shared edges**: two edges are merged when (a) their endpoint
  representatives match (in either direction) and (b) the sample-based
  Hausdorff distance between the two sampled polylines (``samples=8``)
  is below ``tol``.  Coedges are repointed; if the survivor runs in the
  opposite direction the moved coedge's orientation is flipped.
  ``edge.tol`` is set to ``max(input edge tols, sew tol, incident face
  tol)``.
* **Coedge pairing / closedness**: an edge is manifold-interior when it
  carries exactly two coedges of *opposite* orientation.  A shell is
  closed iff every one of its edges satisfies this; edges that do not are
  reported as *free edges*.
* **Tolerance monotonicity**: post-sew the invariant
  ``vertex.tol >= edge.tol >= face.tol`` is enforced for every reachable
  triple by bumping the larger-numbered field upward, never inward.  The
  invariant is explicit and checkable via
  :func:`check_tolerance_monotonicity`, which returns a list of violation
  strings (empty when the invariant holds).

The heal pass (from kerf body_heal.py) never mutates its input:

* **Sub-tolerance entity removal**: edges shorter than their *own*
  tolerance are removed; faces whose boundary collapses (fewer than two
  surviving coedges) or whose spatial extent is below their own tolerance
  are dropped.  (kerf compared against the single global ``tol``; the
  harness port compares each entity against its own tolerance field,
  which after sewing is always >= the sew tolerance.)
* **Vertex welding**: vertices within ``tol`` are welded via union-find
  to the lowest-index representative (deterministic), with tolerances
  merged by max-plus-gap as in sewing.
* **Sliver snap**: loop gaps ``0 < gap <= 10 * tol`` between one coedge's
  end point and the next coedge's start point are closed by snapping the
  end vertex onto the start point (kerf's ``_snap_loop_gaps``).

Results are structured: :class:`SewResult` carries the sewn topology plus
shells, remaining free edges and merge counts; :class:`HealReport` carries
what the heal pass removed, welded and snapped.

Deterministic: input order drives iteration, cluster representatives are
first-seen (sew) or lowest-index (heal), candidate lists are ordered, and
no set-iteration order is relied upon.  Pure stdlib.

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

_EDGE_SAMPLES = 8  # kerf sew.py samples each edge curve at 8 points


# ---------------------------------------------------------------------------
# Data model (minimal harness-style stand-ins for kerf's B-rep classes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SewFace:
    """Sewing input: an ordered boundary polygon of 3D points.

    ``boundary`` lists the loop's points in order *without* repeating the
    first point at the end; consecutive points (wrapping around) define
    the face's boundary edges.  ``tol`` is the face tolerance; freshly
    instantiated vertices and edges inherit it as a floor.
    """

    boundary: Tuple[Vec3, ...]
    tol: float = 1e-6


@dataclass
class TopVertex:
    """A sewn vertex: position plus tolerance ball radius."""

    index: int
    point: Vec3
    tol: float


@dataclass
class TopEdge:
    """A sewn straight edge between two vertex indices.

    ``samples`` retains the original sampled polyline (8 points, kerf's
    sample count) used for the Hausdorff curve-equality test at merge
    time; endpoints may later be repositioned by vertex welding/snapping.
    """

    index: int
    v_start: int
    v_end: int
    tol: float
    samples: Tuple[Vec3, ...] = ()


@dataclass
class TopCoedge:
    """A face's use of an edge.  ``orientation`` True = along the edge."""

    edge: int
    orientation: bool


@dataclass
class TopFace:
    """A sewn face: one outer loop of coedges plus the face tolerance."""

    index: int
    loop: List[TopCoedge] = field(default_factory=list)
    tol: float = 1e-6


@dataclass(frozen=True)
class SewnShell:
    """A connected component of sewn faces.

    ``is_closed`` is True iff every edge used by the shell's faces carries
    exactly two coedges of opposite orientation (closed 2-manifold rule).
    """

    face_indices: Tuple[int, ...]
    is_closed: bool


@dataclass
class SewResult:
    """Structured sewing result (what kerf's sew + shell report expose).

    ``vertices``/``edges``/``faces`` are the surviving topology (dense,
    reindexed).  ``shells`` groups faces by edge connectivity.
    ``free_edges`` lists edge indices that are not manifold-interior.
    ``vertex_merges``/``edge_merges`` count the merges performed.
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
    """What the heal pass did (kerf simplify_body + heal_body effects)."""

    vertices_welded: int
    edges_removed: int
    faces_removed: int
    gaps_snapped: int


# ---------------------------------------------------------------------------
# Vector helpers (pure stdlib)
# ---------------------------------------------------------------------------


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dist(a: Vec3, b: Vec3) -> float:
    d = _sub(a, b)
    return math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])


def _lerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def _sample_segment(a: Vec3, b: Vec3, n_samples: int = _EDGE_SAMPLES) -> Tuple[Vec3, ...]:
    """Sample a straight edge at ``n_samples`` points (kerf _edge_polyline)."""
    n = max(2, int(n_samples))
    return tuple(_lerp(a, b, i / (n - 1)) for i in range(n))


def _hausdorff_samples(pl_a: Sequence[Vec3], pl_b: Sequence[Vec3]) -> float:
    """Symmetric sample-based Hausdorff distance between two polylines.

    Direct port of kerf's ``_hausdorff_samples``: symmetric max-of-min
    over the full sample sets, cheap at 8 samples per side.
    """
    ab = max(min(_dist(pa, pb) for pb in pl_b) for pa in pl_a)
    ba = max(min(_dist(pb, pa) for pa in pl_a) for pb in pl_b)
    return max(ab, ba)


def _curves_match(
    edge_a: TopEdge,
    edge_b: TopEdge,
    *,
    same_direction: bool,
    tol: float,
) -> bool:
    """Decide whether two edges trace the same curve geometry to ``tol``.

    The endpoint-representative check has already passed for the caller;
    here interior samples are compared, reversing ``b`` when the
    parametric directions differ (kerf ``_curves_match``).
    """
    pl_b: Sequence[Vec3] = edge_b.samples
    if not same_direction:
        pl_b = tuple(reversed(pl_b))
    return _hausdorff_samples(edge_a.samples, pl_b) <= max(tol, 1e-12)


# ---------------------------------------------------------------------------
# Tolerance monotonicity (explicit + checkable)
# ---------------------------------------------------------------------------


def _merge_tol(tol_a: float, tol_b: float, gap: float) -> float:
    """Merged tolerance: max of the merged tolerances plus the gap.

    A merge must never NARROW a tolerance: the survivor takes the max of
    both entity tolerances, further widened to at least the merge gap so
    the merged entity's tolerance ball covers both input positions.
    """
    return max(tol_a, tol_b, gap)


def _propagate_tolerance(
    faces: Sequence[TopFace],
    edges: Sequence[TopEdge],
    vertices: Sequence[TopVertex],
    sew_tol: float,
) -> None:
    """Enforce ``vertex.tol >= edge.tol >= face.tol`` monotonically.

    All bumps are upward on the larger-numbered field, never inward, so
    propagation is idempotent and never narrows a tolerance (port of
    kerf's ``_propagate_tolerance``).
    """
    # First pass: edge.tol >= face.tol and >= sew_tol
    for f in faces:
        for ce in f.loop:
            e = edges[ce.edge]
            if e.tol < f.tol:
                e.tol = f.tol
            if e.tol < sew_tol:
                e.tol = sew_tol
    # Second pass: vertex.tol >= incident edge.tol
    for f in faces:
        for ce in f.loop:
            e = edges[ce.edge]
            for vi in (e.v_start, e.v_end):
                v = vertices[vi]
                if v.tol < e.tol:
                    v.tol = e.tol


def check_tolerance_monotonicity(result: SewResult) -> List[str]:
    """Check ``vertex.tol >= edge.tol >= face.tol`` on a sewn result.

    Returns a list of human-readable violation strings; an empty list
    means the invariant holds for every reachable face/edge/vertex triple.
    """
    violations: List[str] = []
    for f in result.faces:
        for ce in f.loop:
            e = result.edges[ce.edge]
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
# Sewing
# ---------------------------------------------------------------------------


def _vertex_cluster(vertices: Sequence[TopVertex], tol: float) -> Dict[int, int]:
    """Union by spatial coincidence, first-seen representative.

    Two vertices ``A``, ``B`` fall in the same cluster when
    ``||A.point - B.point|| <= max(A.tol, B.tol, tol)``.  Returns
    ``{vertex_index: representative_vertex_index}``.  On a merge the
    representative's tolerance is widened via :func:`_merge_tol` (never
    narrowed).  Port of kerf's ``_vertex_cluster``.
    """
    rep_of: Dict[int, int] = {}
    cluster_reps: List[int] = []
    for v in vertices:
        found: Optional[int] = None
        for ri in cluster_reps:
            rep = vertices[ri]
            gap = _dist(v.point, rep.point)
            if gap <= max(rep.tol, v.tol, tol):
                found = ri
                # tolerance monotonicity on merge: widen, never narrow
                rep.tol = _merge_tol(rep.tol, v.tol, gap)
                break
        if found is None:
            cluster_reps.append(v.index)
            rep_of[v.index] = v.index
        else:
            rep_of[v.index] = found
    return rep_of


def _shells_and_free_edges(
    faces: Sequence[TopFace],
    edges: Sequence[TopEdge],
) -> Tuple[Tuple[SewnShell, ...], Tuple[int, ...]]:
    """Group faces into edge-connected shells and list free edges.

    A shell is closed iff every edge used by its faces carries exactly
    two coedges of opposite orientation (kerf sew.py step 5, applied per
    connected component).  Free edges are those violating that rule.
    Deterministic: components keyed by their smallest face index.
    """
    # edge index -> list of (face_index, orientation) in input order
    edge_use: Dict[int, List[Tuple[int, bool]]] = {}
    for f in faces:
        for ce in f.loop:
            edge_use.setdefault(ce.edge, []).append((f.index, ce.orientation))

    # Union-find over face indices, joined by shared edges.
    parent: Dict[int, int] = {f.index: f.index for f in faces}

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for ei in sorted(edge_use):
        uses = edge_use[ei]
        first = root(uses[0][0])
        for fi, _ in uses[1:]:
            r = root(fi)
            if r != first:
                # deterministic: smaller root index wins
                if r < first:
                    first, r = r, first
                parent[r] = first

    groups: Dict[int, List[int]] = {}
    for f in faces:
        groups.setdefault(root(f.index), []).append(f.index)

    free: List[int] = []
    for ei in sorted(edge_use):
        uses = edge_use[ei]
        if len(uses) != 2 or uses[0][1] == uses[1][1]:
            free.append(ei)
    free_set = set(free)

    shells: List[SewnShell] = []
    for key in sorted(groups):
        face_indices = tuple(sorted(groups[key]))
        member_edges = set()
        for fi in face_indices:
            for ce in faces[fi].loop:
                member_edges.add(ce.edge)
        is_closed = not any(ei in free_set for ei in member_edges)
        shells.append(SewnShell(face_indices=face_indices, is_closed=is_closed))
    return tuple(shells), tuple(free)


def sew_faces(faces: Iterable[SewFace], tol: float = 1e-6) -> SewResult:
    """Sew independent boundary-polygon faces into shells.

    Parameters
    ----------
    faces:
        Independent :class:`SewFace` polygons; each gets its own fresh
        vertices and edges before stitching, mirroring kerf's contract of
        faces with independent ``Edge``/``Vertex`` objects.
    tol:
        Linear sew tolerance for vertex coincidence and edge curve
        equality.  Default ``1e-6`` (kerf's analytic-build default).

    Returns
    -------
    SewResult
        Sewn topology plus shells (one per edge-connected component,
        each flagged closed iff every member edge is manifold-interior),
        the free edges remaining and the merge counts.  Tolerances
        satisfy ``vertex.tol >= edge.tol >= face.tol`` on return.
    """
    if tol <= 0:
        raise ValueError("tol must be positive, got %r" % (tol,))
    face_list = list(faces)
    if not face_list:
        raise ValueError("sew_faces requires at least one face")

    # 0. Instantiate fresh vertices/edges/coedges per face -------------------
    vertices: List[TopVertex] = []
    edges: List[TopEdge] = []
    top_faces: List[TopFace] = []
    for fi, sf in enumerate(face_list):
        pts = tuple(sf.boundary)
        if len(pts) < 3:
            raise ValueError(
                "face %d needs at least 3 boundary points, got %d" % (fi, len(pts))
            )
        v_ids: List[int] = []
        for p in pts:
            v = TopVertex(index=len(vertices), point=p, tol=sf.tol)
            vertices.append(v)
            v_ids.append(v.index)
        loop: List[TopCoedge] = []
        n = len(v_ids)
        for i in range(n):
            a, b = v_ids[i], v_ids[(i + 1) % n]
            e = TopEdge(
                index=len(edges),
                v_start=a,
                v_end=b,
                tol=sf.tol,
                samples=_sample_segment(vertices[a].point, vertices[b].point),
            )
            edges.append(e)
            loop.append(TopCoedge(edge=e.index, orientation=True))
        top_faces.append(TopFace(index=fi, loop=loop, tol=sf.tol))

    # 1. Cluster vertices -----------------------------------------------------
    rep_of_vertex = _vertex_cluster(vertices, tol)
    vertex_merges = sum(1 for vi, ri in rep_of_vertex.items() if vi != ri)

    # 2. Repoint every edge's endpoints at the cluster reps ------------------
    for e in edges:
        e.v_start = rep_of_vertex[e.v_start]
        e.v_end = rep_of_vertex[e.v_end]

    # 3. Edge merge: same endpoint-rep pair (either direction) AND
    #    sample-Hausdorff equality (kerf sew.py step 3).
    survivor_of: Dict[int, Tuple[int, bool]] = {}  # edge idx -> (survivor, same_dir)
    edge_buckets: List[int] = []
    for e in edges:
        survivor: Optional[int] = None
        same_direction = False
        key_fwd = (e.v_start, e.v_end)
        for ci in edge_buckets:
            cand = edges[ci]
            ck = (cand.v_start, cand.v_end)
            if ck == key_fwd:
                if _curves_match(e, cand, same_direction=True, tol=tol):
                    survivor = ci
                    same_direction = True
                    break
            elif ck == (key_fwd[1], key_fwd[0]):
                if _curves_match(e, cand, same_direction=False, tol=tol):
                    survivor = ci
                    same_direction = False
                    break
        if survivor is None:
            edge_buckets.append(e.index)
            survivor_of[e.index] = (e.index, True)
        else:
            survivor_of[e.index] = (survivor, same_direction)
            # never narrow: survivor's tol envelopes the merged edge's tol,
            # widened by the residual curve gap
            surv = edges[survivor]
            gap = _hausdorff_samples(
                e.samples,
                surv.samples if same_direction else tuple(reversed(surv.samples)),
            )
            surv.tol = _merge_tol(surv.tol, e.tol, gap)
    edge_merges = sum(1 for ei, (si, _) in survivor_of.items() if ei != si)

    # 4. Repoint every coedge onto its edge's survivor; flip orientation
    #    when the survivor runs in the opposite direction.
    for f in top_faces:
        for ce in f.loop:
            surv, same_dir = survivor_of[ce.edge]
            if surv == ce.edge:
                continue
            if not same_dir:
                ce.orientation = not ce.orientation
            ce.edge = surv

    # 5. Compact surviving vertices/edges into dense indices -----------------
    live_edge_ids = sorted({ce.edge for f in top_faces for ce in f.loop})
    edge_remap = {old: new for new, old in enumerate(live_edge_ids)}
    live_vertex_ids = sorted(
        {edges[ei].v_start for ei in live_edge_ids}
        | {edges[ei].v_end for ei in live_edge_ids}
    )
    vertex_remap = {old: new for new, old in enumerate(live_vertex_ids)}

    new_vertices = [
        replace(vertices[old], index=new) for old, new in
        sorted(vertex_remap.items(), key=lambda kv: kv[1])
    ]
    new_edges = [
        replace(
            edges[old],
            index=new,
            v_start=vertex_remap[edges[old].v_start],
            v_end=vertex_remap[edges[old].v_end],
        )
        for old, new in sorted(edge_remap.items(), key=lambda kv: kv[1])
    ]
    for f in top_faces:
        for ce in f.loop:
            ce.edge = edge_remap[ce.edge]

    # 6. Tolerance propagation (post-merge, with the now-final edge tols) ----
    _propagate_tolerance(top_faces, new_edges, new_vertices, tol)

    # 7. Shells + free edges (closedness per connected component) ------------
    shells, free_edges = _shells_and_free_edges(top_faces, new_edges)

    return SewResult(
        vertices=new_vertices,
        edges=new_edges,
        faces=top_faces,
        shells=shells,
        free_edges=free_edges,
        vertex_merges=vertex_merges,
        edge_merges=edge_merges,
    )


# ---------------------------------------------------------------------------
# Heal pass (from kerf body_heal.py)
# ---------------------------------------------------------------------------


def _weld_vertices(vertices: Sequence[TopVertex], tol: float) -> Dict[int, int]:
    """Union-find weld of near-duplicate vertices within ``tol``.

    The lowest-index vertex of each cluster is canonical, so the mapping
    is deterministic (kerf ``_weld_vertices``).  O(N^2) as in kerf --
    bodies are small; a spatial grid is out of scope for this pass.
    """
    parent: Dict[int, int] = {v.index: v.index for v in vertices}

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = root(a), root(b)
        if ra == rb:
            return
        if ra <= rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for i, vi in enumerate(vertices):
        for j in range(i + 1, len(vertices)):
            vj = vertices[j]
            if _dist(vi.point, vj.point) <= tol:
                union(vi.index, vj.index)

    return {v.index: root(v.index) for v in vertices}


def _face_extent(face: TopFace, edges: Sequence[TopEdge], vertices: Sequence[TopVertex]) -> float:
    """Max pairwise distance between the face's loop vertices."""
    pts: List[Vec3] = []
    seen = set()
    for ce in face.loop:
        e = edges[ce.edge]
        for vi in (e.v_start, e.v_end):
            if vi not in seen:
                seen.add(vi)
                pts.append(vertices[vi].point)
    best = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = _dist(pts[i], pts[j])
            if d > best:
                best = d
    return best


def heal_sewn(result: SewResult, tol: float = 1e-6) -> Tuple[SewResult, HealReport]:
    """Heal a sewn result: simplify sub-tolerance geometry + close slivers.

    Port of kerf ``heal_body`` / ``simplify_body`` (``_rebuild_body``):

    1. Copy the topology (the input ``result`` is not mutated).
    2. Weld near-duplicate vertices within ``tol`` (lowest-index
       canonical; merged tolerance widened, never narrowed).
    3. Rewire edge endpoint references onto the canonical vertices.
    4. Remove edges shorter than their own tolerance (sub-tolerance
       entity removal; kerf compared against the global ``tol``).
    5. Drop faces whose loop keeps fewer than 2 surviving coedges
       (degenerate, kerf's rule) or whose spatial extent is below the
       face's own tolerance.
    6. Snap loop gaps within ``10 * tol`` (sliver heal, kerf's
       ``_snap_loop_gaps``): a coedge end vertex is moved onto the next
       coedge's start point when ``0 < gap <= 10 * tol``.
    7. Recompute shells, free edges and the monotonic tolerance chain.

    Returns the healed :class:`SewResult` plus a :class:`HealReport`.
    """
    if tol <= 0:
        raise ValueError("tol must be positive, got %r" % (tol,))

    # Step 1 -- copy
    vertices = [replace(v) for v in result.vertices]
    edges = [replace(e) for e in result.edges]
    faces = [
        TopFace(
            index=f.index,
            loop=[TopCoedge(edge=ce.edge, orientation=ce.orientation) for ce in f.loop],
            tol=f.tol,
        )
        for f in result.faces
    ]

    # Step 2 -- weld map
    weld_map = _weld_vertices(vertices, tol)
    vertices_welded = 0
    for vi, ri in sorted(weld_map.items()):
        if vi != ri:
            vertices_welded += 1
            rep = vertices[ri]
            v = vertices[vi]
            rep.tol = _merge_tol(rep.tol, v.tol, _dist(rep.point, v.point))

    # Step 3 -- rewire edge endpoints
    for e in edges:
        e.v_start = weld_map[e.v_start]
        e.v_end = weld_map[e.v_end]

    # Step 4 -- remove sub-tolerance edges (own tolerance, see docstring)
    removed_edge_ids = set()
    for e in edges:
        length = _dist(vertices[e.v_start].point, vertices[e.v_end].point)
        if length < e.tol:
            removed_edge_ids.add(e.index)
    edges_removed = len(removed_edge_ids)

    # Step 5 -- drop degenerate / sub-tolerance faces
    new_faces: List[TopFace] = []
    faces_removed = 0
    for f in faces:
        surviving = [ce for ce in f.loop if ce.edge not in removed_edge_ids]
        if len(surviving) < 2:
            faces_removed += 1
            continue
        f.loop = surviving
        if _face_extent(f, edges, vertices) < f.tol:
            faces_removed += 1
            continue
        new_faces.append(f)

    # Step 6 -- sliver gap snap (in place on the copies)
    snap_tol = 10.0 * tol
    gaps_snapped = 0
    for f in new_faces:
        n = len(f.loop)
        if n < 1:
            continue
        for i, ce in enumerate(f.loop):
            nxt = f.loop[(i + 1) % n]
            e = edges[ce.edge]
            en = edges[nxt.edge]
            end_vi = e.v_end if ce.orientation else e.v_start
            start_vi = en.v_start if nxt.orientation else en.v_end
            gap = _dist(vertices[end_vi].point, vertices[start_vi].point)
            if 0.0 < gap <= snap_tol:
                target = vertices[start_vi].point
                end_v = vertices[end_vi]
                end_v.point = target
                # widen, never narrow, so the snapped vertex still covers
                # its pre-snap position
                end_v.tol = _merge_tol(end_v.tol, end_v.tol, gap)
                gaps_snapped += 1

    # Step 7 -- compact + recompute shells / free edges / tolerance chain
    live_edge_ids = sorted({ce.edge for f in new_faces for ce in f.loop})
    edge_remap = {old: new for new, old in enumerate(live_edge_ids)}
    live_vertex_ids = sorted(
        {edges[ei].v_start for ei in live_edge_ids}
        | {edges[ei].v_end for ei in live_edge_ids}
    )
    vertex_remap = {old: new for new, old in enumerate(live_vertex_ids)}

    final_vertices = [
        replace(vertices[old], index=new)
        for old, new in sorted(vertex_remap.items(), key=lambda kv: kv[1])
    ]
    final_edges = [
        replace(
            edges[old],
            index=new,
            v_start=vertex_remap[edges[old].v_start],
            v_end=vertex_remap[edges[old].v_end],
        )
        for old, new in sorted(edge_remap.items(), key=lambda kv: kv[1])
    ]
    final_faces: List[TopFace] = []
    for new_index, f in enumerate(new_faces):
        for ce in f.loop:
            ce.edge = edge_remap[ce.edge]
        f.index = new_index
        final_faces.append(f)

    _propagate_tolerance(final_faces, final_edges, final_vertices, tol)
    shells, free_edges = _shells_and_free_edges(final_faces, final_edges)

    healed = SewResult(
        vertices=final_vertices,
        edges=final_edges,
        faces=final_faces,
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


def _jitter(p: Vec3, k: int) -> Vec3:
    """Deterministic sub-tolerance coordinate jitter (magnitude ~3e-9)."""
    s = 3e-9
    return (
        p[0] + s * (1 if (k % 2) else -1),
        p[1] + s * (1 if ((k // 2) % 2) else -1),
        p[2] + s * (1 if ((k // 4) % 2) else -1),
    )


def _unit_cube_faces(tol: float = 1e-6) -> List[SewFace]:
    """Six independent quads of a unit cube, outward-CCW, with jitter."""
    quads: Tuple[Tuple[Vec3, ...], ...] = (
        ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),  # -Z
        ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),  # +Z
        ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),  # -Y
        ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)),  # +Y
        ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),  # -X
        ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)),  # +X
    )
    out: List[SewFace] = []
    for fi, quad in enumerate(quads):
        pts = tuple(
            _jitter(tuple(float(c) for c in p), fi * 4 + i)  # type: ignore[arg-type]
            for i, p in enumerate(quad)
        )
        out.append(SewFace(boundary=pts, tol=tol))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.geometry.topology.sew",
        description="Tolerant face-to-shell sewing with tolerance "
                    "monotonicity and heal pass (kerf-cad-core port).",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="sew a jittered unit cube and a plane pair, "
                             "check tolerance monotonicity, and heal a "
                             "deliberately inserted sliver edge.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    tol = 1e-6

    # 1. Six jittered quads of a unit cube -> one closed shell, no free edges
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

    # 2. Tolerance monotonicity: vertex.tol >= edge.tol >= face.tol, and
    #    merged tolerances never narrowed below any input tolerance.
    assert check_tolerance_monotonicity(cube) == []
    input_tol = max(f.tol for f in _unit_cube_faces(tol))
    assert all(v.tol >= input_tol for v in cube.vertices)
    assert all(e.tol >= input_tol for e in cube.edges)
    print("[selfcheck] tolerance monotonicity holds; no merge narrowed "
          "below the input tolerance %.1e" % input_tol)

    # 3. Plane pair sharing one edge within tolerance -> one open shell,
    #    the shared edge merged (manifold), six free boundary edges.
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

    # 4. Heal pass removes a deliberately inserted sliver edge: a quad
    #    with two boundary points only 1e-8 apart carries a sub-tolerance
    #    edge that welding + removal must eliminate.
    sliver_quad = SewFace(
        boundary=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0 + 1e-8, 1e-8, 0.0),  # sliver: 1e-8 from previous point
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        tol=1e-9,  # tight face tol so the sliver survives the sew
    )
    sewn = sew_faces([sliver_quad], tol=1e-9)  # sew below the sliver size
    assert len(sewn.edges) == 5  # sliver edge survives the sew
    healed, report = heal_sewn(sewn, tol=tol)
    assert report.edges_removed >= 1
    assert report.vertices_welded >= 1
    assert len(healed.edges) == 4 and len(healed.faces) == 1
    assert check_tolerance_monotonicity(healed) == []
    # original result untouched
    assert len(sewn.edges) == 5
    print("[selfcheck] heal: sliver edge removed (%d welded, %d removed), "
          "4 edges remain, input not mutated"
          % (report.vertices_welded, report.edges_removed))

    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
