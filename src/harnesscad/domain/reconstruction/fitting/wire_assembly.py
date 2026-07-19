"""Weld a bag of edge polylines into oriented closed loops (B-Rep wires).

A learned edge predictor emits B-Rep edges as *independent* 3D polylines: an
unordered bag with no shared topology.  Before a face or a solid can be built
from them the bag has to become a graph -- endpoints that land on the same
physical corner must be identified with one another -- and that graph has to be
walked into oriented cycles.  This module is that front end:

* :func:`circle_from_three_points` -- the exact 3D circumcircle (centre, radius,
  unit normal) of three non-collinear points, used to recover arc geometry from
  sampled polyline points.
* :func:`weld_endpoints` -- collapse coincident polyline endpoints (within a
  tolerance) into shared corner nodes and derive the corner/edge incidence.
* :func:`assemble_loops` -- walk that incidence into oriented closed loops, each
  a list of ``(edge_index, direction)`` with ``direction`` recording whether the
  polyline was traversed forwards (``+1``) or backwards (``-1``).  Edges that do
  not close a cycle come back as open chains rather than being dropped.

Where this sits: :mod:`harnesscad.domain.reconstruction.brep.chain_complex`
extracts loops from *precomputed integer incidence* -- which curve touches which
corner.  It cannot start from raw coordinates.  This module produces exactly
that incidence from coordinates, so the two compose rather than overlap; nothing
here is shared with, or duplicated from, the index-based complex.

Pure stdlib and deterministic: endpoints are welded in edge-index order, loops
are seeded from the lowest unused edge, and at each corner the lowest-indexed
unused incident edge is taken.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "circle_from_three_points",
    "weld_endpoints",
    "WeldResult",
    "assemble_loops",
    "LoopAssembly",
]

Vec3 = Tuple[float, float, float]
Edge = Sequence[Sequence[float]]  # a polyline: >= 2 points

#: Squared length below which a cross product counts as the zero vector, i.e.
#: the three points are collinear to within double precision.
DEGENERACY_EPS_SQ = 1e-18


# --------------------------------------------------------------------------
# small vector helpers
# --------------------------------------------------------------------------

def _as_point(p: Sequence[float]) -> Vec3:
    return (float(p[0]), float(p[1]), float(p[2]))


def _delta(p: Vec3, q: Vec3) -> Vec3:
    """``p - q``."""
    return (p[0] - q[0], p[1] - q[1], p[2] - q[2])


def _cross(u: Vec3, v: Vec3) -> Vec3:
    return (u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0])


def _norm_sq(u: Vec3) -> float:
    return u[0] * u[0] + u[1] * u[1] + u[2] * u[2]


def _distance_sq(p: Vec3, q: Vec3) -> float:
    return _norm_sq(_delta(p, q))


def _combine(points: Sequence[Vec3], weights: Sequence[float],
             total: float) -> Vec3:
    """The affine combination ``sum(w_i * P_i) / total``."""
    x = y = z = 0.0
    for (px, py, pz), w in zip(points, weights):
        x += w * px
        y += w * py
        z += w * pz
    return (x / total, y / total, z / total)


# --------------------------------------------------------------------------
# exact 3D circle through three points
# --------------------------------------------------------------------------

def circle_from_three_points(
    p1: Sequence[float], p2: Sequence[float], p3: Sequence[float]
) -> Tuple[Vec3, float, Vec3]:
    """Return ``(center, radius, unit_normal)`` of the circle through p1,p2,p3.

    The centre is built as the triangle's circumcentre in *barycentric*
    coordinates: with the squared side lengths ``a2 = |p3-p2|^2`` and so on, the
    circumcentre carries weight ``a2 * (b2 + c2 - a2)`` at the opposite vertex.
    The weights sum to sixteen times the squared triangle area, which is zero
    exactly when the points are collinear, so the same quantity that normalises
    the combination is also the degeneracy test.

    The normal is the unit vector along ``(p2-p1) x (p3-p1)``, so the returned
    frame follows the winding of the arguments.

    Raises ``ValueError`` when the three points are collinear and therefore lie
    on no unique circle.
    """
    a_pt = _as_point(p1)
    b_pt = _as_point(p2)
    c_pt = _as_point(p3)

    span_ab = _delta(b_pt, a_pt)
    span_ac = _delta(c_pt, a_pt)
    plane_normal = _cross(span_ab, span_ac)
    normal_sq = _norm_sq(plane_normal)
    if normal_sq <= DEGENERACY_EPS_SQ:
        raise ValueError("collinear points have no unique circle")

    # Squared side lengths, each named for the vertex it faces.
    facing_a = _distance_sq(c_pt, b_pt)
    facing_b = _distance_sq(a_pt, c_pt)
    facing_c = _distance_sq(b_pt, a_pt)

    weights = (
        facing_a * (facing_b + facing_c - facing_a),
        facing_b * (facing_c + facing_a - facing_b),
        facing_c * (facing_a + facing_b - facing_c),
    )
    weight_sum = weights[0] + weights[1] + weights[2]
    center = _combine((a_pt, b_pt, c_pt), weights, weight_sum)

    radius = math.sqrt(_distance_sq(center, a_pt))
    scale = 1.0 / math.sqrt(normal_sq)
    normal = (plane_normal[0] * scale,
              plane_normal[1] * scale,
              plane_normal[2] * scale)
    return center, radius, normal


# --------------------------------------------------------------------------
# endpoint welding
# --------------------------------------------------------------------------

@dataclass
class WeldResult:
    """Welded corner graph derived from a bag of edge polylines."""
    node_coords: List[Vec3]                      # welded corner positions
    edge_nodes: List[Tuple[int, int]]            # (start_node, end_node) per edge
    node_incident: List[List[Tuple[int, int]]]   # per node: (edge_idx, +1 at start / -1 at end)


def _terminal_points(edges: Sequence[Edge]) -> List[Vec3]:
    """Flatten the bag into ``[e0.start, e0.end, e1.start, e1.end, ...]``.

    Welding only ever looks at the two extreme points of each polyline; the
    interior samples describe the edge's shape, not its topology.
    """
    flat: List[Vec3] = []
    for polyline in edges:
        flat.append(_as_point(polyline[0]))
        flat.append(_as_point(polyline[-1]))
    return flat


def _cluster_terminals(terminals: Sequence[Vec3],
                       tol: float) -> Tuple[List[Vec3], List[int]]:
    """Assign each terminal a corner id, creating corners on first sight.

    Returns ``(corner positions, corner id per terminal)``.  A terminal joins
    the first already-created corner it falls within ``tol`` of; because the
    terminals arrive in a fixed order this is fully deterministic, unlike a
    "nearest corner wins" rule whose ties depend on floating-point order.
    """
    corners: List[Vec3] = []
    assignment: List[int] = []
    tol_sq = tol * tol
    for point in terminals:
        chosen = -1
        for corner_id, corner in enumerate(corners):
            if _distance_sq(point, corner) < tol_sq:
                chosen = corner_id
                break
        if chosen < 0:
            chosen = len(corners)
            corners.append(point)
        assignment.append(chosen)
    return corners, assignment


def weld_endpoints(edges: Sequence[Edge], tol: float = 1e-4) -> WeldResult:
    """Cluster all edge endpoints into welded corner nodes within ``tol``.

    Deterministic: endpoints are welded to the first-created node within
    tolerance, scanning edges in index order (start endpoint before end).
    """
    corners, assignment = _cluster_terminals(_terminal_points(edges), tol)

    edge_nodes: List[Tuple[int, int]] = []
    incident: List[List[Tuple[int, int]]] = [[] for _ in corners]
    for edge_index in range(len(assignment) // 2):
        head = assignment[2 * edge_index]
        tail = assignment[2 * edge_index + 1]
        edge_nodes.append((head, tail))
        incident[head].append((edge_index, +1))
        incident[tail].append((edge_index, -1))

    return WeldResult(corners, edge_nodes, incident)


# --------------------------------------------------------------------------
# oriented loop assembly
# --------------------------------------------------------------------------

@dataclass
class LoopAssembly:
    """Result of assembling welded edges into oriented loops."""
    loops: List[List[Tuple[int, int]]] = field(default_factory=list)   # closed
    open_chains: List[List[Tuple[int, int]]] = field(default_factory=list)


class _CornerWalker:
    """Consumes a welded graph one chain at a time, never reusing an edge."""

    def __init__(self, weld: WeldResult) -> None:
        self._edge_nodes = weld.edge_nodes
        self._incident = weld.node_incident
        self._spent = [False] * len(weld.edge_nodes)

    def __len__(self) -> int:
        return len(self._edge_nodes)

    def is_spent(self, edge_index: int) -> bool:
        return self._spent[edge_index]

    def _claim_at(self, corner: int) -> Optional[int]:
        """Take the lowest-indexed unspent edge meeting ``corner``, or None."""
        available = [entry for entry in self._incident[corner]
                     if not self._spent[entry[0]]]
        if not available:
            return None
        edge_index = min(available, key=lambda entry: entry[0])[0]
        self._spent[edge_index] = True
        return edge_index

    def _step(self, edge_index: int, corner: int) -> Tuple[int, int]:
        """Orient ``edge_index`` so it leaves ``corner``; return (dir, far end)."""
        head, tail = self._edge_nodes[edge_index]
        if head == corner:
            return +1, tail
        return -1, head

    def walk_from(self, seed: int) -> Tuple[List[Tuple[int, int]], bool]:
        """Grow a chain out of ``seed``; return ``(chain, closed)``.

        The seed is always traversed forwards, which fixes the orientation of
        the whole chain: every later edge is flipped as needed to leave the
        corner the previous one arrived at.  The walk stops when it returns to
        the seed's start corner (closed) or runs out of unspent edges (open).
        """
        origin, frontier = self._edge_nodes[seed]
        self._spent[seed] = True
        chain: List[Tuple[int, int]] = [(seed, +1)]

        while frontier != origin:
            edge_index = self._claim_at(frontier)
            if edge_index is None:
                return chain, False
            direction, frontier = self._step(edge_index, frontier)
            chain.append((edge_index, direction))
        return chain, True


def assemble_loops(edges: Sequence[Edge], tol: float = 1e-4) -> LoopAssembly:
    """Weld ``edges`` and walk them into oriented closed loops.

    Each loop is a list of ``(edge_index, direction)`` where ``direction`` is
    +1 if the edge is traversed start->end and -1 if reversed, so the loop is a
    continuous corner-to-corner cycle.  Edge polylines whose endpoints do not
    close a cycle are returned in :attr:`LoopAssembly.open_chains`.

    Determinism: loops are grown from the lowest unused edge index; at each
    corner the lowest-index unused incident edge is chosen.
    """
    walker = _CornerWalker(weld_endpoints(edges, tol=tol))
    assembly = LoopAssembly()

    for seed in range(len(walker)):
        if walker.is_spent(seed):
            continue
        chain, closed = walker.walk_from(seed)
        bucket = assembly.loops if closed else assembly.open_chains
        bucket.append(chain)

    return assembly
