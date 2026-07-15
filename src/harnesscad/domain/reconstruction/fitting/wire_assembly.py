"""Weld a bag of edge polylines into oriented closed loops (faceformer wires).

Ported from *faceformer* (``reconstruction/reconstruction_utils.py``:
``construct_connected_cycle`` / ``construct_connected_cylinder`` /
``find_circle_center``).  After faceformer predicts a set of B-Rep edges as
independent 3D polylines, they are an unordered *bag*: to turn them into face
boundaries it welds edge endpoints that coincide (within a tolerance) into shared
corners, then walks the resulting corner graph to assemble oriented **closed
loops**, recording for each edge whether it was traversed forward (+1) or
reversed (-1).  It also fits an exact circle through three points to recover arc
geometry.  All of that is pure, deterministic geometry; the neural predictor is
not, and is not needed here.

Why this helps the harness: reconstruction produces predicted edges that must be
assembled into wires before a face/solid can be built.  The existing chain-complex
loop extraction (``reconstruction/brep/chain_complex.py``) works on *precomputed
integer incidence* (which curve touches which corner); it cannot start from raw
coordinate polylines.  This module does the geometric step *before* that: it
welds coincident endpoints and derives the incidence, then assembles oriented
loops -- the missing front end of wire reconstruction.  It reuses nothing from,
and duplicates nothing in, the index-based complex.

Contents:

* :func:`circle_from_three_points` -- exact 3D circumcircle (centre, radius,
  unit normal) via faceformer's ``find_circle_center`` identity.
* :func:`weld_endpoints` -- cluster edge endpoints into welded corner nodes.
* :func:`assemble_loops` -- walk the welded graph into oriented closed loops,
  each a list of ``(edge_index, direction)``; leftover open chains are reported
  separately.

Pure stdlib, deterministic (edges processed in index order; ties broken by id).
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


def _sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def _dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])
def _add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def _scale(a, s): return (a[0] * s, a[1] * s, a[2] * s)
def _dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


# --------------------------------------------------------------------------
# exact 3D circle through three points (faceformer find_circle_center)
# --------------------------------------------------------------------------

def circle_from_three_points(
    p1: Sequence[float], p2: Sequence[float], p3: Sequence[float]
) -> Tuple[Vec3, float, Vec3]:
    """Return ``(center, radius, unit_normal)`` of the circle through p1,p2,p3.

    Uses faceformer's closed-form identity.  Raises ``ValueError`` if the three
    points are collinear (degenerate circle).
    """
    p1 = tuple(float(c) for c in p1)
    p2 = tuple(float(c) for c in p2)
    p3 = tuple(float(c) for c in p3)
    t = _sub(p2, p1)
    u = _sub(p3, p1)
    v = _sub(p3, p2)
    w = _cross(t, u)
    wsl = _dot(w, w)
    if wsl <= 1e-18:
        raise ValueError("collinear points have no unique circle")
    iwsl2 = 1.0 / (2.0 * wsl)
    tt = _dot(t, t)
    uu = _dot(u, u)
    center = _add(
        p1,
        _scale(
            _sub(_scale(u, tt * _dot(u, v)), _scale(t, uu * _dot(t, v))),
            iwsl2,
        ),
    )
    radius = math.sqrt(tt * uu * _dot(v, v) * iwsl2 / 2.0)
    inv = 1.0 / math.sqrt(wsl)
    normal = _scale(w, inv)
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


def _endpoints(edge: Edge) -> Tuple[Vec3, Vec3]:
    start = tuple(float(c) for c in edge[0])
    end = tuple(float(c) for c in edge[-1])
    return start, end  # type: ignore


def weld_endpoints(edges: Sequence[Edge], tol: float = 1e-4) -> WeldResult:
    """Cluster all edge endpoints into welded corner nodes within ``tol``.

    Deterministic: endpoints are welded to the first-created node within
    tolerance, scanning edges in index order (start endpoint before end).
    """
    node_coords: List[Vec3] = []
    edge_nodes: List[Tuple[int, int]] = []
    incident: List[List[Tuple[int, int]]] = []

    def find_or_add(pt: Vec3) -> int:
        for idx, c in enumerate(node_coords):
            if _dist(pt, c) < tol:
                return idx
        node_coords.append(pt)
        incident.append([])
        return len(node_coords) - 1

    for ei, edge in enumerate(edges):
        s, e = _endpoints(edge)
        sn = find_or_add(s)
        en = find_or_add(e)
        edge_nodes.append((sn, en))
        incident[sn].append((ei, +1))
        incident[en].append((ei, -1))

    return WeldResult(node_coords, edge_nodes, incident)


# --------------------------------------------------------------------------
# oriented loop assembly
# --------------------------------------------------------------------------

@dataclass
class LoopAssembly:
    """Result of assembling welded edges into oriented loops."""
    loops: List[List[Tuple[int, int]]] = field(default_factory=list)   # closed
    open_chains: List[List[Tuple[int, int]]] = field(default_factory=list)


def assemble_loops(edges: Sequence[Edge], tol: float = 1e-4) -> LoopAssembly:
    """Weld ``edges`` and walk them into oriented closed loops.

    Each loop is a list of ``(edge_index, direction)`` where ``direction`` is
    +1 if the edge is traversed start->end and -1 if reversed, so the loop is a
    continuous corner-to-corner cycle.  Edge polylines whose endpoints do not
    close a cycle are returned in :attr:`LoopAssembly.open_chains`.

    Determinism: loops are grown from the lowest unused edge index; at each
    corner the lowest-index unused incident edge is chosen.
    """
    weld = weld_endpoints(edges, tol=tol)
    edge_nodes = weld.edge_nodes
    incident = weld.node_incident
    n_edges = len(edge_nodes)
    used = [False] * n_edges
    result = LoopAssembly()

    def next_edge_from(node: int) -> Optional[Tuple[int, int]]:
        best: Optional[Tuple[int, int]] = None
        for (ei, d) in incident[node]:
            if not used[ei]:
                if best is None or ei < best[0]:
                    best = (ei, d)
        return best

    for seed in range(n_edges):
        if used[seed]:
            continue
        start_node, end_node = edge_nodes[seed]
        chain: List[Tuple[int, int]] = [(seed, +1)]
        used[seed] = True
        current = end_node
        closed = False
        while True:
            if current == start_node:
                closed = True
                break
            nxt = next_edge_from(current)
            if nxt is None:
                break
            ei, _d = nxt
            used[ei] = True
            a, b = edge_nodes[ei]
            if a == current:
                chain.append((ei, +1))
                current = b
            else:
                chain.append((ei, -1))
                current = a
        if closed:
            result.loops.append(chain)
        else:
            result.open_chains.append(chain)

    return result
