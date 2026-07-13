"""UV-Net face-adjacency graph carrying UV-grid / U-grid features.

UV-Net (Jayaraman et al., CVPR 2021) turns a solid into a graph whose *nodes*
are the B-rep faces (feature = the ``num_u x num_v x 7`` UV-grid of
:mod:`geometry.uvnet_uv_grid`) and whose *edges* are the B-rep edges shared by
two faces (feature = the ``num_u x 6`` U-grid of :mod:`geometry.uvnet_u_grid`).
``process/solid_to_graph.py`` builds it with ``occwl.graph.face_adjacency`` and
then *drops* every edge that has no curve (cone apex / degenerate seams).

This module rebuilds that construction on top of a plain topological
description -- no OCC, no DGL:

* :class:`FaceEntry` / :class:`EdgeEntry` -- a face with its surface and trim
  loops, an edge with its curve and the pair of faces it joins.
* :func:`build_face_adjacency` -- samples every face and every edge and returns
  a :class:`UVNetGraph` with node features, edge features, and the ``src``/``dst``
  index arrays (one directed pair per B-rep edge, as in the paper).
* :func:`to_bidirectional` -- the symmetric message-passing form.
* :func:`adjacency_matrix`, :func:`degrees`, :func:`connected_components`,
  :func:`is_connected` -- deterministic graph checks (a watertight solid's
  face-adjacency graph is connected).
* :func:`graph_summary` -- shapes + counts, the sanity report the paper's
  preprocessing prints per file.

Note the difference from :mod:`reconstruction.cadparser_brep_graph`: that graph
uses *categorical* entity features (surface type one-hots, coedge orientation).
This one uses *sampled geometry* -- the UV-grids -- as the feature, which is the
whole point of UV-Net.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence, Tuple

from harnesscad.domain.geometry import uvnet_u_grid as ug
from harnesscad.domain.geometry import uvnet_uv_grid as uvg


@dataclass(frozen=True)
class FaceEntry:
    """A B-rep face: a parametric surface plus its trimming loops in ``(u, v)``."""

    surface: Any
    trim_loops: Sequence[Sequence[Tuple[float, float]]] | None = None
    name: str = ""


@dataclass(frozen=True)
class EdgeEntry:
    """A B-rep edge joining ``faces = (i, j)`` (indices into the face list)."""

    curve: Any
    faces: Tuple[int, int] = (0, 0)
    name: str = ""


@dataclass
class UVNetGraph:
    num_nodes: int
    node_features: list = field(default_factory=list)   # per face: uxvx7 grid
    edge_features: list = field(default_factory=list)   # per edge: ux6 grid
    src: list = field(default_factory=list)
    dst: list = field(default_factory=list)
    skipped_edges: list = field(default_factory=list)   # degenerate edge indices

    @property
    def num_edges(self) -> int:
        return len(self.src)


def build_face_adjacency(faces: Sequence[FaceEntry],
                         edges: Sequence[EdgeEntry],
                         curv_num_u: int = 10,
                         surf_num_u: int = 10,
                         surf_num_v: int = 10) -> UVNetGraph:
    """Sample all faces/edges and assemble the featured face-adjacency graph.

    Degenerate edges (:func:`geometry.uvnet_u_grid.is_degenerate`) and edges
    referencing an out-of-range face are skipped, exactly like the paper's
    ``if not edge.has_curve(): continue``; their indices land in
    ``skipped_edges``.  Self-loops (an edge whose two sides are the same face,
    e.g. a cylinder seam) are kept -- occwl's face-adjacency graph records them
    as such -- but never duplicated.
    """
    if surf_num_u < 2 or surf_num_v < 2 or curv_num_u < 2:
        raise ValueError("grids need at least 2 samples per direction")

    graph = UVNetGraph(num_nodes=len(faces))
    for face in faces:
        graph.node_features.append(
            uvg.face_feature_grid(face.surface, surf_num_u, surf_num_v,
                                  trim_loops=face.trim_loops))

    for index, edge in enumerate(edges):
        a, b = edge.faces
        if not (0 <= a < len(faces) and 0 <= b < len(faces)):
            graph.skipped_edges.append(index)
            continue
        if ug.is_degenerate(edge.curve):
            graph.skipped_edges.append(index)
            continue
        graph.edge_features.append(ug.edge_feature_grid(edge.curve, curv_num_u))
        graph.src.append(a)
        graph.dst.append(b)
    return graph


def to_bidirectional(graph: UVNetGraph) -> UVNetGraph:
    """Duplicate every edge in the reverse direction (message passing both ways).

    The reversed copy carries the reversed U-grid (samples flipped, tangents
    negated), which is what the opposite coedge traverses.
    """
    out = UVNetGraph(num_nodes=graph.num_nodes,
                     node_features=list(graph.node_features),
                     skipped_edges=list(graph.skipped_edges))
    for s, d, feat in zip(graph.src, graph.dst, graph.edge_features):
        out.src.append(s)
        out.dst.append(d)
        out.edge_features.append(feat)
        out.src.append(d)
        out.dst.append(s)
        out.edge_features.append(ug.reverse_grid(feat))
    return out


def adjacency_matrix(graph: UVNetGraph, symmetric: bool = True) -> tuple:
    n = graph.num_nodes
    mat = [[0] * n for _ in range(n)]
    for s, d in zip(graph.src, graph.dst):
        mat[s][d] = 1
        if symmetric:
            mat[d][s] = 1
    return tuple(tuple(row) for row in mat)


def degrees(graph: UVNetGraph) -> list:
    """Undirected degree of each face node (self-loops count once)."""
    deg = [0] * graph.num_nodes
    for s, d in zip(graph.src, graph.dst):
        if s == d:
            deg[s] += 1
        else:
            deg[s] += 1
            deg[d] += 1
    return deg


def connected_components(graph: UVNetGraph) -> list:
    """Sorted list of sorted node-index components (deterministic)."""
    neighbours = {i: set() for i in range(graph.num_nodes)}
    for s, d in zip(graph.src, graph.dst):
        neighbours[s].add(d)
        neighbours[d].add(s)
    seen = set()
    comps = []
    for start in range(graph.num_nodes):
        if start in seen:
            continue
        stack = [start]
        comp = []
        seen.add(start)
        while stack:
            node = stack.pop()
            comp.append(node)
            for nb in sorted(neighbours[node]):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append(sorted(comp))
    return sorted(comps)


def is_connected(graph: UVNetGraph) -> bool:
    return graph.num_nodes > 0 and len(connected_components(graph)) == 1


def graph_summary(graph: UVNetGraph) -> dict:
    node_shape = (uvg.grid_shape(graph.node_features[0])
                  if graph.node_features else (0, 0, 0))
    edge_shape = ((len(graph.edge_features[0]), len(graph.edge_features[0][0]))
                  if graph.edge_features else (0, 0))
    return {
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "node_feature_shape": node_shape,
        "edge_feature_shape": edge_shape,
        "skipped_edges": len(graph.skipped_edges),
        "degrees": degrees(graph),
        "connected": is_connected(graph),
    }


def all_masked_points(graph: UVNetGraph) -> list:
    """Every in-face grid point of the graph (the UV-Net point cloud proxy)."""
    pts = []
    for grid in graph.node_features:
        pts.extend(uvg.masked_points(grid))
    return pts
