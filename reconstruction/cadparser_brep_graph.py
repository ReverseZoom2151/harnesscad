"""B-Rep -> graph construction for CADParser's encoder input.

CADParser treats a B-Rep model as a graph ``G = (V, E)`` where the vertices are
the *faces, edges and coedges* and ``E`` encodes their adjacency, following
BRepNet [Lambourne et al., 2021] and UV-Net [Jayaraman et al., 2021]. The learned
"topological walk" convolution runs on top of this graph; building the graph and
its node/edge feature tensors, however, is a pure, deterministic pre-processing
step, and that is what this module provides.

Adjacency follows the paper's ``faces -> edge -> half-edge`` (coedge) relation:

  * each face owns an ordered set of coedges (its boundary loops);
  * each coedge lies on exactly one edge and belongs to exactly one face;
  * the two coedges sharing an edge are *mates* (opposite orientation);
  * consecutive coedges around a loop are *next/previous* neighbours.

Node features are geometry-only (per the paper's edge/coedge choice): a surface- /
curve-type one-hot plus a scalar magnitude (face area proxy, edge length) and, for
coedges, an orientation flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --- input B-Rep description (kernel-agnostic, deterministic) ----------------
SURFACE_TYPES: tuple[str, ...] = ("plane", "cylinder", "cone", "sphere", "torus", "spline")
CURVE_TYPES: tuple[str, ...] = ("line", "circle", "arc", "ellipse", "spline")


@dataclass(frozen=True)
class EdgeDef:
    id: str
    curve_type: str = "line"
    length: float = 0.0


@dataclass(frozen=True)
class FaceDef:
    """A face given by its surface type and its boundary loops.

    Each loop is an ordered tuple of ``(edge_id, orientation)`` coedges, where
    ``orientation`` is True when the coedge runs with the underlying edge.
    """

    id: str
    surface_type: str = "plane"
    area: float = 0.0
    loops: tuple[tuple[tuple[str, bool], ...], ...] = ()


@dataclass(frozen=True)
class BRep:
    faces: tuple[FaceDef, ...]
    edges: tuple[EdgeDef, ...]


# --- graph value objects ----------------------------------------------------
@dataclass(frozen=True)
class Coedge:
    id: str
    face_id: str
    edge_id: str
    orientation: bool
    loop: int
    position: int


@dataclass(frozen=True)
class CADGraph:
    faces: tuple[FaceDef, ...]
    edges: tuple[EdgeDef, ...]
    coedges: tuple[Coedge, ...]
    # node id -> dense index (faces first, then edges, then coedges)
    index: dict = field(default_factory=dict)
    # relation name -> tuple of (src_index, dst_index) pairs
    relations: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.faces) + len(self.edges) + len(self.coedges)


def build_graph(brep: BRep) -> CADGraph:
    """Construct the face/edge/coedge graph with typed adjacency relations."""
    edge_by_id = {e.id: e for e in brep.edges}
    for face in brep.faces:
        for loop in face.loops:
            for edge_id, _ in loop:
                if edge_id not in edge_by_id:
                    raise ValueError(f"face {face.id!r} references unknown edge {edge_id!r}")

    coedges: list[Coedge] = []
    for face in brep.faces:
        for loop_i, loop in enumerate(face.loops):
            for pos, (edge_id, orient) in enumerate(loop):
                cid = f"{face.id}:{loop_i}:{pos}"
                coedges.append(Coedge(cid, face.id, edge_id, bool(orient), loop_i, pos))

    # Stable dense node ordering: faces, then edges, then coedges.
    index: dict[str, int] = {}
    offset = 0
    for face in brep.faces:
        index[("face", face.id)] = offset
        offset += 1
    for edge in brep.edges:
        index[("edge", edge.id)] = offset
        offset += 1
    for coedge in coedges:
        index[("coedge", coedge.id)] = offset
        offset += 1

    face_ci = {("face", c.face_id): [] for c in coedges}
    edge_cs: dict = {}
    for coedge in coedges:
        edge_cs.setdefault(coedge.edge_id, []).append(coedge)

    relations: dict[str, list[tuple[int, int]]] = {
        "face_coedge": [], "coedge_edge": [], "coedge_mate": [],
        "coedge_next": [], "coedge_prev": [],
    }

    def ci(coedge: Coedge) -> int:
        return index[("coedge", coedge.id)]

    for coedge in coedges:
        fi = index[("face", coedge.face_id)]
        ei = index[("edge", coedge.edge_id)]
        relations["face_coedge"].append((fi, ci(coedge)))
        relations["coedge_edge"].append((ci(coedge), ei))

    # Mates: the coedges sharing an edge (opposite orientation) are partners.
    for edge_id, group in edge_cs.items():
        for a in group:
            for b in group:
                if a.id != b.id and a.orientation != b.orientation:
                    relations["coedge_mate"].append((ci(a), ci(b)))

    # Next / previous around each face loop.
    loops: dict = {}
    for coedge in coedges:
        loops.setdefault((coedge.face_id, coedge.loop), []).append(coedge)
    for group in loops.values():
        ordered = sorted(group, key=lambda c: c.position)
        n = len(ordered)
        for i, coedge in enumerate(ordered):
            nxt = ordered[(i + 1) % n]
            prv = ordered[(i - 1) % n]
            relations["coedge_next"].append((ci(coedge), ci(nxt)))
            relations["coedge_prev"].append((ci(coedge), ci(prv)))

    return CADGraph(
        faces=brep.faces, edges=brep.edges, coedges=tuple(coedges),
        index=index,
        relations={k: tuple(sorted(set(v))) for k, v in relations.items()},
    )


def adjacency_matrix(graph: CADGraph, symmetric: bool = True) -> tuple[tuple[int, ...], ...]:
    """Dense 0/1 adjacency over all nodes (the paper's topological-walk support).

    Aggregates every typed relation into one matrix. With ``symmetric`` the matrix
    is made undirected (both directions set), matching message passing that walks
    the face-edge-coedge hierarchy in either direction.
    """
    n = graph.n_nodes
    matrix = [[0] * n for _ in range(n)]
    for pairs in graph.relations.values():
        for src, dst in pairs:
            matrix[src][dst] = 1
            if symmetric:
                matrix[dst][src] = 1
    return tuple(tuple(row) for row in matrix)


# --- geometry-only node features --------------------------------------------
def _onehot(value: str, vocabulary: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(1 if value == item else 0 for item in vocabulary)


def face_feature(face: FaceDef) -> tuple[float, ...]:
    """Surface-type one-hot + area magnitude for one face node."""
    if face.surface_type not in SURFACE_TYPES:
        raise ValueError(f"unknown surface type: {face.surface_type!r}")
    return tuple(float(v) for v in _onehot(face.surface_type, SURFACE_TYPES)) + (float(face.area),)


def edge_feature(edge: EdgeDef) -> tuple[float, ...]:
    """Curve-type one-hot + length magnitude for one edge node."""
    if edge.curve_type not in CURVE_TYPES:
        raise ValueError(f"unknown curve type: {edge.curve_type!r}")
    return tuple(float(v) for v in _onehot(edge.curve_type, CURVE_TYPES)) + (float(edge.length),)


def coedge_feature(coedge: Coedge) -> tuple[float, ...]:
    """Orientation flag for one coedge node (geometry carried by its edge)."""
    return (1.0 if coedge.orientation else 0.0,)


def node_features(graph: CADGraph) -> dict:
    """Per-family feature tensors keyed by node kind, in dense-index order."""
    return {
        "face": tuple(face_feature(f) for f in graph.faces),
        "edge": tuple(edge_feature(e) for e in graph.edges),
        "coedge": tuple(coedge_feature(c) for c in graph.coedges),
    }
