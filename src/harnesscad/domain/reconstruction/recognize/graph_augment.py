"""Part-graph representation and self-supervised graph augmentation.

A self-supervised graph neural network for mechanical CAD retrieval converts a
B-Rep part into a simple graph where *faces are nodes and curves are edges*, then
trains a GNN with a contrastive pretext task built entirely from **graph
augmentation**. The GNN weights are external; the graph construction and the
augmentation operators are deterministic pre-processing and that is what this
module provides.

This graph is deliberately simpler than a heterogeneous face/edge/coedge graph
(:mod:`reconstruction.cadparser_brep_graph`) or a weighted surface-adjacency
matrix (:mod:`reconstruction.graphbrep_surface_graph`): here every curve connects
*exactly two* faces, so the graph is an ordinary undirected graph with a feature
vector on every node (face) and every edge (curve).

A graph is augmented in two aspects:

* **Feature masking** (ratio ``alpha``) -- for each node and edge, randomly mask
  part of the original feature vector to zero. Corresponds to modifying the local
  shape of the CAD part.
* **Structure augmentation** (ratio ``beta``) -- three delete-only schemes:

  1. ``"node"``          -- randomly remove nodes (and their incident edges);
  2. ``"node_1hop"``     -- randomly remove nodes together with their 1-hop
                            neighbours (and all incident edges);
  3. ``"edge_vertices"`` -- randomly remove edges and the two nodes each is
                            incident to.

  Only *deletions* are performed: adding random faces/curves would produce an
  illogical (self-penetrating) CAD part.

Applying feature masking then structure augmentation twice to one graph yields
the positive pair ``(g', g'')`` whose embeddings ``z'`` and ``z''`` feed the
graph-level NT-Xent loss in :mod:`bench.ssgnn_graph_contrastive`.

Everything is deterministic given ``random.Random(seed)`` and stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Sequence, Tuple

Feature = Tuple[float, ...]

STRUCTURE_SCHEMES: Tuple[str, ...] = ("node", "node_1hop", "edge_vertices")


@dataclass(frozen=True)
class CADGraph:
    """A part graph: faces are nodes, curves are edges.

    ``node_features[i]`` is the feature vector of face ``i``. ``edges`` is a tuple
    of ``(u, v, feature)`` triples with ``u != v`` node indices (each curve is
    adjacent to exactly two faces). The graph is treated as undirected.
    """

    node_features: Tuple[Feature, ...]
    edges: Tuple[Tuple[int, int, Feature], ...]

    @property
    def n_nodes(self) -> int:
        return len(self.node_features)

    @property
    def n_edges(self) -> int:
        return len(self.edges)


def build_graph(node_features: Sequence[Sequence[float]],
                edges: Sequence[Tuple[int, int, Sequence[float]]]) -> CADGraph:
    """Validate and freeze a face-node / curve-edge CAD graph.

    Rejects edges whose endpoints are out of range or equal (a curve must join two
    distinct faces, matching the "every edge connects exactly two nodes"
    assumption after seam-splitting).
    """
    nodes = tuple(tuple(float(x) for x in f) for f in node_features)
    n = len(nodes)
    out_edges: List[Tuple[int, int, Feature]] = []
    for u, v, feat in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge references node out of range: {(u, v)!r}")
        if u == v:
            raise ValueError("self-edge: a curve must join two distinct faces")
        a, b = (u, v) if u <= v else (v, u)
        out_edges.append((a, b, tuple(float(x) for x in feat)))
    return CADGraph(nodes, tuple(out_edges))


# --- adjacency helpers -------------------------------------------------------
def neighbours(graph: CADGraph) -> Tuple[frozenset, ...]:
    """Per-node set of adjacent node indices (undirected)."""
    adj: List[set] = [set() for _ in range(graph.n_nodes)]
    for u, v, _ in graph.edges:
        adj[u].add(v)
        adj[v].add(u)
    return tuple(frozenset(s) for s in adj)


def degrees(graph: CADGraph) -> Tuple[int, ...]:
    """Number of incident edges per node (parallel curves counted separately)."""
    deg = [0] * graph.n_nodes
    for u, v, _ in graph.edges:
        deg[u] += 1
        deg[v] += 1
    return tuple(deg)


# --- feature masking (ratio alpha) -------------------------------------------
def mask_features(graph: CADGraph, ratio: float, rng: random.Random) -> CADGraph:
    """Randomly zero each node/edge feature coordinate with probability ``ratio``.

    Implements the feature augmentation: for each node and edge on the graph, part
    of the original features is randomly masked. A ratio of 0 returns an identical
    graph; the topology is untouched.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be in [0, 1]")

    def _mask(feat: Feature) -> Feature:
        return tuple(0.0 if rng.random() < ratio else x for x in feat)

    nodes = tuple(_mask(f) for f in graph.node_features)
    edges = tuple((u, v, _mask(f)) for u, v, f in graph.edges)
    return CADGraph(nodes, edges)


# --- structure augmentation (ratio beta) -------------------------------------
def subgraph(graph: CADGraph, keep: Sequence[int]) -> Tuple[CADGraph, Tuple[int, ...]]:
    """Induced subgraph over the kept node indices, with reindexing.

    Returns the new graph and the tuple of original node ids in new-index order.
    Edges with any endpoint removed are dropped.
    """
    keep_sorted = tuple(sorted(set(keep)))
    remap = {old: new for new, old in enumerate(keep_sorted)}
    nodes = tuple(graph.node_features[i] for i in keep_sorted)
    edges = tuple(
        (remap[u], remap[v], f)
        for u, v, f in graph.edges
        if u in remap and v in remap
    )
    return CADGraph(nodes, edges), keep_sorted


def _n_to_remove(n: int, ratio: float) -> int:
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be in [0, 1]")
    return int(round(ratio * n))


def remove_nodes(graph: CADGraph, ratio: float, rng: random.Random) -> CADGraph:
    """Scheme 1: randomly remove ``round(ratio * n)`` nodes and incident edges."""
    n = graph.n_nodes
    k = _n_to_remove(n, ratio)
    if k <= 0:
        return graph
    if k >= n:
        return CADGraph((), ())
    victims = set(rng.sample(range(n), k))
    keep = [i for i in range(n) if i not in victims]
    return subgraph(graph, keep)[0]


def remove_nodes_1hop(graph: CADGraph, ratio: float, rng: random.Random) -> CADGraph:
    """Scheme 2: remove seed nodes together with their 1-hop neighbours.

    ``round(ratio * n)`` seed nodes are chosen; each seed and every node adjacent
    to it are deleted (along with their incident edges). Because whole
    neighbourhoods vanish, this scheme removes at least as many nodes as scheme 1
    for the same ratio -- it is the most aggressive of the three schemes.
    """
    n = graph.n_nodes
    k = _n_to_remove(n, ratio)
    if k <= 0:
        return graph
    adj = neighbours(graph)
    seeds = rng.sample(range(n), min(k, n))
    victims: set = set()
    for s in seeds:
        victims.add(s)
        victims.update(adj[s])
    keep = [i for i in range(n) if i not in victims]
    if not keep:
        return CADGraph((), ())
    return subgraph(graph, keep)[0]


def remove_edges_with_vertices(graph: CADGraph, ratio: float,
                               rng: random.Random) -> CADGraph:
    """Scheme 3: remove ``round(ratio * m)`` edges and their two incident nodes.

    Corresponds to deleting curves together with the two faces they join.
    """
    m = graph.n_edges
    k = _n_to_remove(m, ratio)
    if k <= 0 or m == 0:
        return graph
    chosen = rng.sample(range(m), min(k, m))
    victims: set = set()
    for e in chosen:
        u, v, _ = graph.edges[e]
        victims.add(u)
        victims.add(v)
    keep = [i for i in range(graph.n_nodes) if i not in victims]
    if not keep:
        return CADGraph((), ())
    return subgraph(graph, keep)[0]


_STRUCTURE_OPS = {
    "node": remove_nodes,
    "node_1hop": remove_nodes_1hop,
    "edge_vertices": remove_edges_with_vertices,
}


def augment(graph: CADGraph, rng: random.Random, *, scheme: str = "node",
            feature_ratio: float = 0.1, structure_ratio: float = 0.1) -> CADGraph:
    """Apply feature masking (``alpha``) then structure augmentation (``beta``).

    ``scheme`` selects one of :data:`STRUCTURE_SCHEMES`. A grid search over the
    ratios found ``feature_ratio = structure_ratio = 0.1`` and ``scheme = "node"``
    best. Deterministic given ``rng``.
    """
    if scheme not in _STRUCTURE_OPS:
        raise ValueError(f"unknown scheme {scheme!r}; choose from {STRUCTURE_SCHEMES}")
    masked = mask_features(graph, feature_ratio, rng)
    return _STRUCTURE_OPS[scheme](masked, structure_ratio, rng)


def positive_pair(graph: CADGraph, seed, *, scheme: str = "node",
                  feature_ratio: float = 0.1, structure_ratio: float = 0.1
                  ) -> Tuple[CADGraph, CADGraph]:
    """Two independent augmentations of one graph -> a contrastive positive pair.

    Both views come from one seeded RNG stream, so ``(g', g'')`` is byte
    reproducible from ``seed``. These are the augmented graphs whose embeddings
    ``z'`` and ``z''`` form a positive pair in the NT-Xent objective; every other
    part in the batch is a negative.
    """
    rng = random.Random(seed)
    view_a = augment(graph, rng, scheme=scheme, feature_ratio=feature_ratio,
                     structure_ratio=structure_ratio)
    view_b = augment(graph, rng, scheme=scheme, feature_ratio=feature_ratio,
                     structure_ratio=structure_ratio)
    return view_a, view_b
