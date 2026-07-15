"""Text-enhanced primitive graph with type-aware edge features for symbol spotting.

Paper: *Text-Enhanced Panoptic Symbol Spotting in CAD Drawings* (Liu et al.,
2025). Beyond the geometry-only CADTransformer graph (already reimplemented in
:mod:`primitive_graph`), this paper adds two deterministic, handcrafted pieces
that a downstream network consumes as fixed inputs (Sec. III):

  * a **Text Primitives Integration** step -- text annotations become a distinct
    node type in the primitive graph, but only after *low-frequency annotations
    are eliminated* against a corpus-statistics threshold, "ensuring that only
    representative and commonly used textual labels contribute to the graph";
  * a **type-aware edge feature** encoding -- for each node and its k nearest
    neighbours, an edge feature ``E = (t || e)`` where ``t`` is a type indicator
    over the pair category (geometry-geometry, geometry-text, text-text) and
    ``e`` is a geometric relation vector (relative distance, position, angle).

Both are deterministic and stdlib-only; the neural type-aware *attention* that
consumes them is out of scope (trained weights). This module builds exactly the
fixed inputs: annotation filtering, the unified node set, KNN neighbours over
node centres, and the 3+7 = 10-D type-aware edge feature per (node, neighbour).

It is distinct from :mod:`primitive_graph` (geometry nodes + endpoint KNN, no
text, no typed edges) and from :mod:`annotation_parser` (text extraction only).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

__all__ = [
    "Node",
    "annotation_counts",
    "filter_annotations",
    "build_nodes",
    "knn_neighbors",
    "edge_type_onehot",
    "edge_relation",
    "type_aware_edge_features",
]

# pair-type indicator order: geometry-geometry, geometry-text, text-text.
_PAIR_TYPES = ("gg", "gt", "tt")


@dataclass
class Node:
    """A graph node: a geometry primitive or a text annotation.

    ``kind`` is ``"geom"`` or ``"text"``; ``cx``/``cy`` is the node centre;
    ``orient`` is a primitive orientation angle (radians) used for the relation
    vector (0 for text, which has no intrinsic direction); ``label`` carries the
    text string for text nodes (empty for geometry).
    """

    kind: str
    cx: float
    cy: float
    orient: float = 0.0
    label: str = ""


def annotation_counts(labels) -> Counter:
    """Corpus frequency of every text label (the statistics the threshold uses)."""
    return Counter(labels)


def filter_annotations(labels, min_count: int) -> list:
    """Keep only labels whose corpus frequency is >= ``min_count``.

    Returns the surviving labels in input order. This is the paper's low-frequency
    elimination: rare, idiosyncratic annotations are dropped as noise before they
    become graph nodes.
    """
    counts = annotation_counts(labels)
    return [lbl for lbl in labels if counts[lbl] >= min_count]


def build_nodes(geom_centers, text_items, min_count: int = 1) -> list:
    """Assemble the unified node list from geometry centres and text items.

    ``geom_centers`` is a sequence of ``(cx, cy)`` or ``(cx, cy, orient)``.
    ``text_items`` is a sequence of ``(cx, cy, label)``. Text nodes whose label
    is below the frequency threshold are dropped (paper's integration step).
    """
    nodes: list = []
    for g in geom_centers:
        cx, cy = g[0], g[1]
        orient = g[2] if len(g) > 2 else 0.0
        nodes.append(Node(kind="geom", cx=cx, cy=cy, orient=orient))
    labels = [t[2] for t in text_items]
    counts = annotation_counts(labels)
    for cx, cy, label in text_items:
        if counts[label] >= min_count:
            nodes.append(Node(kind="text", cx=cx, cy=cy, label=label))
    return nodes


def _sqdist(a: Node, b: Node) -> float:
    return (a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2


def knn_neighbors(nodes, k: int) -> list:
    """For each node, the indices of its ``k`` nearest neighbours by centre distance.

    Deterministic: ties break by ascending index. Returns a list of neighbour
    index lists parallel to ``nodes`` (excluding the node itself).
    """
    out: list = []
    for i, ni in enumerate(nodes):
        order = sorted(
            (j for j in range(len(nodes)) if j != i),
            key=lambda j: (_sqdist(ni, nodes[j]), j),
        )
        out.append(order[:k])
    return out


def edge_type_onehot(a: Node, b: Node) -> tuple:
    """3-D one-hot of the pair category (gg / gt / tt), order :data:`_PAIR_TYPES`."""
    if a.kind == "text" and b.kind == "text":
        key = "tt"
    elif a.kind == "text" or b.kind == "text":
        key = "gt"
    else:
        key = "gg"
    return tuple(1.0 if pt == key else 0.0 for pt in _PAIR_TYPES)


def edge_relation(a: Node, b: Node, diag: float = 1.0) -> tuple:
    """7-D geometric relation vector from node ``a`` to neighbour ``b``.

    Components (relative distance, position and angle, per the paper): normalized
    centre distance, normalized dx, dy, the bearing from a to b (sin, cos), and
    the two nodes' orientations relative to that bearing (their sin difference).
    ``diag`` normalizes distances (e.g. the drawing/block diagonal).
    """
    dx = b.cx - a.cx
    dy = b.cy - a.cy
    dist = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx)
    d = diag if diag else 1.0
    return (
        dist / d,
        dx / d,
        dy / d,
        math.sin(bearing),
        math.cos(bearing),
        math.sin(a.orient - bearing),
        math.sin(b.orient - bearing),
    )


def type_aware_edge_features(nodes, neighbors=None, k: int = 16,
                             diag: float = 1.0) -> list:
    """Per node, the list of ``(t || e)`` 10-D edge features to its neighbours.

    If ``neighbors`` is None it is computed via :func:`knn_neighbors` with degree
    ``k``. Returns a list parallel to ``nodes``; each element is a list of
    ``(neighbor_index, feature_tuple)`` pairs, ``feature_tuple`` being the 3-D
    type one-hot concatenated with the 7-D relation vector. Deterministic.
    """
    if neighbors is None:
        neighbors = knn_neighbors(nodes, k)
    out: list = []
    for i, ni in enumerate(nodes):
        feats = []
        for j in neighbors[i]:
            t = edge_type_onehot(ni, nodes[j])
            e = edge_relation(ni, nodes[j], diag=diag)
            feats.append((j, t + e))
        out.append(feats)
    return out
