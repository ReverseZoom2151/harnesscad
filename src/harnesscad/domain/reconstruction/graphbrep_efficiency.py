"""GraphBrep efficiency / compactness metric (paper Sec. 4.2.3).

GraphBrep's headline claim is *efficiency*: an explicit graph topology replaces
BrepGen's tree/fixed-edge representation, shrinking the edge-generation sequence
length and thus the (quadratic) self-attention cost.

BrepGen assigns every surface a *fixed* maximum edge count, so its edge sequence
length is ``n_faces_max * max_edges_per_face`` -- e.g. DeepCAD ``30 * 20 = 600``
at training and ``30 * 30 = 900`` at inference; ABC ``50 * 30 = 1500`` at
training. GraphBrep instead handles only the *actual* edges, giving sequence
lengths of ``120`` (DeepCAD) and ``150`` (ABC). Because attention scales as
``O(L^2)``, the compute saving is quadratic in that reduction.

This module turns those relationships into deterministic metrics: sequence-length
reduction, quadratic attention-cost reduction, redundancy ratio, and a per-model
comparison driven from an actual surface-adjacency matrix. Pure arithmetic,
stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass

from harnesscad.domain.reconstruction.graphbrep_surface_graph import Matrix, total_edges


def tree_sequence_length(max_faces: int, max_edges_per_face: int) -> int:
    """BrepGen fixed edge-sequence length ``n_faces * max_edges_per_face``."""
    if max_faces < 0 or max_edges_per_face < 0:
        raise ValueError("lengths must be non-negative")
    return max_faces * max_edges_per_face


def graph_sequence_length(total_edge_count: int) -> int:
    """GraphBrep edge-sequence length: the actual (max) number of edges."""
    if total_edge_count < 0:
        raise ValueError("edge count must be non-negative")
    return total_edge_count


def attention_cost(length: int) -> int:
    """Self-attention cost, quadratic in sequence length ``L^2``."""
    return length * length


def _reduction(baseline: float, improved: float) -> float:
    """Fractional reduction ``1 - improved / baseline`` in ``[0, 1]``-ish."""
    if baseline == 0:
        return 0.0
    return 1.0 - improved / baseline


def sequence_reduction(tree_length: int, graph_length: int) -> float:
    """Fraction the edge sequence shrinks (e.g. 0.80 for 600 -> 120)."""
    return _reduction(tree_length, graph_length)


def attention_reduction(tree_length: int, graph_length: int) -> float:
    """Fraction the quadratic attention cost shrinks (``1 - (g/t)^2``)."""
    return _reduction(attention_cost(tree_length), attention_cost(graph_length))


def redundancy_ratio(tree_length: int, graph_length: int) -> float:
    """How many times longer the tree sequence is than the graph one."""
    if graph_length == 0:
        raise ValueError("graph_length must be positive")
    return tree_length / graph_length


@dataclass(frozen=True)
class EfficiencyReport:
    tree_length: int
    graph_length: int
    tree_attention: int
    graph_attention: int
    sequence_reduction: float
    attention_reduction: float
    redundancy_ratio: float


def compare(max_faces: int, max_edges_per_face: int, total_edge_count: int) -> EfficiencyReport:
    """Full tree-vs-graph efficiency comparison from the size parameters."""
    tree = tree_sequence_length(max_faces, max_edges_per_face)
    graph = graph_sequence_length(total_edge_count)
    return EfficiencyReport(
        tree_length=tree,
        graph_length=graph,
        tree_attention=attention_cost(tree),
        graph_attention=attention_cost(graph),
        sequence_reduction=sequence_reduction(tree, graph),
        attention_reduction=attention_reduction(tree, graph),
        redundancy_ratio=redundancy_ratio(tree, graph) if graph else float("inf"),
    )


def compare_model(matrix: Matrix, max_edges_per_face: int) -> EfficiencyReport:
    """Efficiency comparison for one concrete surface-adjacency matrix.

    The tree baseline pads each of the model's surfaces to ``max_edges_per_face``
    edges; the graph representation stores only the model's real edges (the sum
    over the adjacency upper triangle).
    """
    n_faces = len(matrix)
    return compare(n_faces, max_edges_per_face, total_edges(matrix))
