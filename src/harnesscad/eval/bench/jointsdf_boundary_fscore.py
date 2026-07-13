"""Boundary F-score for mesh / point-cloud part segmentation.

The joint SDF paper singles out *boundary precision* as its main limitation
("occasional leaks or frayed borders ... boundaries inherit face-scale
discretization").  A standard deterministic way to score that is a boundary
F-score: identify the set of label boundaries in a prediction and in the ground
truth, then measure precision / recall / F1 of predicted boundaries against GT
boundaries with a small tolerance.

Here a *boundary element* is an adjacency edge whose two endpoints carry
different labels.  Given a shared adjacency (the mesh / kNN graph is fixed, only
the labels differ between pred and GT), this is palette-invariant and fully
deterministic.  A ``tolerance`` in graph hops lets a predicted boundary edge
count as a match if a GT boundary edge lies within that many hops (and vice
versa), tolerating the face-scale slack the paper describes.
"""

from __future__ import annotations


def boundary_edges(adjacency, labels):
    """Set of ``(u, v)`` edges (u < v) whose endpoints have different labels."""
    out = set()
    for u, nbrs in adjacency.items():
        for v in nbrs:
            if labels[u] != labels[v]:
                out.add((u, v) if u < v else (v, u))
    return out


def _incident_nodes(edges):
    nodes = set()
    for u, v in edges:
        nodes.add(u)
        nodes.add(v)
    return nodes


def _within_tolerance(edge, target_nodes, adjacency, tolerance):
    """True if either endpoint of ``edge`` is within ``tolerance`` hops of a
    node incident to a target boundary edge (BFS on ``adjacency``)."""
    if tolerance <= 0:
        u, v = edge
        return u in target_nodes or v in target_nodes
    frontier = set(edge)
    seen = set(frontier)
    for _ in range(tolerance):
        if frontier & target_nodes:
            return True
        nxt = set()
        for n in frontier:
            for m in adjacency.get(n, ()):  # neighbours
                if m not in seen:
                    seen.add(m)
                    nxt.add(m)
        frontier = nxt
    return bool(frontier & target_nodes) or bool(seen & target_nodes)


def boundary_prf(adjacency, pred_labels, gt_labels, *, tolerance=0):
    """Precision / recall / F1 of predicted vs GT label boundaries.

    Returns ``(precision, recall, f1)``.  ``tolerance`` is the graph-hop slack.
    """
    pred_b = boundary_edges(adjacency, pred_labels)
    gt_b = boundary_edges(adjacency, gt_labels)

    gt_nodes = _incident_nodes(gt_b)
    pred_nodes = _incident_nodes(pred_b)

    if not pred_b and not gt_b:
        return (1.0, 1.0, 1.0)

    tp_p = sum(
        1 for e in pred_b if _within_tolerance(e, gt_nodes, adjacency, tolerance)
    )
    tp_r = sum(
        1 for e in gt_b if _within_tolerance(e, pred_nodes, adjacency, tolerance)
    )
    precision = tp_p / len(pred_b) if pred_b else 0.0
    recall = tp_r / len(gt_b) if gt_b else 0.0
    if precision + recall == 0.0:
        return (precision, recall, 0.0)
    f1 = 2 * precision * recall / (precision + recall)
    return (precision, recall, f1)


def boundary_f1(adjacency, pred_labels, gt_labels, *, tolerance=0):
    """Just the boundary F1 (higher is better)."""
    return boundary_prf(adjacency, pred_labels, gt_labels, tolerance=tolerance)[2]
