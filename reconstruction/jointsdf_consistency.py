"""Segmentation-consistency metric from Fan & Musialski's joint SDF paper.

The paper introduces a *Segmentation Consistency* metric that measures local
label smoothness in prediction space and is invariant to palette permutations.
For predicted samples ``P = {(x_i, l_i)}`` it computes, for a set of anchors,

    c_i = (1 / k) * sum_{j in N_k(x_i)} 1[l_j == l_i]

where ``N_k(x_i)`` are the k nearest *other* samples of ``x_i`` in 3D, and
averages ``c_i`` over ``M = min(1000, |P|)`` anchors (k = 10 in the paper).

Everything here is deterministic given the point coordinates and labels: no
learned network is involved.  The learned SDF / segmentation heads are external;
we only score their *outputs*.

This module also adds an SDF-aware variant that restricts the consistency
computation to on-surface samples (``|sdf| <= tau``), which is the natural
consistency check between a segmentation head and the SDF trunk of a joint
network: labels are only meaningful on the reconstructed surface.
"""

from __future__ import annotations


def _sqdist(a, b):
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return s


def _knn_indices(points, anchor_index, k):
    """Indices of the ``k`` nearest points to ``points[anchor_index]``.

    Ties are broken by ascending index so the result is fully deterministic.
    The anchor itself is excluded.
    """
    ax = points[anchor_index]
    order = []
    for j, p in enumerate(points):
        if j == anchor_index:
            continue
        order.append((_sqdist(ax, p), j))
    order.sort()
    return [j for _, j in order[:k]]


def _anchor_indices(n, m, rng):
    """The first ``m`` anchor indices.

    With ``rng`` given (a ``random.Random``) a deterministic sample without
    replacement is drawn; otherwise the natural prefix ``0..m-1`` is used so the
    metric is reproducible without any randomness.
    """
    m = min(m, n)
    if rng is None:
        return list(range(m))
    idx = list(range(n))
    rng.shuffle(idx)
    return sorted(idx[:m])


def per_anchor_consistency(points, labels, *, k=10, max_anchors=1000, rng=None):
    """Return ``(anchor_index, c_i)`` pairs for the evaluated anchors."""
    n = len(points)
    if n != len(labels):
        raise ValueError("points and labels length mismatch")
    if n == 0:
        return []
    if k < 1:
        raise ValueError("k must be >= 1")
    anchors = _anchor_indices(n, max_anchors, rng)
    out = []
    for i in anchors:
        neigh = _knn_indices(points, i, k)
        if not neigh:
            out.append((i, 1.0))
            continue
        same = sum(1 for j in neigh if labels[j] == labels[i])
        out.append((i, same / len(neigh)))
    return out


def segmentation_consistency(points, labels, *, k=10, max_anchors=1000, rng=None):
    """Mean local label agreement ``c`` over the evaluated anchors.

    Higher is better (1.0 = every neighbourhood is single-label).  Invariant to
    any relabelling (palette permutation) of ``labels`` because only label
    *equality* is used.
    """
    per = per_anchor_consistency(
        points, labels, k=k, max_anchors=max_anchors, rng=rng
    )
    if not per:
        return 1.0
    return sum(c for _, c in per) / len(per)


def surface_consistency(points, labels, sdf, *, tau, k=10, max_anchors=1000, rng=None):
    """SDF-segmentation consistency on the on-surface band ``|sdf| <= tau``.

    Neighbours are searched only among on-surface samples, so the score reflects
    label smoothness of the segmentation head restricted to the SDF trunk's
    zero-level set.  Returns ``(consistency, n_surface)``.
    """
    if len(points) != len(sdf):
        raise ValueError("points and sdf length mismatch")
    keep = [i for i, s in enumerate(sdf) if abs(s) <= tau]
    pts = [points[i] for i in keep]
    labs = [labels[i] for i in keep]
    return (
        segmentation_consistency(pts, labs, k=k, max_anchors=max_anchors, rng=rng),
        len(keep),
    )
