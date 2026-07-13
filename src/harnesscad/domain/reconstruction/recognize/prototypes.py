"""Deterministic multi-prototype nearest-prototype inference for few-shot part segmentation.

The learned feature *extractor* of Wang et al. is external, but the paper's
inference-time classifier is a fully deterministic non-parametric method: a
multi-prototype network (an extension of Snell et al.'s prototypical network).
Given support-point features and their labels, it

  1. selects ``n_prototypes`` anchor points per class with Farthest Point
     Sampling (FPS),
  2. assigns each class point to its nearest anchor and averages the members to
     form the prototype (Eq. 5 of the paper), and
  3. labels a query point by the class of its nearest prototype.

``n_prototypes = 1`` degenerates to the classic single-prototype network (class
mean). A dedicated background class is supported by simply passing it as one of
the label values. All ties break by index / label ordering so the result is
reproducible.
"""

from __future__ import annotations


def _sqdist(a, b):
    return sum((a[d] - b[d]) ** 2 for d in range(len(a)))


def farthest_point_sampling(features, count, seed=0):
    """Return indices of ``count`` FPS anchors over ``features``.

    The first seed point is chosen deterministically as ``seed % n`` so no wall
    clock or global RNG is involved; each subsequent anchor maximises the
    minimum distance to the already-chosen set (ties broken by index).
    """
    n = len(features)
    if count <= 0 or n == 0:
        return ()
    count = min(count, n)
    start = seed % n
    chosen = [start]
    min_dist = [_sqdist(features[i], features[start]) for i in range(n)]
    while len(chosen) < count:
        best_i, best_d = -1, -1.0
        for i in range(n):
            if i in chosen:
                continue
            if min_dist[i] > best_d:
                best_d, best_i = min_dist[i], i
        if best_i < 0:
            break
        chosen.append(best_i)
        for i in range(n):
            d = _sqdist(features[i], features[best_i])
            if d < min_dist[i]:
                min_dist[i] = d
    return tuple(chosen)


def _mean(vectors):
    m = len(vectors)
    dim = len(vectors[0])
    return tuple(sum(v[d] for v in vectors) / m for d in range(dim))


def class_prototypes(features, indices, n_prototypes, seed=0):
    """Compute up to ``n_prototypes`` prototypes for one class.

    ``indices`` are the positions in ``features`` belonging to the class. FPS
    picks anchors, each class point is bucketed to its nearest anchor, and the
    prototype is the mean of its bucket. Empty buckets are dropped, so the
    number of returned prototypes may be smaller than ``n_prototypes`` for tiny
    classes (matching the paper's arrangement of points to closest anchors).
    """
    if not indices:
        return ()
    class_feats = [features[i] for i in indices]
    if n_prototypes <= 1:
        return (_mean(class_feats),)
    anchor_local = farthest_point_sampling(class_feats, n_prototypes, seed)
    buckets = {a: [] for a in anchor_local}
    for cf in class_feats:
        best_a, best_d = anchor_local[0], _sqdist(cf, class_feats[anchor_local[0]])
        for a in anchor_local[1:]:
            d = _sqdist(cf, class_feats[a])
            if d < best_d:
                best_d, best_a = d, a
        buckets[best_a].append(cf)
    protos = []
    for a in anchor_local:
        if buckets[a]:
            protos.append(_mean(buckets[a]))
    return tuple(protos)


def build_prototypes(support_features, support_labels, n_prototypes=1, seed=0):
    """Prototypes for every class label present in the support set.

    Returns a tuple of ``(label, prototype_vector)`` pairs, sorted by label then
    by insertion order, so ``n_prototypes`` prototypes can represent each class.
    """
    if len(support_features) != len(support_labels):
        raise ValueError("features and labels length mismatch")
    by_label = {}
    for i, lab in enumerate(support_labels):
        by_label.setdefault(lab, []).append(i)
    out = []
    for lab in sorted(by_label, key=repr):
        for proto in class_prototypes(support_features, by_label[lab],
                                      n_prototypes, seed):
            out.append((lab, proto))
    return tuple(out)


def assign(query_features, prototypes):
    """Nearest-prototype label for each query point.

    ``prototypes`` is the ``(label, vector)`` sequence from ``build_prototypes``.
    Ties break by prototype order (which is label-sorted), so identical distances
    resolve to the lower label deterministically.
    """
    if not prototypes:
        raise ValueError("no prototypes")
    labels = []
    for qf in query_features:
        best_lab, best_d = prototypes[0][0], _sqdist(qf, prototypes[0][1])
        for lab, proto in prototypes[1:]:
            d = _sqdist(qf, proto)
            if d < best_d:
                best_d, best_lab = d, lab
        labels.append(best_lab)
    return tuple(labels)


def segment(support_features, support_labels, query_features,
            n_prototypes=1, seed=0):
    """End-to-end: build prototypes from the support set, label the query set."""
    protos = build_prototypes(support_features, support_labels,
                              n_prototypes, seed)
    return assign(query_features, protos)
