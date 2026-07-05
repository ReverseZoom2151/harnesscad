"""Deterministic transductive label propagation for few-shot part segmentation.

Section II-D's classification head is a graph-based label propagation algorithm
(LPA, Algorithm 1). It is a *non-parametric* transductive inference step -- no
optimal parameters are trained -- so it is deterministic and buildable given a
feature graph. This module implements both forms the paper cites:

  * the iterative update ``Z_{t+1} = alpha S Z_t + (1 - alpha) Y`` (Eq. 7), and
  * the closed-form optimum ``Z* = (I - alpha S)^{-1} Y`` (Eq. 8),

where ``S = D^{-1/2}(W + W^T)D^{-1/2}`` is the symmetric-normalised affinity of a
Gaussian kNN graph (Eq. of Algorithm 1, step 1). Final class probabilities use
the softmax of Eq. 9. Labelled rows are one-hot; unlabelled rows are zero and
receive propagated labels. A small Gauss-Jordan solver keeps this stdlib-only.
"""

from __future__ import annotations

import math


def _sqdist(a, b):
    return sum((a[d] - b[d]) ** 2 for d in range(len(a)))


def gaussian_knn_weights(features, k, sigma=1.0):
    """Symmetric Gaussian kNN affinity matrix ``W``.

    ``w_ij = exp(-||Vi - Vj||^2 / sigma^2)`` for j among i's k nearest
    neighbours (Algorithm 1). The graph is symmetrised (union of directed
    edges), the diagonal is zero, and ties break by index.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    n = len(features)
    limit = min(k, n - 1) if n else 0
    w = [[0.0] * n for _ in range(n)]
    for i in range(n):
        order = sorted((j for j in range(n) if j != i),
                       key=lambda j: (_sqdist(features[i], features[j]), j))
        for j in order[:limit]:
            val = math.exp(-_sqdist(features[i], features[j]) / (sigma ** 2))
            w[i][j] = max(w[i][j], val)
            w[j][i] = max(w[j][i], val)
    return w


def symmetric_normalise(w):
    """Return ``S = D^{-1/2}(W + W^T)D^{-1/2}``.

    ``W`` produced above is already symmetric, so ``W + W^T = 2W``; the routine
    still symmetrises defensively for arbitrary inputs.
    """
    n = len(w)
    sym = [[w[i][j] + w[j][i] for j in range(n)] for i in range(n)]
    deg = [sum(sym[i]) for i in range(n)]
    inv_sqrt = [1.0 / math.sqrt(d) if d > 0 else 0.0 for d in deg]
    return [[sym[i][j] * inv_sqrt[i] * inv_sqrt[j] for j in range(n)]
            for i in range(n)]


def one_hot(labels, n_classes):
    """Build the class matrix ``Y``: labelled rows one-hot, unlabelled zero.

    ``labels`` entries of ``None`` mark unlabelled points; integer entries in
    ``[0, n_classes)`` mark labelled points.
    """
    y = []
    for lab in labels:
        row = [0.0] * n_classes
        if lab is not None:
            if not 0 <= lab < n_classes:
                raise ValueError("label out of range")
            row[lab] = 1.0
        y.append(row)
    return y


def propagate_iterative(s, y, alpha=0.99, epochs=50):
    """Iterative label propagation ``Z_{t+1} = alpha S Z_t + (1 - alpha) Y``."""
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    n = len(s)
    c = len(y[0]) if y else 0
    z = [row[:] for row in y]
    for _ in range(epochs):
        nz = [[0.0] * c for _ in range(n)]
        for i in range(n):
            srow = s[i]
            for j in range(n):
                sij = srow[j]
                if sij:
                    zj = z[j]
                    for k in range(c):
                        nz[i][k] += sij * zj[k]
        for i in range(n):
            for k in range(c):
                nz[i][k] = alpha * nz[i][k] + (1.0 - alpha) * y[i][k]
        z = nz
    return z


def _solve(a, b):
    """Solve ``A X = B`` by Gauss-Jordan elimination with partial pivoting."""
    n = len(a)
    m = len(b[0])
    aug = [a[i][:] + b[i][:] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-15:
            raise ValueError("matrix is singular")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pv = aug[col][col]
        aug[col] = [v / pv for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col]:
                factor = aug[r][col]
                aug[r] = [aug[r][t] - factor * aug[col][t]
                          for t in range(n + m)]
    return [row[n:] for row in aug]


def propagate_closed_form(s, y, alpha=0.99):
    """Closed-form optimum ``Z* = (I - alpha S)^{-1} Y`` (Eq. 8)."""
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    n = len(s)
    a = [[(1.0 if i == j else 0.0) - alpha * s[i][j] for j in range(n)]
         for i in range(n)]
    return _solve(a, y)


def softmax_rows(z):
    """Row-wise softmax (Eq. 9) turning scores into class probabilities."""
    out = []
    for row in z:
        mx = max(row) if row else 0.0
        exps = [math.exp(v - mx) for v in row]
        total = sum(exps) or 1.0
        out.append(tuple(e / total for e in exps))
    return tuple(out)


def predict(features, labels, n_classes, *, k=8, sigma=1.0, alpha=0.99,
            method="closed_form", epochs=50):
    """Full LPA classification head.

    Builds the Gaussian kNN graph over ``features``, propagates the one-hot
    ``labels`` (``None`` = unlabelled/query point), and returns the argmax class
    per point together with the softmax probability matrix.
    """
    w = gaussian_knn_weights(features, k, sigma)
    s = symmetric_normalise(w)
    y = one_hot(labels, n_classes)
    if method == "iterative":
        z = propagate_iterative(s, y, alpha, epochs)
    elif method == "closed_form":
        z = propagate_closed_form(s, y, alpha)
    else:
        raise ValueError("method must be 'closed_form' or 'iterative'")
    probs = softmax_rows(z)
    preds = tuple(max(range(n_classes), key=lambda c: (row[c], -c))
                  for row in probs)
    return preds, probs
