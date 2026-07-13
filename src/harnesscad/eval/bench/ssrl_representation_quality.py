"""Representation-quality evaluation protocol for self-supervised CAD embeddings.

Jones, Hu, Kim & Schulz, *Self-Supervised Representation Learning for CAD* (2022),
Sections 4-5. The paper judges its *frozen, unlabelled* face embeddings by how well
they support downstream tasks with only small labelled sets: it plugs the embeddings
into a linear SVM, an MLP, and a message-passing head (Table 2/3), reporting
accuracy at growing training-set sizes. This module provides the fully deterministic
*evaluation* half of that protocol -- the standard toolkit for scoring a learned
representation *without* training the encoder:

* **linear probe** -- fit a linear classifier on frozen embeddings by closed-form
  ridge regression (one-hot least squares), then report accuracy. This is the
  deterministic analogue of the paper's "Self-Supervision + SVM" linear evaluation:
  a good representation is linearly separable, so probe accuracy measures quality.
* **k-NN classification** -- non-parametric accuracy of majority vote over the ``k``
  nearest labelled neighbours. Requires no training at all, isolating the geometry
  of the embedding space.
* **alignment & uniformity** (Wang & Isola, 2020) -- the two asymptotic properties
  a contrastive/SSL embedding should have: *alignment* is the mean squared distance
  between L2-normalised positive pairs (lower = closer), *uniformity* is the log of
  the mean Gaussian potential over all pairs (lower = features spread evenly on the
  hypersphere). Together they diagnose *why* a representation probes well or badly.

All routines are stdlib-only and deterministic. The learned encoder that produces
the embeddings is out of scope; embeddings and labels are supplied by the caller.
This is the representation-quality complement to ``bench.contrastcad_latent_metrics``
(which scores clustering: silhouette / SSE / K-means), and does not duplicate it.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

Vector = Sequence[float]


# --------------------------------------------------------------------------- #
# Small linear-algebra helpers (stdlib only).                                  #
# --------------------------------------------------------------------------- #
def _l2_normalize(v: Vector) -> List[float]:
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0.0:
        raise ValueError("cannot normalise a zero vector")
    return [x / norm for x in v]


def _sqdist(a: Vector, b: Vector) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _solve_symmetric(matrix: List[List[float]], rhs: List[float]) -> List[float]:
    """Solve ``A x = b`` for symmetric positive-definite ``A`` via Gaussian
    elimination with partial pivoting. Stdlib only."""
    n = len(matrix)
    aug = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-15:
            aug[pivot][col] += 1e-12  # nudge singular pivots (ridge safety)
        aug[col], aug[pivot] = aug[pivot], aug[col]
        piv = aug[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col] / piv
            for c in range(col, n + 1):
                aug[r][c] -= factor * aug[col][c]
    return [aug[i][n] / aug[i][i] for i in range(n)]


# --------------------------------------------------------------------------- #
# Alignment & uniformity (Wang & Isola, 2020).                                 #
# --------------------------------------------------------------------------- #
def alignment(pairs: Sequence[Tuple[Vector, Vector]], *, alpha: float = 2.0,
              normalize: bool = True) -> float:
    """Alignment metric: mean ``||f(x) - f(x+)||_2^alpha`` over positive pairs.

    Lower is better -- positive pairs should map to nearby points. Vectors are
    L2-normalised onto the unit hypersphere first (the metric is defined for
    normalised features), unless ``normalize=False``.
    """
    if not pairs:
        raise ValueError("need at least one positive pair")
    total = 0.0
    for a, b in pairs:
        if normalize:
            a, b = _l2_normalize(a), _l2_normalize(b)
        total += math.sqrt(_sqdist(a, b)) ** alpha
    return total / len(pairs)


def uniformity(vectors: Sequence[Vector], *, t: float = 2.0,
               normalize: bool = True) -> float:
    """Uniformity metric: ``log E[exp(-t ||f(x) - f(y)||^2)]`` over all pairs.

    Lower (more negative) is better -- features spread uniformly on the sphere.
    Computed over all unordered pairs ``i < j``. Vectors are L2-normalised first
    unless ``normalize=False``.
    """
    n = len(vectors)
    if n < 2:
        raise ValueError("need at least two vectors")
    vs = [_l2_normalize(v) for v in vectors] if normalize else [list(v) for v in vectors]
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += math.exp(-t * _sqdist(vs[i], vs[j]))
            count += 1
    return math.log(total / count)


# --------------------------------------------------------------------------- #
# k-NN classification.                                                         #
# --------------------------------------------------------------------------- #
def knn_classify(train_x: Sequence[Vector], train_y: Sequence,
                 query_x: Sequence[Vector], k: int = 5) -> List:
    """Majority-vote k-NN label for each query embedding.

    Ties in distance break by training index; ties in the vote break by choosing
    the label whose nearest supporting neighbour is closest (then by label repr),
    so results are fully deterministic. ``k`` is clamped to the train-set size.
    """
    if len(train_x) != len(train_y):
        raise ValueError("train features and labels must align")
    if not train_x:
        raise ValueError("empty training set")
    k = min(k, len(train_x))
    preds = []
    for q in query_x:
        order = sorted(range(len(train_x)),
                       key=lambda i: (_sqdist(q, train_x[i]), i))
        neighbours = order[:k]
        votes: Dict = {}
        best_rank: Dict = {}
        for rank, i in enumerate(neighbours):
            lab = train_y[i]
            votes[lab] = votes.get(lab, 0) + 1
            if lab not in best_rank:
                best_rank[lab] = rank
        # Prefer more votes, then the label whose nearest support is closest,
        # then label repr -- fully deterministic tie-breaking.
        preds.append(_argmax_vote(votes, best_rank))
    return preds


def _argmax_vote(votes: Dict, best_rank: Dict):
    return max(votes, key=lambda lab: (votes[lab], -best_rank[lab], _lab_key(lab)))


def _lab_key(lab):
    # Deterministic, type-agnostic secondary key.
    return repr(lab)


def knn_accuracy(train_x, train_y, test_x, test_y, k: int = 5) -> float:
    """Fraction of test embeddings correctly classified by k-NN."""
    if len(test_x) != len(test_y):
        raise ValueError("test features and labels must align")
    if not test_x:
        raise ValueError("empty test set")
    preds = knn_classify(train_x, train_y, test_x, k)
    correct = sum(1 for p, y in zip(preds, test_y) if p == y)
    return correct / len(test_y)


# --------------------------------------------------------------------------- #
# Linear probe (ridge-regression, one-vs-rest least squares).                  #
# --------------------------------------------------------------------------- #
class LinearProbe:
    """Closed-form linear classifier over frozen embeddings.

    Fits one ridge-regression weight vector per class against a one-hot target
    (least-squares one-vs-rest), which is deterministic and needs no iterative
    optimiser. At inference the class with the highest linear score wins. This is
    the deterministic stand-in for the paper's "Self-Supervision + SVM" linear
    evaluation head: probe accuracy measures how linearly separable -- i.e. how
    good -- the representation is.
    """

    def __init__(self, weights: List[List[float]], classes: List, ridge: float):
        self._weights = weights          # per-class weight vectors (with bias)
        self._classes = classes
        self.ridge = ridge

    @property
    def classes(self) -> List:
        return list(self._classes)

    @staticmethod
    def _augment(x: Vector) -> List[float]:
        return list(x) + [1.0]  # bias term

    @classmethod
    def fit(cls, train_x: Sequence[Vector], train_y: Sequence,
            ridge: float = 1.0) -> "LinearProbe":
        if len(train_x) != len(train_y):
            raise ValueError("features and labels must align")
        if not train_x:
            raise ValueError("empty training set")
        if ridge < 0.0:
            raise ValueError("ridge must be non-negative")
        classes = sorted(set(train_y), key=_lab_key)
        xs = [cls._augment(x) for x in train_x]
        dim = len(xs[0])
        # Normal-equation matrix A = X^T X + ridge * I (shared across classes).
        a = [[0.0] * dim for _ in range(dim)]
        for row in xs:
            for i in range(dim):
                ri = row[i]
                if ri == 0.0:
                    continue
                for j in range(dim):
                    a[i][j] += ri * row[j]
        for i in range(dim):
            a[i][i] += ridge
        weights = []
        for lab in classes:
            targets = [1.0 if y == lab else 0.0 for y in train_y]
            b = [0.0] * dim
            for row, t in zip(xs, targets):
                if t == 0.0:
                    continue
                for i in range(dim):
                    b[i] += row[i] * t
            weights.append(_solve_symmetric([r[:] for r in a], b))
        return cls(weights, classes, ridge)

    def scores(self, x: Vector) -> List[float]:
        row = self._augment(x)
        return [sum(w * r for w, r in zip(weight, row)) for weight in self._weights]

    def predict_one(self, x: Vector):
        s = self.scores(x)
        best = max(range(len(s)), key=lambda i: (s[i], -i))
        return self._classes[best]

    def predict(self, xs: Sequence[Vector]) -> List:
        return [self.predict_one(x) for x in xs]


def linear_probe_accuracy(train_x, train_y, test_x, test_y,
                          ridge: float = 1.0) -> float:
    """Fit a linear probe on the train split, report test accuracy.

    This is the headline representation-quality number: high linear-probe
    accuracy means the self-supervised embedding already separates the labelled
    classes without a deep task head.
    """
    if len(test_x) != len(test_y):
        raise ValueError("test features and labels must align")
    if not test_x:
        raise ValueError("empty test set")
    probe = LinearProbe.fit(train_x, train_y, ridge)
    preds = probe.predict(test_x)
    correct = sum(1 for p, y in zip(preds, test_y) if p == y)
    return correct / len(test_y)
