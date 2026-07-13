"""Few-shot scaling-curve protocol for self-supervised CAD embeddings.

Jones, Hu, Kim & Schulz, *Self-Supervised Representation Learning for CAD* (2022),
Section 5 and Tables 2-3. The paper's headline evaluation is not a single accuracy
number but a *scaling curve*: freeze the self-supervised face embeddings, then train
a small task head on labelled subsets of increasing size (``@10, @100, @1000, ...``
or fractions ``1%, 5%, ...``), repeating each size over several random seeds and
reporting the mean accuracy (Table 2 records "the mean of 10 runs at each data set
size with the train set subset at different random seeds; each model sees the same
random subsets"). The point being demonstrated is that a good self-supervised
representation wins *most* in the few-shot regime.

This module is the deterministic harness for that protocol. It:

* draws **stratified, seeded** labelled subsets at each requested size (matching the
  paper's "each model sees the same random subsets" -- the same seed reproduces the
  same subset, and stratification keeps at least one example per class when
  possible, echoing the paper's stratified sampling for FabWave);
* fits a frozen-embedding classifier (linear probe or k-NN, reusing
  ``bench.ssrl_representation_quality``) on each subset and scores it on a fixed
  held-out test set;
* averages accuracy over the repeats to produce the ``(size -> mean accuracy)``
  scaling curve, plus the per-seed spread.

Stdlib only; ``random.Random(seed)`` for all sampling (no wall clock). The learned
encoder is out of scope -- embeddings and labels are supplied by the caller.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

from harnesscad.eval.bench.retrieval.ssrl_representation_quality import (
    linear_probe_accuracy,
    knn_accuracy,
)

Vector = Sequence[float]


# --------------------------------------------------------------------------- #
# Stratified seeded subset selection.                                          #
# --------------------------------------------------------------------------- #
def stratified_subset(labels: Sequence, size: int, rng: random.Random) -> List[int]:
    """Indices of a class-stratified subset of ``size`` items.

    Guarantees at least one item per class while the budget allows (paper's
    stratified sampling), then fills the remaining budget uniformly at random
    from the leftover pool. Deterministic given ``rng``. ``size`` is clamped to
    the dataset size.
    """
    n = len(labels)
    size = min(size, n)
    if size <= 0:
        return []
    by_label: Dict = {}
    for i, lab in enumerate(labels):
        by_label.setdefault(lab, []).append(i)
    chosen: List[int] = []
    # One per class first (in sorted-label order for determinism).
    for lab in sorted(by_label, key=repr):
        if len(chosen) >= size:
            break
        pool = by_label[lab]
        pick = pool[rng.randrange(len(pool))]
        chosen.append(pick)
    chosen_set = set(chosen)
    remaining = [i for i in range(n) if i not in chosen_set]
    need = size - len(chosen)
    if need > 0 and remaining:
        need = min(need, len(remaining))
        extra = rng.sample(remaining, need)
        chosen.extend(extra)
    return sorted(chosen)


# --------------------------------------------------------------------------- #
# Classifier adapters (reuse the representation-quality heads).                #
# --------------------------------------------------------------------------- #
def _linear_evaluator(ridge: float) -> Callable:
    def run(tx, ty, ex, ey):
        return linear_probe_accuracy(tx, ty, ex, ey, ridge=ridge)
    return run


def _knn_evaluator(k: int) -> Callable:
    def run(tx, ty, ex, ey):
        return knn_accuracy(tx, ty, ex, ey, k=k)
    return run


def make_evaluator(kind: str = "linear", **kwargs) -> Callable:
    """Build a ``(train_x, train_y, test_x, test_y) -> accuracy`` evaluator.

    ``kind`` is ``"linear"`` (ridge linear probe; ``ridge`` kwarg) or ``"knn"``
    (``k`` kwarg). These wrap the deterministic heads in
    :mod:`bench.ssrl_representation_quality`.
    """
    if kind == "linear":
        return _linear_evaluator(kwargs.get("ridge", 1.0))
    if kind == "knn":
        return _knn_evaluator(kwargs.get("k", 5))
    raise ValueError(f"unknown evaluator kind: {kind!r}")


# --------------------------------------------------------------------------- #
# Scaling-curve computation.                                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScalingPoint:
    """Mean and per-seed accuracies at one training-set size."""

    size: int
    mean_accuracy: float
    accuracies: Tuple[float, ...]

    @property
    def spread(self) -> float:
        """Max-minus-min accuracy across seeds (a simple stability proxy)."""
        return max(self.accuracies) - min(self.accuracies)


def scaling_curve(train_x: Sequence[Vector], train_y: Sequence,
                  test_x: Sequence[Vector], test_y: Sequence,
                  sizes: Sequence[int], *, repeats: int = 10, seed: int = 0,
                  evaluator: Callable | None = None) -> List[ScalingPoint]:
    """Accuracy-vs-training-size scaling curve (paper Table 2/3 protocol).

    For each ``size`` in ``sizes``, draw ``repeats`` stratified seeded subsets of
    the labelled training pool, fit the ``evaluator`` classifier on each, score it
    on the fixed ``(test_x, test_y)`` set, and average. The same base ``seed``
    reproduces the whole curve exactly (subset ``r`` at size ``s`` uses a derived
    seed), matching "each model sees the same random subsets". Defaults to a
    ridge linear probe.

    Returns one :class:`ScalingPoint` per size, in the order given.
    """
    if len(train_x) != len(train_y):
        raise ValueError("train features and labels must align")
    if not train_x:
        raise ValueError("empty training pool")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    ev = evaluator or make_evaluator("linear")
    curve: List[ScalingPoint] = []
    for size in sizes:
        if size <= 0:
            raise ValueError("sizes must be positive")
        accs: List[float] = []
        for r in range(repeats):
            rng = random.Random("ssrl-scaling:%d:%d:%d" % (seed, size, r))
            idx = stratified_subset(train_y, size, rng)
            sub_x = [train_x[i] for i in idx]
            sub_y = [train_y[i] for i in idx]
            accs.append(ev(sub_x, sub_y, test_x, test_y))
        mean = sum(accs) / len(accs)
        curve.append(ScalingPoint(min(size, len(train_x)), mean, tuple(accs)))
    return curve


def few_shot_advantage(curve_a: Sequence[ScalingPoint],
                       curve_b: Sequence[ScalingPoint]) -> List[float]:
    """Per-size accuracy gap ``A - B`` between two scaling curves.

    Reproduces the paper's core comparison: the self-supervised curve minus a
    baseline curve, size by size. Positive entries are where representation ``A``
    wins; the paper's claim is that the gap is largest at the smallest sizes.
    The two curves must cover the same sizes in the same order.
    """
    if len(curve_a) != len(curve_b):
        raise ValueError("curves must have the same number of sizes")
    gaps = []
    for pa, pb in zip(curve_a, curve_b):
        if pa.size != pb.size:
            raise ValueError("curves must be aligned on size")
        gaps.append(pa.mean_accuracy - pb.mean_accuracy)
    return gaps
