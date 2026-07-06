"""Leakage-safe grouped k-fold cross-validation with augment-within-fold.

Evaluation protocol from del Rio & England, *Lessons on Datasets and Paradigms in
Machine Learning for Symbolic Computation: A Case Study on CAD* (Sections 3.4 and
4.3.1). Two deterministic ideas, both domain-agnostic:

1. **Grouped k-fold** (Section 3.4): instances that share a *source group* (their
   SMT-LIB directory) must all land in the SAME fold, "to prevent data leakage: a
   situation in which a model is tested using the same or similar data to the data
   it was trained on". The existing ``bench/brep_splits.grouped_split`` produces a
   single 70/15/15 split; this produces the k rotating cross-validation folds the
   paper's 5-fold protocol needs, still keeping each group whole.

2. **Augment-after-split** (Section 4.3.1): "instances are first separated in
   folds and then either balanced or augmented. This ensures that even when
   augmenting the dataset there is no data leakage." Applying augmentation *inside*
   each fold guarantees an augmented copy can never appear in a different fold from
   its origin. :func:`augment_within_folds` enforces exactly this, and
   :func:`train_test_folds` yields (train, test) pairs with augmentation applied
   only to the training portion.

Both apply to any mechanical-CAD benchmark whose items carry a family/source group
(a base part, a CAD file, a template) that must not straddle the split.

Determinism: folds are assigned by hashing the group id (stable across runs and
input order). Stdlib-only, no wall clock, no unseeded randomness.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable, Dict, Iterable, List, Sequence, Tuple, TypeVar

T = TypeVar("T")


def _group_fold(group_id: str, k: int) -> int:
    digest = hashlib.sha256(group_id.encode()).hexdigest()
    return int(digest[:16], 16) % k


def grouped_kfold(
    records: Sequence[T],
    *,
    k: int = 5,
    group: Callable[[T], object] = lambda item: item,
) -> List[List[T]]:
    """Partition ``records`` into ``k`` folds keeping each group whole.

    Every record whose ``group(record)`` is equal lands in the same fold. Fold
    assignment is a deterministic hash of the group id, so it is stable regardless
    of input order. Records within a fold preserve their original relative order.
    Returns a list of ``k`` folds (some may be empty if there are few groups).
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    folds: List[List[T]] = [[] for _ in range(k)]
    for record in records:
        gid = str(group(record))
        folds[_group_fold(gid, k)].append(record)
    return folds


def group_leakage(folds: Sequence[Sequence[T]], *, group: Callable[[T], object]) -> List[object]:
    """Return any group ids that appear in more than one fold (should be empty).

    A non-empty result signals a leakage bug in fold construction.
    """
    where: Dict[object, set] = {}
    for fi, fold in enumerate(folds):
        for record in fold:
            where.setdefault(group(record), set()).add(fi)
    return sorted(
        (gid for gid, folds_seen in where.items() if len(folds_seen) > 1),
        key=str,
    )


def augment_within_folds(
    folds: Sequence[Sequence[T]],
    augment: Callable[[T], Iterable[T]],
) -> List[List[T]]:
    """Apply ``augment`` to every record, keeping augmented copies in-fold.

    ``augment(record)`` returns an iterable of derived records (include the
    original if you want to keep it). Because expansion happens per fold, no
    augmented instance can leak into a different fold than its origin.
    """
    out: List[List[T]] = []
    for fold in folds:
        expanded: List[T] = []
        for record in fold:
            expanded.extend(augment(record))
        out.append(expanded)
    return out


@dataclass(frozen=True)
class FoldSplit:
    fold_index: int
    train: Tuple[T, ...]
    test: Tuple[T, ...]


def train_test_folds(
    folds: Sequence[Sequence[T]],
    *,
    augment: Callable[[T], Iterable[T]] = lambda item: (item,),
) -> List[FoldSplit]:
    """Yield the k cross-validation (train, test) splits.

    For each fold ``i``: that fold is the test set (never augmented, so evaluation
    stays on genuine instances) and the union of the other folds is the training
    set, with ``augment`` applied ONLY to training records. This mirrors the
    paper: balancing/augmentation happens on training folds after the split, so
    test data is untouched and leakage-free.
    """
    splits: List[FoldSplit] = []
    for i, test_fold in enumerate(folds):
        train: List[T] = []
        for j, other in enumerate(folds):
            if j == i:
                continue
            for record in other:
                train.extend(augment(record))
        splits.append(
            FoldSplit(fold_index=i, train=tuple(train), test=tuple(test_fold))
        )
    return splits
