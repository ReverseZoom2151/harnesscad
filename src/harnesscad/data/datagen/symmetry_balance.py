"""Symmetry-group data augmentation and class balancing for ordering labels.

Transferable, domain-agnostic core extracted from del Rio & England, *Lessons on
Datasets and Paradigms in Machine Learning for Symbolic Computation: A Case Study
on CAD* (Sections 1.2, 4.3). The paper studies choosing a *variable ordering* for
cylindrical algebraic decomposition, but its data-augmentation idea is purely
combinatorial and applies to ANY labelled dataset whose instance carries one
feature block per item and whose label is an *ordering* (a permutation) of those
items -- e.g. choosing the order in which to resolve N interchangeable design
constraints, mesh N symmetric faces, or apply N boolean features.

The key observation (their arrow-rotation analogy): renaming/relabelling the
items by a permutation ``p`` produces a genuinely new labelled instance whose
label follows deterministically from the original -- *no re-labelling cost*.
Because labelling (running CAD, or, for us, an expensive solver / geometric
build) is the costly part, symmetry augmentation multiplies data for free and,
crucially, lets an imbalanced dataset be balanced.

An ``OrderingInstance`` is:
  * ``blocks`` -- a tuple of per-item feature vectors, ``blocks[i]`` describing
    item ``i`` (0-based).
  * ``label``  -- a permutation of ``range(n)`` giving the optimal ordering:
    ``label[0]`` is the item placed first, etc.

Applying a *renaming* permutation ``p`` (new item ``j`` is old item ``p[j]``)
maps the instance to a new one WITHOUT recomputation:

  * ``blocks_new[j] = blocks_old[p[j]]``
  * ``label_new     = tuple(p_inverse[o] for o in label_old)``

The full orbit over all ``n!`` renamings yields exactly one instance per ordering
class (a perfectly balanced "augmented" dataset). ``balance_by_symmetry`` instead
keeps the dataset size fixed and relabels each instance greedily toward the
least-populated class -- a deterministic realisation of the paper's "balanced
dataset" (they relabel randomly; greedy gives an exact balance).

Stdlib-only, deterministic (any randomness seeded via ``random.Random``).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import permutations
import random
from typing import Dict, Iterable, List, Sequence, Tuple

Block = Tuple[float, ...]


@dataclass(frozen=True)
class OrderingInstance:
    """A labelled instance: per-item feature blocks + an optimal ordering."""

    blocks: Tuple[Block, ...]
    label: Tuple[int, ...]

    def __post_init__(self) -> None:
        n = len(self.blocks)
        if tuple(sorted(self.label)) != tuple(range(n)):
            raise ValueError("label must be a permutation of range(len(blocks))")

    @property
    def n_items(self) -> int:
        return len(self.blocks)


def _as_instance(obj) -> OrderingInstance:
    if isinstance(obj, OrderingInstance):
        return obj
    blocks, label = obj
    return OrderingInstance(
        tuple(tuple(float(x) for x in b) for b in blocks),
        tuple(int(i) for i in label),
    )


def inverse_permutation(p: Sequence[int]) -> Tuple[int, ...]:
    """Return ``q`` with ``q[p[j]] == j`` for all ``j``."""
    q = [0] * len(p)
    for j, pj in enumerate(p):
        q[pj] = j
    return tuple(q)


def apply_permutation(instance, p: Sequence[int]) -> OrderingInstance:
    """Relabel ``instance`` by renaming permutation ``p`` (new j <- old p[j])."""
    inst = _as_instance(instance)
    p = tuple(int(x) for x in p)
    if tuple(sorted(p)) != tuple(range(inst.n_items)):
        raise ValueError("p must be a permutation of range(n_items)")
    p_inv = inverse_permutation(p)
    new_blocks = tuple(inst.blocks[p[j]] for j in range(inst.n_items))
    new_label = tuple(p_inv[o] for o in inst.label)
    return OrderingInstance(new_blocks, new_label)


def orbit(instance) -> List[OrderingInstance]:
    """All ``n!`` renamings of ``instance`` (full augmentation).

    Deterministic order: permutations in lexicographic order. Each distinct
    ordering label appears exactly once, so the orbit is class-balanced.
    """
    inst = _as_instance(instance)
    return [apply_permutation(inst, p) for p in permutations(range(inst.n_items))]


def augment_full(instances: Iterable) -> List[OrderingInstance]:
    """Concatenate the full orbit of every instance (the 'augmented' dataset)."""
    out: List[OrderingInstance] = []
    for obj in instances:
        out.extend(orbit(obj))
    return out


def class_distribution(instances: Iterable) -> Dict[Tuple[int, ...], int]:
    """Count instances per ordering-label class (sorted keys)."""
    counter: Counter = Counter(_as_instance(o).label for o in instances)
    return {label: counter[label] for label in sorted(counter)}


def imbalance_ratio(instances: Iterable) -> float:
    """max_class / min_class count; 1.0 is perfectly balanced.

    Minimum is taken over *observed* classes only. Empty input -> 1.0.
    """
    dist = class_distribution(instances)
    if not dist:
        return 1.0
    counts = list(dist.values())
    return max(counts) / min(counts)


def balance_by_symmetry(instances: Iterable, *, seed: int = 0) -> List[OrderingInstance]:
    """Relabel each instance toward the least-populated reachable class.

    Keeps the dataset size unchanged (unlike :func:`augment_full`). Each instance
    is moved, via one of its orbit renamings, into whichever ordering class is
    currently least populated -- a deterministic exact-balance realisation of the
    paper's "balanced dataset" (Section 4.3.1). Ties broken by lexicographic
    label; ``seed`` only randomises the *processing order* of instances so the
    result does not depend on input order in a biased way.
    """
    items = [_as_instance(o) for o in instances]
    rng = random.Random(seed)
    order = list(range(len(items)))
    rng.shuffle(order)
    counts: Counter = Counter()
    result: List[Tuple[int, OrderingInstance]] = []
    for idx in order:
        inst = items[idx]
        best = None
        best_key = None
        for cand in orbit(inst):
            key = (counts[cand.label], cand.label)
            if best_key is None or key < best_key:
                best_key = key
                best = cand
        counts[best.label] += 1
        result.append((idx, best))
    result.sort(key=lambda pair: pair[0])
    return [inst for _idx, inst in result]


def random_relabel(instances: Iterable, *, seed: int = 0) -> List[OrderingInstance]:
    """The paper's exact approach: relabel each instance by a random renaming.

    Approximately balances classes over a large dataset. Deterministic given
    ``seed`` (drives a single ``random.Random``).
    """
    rng = random.Random(seed)
    out: List[OrderingInstance] = []
    for obj in instances:
        inst = _as_instance(obj)
        p = list(range(inst.n_items))
        rng.shuffle(p)
        out.append(apply_permutation(inst, p))
    return out
