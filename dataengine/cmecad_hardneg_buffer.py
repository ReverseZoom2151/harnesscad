"""CME-CAD Hard Negative Sample Buffering Mechanism (CME-CAD, 2025).

MERL suffers reward sparsity when *all* experts fail a query. CME-CAD's Hard
Negative Sample Buffering Mechanism (Sec. 3.2) mitigates this:

1. Split the RL data into ``M`` parts. After training on one part (1/M of the
   data), the next 1/M portion is used as an in-loop test set.
2. Maintain a buffer ``B`` of difficult samples. For each input ``I_i``, if an
   expert produces more than ``K`` incorrect answers out of its ``G`` responses,
   the sample is added to ``B`` with probability ``K/G``.
3. Periodically run supervised fine-tuning on ``B`` (Eq. 12,
   ``L_SFT = - sum_B log p_theta(A_correct | P_n, I_i)``), so hard cases are
   revisited rather than discarded.

This differs from ``cadrille_reward.mine_hard_examples`` (a mean-reward
threshold filter with no probabilistic buffer, no M-way partition, no
rotating test split). Pure-stdlib, deterministic: the ``K/G`` admission draw
uses a seeded ``random.Random`` so runs are reproducible.
"""

from __future__ import annotations

import random


def partition_rounds(items, m: int):
    """Split ``items`` into ``M`` contiguous, near-equal parts (train chunks).

    Returns a list of ``M`` lists. Extra items are distributed to the earliest
    parts so sizes differ by at most 1. ``M`` must be >= 1 and <= len(items).
    """
    items = list(items)
    n = len(items)
    if m < 1:
        raise ValueError("m must be >= 1")
    if m > n:
        raise ValueError("m cannot exceed the number of items")
    base, extra = divmod(n, m)
    parts = []
    start = 0
    for i in range(m):
        size = base + (1 if i < extra else 0)
        parts.append(items[start:start + size])
        start += size
    return parts


def train_test_rounds(items, m: int):
    """Yield ``(train_part, test_part)`` pairs where the test part is the *next*
    1/M portion after the trained part (the last part wraps to the first)."""
    parts = partition_rounds(items, m)
    pairs = []
    for i in range(m):
        train = parts[i]
        test = parts[(i + 1) % m]
        pairs.append((train, test))
    return pairs


def count_incorrect(correctness) -> int:
    """Number of incorrect responses given a per-response boolean correctness
    sequence (True == correct)."""
    return sum(0 if bool(c) else 1 for c in correctness)


def is_hard(correctness, k: int) -> bool:
    """A sample is *hard* for an expert when it yields MORE than ``K`` incorrect
    answers out of its ``G`` responses."""
    return count_incorrect(correctness) > int(k)


def admission_probability(k: int, g: int) -> float:
    """Buffer admission probability ``K/G`` for a hard sample."""
    if g <= 0:
        raise ValueError("g (group size) must be positive")
    return float(k) / float(g)


class HardNegativeBuffer:
    """Probabilistic buffer of difficult samples (mechanism state).

    Deterministic given a seed: the ``K/G`` admission draw is reproducible.
    """

    def __init__(self, k: int, g: int, seed: int = 0):
        if g <= 0:
            raise ValueError("g (group size) must be positive")
        if k < 0:
            raise ValueError("k must be >= 0")
        self.k = int(k)
        self.g = int(g)
        self._rng = random.Random(seed)
        self._buffer = []

    def offer(self, key, correctness) -> bool:
        """Consider adding a sample keyed by ``key`` given its per-response
        correctness. Returns True iff the sample was admitted to the buffer.

        The sample must be *hard* (> K incorrect); if so it is admitted with
        probability ``K/G``.
        """
        correctness = list(correctness)
        if len(correctness) != self.g:
            raise ValueError("correctness length must equal group size g")
        return self._offer(key, correctness)

    def _offer(self, key, correctness):
        if not is_hard(correctness, self.k):
            return False
        prob = admission_probability(self.k, self.g)
        if self._rng.random() < prob:
            self._buffer.append(key)
            return True
        return False

    def offer_many(self, samples):
        """Offer an iterable of ``(key, correctness)`` pairs; return admitted
        keys in offer order."""
        admitted = []
        for key, correctness in samples:
            correctness = list(correctness)
            if len(correctness) != self.g:
                raise ValueError("correctness length must equal group size g")
            if self._offer(key, correctness):
                admitted.append(key)
        return admitted

    @property
    def buffer(self):
        """Current buffered keys (order of admission)."""
        return list(self._buffer)

    def __len__(self):
        return len(self._buffer)

    def clear(self):
        """Empty the buffer (e.g. after an SFT pass over B)."""
        self._buffer = []


def sft_loss(log_probs) -> float:
    """L_SFT = - sum_B log p_theta(A_correct | P_n, I_i) (Eq. 12).

    ``log_probs`` are the ground-truth-answer log-probabilities of the buffered
    samples. Returns the summed negative log-likelihood; 0.0 for an empty buffer.
    """
    values = [float(x) for x in log_probs]
    return -sum(values)
