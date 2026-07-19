"""Nearest-codebook neural-code assignment for hierarchical CAD codebooks.

The learned VQ-VAE extracts three discrete codebooks (Loop, Profile,
Solid). *Training* the codebook is not deterministic, but the **code assignment given
a fixed codebook is** -- it is a plain nearest-neighbour lookup:

    c <- b_k,  where  k = argmin_i || E(TE) - b_i ||_2

after the encoder outputs are **average-pooled** over the sequence during vector
quantization. This module implements that deterministic quantization core:

* :func:`average_pool` -- mean of a set of encoder token vectors (E(TE)).
* :class:`Codebook` -- a fixed set of code vectors with nearest-neighbour
  :meth:`assign` (Euclidean, deterministic tie-break to the lowest index) and batch
  assignment.
* :class:`SPLCodebooks` -- the three independent Loop/Profile/Solid codebooks.
* codebook health, matching the training bookkeeping (which is deterministic
  *given* the assignments): :func:`utilization`, :func:`underutilized_codes`
  (the codebook re-initialisation rule: a code with fewer than 7 mapped samples), and
  :func:`compression_ratio` (unique-data / codebook-size, typically ~60x loop, 17x
  profile, 29x solid).

The transformer that *generates* code trees is out of scope; consuming a fixed
codebook here is fully deterministic. Pure stdlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Vector = tuple[float, ...]

# Under-utilization threshold: a code mapped by fewer than this many samples is
# re-initialized.
REINIT_THRESHOLD = 7


def average_pool(vectors: list[Vector]) -> Vector:
    """Mean over a list of equal-length encoder token vectors (the average-pool step).

    Returns E(TE) -- the pooled representation quantized by the codebook.
    """
    if not vectors:
        raise ValueError("cannot average-pool an empty set of vectors")
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        raise ValueError("all vectors must share the same dimension")
    return tuple(sum(v[i] for v in vectors) / len(vectors) for i in range(dim))


def _sq_dist(a: Vector, b: Vector) -> float:
    return sum((x - y) * (x - y) for x, y in zip(a, b))


@dataclass(frozen=True)
class Codebook:
    """A fixed set of code vectors supporting nearest-neighbour code assignment."""

    codes: tuple[Vector, ...]

    def __post_init__(self):
        if not self.codes:
            raise ValueError("codebook must contain at least one code")
        dim = len(self.codes[0])
        if any(len(c) != dim for c in self.codes):
            raise ValueError("all codes must share the same dimension")

    @property
    def size(self) -> int:
        return len(self.codes)

    @property
    def dim(self) -> int:
        return len(self.codes[0])

    def assign(self, vec: Vector) -> int:
        """Nearest-neighbour code index for ``vec``, ties -> lowest index."""
        if len(vec) != self.dim:
            raise ValueError(f"vector dim {len(vec)} != codebook dim {self.dim}")
        best_k, best_d = 0, _sq_dist(vec, self.codes[0])
        for k in range(1, self.size):
            d = _sq_dist(vec, self.codes[k])
            if d < best_d:
                best_k, best_d = k, d
        return best_k

    def quantize(self, vec: Vector) -> Vector:
        """Return the nearest code vector c = b_k (the quantized representation)."""
        return self.codes[self.assign(vec)]

    def assign_batch(self, vectors: list[Vector]) -> tuple[int, ...]:
        """Assign a code index to each vector, in order."""
        return tuple(self.assign(v) for v in vectors)


@dataclass(frozen=True)
class SPLCodebooks:
    """The three independent codebooks for the S-P-L levels."""

    loop: Codebook
    profile: Codebook
    solid: Codebook

    def assign(self, level: str, vec: Vector) -> int:
        return self._book(level).assign(vec)

    def _book(self, level: str) -> Codebook:
        book = {"loop": self.loop, "profile": self.profile, "solid": self.solid}.get(level)
        if book is None:
            raise ValueError(f"unknown level {level!r} (expected loop/profile/solid)")
        return book


# --- codebook health (deterministic given the assignments) -----------------
def utilization(assignments: list[int], size: int) -> tuple[int, ...]:
    """Per-code usage histogram over ``size`` codes given the assigned indices."""
    hist = [0] * size
    for a in assignments:
        if not 0 <= a < size:
            raise ValueError(f"assignment {a} outside [0, {size})")
        hist[a] += 1
    return tuple(hist)


def underutilized_codes(assignments: list[int], size: int,
                        threshold: int = REINIT_THRESHOLD) -> tuple[int, ...]:
    """Indices of codes mapped by fewer than ``threshold`` samples (re-init targets)."""
    hist = utilization(assignments, size)
    return tuple(i for i, n in enumerate(hist) if n < threshold)


def active_code_fraction(assignments: list[int], size: int) -> float:
    """Fraction of codes that are used at least once (1 - dead-code rate)."""
    if size == 0:
        return 0.0
    hist = utilization(assignments, size)
    return sum(1 for n in hist if n > 0) / size


def codebook_perplexity(assignments: list[int], size: int) -> float:
    """Perplexity of the code-usage distribution (higher = more uniform usage).

    ``exp(H)`` of the empirical code distribution; ranges from 1 (all mass on one
    code) up to ``size`` (uniform). A soft, deterministic diversity summary.
    """
    n = len(assignments)
    if n == 0:
        return 0.0
    hist = utilization(assignments, size)
    entropy = 0.0
    for count in hist:
        if count:
            p = count / n
            entropy -= p * math.log(p)
    return math.exp(entropy)


def compression_ratio(n_unique_data: int, codebook_size: int) -> float:
    """Compression ratio = unique-data / codebook-size."""
    if codebook_size <= 0:
        raise ValueError("codebook size must be positive")
    return n_unique_data / codebook_size
