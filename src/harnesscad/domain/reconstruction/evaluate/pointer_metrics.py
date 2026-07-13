"""Pointer-accuracy and topological-soundness metrics for Pointer-CAD.

Pointer-CAD is trained so that a predicted pointer is *correct* whenever it lands on
any of the geometrically-equivalent ground-truth candidates (paper Sec. 4.1.1: "the
ground-truth pointer is defined as a subset" P; Sec. 10.3 lists the coplanar-face /
collinear-edge special cases that make P a set rather than a single entity). Standard
top-1 accuracy would unfairly penalise a model that picks a different-but-equivalent
face/edge, so this module scores a prediction as a hit iff it is a member of the
valid-candidate set.

It also provides the paper's robustness signals in their deterministic, countable
form (Sec. 9.1): the **Invalidity Ratio** IR = (N_test - N_build) / N_test and a
**dangling-pointer ratio** (the fraction of pointers in a sequence that fail to
resolve). The learned similarity search (cosine over 128-d embeddings) is external;
we expose a pure-Python cosine matcher so a predicted embedding can be resolved to a
candidate deterministically for offline evaluation.

Pure stdlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from harnesscad.domain.reconstruction.brep.entity_index import EntityIndex
from harnesscad.domain.reconstruction.sequences.pointer_commands import PointerCommand, resolve_command


class PointerMetricError(ValueError):
    pass


def pointer_hit(predicted: int, valid_set: set[int] | frozenset[int] | tuple[int, ...]) -> bool:
    """A predicted pointer is correct iff it lies in the valid-candidate set."""
    return predicted in set(valid_set)


@dataclass(frozen=True)
class PointerAccuracy:
    """Set-membership pointer accuracy over a batch of predictions."""
    hits: int
    total: int

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.hits / self.total


def pointer_accuracy(
    predictions: list[int],
    valid_sets: list[set[int] | frozenset[int] | tuple[int, ...]],
) -> PointerAccuracy:
    """Score a batch: each prediction is a hit iff it is in its valid set (Sec. 4.1.1)."""
    if len(predictions) != len(valid_sets):
        raise PointerMetricError("predictions and valid_sets length mismatch")
    hits = sum(1 for p, vs in zip(predictions, valid_sets) if pointer_hit(p, vs))
    return PointerAccuracy(hits=hits, total=len(predictions))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if len(a) != len(b):
        raise PointerMetricError("vectors must be equal length")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        raise PointerMetricError("cannot match a zero-norm embedding")
    return dot / (na * nb)


def match_pointer(
    predicted_embedding: list[float],
    candidate_embeddings: list[list[float]],
) -> int:
    """Resolve a predicted 128-d vector to the highest-cosine candidate (Sec. 11.2).

    Returns the index of the best-matching candidate; ties break to the lowest index
    for determinism.
    """
    if not candidate_embeddings:
        raise PointerMetricError("no candidate embeddings to match against")
    best_i = -1
    best_s = -math.inf
    for i, cand in enumerate(candidate_embeddings):
        s = cosine_similarity(predicted_embedding, cand)
        if s > best_s:
            best_s = s
            best_i = i
    return best_i


def invalidity_ratio(n_test: int, n_build: int) -> float:
    """IR = (N_test - N_build) / N_test (Sec. 9.1, Eq. 2).

    ``n_build`` is the count of generated representations that build into a non-zero
    volume solid without post-processing; anything malformed is invalid.
    """
    if n_test <= 0:
        raise PointerMetricError("n_test must be positive")
    if not 0 <= n_build <= n_test:
        raise PointerMetricError("n_build must be in [0, n_test]")
    return (n_test - n_build) / n_test


@dataclass(frozen=True)
class DanglingReport:
    """Dangling-pointer accounting over a command sequence."""
    total_pointers: int
    dangling_pointers: int

    @property
    def ratio(self) -> float:
        if self.total_pointers == 0:
            return 0.0
        return self.dangling_pointers / self.total_pointers

    @property
    def is_sound(self) -> bool:
        return self.dangling_pointers == 0


def dangling_pointer_ratio(cmds: list[PointerCommand], index: EntityIndex) -> DanglingReport:
    """Fraction of pointers across ``cmds`` that fail to resolve against ``index``.

    This is the pointer-level analogue of the paper's dangling-edge signal: every
    pointer that indexes a non-existent entity is a topological fault.
    """
    total = 0
    dangling = 0
    for cmd in cmds:
        total += len(cmd.face_pointers) + len(cmd.edge_pointers)
        res = resolve_command(cmd, index)
        dangling += len(res.dangling_faces) + len(res.dangling_edges)
    return DanglingReport(total_pointers=total, dangling_pointers=dangling)
