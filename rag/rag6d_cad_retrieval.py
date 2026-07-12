"""CAD-as-knowledge-base retrieval and pose-hypothesis ranking (RAG-6DPose).

RAG-6DPose treats a set of 3D CAD models as a *knowledge base*: each model is
rendered from multiple viewpoints, every view is described by a feature vector,
and at query time the most relevant CAD view is retrieved for the input image
(Sec. III-B/-C).  Each retrieved view carries a coarse pose hypothesis (the
render camera's rotation), and several hypotheses are then ranked by a
correspondence / similarity score to pick the best one (Sec. III-D).

This module implements the *deterministic* retrieval-and-ranking bookkeeping --
not the learned DINOv2/ReSPC descriptor network.  Descriptors are supplied by the
caller; the module provides:

  * :class:`CadKnowledgeBase` -- an index of ``(model_id, view_id, descriptor,
    pose)`` entries with cosine-similarity nearest-view retrieval;
  * :func:`retrieve_views` -- top-k nearest CAD views for a query descriptor;
  * :func:`retrieve_best_model` -- the CAD model whose best-matching view scores
    highest (nearest-model identification);
  * :func:`rank_pose_hypotheses` -- rank candidate poses by a combined
    retrieval-similarity and correspondence-inlier score, mirroring the paper's
    "training-loss score helps select the best pose hypothesis".

All ordering is total and deterministic (ties broken by ``(model_id, view_id)``);
no wall clock, no randomness.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in ``[-1, 1]``; 0 when either vector is null.

    Raises ``ValueError`` on a dimension mismatch.
    """
    if len(a) != len(b):
        raise ValueError("descriptor dimension mismatch")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class CadViewEntry:
    """One rendered CAD view in the knowledge base."""

    __slots__ = ("model_id", "view_id", "descriptor", "pose")

    def __init__(self, model_id: str, view_id: str,
                 descriptor: Sequence[float], pose=None) -> None:
        self.model_id = str(model_id)
        self.view_id = str(view_id)
        self.descriptor = list(descriptor)
        self.pose = pose

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "CadViewEntry(%r, %r)" % (self.model_id, self.view_id)


class CadKnowledgeBase:
    """Multi-view CAD knowledge base with nearest-view retrieval."""

    def __init__(self) -> None:
        self._entries: List[CadViewEntry] = []

    def add_view(self, model_id: str, view_id: str,
                 descriptor: Sequence[float], pose=None) -> None:
        """Register one rendered view's descriptor (and optional pose)."""
        self._entries.append(CadViewEntry(model_id, view_id, descriptor, pose))

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> Tuple[CadViewEntry, ...]:
        return tuple(self._entries)

    def model_ids(self) -> Tuple[str, ...]:
        """Distinct model ids in insertion-stable, sorted order."""
        return tuple(sorted({e.model_id for e in self._entries}))

    def retrieve_views(self, query: Sequence[float], k: int = 1
                       ) -> List[Tuple[float, CadViewEntry]]:
        """Top-``k`` views by descending cosine similarity to ``query``.

        Ties (equal similarity) are broken deterministically by
        ``(model_id, view_id)``.  Returns ``(score, entry)`` pairs.
        """
        scored = [(cosine_similarity(query, e.descriptor), e) for e in self._entries]
        scored.sort(key=lambda se: (-se[0], se[1].model_id, se[1].view_id))
        if k < 0:
            raise ValueError("k must be non-negative")
        return scored[:k]

    def retrieve_best_model(self, query: Sequence[float]
                            ) -> Optional[Tuple[str, float, CadViewEntry]]:
        """Identify the CAD model whose best view matches ``query`` best.

        Returns ``(model_id, score, entry)`` for the single highest-scoring view,
        or ``None`` if the base is empty.
        """
        top = self.retrieve_views(query, k=1)
        if not top:
            return None
        score, entry = top[0]
        return entry.model_id, score, entry


class PoseCandidate:
    """A candidate pose paired with the scores used to rank it."""

    __slots__ = ("pose", "retrieval_score", "num_inliers", "label")

    def __init__(self, pose, retrieval_score: float, num_inliers: int,
                 label: str = "") -> None:
        self.pose = pose
        self.retrieval_score = float(retrieval_score)
        self.num_inliers = int(num_inliers)
        self.label = str(label)


def hypothesis_score(candidate: PoseCandidate,
                     inlier_weight: float = 1.0,
                     retrieval_weight: float = 1.0) -> float:
    """Combined ranking score: inliers and retrieval similarity.

    ``score = inlier_weight * num_inliers + retrieval_weight * retrieval_score``.
    Higher is better, matching the paper's use of the correspondence score and
    the retrieval match to pick the winning hypothesis.
    """
    return (inlier_weight * candidate.num_inliers
            + retrieval_weight * candidate.retrieval_score)


def rank_pose_hypotheses(candidates: Sequence[PoseCandidate],
                         inlier_weight: float = 1.0,
                         retrieval_weight: float = 1.0
                         ) -> List[PoseCandidate]:
    """Rank pose candidates best-first by :func:`hypothesis_score`.

    Ties are broken deterministically by descending ``num_inliers`` then by
    ``label`` so the ordering is total and reproducible.
    """
    return sorted(
        candidates,
        key=lambda c: (
            -hypothesis_score(c, inlier_weight, retrieval_weight),
            -c.num_inliers,
            c.label,
        ),
    )


def select_best_hypothesis(candidates: Sequence[PoseCandidate],
                           inlier_weight: float = 1.0,
                           retrieval_weight: float = 1.0
                           ) -> Optional[PoseCandidate]:
    """Return the single best pose candidate, or ``None`` if none supplied."""
    ranked = rank_pose_hypotheses(candidates, inlier_weight, retrieval_weight)
    return ranked[0] if ranked else None
