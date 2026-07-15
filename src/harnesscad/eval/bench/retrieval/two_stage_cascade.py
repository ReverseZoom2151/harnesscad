"""Two-stage open-set retrieval cascade (OSCAR).

Mined from *OSCAR: Open-Set CAD Retrieval from a Language Prompt and a Single
Image*. OSCAR uses trained encoders (CLIP, DINOv2, GroundedSAM), but its
**retrieval logic** is a deterministic two-stage cascade over precomputed
embeddings:

*   **Stage 1 -- text filtering.** For every database model with multi-view caption
    embeddings, score it by the maximum cosine similarity between the query (image)
    embedding and any of its caption embeddings; keep only models whose score
    passes a threshold (the candidate set).
*   **Stage 2 -- visual refinement.** Among the survivors, compare the query ROI
    embedding to each model's pre-rendered view embeddings (again max over views)
    and return the single most visually similar model.

This module ports that cascade. Embeddings are plain vectors supplied by the caller
(no model is run). Deterministic: ties break by database index. Stdlib-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "cosine_similarity",
    "DatabaseModel",
    "stage1_text_filter",
    "stage2_visual_refine",
    "retrieve",
]

Vector = Sequence[float]


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity of two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must be the same length")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        raise ValueError("cannot take cosine similarity of a zero vector")
    return dot / (na * nb)


def _max_sim(query: Vector, views: Sequence[Vector]) -> float:
    if not views:
        raise ValueError("model has no view embeddings")
    return max(cosine_similarity(query, v) for v in views)


@dataclass(frozen=True)
class DatabaseModel:
    """One onboarded database model with multi-view text and image embeddings."""

    name: str
    caption_embeddings: Tuple[Tuple[float, ...], ...]  # CLIP text embeddings per view
    view_embeddings: Tuple[Tuple[float, ...], ...]     # DINOv2 image embeddings per view


def stage1_text_filter(
    query_text_embedding: Vector,
    database: Sequence[DatabaseModel],
    threshold: float,
) -> List[Tuple[int, float]]:
    """Stage 1: keep ``(index, score)`` for models whose max caption sim >= threshold.

    Results are sorted by descending score, ties by database index (deterministic).
    """
    scored: List[Tuple[int, float]] = []
    for i, model in enumerate(database):
        score = _max_sim(query_text_embedding, model.caption_embeddings)
        if score >= threshold:
            scored.append((i, score))
    scored.sort(key=lambda it: (-it[1], it[0]))
    return scored


def stage2_visual_refine(
    query_image_embedding: Vector,
    database: Sequence[DatabaseModel],
    candidates: Sequence[Tuple[int, float]],
) -> Optional[Tuple[int, float]]:
    """Stage 2: among candidates, return the ``(index, score)`` of best visual match.

    ``None`` when there are no candidates.
    """
    best: Optional[Tuple[int, float]] = None
    for idx, _ in candidates:
        score = _max_sim(query_image_embedding, database[idx].view_embeddings)
        if best is None or score > best[1]:
            best = (idx, score)
    return best


def retrieve(
    query_text_embedding: Vector,
    query_image_embedding: Vector,
    database: Sequence[DatabaseModel],
    threshold: float,
) -> Optional[DatabaseModel]:
    """Run the full OSCAR cascade; return the matched model (or ``None``)."""
    candidates = stage1_text_filter(query_text_embedding, database, threshold)
    best = stage2_visual_refine(query_image_embedding, database, candidates)
    return database[best[0]] if best is not None else None
