"""OSCAR two-stage multimodal (text + image) CAD retrieval fusion.

Pulli et al., *OSCAR: Open-Set CAD Retrieval from a Language Prompt and a Single
Image* (2024), Section 3.2.

OSCAR retrieves a 3D CAD model from an *unlabelled* database given a single RGB
image (Region-of-Interest) and a language prompt. Each database model ``s_i`` is
onboarded with two banks of precomputed embeddings:

* ``text_embeddings`` -- CLIP text embeddings ``t_ij`` of ``K`` auto-generated
  captions (one per rendered view);
* ``image_embeddings`` -- DINOv2 image embeddings ``v_ik`` of the ``K``
  pre-rendered views.

At inference the ROI is embedded twice: ``q_clip`` (CLIP image encoder, shares
CLIP's joint text-image space) and ``q_dino`` (DINOv2). Retrieval is two-stage:

1. **Text filtering.** For each model, ``sim_text(s_i) = max_j cos(q_clip,
   t_ij)`` -- the best cross-modal image-to-caption similarity. Keep every model
   whose ``sim_text >= tau_text`` (paper uses ``tau_text = 0.37``). If the
   candidate set is empty, fall back to the top-``k`` models by ``sim_text``.

2. **Image refinement.** For each surviving candidate, ``sim_img(s_i) = max_k
   cos(q_dino, v_ik)`` -- the best view-to-ROI visual similarity. The retrieved
   model is ``argmax_{s_i in S'} sim_img(s_i)``.

This module implements the *deterministic fusion protocol* only. The learned
CLIP / DINOv2 / captioning encoders are external -- callers pass precomputed
embeddings. It reuses :func:`bench.geomretr_eval.cosine_distance` for scoring so
that similarity ordering is consistent with the repo's other retrieval evals.
Distinct from the single-modality closed-set rankers in
:mod:`bench.gencad_retrieval` and :mod:`bench.ranked_retrieval_metrics`: the
novelty here is the *late fusion of two similarity scores across two modalities*
with a semantic sanity-check threshold and top-k fallback.

Pure stdlib, deterministic (ties broken by database index).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from bench.geomretr_eval import cosine_distance

Vector = Sequence[float]

__all__ = [
    "OscarModel",
    "cosine_similarity",
    "text_similarity",
    "image_similarity",
    "filter_candidates",
    "late_fusion_score",
    "retrieve",
    "RetrievalResult",
]


def cosine_similarity(u: Vector, v: Vector) -> float:
    """Cosine similarity in ``[-1, 1]``; 0.0 if either vector is zero.

    Thin wrapper over :func:`bench.geomretr_eval.cosine_distance` so the whole
    OSCAR pipeline scores identically to the rest of ``bench``.
    """
    return 1.0 - cosine_distance(u, v)


@dataclass(frozen=True)
class OscarModel:
    """A database CAD model onboarded with multi-view text and image embeddings.

    ``text_embeddings``  -- CLIP text embeddings of the per-view captions.
    ``image_embeddings`` -- DINOv2 embeddings of the pre-rendered views.
    ``model_id``         -- opaque identifier mapped back to the CAD model.
    """

    model_id: str
    text_embeddings: Tuple[Tuple[float, ...], ...]
    image_embeddings: Tuple[Tuple[float, ...], ...]


def text_similarity(q_clip: Vector, model: OscarModel) -> float:
    """``sim_text(s_i) = max_j cos(q_clip, t_ij)`` (Eq. for the filtering stage).

    Aggregates over the model's captions by taking the *best* match. Returns
    ``-1.0`` (the cosine floor) when the model has no captions.
    """
    if not model.text_embeddings:
        return -1.0
    return max(cosine_similarity(q_clip, t) for t in model.text_embeddings)


def image_similarity(q_dino: Vector, model: OscarModel) -> float:
    """``sim_img(s_i) = max_k cos(q_dino, v_ik)`` (image refinement stage).

    Best view-to-ROI similarity across the model's rendered views. Returns
    ``-1.0`` when the model has no rendered views.
    """
    if not model.image_embeddings:
        return -1.0
    return max(cosine_similarity(q_dino, v) for v in model.image_embeddings)


def filter_candidates(q_clip: Vector,
                      models: Sequence[OscarModel],
                      tau_text: float = 0.37,
                      top_k: int = 5) -> List[int]:
    """Stage 1: indices of models passing the CLIP-text sanity check.

    Returns the indices of every model with ``sim_text >= tau_text``. When no
    model passes the threshold, falls back to the ``top_k`` models by descending
    ``sim_text`` (paper's "If S' is empty ... fall back to the top-k text
    candidates"). Ordering of the returned survivors is by database index; the
    top-k fallback is ordered by descending similarity then index.

    ``top_k`` is clamped to the number of models and to be at least 1.
    """
    if not models:
        return []
    sims = [text_similarity(q_clip, m) for m in models]
    passed = [i for i, s in enumerate(sims) if s >= tau_text]
    if passed:
        return passed
    k = max(1, min(top_k, len(models)))
    order = sorted(range(len(models)), key=lambda i: (-sims[i], i))
    return order[:k]


def late_fusion_score(sim_text: float,
                      sim_img: float,
                      image_weight: float = 1.0) -> float:
    """Deterministic late fusion of the two per-modality similarity scores.

    A convex combination ``(1 - w) * sim_text + w * sim_img`` with
    ``w = image_weight`` in ``[0, 1]``. OSCAR's default pipeline selects the
    final model purely by image similarity among the text-filtered candidates,
    i.e. ``image_weight = 1.0``; intermediate weights let callers blend the
    semantic (CLIP text) and visual (DINOv2) scores. Raises ``ValueError`` if the
    weight is outside ``[0, 1]``.
    """
    if not 0.0 <= image_weight <= 1.0:
        raise ValueError("image_weight must be in [0, 1]")
    return (1.0 - image_weight) * sim_text + image_weight * sim_img


@dataclass(frozen=True)
class RetrievalResult:
    """Outcome of an OSCAR query.

    ``model_index``     -- database index of the retrieved model (``-1`` if the
                           database is empty).
    ``model_id``        -- its identifier (``None`` if empty).
    ``fused_score``     -- late-fusion score of the winner.
    ``candidates``      -- indices that survived text filtering (stage 1).
    ``used_fallback``   -- ``True`` when the top-k fallback was triggered
                           (no model passed ``tau_text``).
    """

    model_index: int
    model_id: Optional[str]
    fused_score: float
    candidates: Tuple[int, ...]
    used_fallback: bool


def retrieve(q_clip: Vector,
             q_dino: Vector,
             models: Sequence[OscarModel],
             tau_text: float = 0.37,
             top_k: int = 5,
             image_weight: float = 1.0) -> RetrievalResult:
    """Full two-stage OSCAR retrieval: text filter then image refinement.

    Returns the best model under :func:`late_fusion_score` among the
    text-filtered candidates. Ties in the fused score break to the lowest
    database index for determinism.
    """
    if not models:
        return RetrievalResult(-1, None, float("-inf"), (), False)

    sims_text = [text_similarity(q_clip, m) for m in models]
    passed = [i for i, s in enumerate(sims_text) if s >= tau_text]
    used_fallback = not passed
    candidates = filter_candidates(q_clip, models, tau_text, top_k)

    best_i = candidates[0]
    best_score = None
    for i in candidates:
        s_img = image_similarity(q_dino, models[i])
        fused = late_fusion_score(sims_text[i], s_img, image_weight)
        if best_score is None or fused > best_score:
            best_score = fused
            best_i = i
    return RetrievalResult(
        model_index=best_i,
        model_id=models[best_i].model_id,
        fused_score=best_score,
        candidates=tuple(candidates),
        used_fallback=used_fallback,
    )
