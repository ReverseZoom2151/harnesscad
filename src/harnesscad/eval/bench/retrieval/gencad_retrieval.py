"""Image-based CAD-program retrieval accuracy (R_B) for GenCAD's CCIP model.

Alam & Ahmed, *GenCAD* (2024), Section 4.3.2 / 5.2 (Table 2).

GenCAD evaluates its Contrastive CAD-Image Pre-training (CCIP) model with a
**batch retrieval accuracy** protocol.  Given a pool of paired (image latent,
CAD latent) examples, the protocol is:

  1. draw a batch of ``n_b`` examples from the pool;
  2. pick one example at random and use its *image* latent as the query;
  3. score cosine similarity between the query image latent and every one of the
     ``n_b`` *CAD* latents in the batch;
  4. count a hit if the highest-similarity CAD latent is the query's true match;
  5. repeat many times and report mean +/- std accuracy R_B (%).

The paper runs ``n_b = 10, 128, 1024, 2048`` and reports that CCIP is >15x more
accurate than an image-to-image search whose random-guess floor is ``1/n_b``.

This module implements the *protocol* deterministically on top of existing
primitives -- it does NOT reimplement contrastive training (learned/external) or
the cosine metric.  It reuses ``bench.geomretr_eval.cosine_distance`` for
scoring, so retrieval here is consistent with the repo's other retrieval evals;
the novelty is the batched-sampling accuracy estimator and its random-guess
baseline, which neither ``geomretr_eval`` (NDCG/mAP over a fixed gallery) nor
``ranked_retrieval_metrics`` (success@k on given rankings) provide.

Pure stdlib; all randomness flows through ``random.Random(seed)``.
"""

from __future__ import annotations

import random
from math import sqrt
from typing import List, Sequence, Tuple

from harnesscad.eval.bench.retrieval.geomretr_eval import cosine_distance

Vector = List[float]


def _argmin_distance(query: Vector, candidates: Sequence[Vector]) -> int:
    """Index of the candidate with smallest cosine distance (highest similarity).

    Ties break to the lowest index for determinism.
    """
    best_i = 0
    best_d = None
    for i, c in enumerate(candidates):
        d = cosine_distance(query, c)
        if best_d is None or d < best_d:
            best_d = d
            best_i = i
    return best_i


def batch_retrieval_hit(image_latents: Sequence[Vector],
                        cad_latents: Sequence[Vector],
                        query_pos: int) -> bool:
    """One retrieval trial: is the query image's nearest CAD latent its match?

    ``image_latents`` and ``cad_latents`` are the parallel latents of a single
    batch (same length); ``query_pos`` indexes which example supplies the query
    image.  A hit means the most-similar CAD latent is at ``query_pos``.
    """
    if len(image_latents) != len(cad_latents):
        raise ValueError("image and CAD latent batches must have equal length")
    if not (0 <= query_pos < len(image_latents)):
        raise ValueError("query_pos out of range")
    query = image_latents[query_pos]
    return _argmin_distance(query, cad_latents) == query_pos


def retrieval_accuracy(image_latents: Sequence[Vector],
                       cad_latents: Sequence[Vector],
                       batch_size: int,
                       repeats: int,
                       seed: int = 0) -> Tuple[float, float]:
    """Mean +/- std batch retrieval accuracy R_B over ``repeats`` random batches.

    Each repeat samples ``batch_size`` distinct examples from the pool (without
    replacement within a batch), picks one as the image query, and checks
    whether its true CAD latent is the top match among the batch.  Returns
    ``(mean, std)`` as *fractions* in ``[0, 1]`` (multiply by 100 for the paper's
    percentages).  The std is the population standard deviation of the per-repeat
    accuracies.
    """
    n = len(image_latents)
    if len(cad_latents) != n:
        raise ValueError("image and CAD latent pools must have equal length")
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2")
    if batch_size > n:
        raise ValueError("batch_size cannot exceed the pool size")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    rng = random.Random(seed)
    per_repeat: List[float] = []
    for _ in range(repeats):
        idx = rng.sample(range(n), batch_size)
        img_batch = [image_latents[i] for i in idx]
        cad_batch = [cad_latents[i] for i in idx]
        query_pos = rng.randrange(batch_size)
        hit = batch_retrieval_hit(img_batch, cad_batch, query_pos)
        per_repeat.append(1.0 if hit else 0.0)
    mean = sum(per_repeat) / len(per_repeat)
    var = sum((x - mean) ** 2 for x in per_repeat) / len(per_repeat)
    return mean, sqrt(var)


def random_guess_accuracy(batch_size: int) -> float:
    """Random-guess retrieval floor ``1 / n_b`` for a batch of ``batch_size``."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return 1.0 / batch_size


def retrieval_curve(image_latents: Sequence[Vector],
                    cad_latents: Sequence[Vector],
                    batch_sizes: Sequence[int],
                    repeats: int,
                    seed: int = 0) -> List[dict]:
    """R_B across several batch sizes (Table 2 row), with the random-guess floor.

    Returns one dict per ``batch_size`` with ``mean``, ``std``, ``random_guess``
    and ``lift`` (mean / random_guess -- the ">15x" figure the paper reports).
    """
    out: List[dict] = []
    for b in batch_sizes:
        mean, std = retrieval_accuracy(image_latents, cad_latents, b, repeats,
                                       seed=seed)
        floor = random_guess_accuracy(b)
        out.append({
            "batch_size": b,
            "mean": mean,
            "std": std,
            "random_guess": floor,
            "lift": (mean / floor) if floor > 0.0 else float("inf"),
        })
    return out
