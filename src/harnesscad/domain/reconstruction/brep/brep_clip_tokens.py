"""BRep dual-vocabulary tokenisation and pooled descriptor (Usama et al., 2026,
"BRepCLIP: Contrastive Multimodal Pretraining on BRep Primitives for CAD
Understanding").

BRepCLIP models a CAD object as a sequence of *face* and *edge* tokens with
*separate discrete vocabularies* for surface and curve geometry, each augmented
with spatial and semantic descriptors (surface type: plane / cylinder / cone /
sphere / torus / nurbs; curve primitive: line / circle / arc / ellipse / bspline).
A transformer aggregates the tokens into a global BRep embedding used as a
CAD-aware similarity metric. The learned transformer/CLIP alignment is out of
scope, but the deterministic front-end -- the two vocabularies, the per-token
feature construction, and a fixed pooling into a global descriptor -- is
buildable and is exactly what makes the representation "structure-aware":

* :data:`SURFACE_VOCAB` / :data:`CURVE_VOCAB` -- the discrete type vocabularies.
* :func:`face_token` / :func:`edge_token` -- one-hot type id + normalised spatial
  descriptor (area or length) for a primitive.
* :func:`brep_descriptor` -- mean-pool the face and edge token features into a
  fixed-length global descriptor (a deterministic stand-in for the embedding).
* :func:`descriptor_similarity` -- cosine similarity between two descriptors, the
  CAD-aware similarity metric the paper proposes for evaluating generation.

Deterministic, stdlib-only (``math`` only).
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence

__all__ = [
    "SURFACE_VOCAB",
    "CURVE_VOCAB",
    "face_token",
    "edge_token",
    "brep_descriptor",
    "descriptor_similarity",
]

SURFACE_VOCAB: Sequence[str] = ("plane", "cylinder", "cone", "sphere", "torus", "nurbs")
CURVE_VOCAB: Sequence[str] = ("line", "circle", "arc", "ellipse", "bspline")


def _one_hot(value: str, vocab: Sequence[str]) -> List[float]:
    if value not in vocab:
        raise ValueError(f"unknown type {value!r}; expected one of {tuple(vocab)}")
    return [1.0 if v == value else 0.0 for v in vocab]


def face_token(surface_type: str, area: float) -> List[float]:
    """One-hot surface type followed by a log-scaled area descriptor."""
    if area < 0:
        raise ValueError("area must be non-negative")
    return _one_hot(surface_type, SURFACE_VOCAB) + [math.log1p(float(area))]


def edge_token(curve_type: str, length: float) -> List[float]:
    """One-hot curve type followed by a log-scaled length descriptor."""
    if length < 0:
        raise ValueError("length must be non-negative")
    return _one_hot(curve_type, CURVE_VOCAB) + [math.log1p(float(length))]


def _mean_pool(tokens: Sequence[Sequence[float]], width: int) -> List[float]:
    if not tokens:
        return [0.0] * width
    acc = [0.0] * width
    for tok in tokens:
        for i, v in enumerate(tok):
            acc[i] += v
    return [v / len(tokens) for v in acc]


def brep_descriptor(
    faces: Sequence[Mapping[str, object]],
    edges: Sequence[Mapping[str, object]],
) -> List[float]:
    """Global BRep descriptor = concat(mean face token, mean edge token).

    ``faces`` items are ``{"type": surface_type, "area": float}`` and ``edges``
    items ``{"type": curve_type, "length": float}``. The descriptor has fixed
    length ``(len(SURFACE_VOCAB)+1) + (len(CURVE_VOCAB)+1)`` regardless of the
    face/edge counts, so any two BReps are directly comparable.
    """
    face_toks = [face_token(str(f["type"]), float(f["area"])) for f in faces]
    edge_toks = [edge_token(str(e["type"]), float(e["length"])) for e in edges]
    fw = len(SURFACE_VOCAB) + 1
    ew = len(CURVE_VOCAB) + 1
    return _mean_pool(face_toks, fw) + _mean_pool(edge_toks, ew)


def descriptor_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two BRep descriptors (0 if either is all-zero)."""
    if len(a) != len(b):
        raise ValueError("descriptors must have equal length")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
