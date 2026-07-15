"""UV-grid Gram style fingerprints (the deterministic core of UVStyle-Net).

Ported from *UVStyle-Net: Unsupervised Few-shot Learning of 3D Style Similarity
Measure for B-Reps* (Meltzer, Shayani et al.).  UVStyle-Net measures *style*
(not category) similarity between solids by taking the **Gram matrix** -- the
matrix of feature-channel correlations -- of a per-face UV-grid feature map, at
several network layers, and comparing two shapes as a weighted sum of per-layer
Gram distances.  The Gram matrix is the classic Gatys style representation: it
discards *where* features occur and keeps only *how channels co-vary*, which is
why it captures style independent of pose and sampling.

The network that produces the feature grids is learned and external.  But the
Gram construction, its normalisation variants (the paper's feature-wise
``fnorm`` and instance-wise ``inorm``), and the weighted per-layer style distance
(``few_shot.euclidean`` / ``few_shot.cosine``) are **pure deterministic linear
algebra**, and they apply verbatim to feature grids the harness can compute
itself with no model: sampling each B-Rep face on a UV grid and reading off
geometric channels (position, normal, principal curvatures, ...).

This module provides exactly that deterministic substrate:

* :func:`gram_vector` -- the (optionally normalised) Gram matrix of one UV-grid
  feature map, flattened to its upper triangle.
* :func:`style_fingerprint` -- a shape's fingerprint: one Gram vector per feature
  "layer" (channel group), pooling every face's samples.
* :func:`style_distance` -- the weighted per-layer distance between two
  fingerprints (Euclidean or cosine), i.e. UVStyle-Net's style metric.

Key deterministic property (tested): a Gram vector is invariant to the ordering
of the UV samples and of the faces, because it is a sum of per-sample outer
products -- so it is a genuine order-independent style signature.

This is a *style* descriptor (channel co-variance).  It is complementary to
``recognize/shape_descriptors`` (D2 distance histograms, radial shells), which
capture global *shape*, not channel style, and share no code with this module.

Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from typing import List, Sequence

__all__ = [
    "gram_vector",
    "style_fingerprint",
    "style_distance",
]

Sample = Sequence[float]        # one feature vector (channels)
Grid = Sequence[Sample]         # a UV-grid feature map: many samples
Fingerprint = List[List[float]]  # one Gram vector per layer


def _feature_normalise(samples: Sequence[Sample], n_ch: int) -> List[List[float]]:
    """L2-normalise each channel column across samples (UVStyle-Net ``fnorm``)."""
    cols_sq = [0.0] * n_ch
    for s in samples:
        for c in range(n_ch):
            cols_sq[c] += s[c] * s[c]
    norms = [math.sqrt(v) if v > 1e-30 else 1.0 for v in cols_sq]
    return [[s[c] / norms[c] for c in range(n_ch)] for s in samples]


def _instance_normalise(samples: Sequence[Sample], n_ch: int) -> List[List[float]]:
    """Subtract each channel's mean across samples (UVStyle-Net ``inorm``)."""
    means = [0.0] * n_ch
    for s in samples:
        for c in range(n_ch):
            means[c] += s[c]
    n = max(1, len(samples))
    means = [m / n for m in means]
    return [[s[c] - means[c] for c in range(n_ch)] for s in samples]


def gram_vector(grid: Grid, normalize: str = "none") -> List[float]:
    """Flattened upper-triangular Gram matrix of a UV-grid feature map.

    ``grid`` is a sequence of per-sample feature vectors, all the same length C.
    The Gram entry ``G[a][b] = mean_s grid[s][a] * grid[s][b]`` is the mean over
    samples of the channel-pair product; the returned vector is the upper
    triangle (including the diagonal), length ``C*(C+1)/2``.

    ``normalize`` selects a variant: ``"none"``, ``"feature"`` (fnorm: L2 each
    channel first), or ``"instance"`` (inorm: centre each channel first, giving
    the channel covariance).  Raises ``ValueError`` on an empty grid.
    """
    samples = [list(map(float, s)) for s in grid]
    if not samples:
        raise ValueError("empty UV grid")
    n_ch = len(samples[0])
    for s in samples:
        if len(s) != n_ch:
            raise ValueError("ragged feature grid (inconsistent channel count)")

    if normalize == "feature":
        samples = _feature_normalise(samples, n_ch)
    elif normalize == "instance":
        samples = _instance_normalise(samples, n_ch)
    elif normalize != "none":
        raise ValueError(f"unknown normalize mode {normalize!r}")

    n = len(samples)
    out: List[float] = []
    for a in range(n_ch):
        for b in range(a, n_ch):
            acc = 0.0
            for s in samples:
                acc += s[a] * s[b]
            out.append(acc / n)
    return out


def style_fingerprint(
    layers: Sequence[Grid], normalize: str = "none"
) -> Fingerprint:
    """A shape's style fingerprint: one :func:`gram_vector` per layer.

    Each element of ``layers`` is a feature grid pooling all UV samples of all
    faces for one channel group (e.g. layer 0 = positions, layer 1 = normals).
    The fingerprint is the list of their Gram vectors, matching UVStyle-Net's
    per-layer Gram stack.
    """
    return [gram_vector(layer, normalize=normalize) for layer in layers]


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-30 or nb <= 1e-30:
        return 0.0 if na <= 1e-30 and nb <= 1e-30 else 1.0
    dot = sum(x * y for x, y in zip(a, b))
    return 1.0 - dot / (na * nb)


def style_distance(
    fp1: Fingerprint,
    fp2: Fingerprint,
    weights: Sequence[float] | None = None,
    metric: str = "euclidean",
) -> float:
    """Weighted per-layer style distance between two fingerprints.

    Mirrors UVStyle-Net's ``few_shot`` distances: sum over layers of a per-layer
    distance (``"euclidean"`` or ``"cosine"``) scaled by ``weights`` (uniform if
    omitted).  The two fingerprints must have the same number of layers and
    matching per-layer lengths.
    """
    if len(fp1) != len(fp2):
        raise ValueError("fingerprints have different layer counts")
    n_layers = len(fp1)
    if weights is None:
        weights = [1.0] * n_layers
    if len(weights) != n_layers:
        raise ValueError("weights length must match number of layers")
    if metric == "euclidean":
        dist = _euclidean
    elif metric == "cosine":
        dist = _cosine
    else:
        raise ValueError(f"unknown metric {metric!r}")

    total = 0.0
    for l in range(n_layers):
        if len(fp1[l]) != len(fp2[l]):
            raise ValueError(f"layer {l} length mismatch")
        total += weights[l] * dist(fp1[l], fp2[l])
    return total
