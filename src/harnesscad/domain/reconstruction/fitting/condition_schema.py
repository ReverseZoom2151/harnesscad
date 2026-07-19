"""Deterministic multi-modal condition-encoding schema.

Generation is conditioned on text, images, or point clouds, encoding
each active voxel with a fixed feature layout ``f_i = [p(v_i); n(v_i);
c(v_i); s(v_i)]`` (visual features, per-view normals, voxel centre, signed
distance).  The learned encoders (large vision backbones, point-cloud encoders,
flow transformers) are research-heavy and out of scope, but the *schema* -- the
deterministic,
fixed-length, normalised feature layout and the modality tag -- is buildable
and testable.

This module provides:

  * ``Modality`` -- the supported conditioning modalities.
  * ``ConditionSchema`` -- a fixed feature layout with a one-hot modality tag
    plus a fixed-width feature block, with deterministic encoders:
      - text  -> hashing-trick bag-of-words (stable across runs),
      - points -> normalised occupancy + centroid + extent statistics,
      - voxel-feature -> the [p; n; c; s] concatenation.

Encoders are deterministic (a fixed hash seed, no wall clock, no RNG state)
and always return an L2-normalised vector of the schema's declared length.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from math import sqrt


class Modality(Enum):
    TEXT = "text"
    IMAGE = "image"
    POINT = "point"
    VOXEL = "voxel"


_MODALITY_ORDER = (Modality.TEXT, Modality.IMAGE, Modality.POINT, Modality.VOXEL)


def _one_hot(modality):
    return [1.0 if modality is m else 0.0 for m in _MODALITY_ORDER]


def _l2_normalise(vec):
    norm = sqrt(sum(x * x for x in vec))
    if norm <= 0.0:
        return list(vec)
    return [x / norm for x in vec]


def _stable_bucket(token, buckets):
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


@dataclass(frozen=True)
class ConditionSchema:
    """Fixed-length condition feature layout.

    The output vector is ``one_hot(modality) (len 4) ++ feature_block`` where
    ``feature_block`` has length ``feature_dim``.  The full vector length is
    ``len(_MODALITY_ORDER) + feature_dim``.
    """

    feature_dim: int = 32

    @property
    def length(self):
        return len(_MODALITY_ORDER) + self.feature_dim

    def _wrap(self, modality, feature_block):
        if len(feature_block) != self.feature_dim:
            raise ValueError("feature block has wrong length")
        vec = _one_hot(modality) + _l2_normalise(feature_block)
        return vec

    def encode_text(self, text):
        """Hashing-trick bag-of-words over whitespace tokens (deterministic)."""
        block = [0.0] * self.feature_dim
        tokens = text.lower().split()
        for token in tokens:
            block[_stable_bucket(token, self.feature_dim)] += 1.0
        return self._wrap(Modality.TEXT, block)

    def encode_points(self, points, *, grid=4):
        """Normalised occupancy grid + centroid + extent for a point cloud.

        Points are normalised to [-0.5, 0.5]^3 before
        binning into a ``grid^3`` occupancy histogram; the tail of the block
        carries the centroid (3) and extent (3).  Requires
        ``feature_dim >= grid**3 + 6``.
        """
        if not points:
            raise ValueError("point cloud is empty")
        occupancy_dim = grid ** 3
        if self.feature_dim < occupancy_dim + 6:
            raise ValueError("feature_dim too small for the requested grid")
        mins = [min(p[d] for p in points) for d in range(3)]
        maxs = [max(p[d] for p in points) for d in range(3)]
        extent = [maxs[d] - mins[d] for d in range(3)]
        span = max(extent) if max(extent) > 0 else 1.0
        centroid = [sum(p[d] for p in points) / len(points) for d in range(3)]

        block = [0.0] * self.feature_dim
        for p in points:
            cell = []
            for d in range(3):
                # normalise to [0, 1) using the uniform span, then bin.
                t = (p[d] - mins[d]) / span
                cell.append(min(grid - 1, int(t * grid)))
            index = (cell[0] * grid + cell[1]) * grid + cell[2]
            block[index] += 1.0
        for d in range(3):
            block[occupancy_dim + d] = centroid[d]
        for d in range(3):
            block[occupancy_dim + 3 + d] = extent[d]
        return self._wrap(Modality.POINT, block)

    def encode_voxel_feature(self, visual, normal, center, sdf):
        """Assemble f_i = [p(v_i); n(v_i); c(v_i); s(v_i)].

        ``visual`` and ``normal`` are arbitrary-length sequences; ``center``
        is a length-3 tuple; ``sdf`` is a scalar.  The concatenation is padded
        or truncated to the schema's ``feature_dim`` and returned tagged VOXEL.
        """
        if len(center) != 3:
            raise ValueError("center must be a 3-tuple")
        raw = list(visual) + list(normal) + list(center) + [float(sdf)]
        if len(raw) >= self.feature_dim:
            block = raw[: self.feature_dim]
        else:
            block = raw + [0.0] * (self.feature_dim - len(raw))
        return self._wrap(Modality.VOXEL, block)

    def modality_of(self, vector):
        """Recover the modality from an encoded vector's one-hot prefix."""
        if len(vector) != self.length:
            raise ValueError("vector length does not match schema")
        prefix = vector[: len(_MODALITY_ORDER)]
        return _MODALITY_ORDER[max(range(len(prefix)), key=prefix.__getitem__)]
