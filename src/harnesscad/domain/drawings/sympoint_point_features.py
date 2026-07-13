"""SymPoint point-cloud feature engineering for CAD primitives (ECCV 2024).

Once every primitive has collapsed to a point (see
:mod:`drawings.sympoint_primitive_points`), SymPoint's dataset layer
(``svgnet/data/svg.py``) turns the point set into the tensors the network eats.
Every step is deterministic and reimplemented here, stdlib-only:

* :func:`normalize_args` -- divide raw drawing coordinates by a fixed scale
  (``COORD_SCALE = 140`` in the reference) rather than by the drawing extent,
  so the same physical size maps to the same number in every plan.
* :func:`point_feature` -- the 6-D per-point feature
  ``[polar_angle, length, onehot(command) x4]`` where

    - ``polar_angle = atan(y / x) / pi`` -- a *scale-invariant* orientation cue
      of the point about the drawing origin (SymPoint's ``arc`` feature), and
    - ``length = clip(raw_length, 0, COORD_SCALE) / COORD_SCALE``.

  Note the feature uses ``atan`` (half-turn, range ``(-1/2, 1/2)``), not
  ``atan2``: SymPoint deliberately identifies antipodal directions.
* :func:`center_coords` -- ``mean`` / ``min`` re-centring of the coordinate
  block (the ``data_norm`` option).
* :func:`recompute_polar_feature` -- after any coordinate change (shift, mix)
  the polar feature must be *recomputed*, since it depends on the coordinates;
  the reference does exactly this after augmentation.
* :func:`pad_to_min_points` -- zero-pad short drawings up to ``min_points``
  with background semantic id and stuff instance id ``-1``.
* :func:`build_batch` -- the collate step: concatenate per-drawing point blocks
  and emit the cumulative ``offset`` vector that delimits them, and offset the
  instance ids so instances stay unique across the batch.

The feature is what makes SymPoint different from the graph methods: no
adjacency at all, just a permutation-invariant point set with hand-built
geometric channels.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

#: Reference coordinate / length normalisation constant.
COORD_SCALE = 140.0

#: Background semantic id and stuff instance id of the reference dataset.
BACKGROUND_SEMANTIC_ID = 35
STUFF_INSTANCE_ID = -1

#: Number of command classes (line, arc, circle, ellipse).
NUM_COMMANDS = 4

#: Length of the per-point feature vector.
FEATURE_DIM = 2 + NUM_COMMANDS

Point = Tuple[float, float]


def normalize_args(args: Sequence[float], scale: float = COORD_SCALE) -> Tuple[float, ...]:
    """Divide a flat ``args`` record by ``scale``."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    return tuple(float(v) / scale for v in args)


def polar_angle(point: Point) -> float:
    """SymPoint's ``arc`` channel: ``atan(y / x) / pi`` in ``(-0.5, 0.5)``."""
    x, y = point
    return math.atan(y / (x + 1e-8)) / math.pi


def normalized_length(length: float, scale: float = COORD_SCALE) -> float:
    """``clip(length, 0, scale) / scale``."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    if length < 0.0:
        length = 0.0
    elif length > scale:
        length = scale
    return length / scale


def command_onehot(command: int) -> Tuple[float, ...]:
    """One-hot encoding of a command id in ``[0, NUM_COMMANDS)``."""
    if not 0 <= command < NUM_COMMANDS:
        raise ValueError("command id out of range: %r" % (command,))
    return tuple(1.0 if i == command else 0.0 for i in range(NUM_COMMANDS))


def point_feature(point: Point, length: float, command: int,
                  scale: float = COORD_SCALE) -> Tuple[float, ...]:
    """The 6-D SymPoint feature of one primitive-point.

    ``point`` must already be in normalised coordinates; ``length`` is the raw
    (unnormalised) arc length.
    """
    return (polar_angle(point), normalized_length(length, scale)) + command_onehot(command)


def recompute_polar_feature(features: Sequence[Sequence[float]],
                            coords: Sequence[Point]) -> List[List[float]]:
    """Refresh channel 0 of every feature from (possibly moved) ``coords``."""
    if len(features) != len(coords):
        raise ValueError("features and coords must be the same length")
    out: List[List[float]] = []
    for feat, pt in zip(features, coords):
        row = [float(v) for v in feat]
        if len(row) != FEATURE_DIM:
            raise ValueError("feature must have %d channels" % FEATURE_DIM)
        row[0] = polar_angle(pt)
        out.append(row)
    return out


def scale_coords(coords: Sequence[Point], features: Sequence[Sequence[float]],
                 factor: float) -> Tuple[List[Point], List[List[float]]]:
    """Scale coordinates by ``factor`` and the length channel with them."""
    new_coords = [(p[0] * factor, p[1] * factor) for p in coords]
    new_feats = []
    for feat in features:
        row = [float(v) for v in feat]
        row[1] = row[1] * factor
        new_feats.append(row)
    return new_coords, recompute_polar_feature(new_feats, new_coords)


def center_coords(coords: Sequence[Point], mode: str = "mean") -> List[Point]:
    """Re-centre a coordinate block; ``mode`` is ``mean``, ``min`` or ``none``."""
    pts = [(float(x), float(y)) for x, y in coords]
    if mode == "none":
        return pts
    if not pts:
        return pts
    if mode == "mean":
        ox = sum(p[0] for p in pts) / len(pts)
        oy = sum(p[1] for p in pts) / len(pts)
    elif mode == "min":
        ox = min(p[0] for p in pts)
        oy = min(p[1] for p in pts)
    else:
        raise ValueError("unknown normalisation mode: %r" % (mode,))
    return [(p[0] - ox, p[1] - oy) for p in pts]


def build_point_cloud(points: Sequence[Point], lengths: Sequence[float],
                      commands: Sequence[int], semantic_ids: Sequence[int],
                      instance_ids: Sequence[int], scale: float = COORD_SCALE,
                      norm: str = "mean") -> Dict[str, object]:
    """Full SymPoint sample: normalised coords, features, labels, raw lengths."""
    n = len(points)
    if not (len(lengths) == len(commands) == len(semantic_ids) == len(instance_ids) == n):
        raise ValueError("all input sequences must be the same length")
    coords = [(p[0] / scale, p[1] / scale) for p in points]
    feats = [point_feature(c, l, cmd, scale)
             for c, l, cmd in zip(coords, lengths, commands)]
    coords = center_coords(coords, norm)
    return {
        "coords": coords,
        "features": [list(f) for f in feats],
        "semantic_ids": [int(s) for s in semantic_ids],
        "instance_ids": [int(i) for i in instance_ids],
        "lengths": [float(l) for l in lengths],
    }


def pad_to_min_points(sample: Dict[str, object], min_points: int) -> Dict[str, object]:
    """Zero-pad a sample up to ``min_points`` points (background / stuff labels)."""
    coords = list(sample["coords"])  # type: ignore[arg-type]
    n = len(coords)
    if n >= min_points:
        return {k: list(v) for k, v in sample.items()}  # type: ignore[union-attr]
    pad = min_points - n
    return {
        "coords": coords + [(0.0, 0.0)] * pad,
        "features": [list(f) for f in sample["features"]] + [[0.0] * FEATURE_DIM] * pad,  # type: ignore[union-attr]
        "semantic_ids": list(sample["semantic_ids"]) + [BACKGROUND_SEMANTIC_ID] * pad,  # type: ignore[arg-type]
        "instance_ids": list(sample["instance_ids"]) + [STUFF_INSTANCE_ID] * pad,  # type: ignore[arg-type]
        "lengths": list(sample["lengths"]) + [0.0] * pad,  # type: ignore[arg-type]
    }


def build_batch(samples: Sequence[Dict[str, object]],
                instance_stride: int = 2048) -> Dict[str, object]:
    """Concatenate samples, emit cumulative offsets, and globalise instance ids.

    Instance ids of sample ``k`` are shifted by ``k * instance_stride`` so that
    identically-numbered instances of different drawings do not collide; stuff
    points (id ``-1``) stay ``-1``.
    """
    coords: List[Point] = []
    feats: List[List[float]] = []
    sems: List[int] = []
    inss: List[int] = []
    lens: List[float] = []
    offsets: List[int] = []
    count = 0
    for k, sample in enumerate(samples):
        block = list(sample["coords"])  # type: ignore[arg-type]
        coords.extend(block)
        feats.extend([list(f) for f in sample["features"]])  # type: ignore[union-attr]
        sems.extend(int(s) for s in sample["semantic_ids"])  # type: ignore[union-attr]
        for ins in sample["instance_ids"]:  # type: ignore[union-attr]
            ins = int(ins)
            inss.append(ins if ins < 0 else ins + k * instance_stride)
        lens.extend(float(v) for v in sample["lengths"])  # type: ignore[union-attr]
        count += len(block)
        offsets.append(count)
    return {"coords": coords, "features": feats, "semantic_ids": sems,
            "instance_ids": inss, "lengths": lens, "offsets": offsets}
