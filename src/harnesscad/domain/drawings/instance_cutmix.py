"""Symbol-instance CutMix and point-set augmentation (SymPoint, ECCV 2024).

Symbol spotting suffers a brutal class imbalance: a floor plan is mostly wall,
and a rare symbol (revolving door, escalator) may appear once per drawing.
SymPoint's answer (``svgnet/data/svg.py::transform_train``) is a *point-level
CutMix*: whole **thing** instances harvested from previously seen drawings are
kept in a bounded FIFO queue and pasted, with a relative shift, into the current
point set.  Because a primitive is just a point, pasting is a concatenation --
no rasterisation, no occlusion reasoning, no re-meshing.

Reimplemented here, stdlib-only, with the randomness lifted out into explicit
arguments (shifts, angles, seeds) so every operation is reproducible:

* :func:`hflip`, :func:`vflip`, :func:`rotate` -- the point-set geometric
  augmentations, all about the *canvas centre* rather than the point centroid.
* :func:`shift`, :func:`scale` -- translation and scaling (scaling also rescales
  the length feature channel).
* :func:`extract_instances` -- split a sample into its thing instances (stuff
  classes and unlabelled points are never harvested).
* :class:`InstanceQueue` -- the bounded FIFO of harvested instances: new
  instances are pushed at the front and the oldest fall off the back once the
  queue is full (``queueK`` in the reference config).
* :func:`cutmix` -- paste the queue into a sample at a relative shift, then
  *recompute the polar feature channel* (it depends on the coordinates, so a
  paste invalidates it -- the reference is careful about this and so are we).
* :func:`shuffle` -- seeded permutation of the point order; the network is
  permutation-invariant but the check keeps the pipeline honest.

Samples are the dicts produced by
:func:`drawings.sympoint_point_features.build_point_cloud`.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple

from harnesscad.domain.drawings.point_features import recompute_polar_feature

Point = Tuple[float, float]
Sample = Dict[str, object]

#: Semantic ids >= this are "stuff" (wall, railing, ...) and are never harvested.
STUFF_START = 30
UNLABELLED_INSTANCE = -1


def _coords(sample: Sample) -> List[Point]:
    return [(float(x), float(y)) for x, y in sample["coords"]]  # type: ignore[union-attr]


def _rebuild(sample: Sample, coords: Sequence[Point],
             features: Sequence[Sequence[float]]) -> Sample:
    return {
        "coords": [(float(x), float(y)) for x, y in coords],
        "features": recompute_polar_feature(features, coords),
        "semantic_ids": [int(s) for s in sample["semantic_ids"]],  # type: ignore[union-attr]
        "instance_ids": [int(i) for i in sample["instance_ids"]],  # type: ignore[union-attr]
        "lengths": [float(v) for v in sample["lengths"]],  # type: ignore[union-attr]
    }


def hflip(sample: Sample, width: float = 1.0) -> Sample:
    """Mirror the point set about the canvas vertical mid-line."""
    coords = [(width - x, y) for x, y in _coords(sample)]
    return _rebuild(sample, coords, sample["features"])  # type: ignore[arg-type]


def vflip(sample: Sample, height: float = 1.0) -> Sample:
    """Mirror the point set about the canvas horizontal mid-line."""
    coords = [(x, height - y) for x, y in _coords(sample)]
    return _rebuild(sample, coords, sample["features"])  # type: ignore[arg-type]


def rotate(sample: Sample, angle_degrees: float, width: float = 1.0,
           height: float = 1.0) -> Sample:
    """Rotate the point set about the canvas centre ``(width/2, height/2)``."""
    rad = math.radians(angle_degrees)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    ax, ay = width / 2.0, height / 2.0
    coords = []
    for x, y in _coords(sample):
        dx, dy = x - ax, y - ay
        coords.append((dx * cos_a - dy * sin_a + ax, dx * sin_a + dy * cos_a + ay))
    return _rebuild(sample, coords, sample["features"])  # type: ignore[arg-type]


def shift(sample: Sample, dx: float, dy: float) -> Sample:
    """Translate the point set."""
    coords = [(x + dx, y + dy) for x, y in _coords(sample)]
    return _rebuild(sample, coords, sample["features"])  # type: ignore[arg-type]


def scale(sample: Sample, factor: float) -> Sample:
    """Scale coordinates about the origin; the length channel scales with them."""
    coords = [(x * factor, y * factor) for x, y in _coords(sample)]
    feats = []
    for row in sample["features"]:  # type: ignore[union-attr]
        row = [float(v) for v in row]
        row[1] = row[1] * factor
        feats.append(row)
    return _rebuild(sample, coords, feats)


def extract_instances(sample: Sample, stuff_start: int = STUFF_START) -> List[Sample]:
    """Harvest the *thing* instances of a sample, sorted by ``(sem, ins)``."""
    groups: Dict[Tuple[int, int], List[int]] = {}
    sems = [int(s) for s in sample["semantic_ids"]]  # type: ignore[union-attr]
    inss = [int(i) for i in sample["instance_ids"]]  # type: ignore[union-attr]
    for idx, (sem, ins) in enumerate(zip(sems, inss)):
        if sem >= stuff_start or ins == UNLABELLED_INSTANCE:
            continue
        groups.setdefault((sem, ins), []).append(idx)
    coords = _coords(sample)
    feats = [[float(v) for v in row] for row in sample["features"]]  # type: ignore[union-attr]
    lengths = [float(v) for v in sample["lengths"]]  # type: ignore[union-attr]
    out: List[Sample] = []
    for key in sorted(groups):
        idxs = groups[key]
        out.append({
            "coords": [coords[i] for i in idxs],
            "features": [list(feats[i]) for i in idxs],
            "semantic_ids": [sems[i] for i in idxs],
            "instance_ids": [inss[i] for i in idxs],
            "lengths": [lengths[i] for i in idxs],
        })
    return out


class InstanceQueue:
    """Bounded FIFO of harvested thing instances (SymPoint ``instance_queues``).

    ``push`` inserts at the front; once the queue holds ``capacity`` instances
    the oldest one falls off the back.  Iteration order is newest-first, which
    is the order the reference pastes them in.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        self.capacity = capacity
        self.items: List[Sample] = []

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def push(self, instance: Sample) -> "InstanceQueue":
        self.items.insert(0, instance)
        while len(self.items) > self.capacity:
            self.items.pop()
        return self

    def push_sample(self, sample: Sample, stuff_start: int = STUFF_START) -> "InstanceQueue":
        """Harvest every thing instance of ``sample`` into the queue."""
        for instance in extract_instances(sample, stuff_start):
            self.push(instance)
        return self


def cutmix(sample: Sample, queue: InstanceQueue, dx: float, dy: float) -> Sample:
    """Paste every queued instance into ``sample``, shifted by ``(dx, dy)``.

    The polar feature channel of the *whole* result is recomputed, since the
    pasted points sit at new positions relative to the origin.
    """
    coords = _coords(sample)
    feats = [[float(v) for v in row] for row in sample["features"]]  # type: ignore[union-attr]
    sems = [int(s) for s in sample["semantic_ids"]]  # type: ignore[union-attr]
    inss = [int(i) for i in sample["instance_ids"]]  # type: ignore[union-attr]
    lengths = [float(v) for v in sample["lengths"]]  # type: ignore[union-attr]
    for instance in queue:
        coords.extend((x + dx, y + dy) for x, y in instance["coords"])  # type: ignore[union-attr]
        feats.extend([float(v) for v in row] for row in instance["features"])  # type: ignore[union-attr]
        sems.extend(int(s) for s in instance["semantic_ids"])  # type: ignore[union-attr]
        inss.extend(int(i) for i in instance["instance_ids"])  # type: ignore[union-attr]
        lengths.extend(float(v) for v in instance["lengths"])  # type: ignore[union-attr]
    return {
        "coords": coords,
        "features": recompute_polar_feature(feats, coords),
        "semantic_ids": sems,
        "instance_ids": inss,
        "lengths": lengths,
    }


def shuffle(sample: Sample, seed: int) -> Sample:
    """Seeded permutation of the point order (all channels move together)."""
    n = len(sample["coords"])  # type: ignore[arg-type]
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return {
        "coords": [tuple(sample["coords"][i]) for i in order],  # type: ignore[index]
        "features": [list(sample["features"][i]) for i in order],  # type: ignore[index]
        "semantic_ids": [int(sample["semantic_ids"][i]) for i in order],  # type: ignore[index]
        "instance_ids": [int(sample["instance_ids"][i]) for i in order],  # type: ignore[index]
        "lengths": [float(sample["lengths"][i]) for i in order],  # type: ignore[index]
    }
