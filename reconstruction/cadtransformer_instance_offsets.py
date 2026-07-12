"""Instance centroid-offset targets for primitive grouping (CADTransformer).

To group vectorised primitives into symbol instances, CADTransformer regresses,
for every primitive, the vector from its own centre to the centroid of all
primitives sharing its instance id (``dataset.get_instance_center_tensor`` /
``offset = xy - instance_center``).  Predicting this offset lets a downstream
step shift each primitive toward its instance centre and cluster them -- the
same "vote to the centroid" trick point-cloud instance segmentation uses.

Everything here is deterministic and stdlib-only:

* :func:`instance_centroids` -- centroid of each instance's primitive centres.
* :func:`offset_targets` -- per-primitive offset to its instance centroid;
  background primitives (instance id ``background_id``, default ``-1``) get a
  sentinel offset.
* :func:`shift_to_centroid` -- apply the offsets (centre + offset), reproducing
  the instance centroids for foreground primitives.
* :func:`group_by_shifted_center` -- cluster shifted centres by proximity,
  recovering instances from (noisy) predicted offsets.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Point = Tuple[float, float]

SENTINEL = (-999.0, -999.0)


def instance_centroids(centers: Sequence[Point], instances: Sequence[int],
                       background_id: int = -1) -> Dict[int, Point]:
    """Mean centre of every non-background instance.

    ``centers[i]`` is the centre of primitive ``i`` and ``instances[i]`` its
    instance id.  Returns ``{instance_id: (mean_x, mean_y)}``.
    """
    if len(centers) != len(instances):
        raise ValueError("centers and instances must have equal length")
    acc: Dict[int, List[float]] = {}
    for (x, y), inst in zip(centers, instances):
        if inst == background_id:
            continue
        if inst not in acc:
            acc[inst] = [0.0, 0.0, 0.0]
        acc[inst][0] += x
        acc[inst][1] += y
        acc[inst][2] += 1.0
    return {inst: (sx / n, sy / n) for inst, (sx, sy, n) in acc.items()}


def offset_targets(centers: Sequence[Point], instances: Sequence[int],
                   background_id: int = -1,
                   sentinel: Point = SENTINEL) -> List[Point]:
    """Per-primitive offset ``centroid - center`` to its instance centroid.

    Background primitives receive ``sentinel`` (default ``(-999, -999)``),
    matching the reference's masked-out target.
    """
    centroids = instance_centroids(centers, instances, background_id)
    out: List[Point] = []
    for (x, y), inst in zip(centers, instances):
        if inst == background_id:
            out.append(sentinel)
        else:
            cx, cy = centroids[inst]
            out.append((cx - x, cy - y))
    return out


def shift_to_centroid(centers: Sequence[Point], offsets: Sequence[Point],
                      instances: Sequence[int],
                      background_id: int = -1) -> List[Point]:
    """Apply offsets: shifted centre ``center + offset`` for foreground nodes.

    Background primitives are returned unchanged.
    """
    if not (len(centers) == len(offsets) == len(instances)):
        raise ValueError("centers, offsets and instances must have equal length")
    out: List[Point] = []
    for (x, y), (ox, oy), inst in zip(centers, offsets, instances):
        if inst == background_id:
            out.append((x, y))
        else:
            out.append((x + ox, y + oy))
    return out


def group_by_shifted_center(shifted: Sequence[Point], instances: Sequence[int],
                            tol: float = 1e-6, background_id: int = -1
                            ) -> List[int]:
    """Cluster shifted centres by proximity into instance labels.

    Two foreground primitives join the same cluster when their shifted centres
    are within ``tol`` (Chebyshev distance).  Returns a per-primitive cluster
    label; background primitives get ``background_id``.  Deterministic:
    clusters are seeded in primitive order.
    """
    if len(shifted) != len(instances):
        raise ValueError("shifted and instances must have equal length")
    labels: List[int] = [background_id] * len(shifted)
    reps: List[Tuple[int, Point]] = []  # (label, representative point)
    next_label = 0
    for i, ((x, y), inst) in enumerate(zip(shifted, instances)):
        if inst == background_id:
            continue
        found = None
        for label, (rx, ry) in reps:
            if abs(x - rx) <= tol and abs(y - ry) <= tol:
                found = label
                break
        if found is None:
            found = next_label
            reps.append((found, (x, y)))
            next_label += 1
        labels[i] = found
    return labels
