"""Interface Match: keep-in / keep-out mating-feature scoring.

The interface axis asks "would the candidate bolt up to the same fixture?".
Every mating feature is authored as a sub-volume of one of two kinds:

- **KOR** (keep-out region): the candidate must be **empty** there (a bolt hole,
  a slot); material in that space would block the mating part.
- **KIR** (keep-in region): the candidate must be **solid** there (a locating
  boss or pin); missing material leaves nothing to mate against.

Sub-volumes that must seat together form one **mating group** (e.g. two bolt
holes plus a slot that a single jig drops into). A part can carry several
independent groups.

This module implements the deterministic scaffolding of the axis: the authored
sub-volume naming contract, the bounded pose search grid, the IoU-to-score
ramp, and the group aggregation. The volumetric IoU itself needs a boolean
kernel and is injected as a callable, so the aggregation is testable with a
stub.

Scoring, exactly as this benchmark axis defines it:

1. Per-feature fit is a volumetric IoU between the region (plus a verification
   shell of the opposite material, which is what makes an *oversize* feature
   fail too) and the candidate.
2. The region is re-scored over a **bounded pose search** (+/- 1 degree and
   +/- 1% of the part size per axis) and the best IoU is kept, so a feature is
   not punished for the residual of whole-part alignment. The search stops
   early once an IoU crosses :data:`SATURATION_THRESHOLD`.
3. Each IoU goes through a **pass/fail ramp**: ``>= 0.95`` maps to ``1.0``,
   ``<= 0.80`` maps to ``0.0``, linear in between. A sloppy fit banks no credit.
4. A **group scores as its worst feature** (the minimum), and the sample scores
   as the **mean over groups**, so nailing one independent interface while
   missing another still earns partial credit.

Nothing equivalent exists in the harness: no module models keep-in/keep-out
regions, the min-within-group / mean-across-groups aggregation, or the
0.80/0.95 pass ramp.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

FIT_TYPES = ("KOR", "KIR")

# Authored sub-volume filename contract: jig_<group>__<index>__<KOR|KIR>.step
SUBVOL_RE = re.compile(r"^jig_(\d+)__(\d+)__(KOR|KIR)\.(?:step|stp)$")

# Pass/fail ramp. Above this IoU a feature is a full pass ...
INTERFACE_FULL_SCORE_IOU = 0.95
# ... and at or below this it is a clean fail; linear in between.
INTERFACE_ZERO_SCORE_IOU = 0.80
# Reported "did this feature pass" threshold (same value as the full-score end).
DEFAULT_IOU_THRESHOLD = 0.95
# The pose search stops as soon as an IoU crosses this: a fit this good cannot
# be meaningfully improved, and every further pose is wasted boolean work.
SATURATION_THRESHOLD = 0.99

# Bounded pose search window.
DEFAULT_MAX_ROTATION_DEG = 1.0
TRANSLATION_FRACTION_OF_BBOX = 0.01
DEFAULT_N_SAMPLES = 32

Pose = Tuple[float, float, float, float, float, float]  # rx, ry, rz (deg), tx, ty, tz
ZERO_POSE: Pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class SubVolumeNameError(ValueError):
    """A sub-volume filename does not follow the authored naming contract."""


@dataclass(frozen=True)
class SubVolume:
    """One authored mating feature."""

    group: int
    index: int
    fit_type: str
    filename: str = ""

    def __post_init__(self) -> None:
        if self.fit_type not in FIT_TYPES:
            raise SubVolumeNameError(f"unknown fit type: {self.fit_type!r}")

    @property
    def name(self) -> str:
        return f"jig_{self.group}__{self.index}__{self.fit_type}"


@dataclass(frozen=True)
class GroupScore:
    group: int
    per_feature_iou: Dict[str, float]
    per_feature_score: Dict[str, float]
    score: float
    worst_feature: str


@dataclass(frozen=True)
class InterfaceResult:
    score: float
    groups: List[GroupScore] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "contexts": {
                str(g.group): {
                    "score": g.score,
                    "worst_feature": g.worst_feature,
                    "per_feature_iou": dict(g.per_feature_iou),
                    "per_feature_score": dict(g.per_feature_score),
                }
                for g in self.groups
            },
        }


# ---------------------------------------------------------------------------
# Sub-volume discovery
# ---------------------------------------------------------------------------


def parse_sub_volume(filename: str) -> SubVolume:
    """Parse ``jig_<group>__<index>__<KOR|KIR>.step`` into a :class:`SubVolume`."""
    match = SUBVOL_RE.match(filename)
    if match is None:
        raise SubVolumeNameError(
            f"{filename!r} does not match the sub-volume contract "
            "jig_<group>__<index>__<KOR|KIR>.step"
        )
    group, index, fit_type = match.groups()
    return SubVolume(
        group=int(group), index=int(index), fit_type=fit_type, filename=filename
    )


def discover_sub_volumes(filenames: Iterable[str]) -> List[SubVolume]:
    """Parse every sub-volume filename, ignoring non-matching files.

    Returned in ``(group, index)`` order, so the scoring is order-independent
    of how the directory happened to be listed.
    """
    found: List[SubVolume] = []
    for name in filenames:
        if SUBVOL_RE.match(name):
            found.append(parse_sub_volume(name))
    return sorted(found, key=lambda sv: (sv.group, sv.index))


def group_sub_volumes(sub_volumes: Iterable[SubVolume]) -> Dict[int, List[SubVolume]]:
    """Bucket sub-volumes into their mating groups (ordered by index)."""
    groups: Dict[int, List[SubVolume]] = {}
    for sv in sub_volumes:
        groups.setdefault(sv.group, []).append(sv)
    for members in groups.values():
        members.sort(key=lambda sv: sv.index)
    return dict(sorted(groups.items()))


# ---------------------------------------------------------------------------
# Ramp
# ---------------------------------------------------------------------------


def iou_to_interface_score(iou: float) -> float:
    """Map a raw IoU onto the pass/fail ramp, in ``[0, 1]``.

    ``>= INTERFACE_FULL_SCORE_IOU`` -> 1.0, ``<= INTERFACE_ZERO_SCORE_IOU`` ->
    0.0, linear between. The steep ramp is what stops a sloppy fit from banking
    partial credit: an IoU of 0.85 is a badly-placed feature, not "85% right".
    """
    if iou >= INTERFACE_FULL_SCORE_IOU:
        return 1.0
    if iou <= INTERFACE_ZERO_SCORE_IOU:
        return 0.0
    span = INTERFACE_FULL_SCORE_IOU - INTERFACE_ZERO_SCORE_IOU
    return (iou - INTERFACE_ZERO_SCORE_IOU) / span


# ---------------------------------------------------------------------------
# Bounded pose search
# ---------------------------------------------------------------------------


def _van_der_corput(n: int, base: int) -> float:
    """Deterministic low-discrepancy sequence value in ``[0, 1)``."""
    value = 0.0
    denom = 1.0
    while n > 0:
        denom *= base
        n, rem = divmod(n, base)
        value += rem / denom
    return value


def pose_grid(
    bbox_diagonal: float,
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
    max_rotation_deg: float = DEFAULT_MAX_ROTATION_DEG,
    translation_fraction: float = TRANSLATION_FRACTION_OF_BBOX,
) -> List[Pose]:
    """Deterministic bounded pose window around the authored pose.

    The identity pose always comes first (so a perfectly-placed feature is
    scored at its authored pose and can saturate immediately); the remaining
    ``n_samples - 1`` poses are a Halton sequence mapped onto the symmetric
    window ``+/- max_rotation_deg`` per rotation axis and
    ``+/- translation_fraction * bbox_diagonal`` per translation axis. A Halton
    grid (rather than an RNG) keeps the search reproducible bit-for-bit.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if bbox_diagonal < 0:
        raise ValueError("bbox_diagonal must be non-negative")
    max_translation = translation_fraction * bbox_diagonal
    bases = (2, 3, 5, 7, 11, 13)
    poses: List[Pose] = [ZERO_POSE]
    for i in range(1, n_samples):
        coords = [_van_der_corput(i, b) * 2.0 - 1.0 for b in bases]
        poses.append(
            (
                coords[0] * max_rotation_deg,
                coords[1] * max_rotation_deg,
                coords[2] * max_rotation_deg,
                coords[3] * max_translation,
                coords[4] * max_translation,
                coords[5] * max_translation,
            )
        )
    return poses


def best_iou_in_context(
    iou_at_pose: Callable[[Pose], float],
    poses: Sequence[Pose],
    *,
    saturation: float = SATURATION_THRESHOLD,
) -> Tuple[float, Optional[Pose], int]:
    """Search *poses* for the best IoU; stop early once one saturates.

    Returns ``(best_iou, best_pose, n_evaluated)``. Early exit is not an
    approximation: an IoU above :data:`SATURATION_THRESHOLD` is already a full
    pass on the ramp, so a better pose cannot change the score.
    """
    best = -1.0
    best_pose: Optional[Pose] = None
    evaluated = 0
    for pose in poses:
        iou = float(iou_at_pose(pose))
        evaluated += 1
        if iou > best:
            best = iou
            best_pose = pose
        if best >= saturation:
            break
    if best_pose is None:
        return 0.0, None, 0
    return max(0.0, best), best_pose, evaluated


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def score_group(group: int, per_feature_iou: Dict[str, float]) -> GroupScore:
    """A group scores as its **worst** feature (minimum of the ramped IoUs)."""
    if not per_feature_iou:
        raise ValueError(f"mating group {group} has no features")
    ramped = {name: iou_to_interface_score(iou) for name, iou in per_feature_iou.items()}
    worst = min(ramped.items(), key=lambda kv: (kv[1], kv[0]))
    return GroupScore(
        group=group,
        per_feature_iou=dict(per_feature_iou),
        per_feature_score=ramped,
        score=worst[1],
        worst_feature=worst[0],
    )


def interface_score(per_group_iou: Dict[int, Dict[str, float]]) -> InterfaceResult:
    """Aggregate per-feature IoUs into the interface-match axis.

    ``per_group_iou`` maps a mating-group id to ``{feature_name: best IoU}``.
    Group score = min over its ramped features; sample score = mean over groups.
    A sample with no authored sub-volumes has **no** interface axis at all; that
    is signalled by an empty result whose score is ``None`` at the composition
    layer, so here it simply raises rather than inventing a zero.
    """
    if not per_group_iou:
        raise ValueError("no mating groups: the interface axis does not apply")
    groups = [score_group(g, ious) for g, ious in sorted(per_group_iou.items())]
    score = sum(g.score for g in groups) / len(groups)
    return InterfaceResult(score=score, groups=groups)


def evaluate_interface(
    sub_volumes: Sequence[SubVolume],
    iou_fn: Callable[[SubVolume, Pose], float],
    *,
    bbox_diagonal: float,
    n_samples: int = DEFAULT_N_SAMPLES,
    max_rotation_deg: float = DEFAULT_MAX_ROTATION_DEG,
    translation_fraction: float = TRANSLATION_FRACTION_OF_BBOX,
) -> InterfaceResult:
    """End-to-end axis: pose-search each sub-volume, ramp, then aggregate.

    ``iou_fn(sub_volume, pose)`` supplies the volumetric IoU of the region (with
    its verification shell) against the candidate at that pose; it is injected
    because the boolean kernel lives outside this module.
    """
    poses = pose_grid(
        bbox_diagonal,
        n_samples=n_samples,
        max_rotation_deg=max_rotation_deg,
        translation_fraction=translation_fraction,
    )
    per_group: Dict[int, Dict[str, float]] = {}
    for sv in sub_volumes:
        best, _, _ = best_iou_in_context(lambda pose, sv=sv: iou_fn(sv, pose), poses)
        per_group.setdefault(sv.group, {})[sv.name] = best
    return interface_score(per_group)


def feature_passes(iou: float, *, threshold: float = DEFAULT_IOU_THRESHOLD) -> bool:
    """Whether one feature's best IoU clears the reported pass threshold."""
    return iou >= threshold and not math.isnan(iou)
