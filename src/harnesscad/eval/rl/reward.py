"""Composite, target-comparing reward for the CAD RL environment.

Why this module exists
----------------------
``harnesscad.io.surfaces.mcp.tools.reward_from_apply`` is a *process* reward: it
returns 1.0 minus 0.1 per WARNING diagnostic, -1.0 for a rejected batch.  It
never looks at the geometry that was built, so an agent can earn a perfect score
by emitting any clean op batch whatsoever.  That is far too thin to train
against.  This module adds the missing *outcome* channel: a reward that compares
the built solid to a TARGET.  ``reward_from_apply`` is untouched and keeps
working for its existing callers.

The formula
-----------
Two papers, combined::

    R = R_exec * [ w_geom * shape(R_geom) + w_process * R_process ]

    R_geom = 0.2 * IoU  +  0.5 * R_MMD  +  0.3 * NC

**R_geom** is RLCAD (arXiv:2503.18549, "RL Training Gym for Revolution Involved
CAD Command Sequence Generation"), weights ``a=0.2, b=0.5, c=0.3`` taken
verbatim.  Their ablation is the whole argument for a composite: IoU alone
reached 0.7045 quality, IoU+MMD 0.7436, IoU+MMD+NC 0.7932.  IoU alone is
*measurably insufficient* -- an agent that only maximises coarse volumetric
overlap fills the right bounding volume and neglects fine detail, which is
precisely what the MMD (surface distance) and NC (surface orientation) terms
restore.

**R_exec** is CAD-RL (arXiv:2508.10118, "From Intent to Execution"), which sets
``R = R_exec * [w_geom*R_geom + w_eval*R_eval]`` with ``R_exec`` BINARY and
MULTIPLICATIVE.  Their stated reason is that this "prevents reward leakage from
malformed samples": a candidate that does not build must score exactly zero, not
partial credit for accidentally overlapping the target.  Implemented here as
:func:`exec_gate`.

Refused: CAD-RL's R_eval
------------------------
CAD-RL's second term ``R_eval`` is a GPT-4o judge scoring the candidate from
rendered views.  It is NOT implemented here and ``w_eval`` does not exist.  A
model-authored score is an unsound instruction channel inside a reward: the
judge reads attacker-influenced content (the candidate's own text, names,
comments) and its output is the training signal, so any text that talks the
judge upward is directly rewarded.  This repository has already paid for that
failure mode -- the v1 pressure experiment lost 8.3 points to exactly it.  A
reward must be a measurement, not a conversation.

``w_process`` is offered in its place, defaulting to **0.0** (off, so the
default is faithful to the papers).  It is a deterministic verifier-derived
score -- the same warning-count signal ``reward_from_apply`` computes, mapped to
``[0, 1]`` -- for callers who want a soundly-computed second term.  It is not a
substitute for R_eval and does not claim to be; it is simply the strongest
second term that can be computed without asking a model anything.

METRIC HONESTY: which chamfer, and with what normalisation
----------------------------------------------------------
This repository's central finding is that metrics sharing a NAME do not share a
DEFINITION: six chamfer implementations here differ by four orders of magnitude
(see the README's "The finding" and ``eval/bench/registry.py``).  So, exactly:

* **Chamfer variant**: ``eval.bench.geometry.chamfer_unit_cube.chamfer_distance``
  -- the CAD-Recode / cadrille variant: *symmetric*, *mean* (not sum), *L2*
  distance (not squared), averaged over the two directed means.  Chosen over
  ``chamfer_unit_sphere`` (squared CD -- squaring re-weights the tail and makes
  the term non-commensurate with IoU) and over the bare
  ``eval.bench.geometry.chamfer.symmetric_chamfer`` (identical arithmetic but no
  normalisation contract at all).
* **Scale**: called with ``scale=1.0``, NOT the module default
  ``CD_SCALE=1000``.  The x1000 is a reporting convenience so published median
  CDs are readable; leaving it on would push the term three orders of magnitude
  outside ``[0, 1]`` and let it swamp IoU and NC.
* **Normalisation**: both clouds are mapped through ONE SHARED frame derived
  from the TARGET's bounding box (:func:`shared_unit_cube_frame`), so the target
  occupies ``[-0.5, 0.5]^3`` and the candidate lands wherever it actually is.
  We deliberately do NOT use ``chamfer_unit_cube.normalize_to_unit_cube``, which
  normalises *each cloud into its own* bounding box: that would erase precisely
  the translation and scale error the reward exists to punish, scoring a
  half-size candidate as perfect.
  **Consequence**: the CD numbers this module reports are NOT comparable to
  published cadrille/CAD-Recode CD figures, which are per-cloud normalised and
  x1000.  They are comparable only to each other.
* **EMD**: ``domain.reconstruction.evaluate.pointcloud_emd.mean_emd`` -- exact
  Hungarian assignment, per-point mean -- in the same shared frame.  It needs
  equal cardinality and is O(n^3), so clouds are deterministically
  stride-subsampled to a shared size capped at ``emd_max_points``; whenever that
  happens it is recorded in ``notes``, never silently.
* **MMD**: RLCAD's MMD-CD / MMD-EMD are *minimum matching distances* over a set
  of samples.  Against a single (candidate, target) pair the minimum over a
  one-element set is the distance itself, so this is the degenerate case of
  their metric, not a different one.  Stated rather than glossed.
* **R_MMD sign and range**: the paper writes ``R_MMD = -(MMD-CD + MMD-EMD)/2``,
  negated so that minimising distance maximises reward.  A raw negated distance
  is unbounded below and would make the weighted sum incomparable across shapes,
  so it is mapped affinely into ``[0, 1]``::

      R_MMD = clamp(1 - mean(CD, EMD) / sqrt(3), 0, 1)

  ``sqrt(3)`` is the diagonal of the unit cube the target was normalised into --
  the largest distance that can separate two points both inside it (the same
  constant ``eval.bench.geometry.hausdorff_iogt.UNIT_CUBE_DIAGONAL`` uses as its
  compile-failure penalty).  Ordering is identical to the paper's; only the
  origin and unit differ.
* **IoU**: ``domain.geometry.volumes.voxel_iou`` on sparse occupancy sets built
  by its own ``voxelize_points`` in UNNORMALISED world units with a shared
  origin and spacing.  IoU is deliberately the one term left in world space: it
  is what catches a candidate that is the right shape at the wrong size, which
  the shared-frame-but-scale-free surface terms are weaker at.
* **NC**: ``domain.geometry.pointcloud.normal_consistency`` (written for this
  module; there was no normal-consistency implementation in the repo), nearest
  neighbour correspondence, ``unoriented=True`` so it lands in ``[0, 1]``.

Saturation guard (RLCAD failure case 7c)
----------------------------------------
RLCAD report training stalling once the primary feature dominates: the score
exceeds 0.99 and subsequent detail ops move it by less than the noise floor, so
the gradient carries no information about the detail.  Two responses, both
OBSERVABLE rather than silent:

1. :class:`RewardBreakdown` reports every term, its weighted contribution, the
   ``dominant`` term and its ``dominance`` share, the remaining ``headroom``
   (``1 - R_geom``), a ``saturated`` flag, and human-readable ``notes``.  A
   saturated or single-term-dominated reward is visible in the returned object
   and in ``to_dict()`` -- a training loop can log or assert on it.
2. ``detail_sensitivity`` (gamma, default 1.0 = off) applies
   ``R_geom ** gamma``.  For gamma > 1 the slope at ``R_geom = 1`` is gamma, so
   gamma=3 triples the reward variation produced by a detail op near the top of
   the range while leaving the ordering of candidates untouched (x**gamma is
   strictly monotone on [0, 1]).  ``saturated`` is computed on the RAW
   ``R_geom``, before shaping, so turning the knob up cannot hide the condition
   it exists to counteract.

Distinct from ``eval.quality.reward.composite_reward`` (a generic named-weight
aggregator over scores the caller has already computed) and from
``eval.quality.reward.execution_reward`` (a single-CD threshold reward): this
module *computes* the geometry terms from clouds and voxels, and is bound to the
CISP ``ApplyOpsResult`` contract.

Stdlib only, deterministic, no model in the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

from harnesscad.domain.geometry.pointcloud.normal_consistency import (
    normal_consistency,
)
from harnesscad.domain.geometry.volumes.voxel_iou import voxel_iou, voxelize_points
from harnesscad.domain.reconstruction.evaluate.pointcloud_emd import mean_emd
from harnesscad.eval.bench.geometry.chamfer_unit_cube import chamfer_distance
from harnesscad.eval.bench.geometry.hausdorff_iogt import UNIT_CUBE_DIAGONAL

#: RLCAD's published weights (arXiv:2503.18549): R = 0.2*IoU + 0.5*R_MMD + 0.3*NC.
RLCAD_WEIGHTS: Dict[str, float] = {"iou": 0.2, "mmd": 0.5, "nc": 0.3}

#: Default voxel edge length (world units) for the IoU occupancy grid.
DEFAULT_VOXEL_SPACING = 1.0

#: Hungarian EMD is O(n^3); above this many points both clouds are
#: stride-subsampled (and a note is emitted).
DEFAULT_EMD_MAX_POINTS = 64

#: R_geom at or above this is reported as saturated (RLCAD failure case 7c).
DEFAULT_SATURATION_THRESHOLD = 0.99

#: One weighted term supplying at least this share of R_geom is reported as
#: dominating -- the other terms are then contributing almost no gradient.
DEFAULT_DOMINANCE_THRESHOLD = 0.9


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


# ===========================================================================
# Shape samples
# ===========================================================================
@dataclass(frozen=True)
class ShapeSample:
    """A sampled shape: world-space surface points, optional normals/voxels.

    ``points`` are in world units (millimetres, in this repo's convention).
    ``normals`` -- when present -- must be parallel to ``points`` and is what
    the NC term consumes; without them NC is skipped and its weight is
    redistributed (recorded in ``notes``).  ``voxels`` may carry a
    pre-computed sparse occupancy set (a set of integer ``(i, j, k)``); when it
    is None the IoU term voxelises ``points`` with the shared grid.
    """

    points: Tuple[Tuple[float, ...], ...]
    normals: Optional[Tuple[Tuple[float, ...], ...]] = None
    voxels: Optional[frozenset] = None

    @classmethod
    def of(cls, points, normals=None, voxels=None) -> "ShapeSample":
        """Build a sample from any iterables, freezing them into tuples."""
        pts = tuple(tuple(float(c) for c in p) for p in points)
        if not pts:
            raise ValueError("a ShapeSample needs at least one point")
        nrm = None
        if normals is not None:
            nrm = tuple(tuple(float(c) for c in n) for n in normals)
            if len(nrm) != len(pts):
                raise ValueError("normals must be parallel to points")
        vox = None if voxels is None else frozenset(
            tuple(int(c) for c in v) for v in voxels)
        return cls(pts, nrm, vox)

    def oriented(self):
        """``(point, normal)`` pairs for the NC metric; None without normals."""
        if self.normals is None:
            return None
        return list(zip(self.points, self.normals))


def _stride_subsample(seq: Sequence, k: int):
    """Deterministically take ``k`` items spread evenly across ``seq``."""
    n = len(seq)
    if k >= n:
        return list(seq)
    return [seq[(i * n) // k] for i in range(k)]


# ===========================================================================
# Shared normalisation frame
# ===========================================================================
def shared_unit_cube_frame(points):
    """``(center, scale)`` mapping ``points``' bbox into ``[-0.5, 0.5]^3``.

    Applied to BOTH clouds so relative translation and scale survive; see the
    module docstring's normalisation paragraph for why per-cloud normalisation
    is refused here.
    """
    pts = [tuple(float(c) for c in p) for p in points]
    if not pts:
        raise ValueError("points must be non-empty")
    dims = len(pts[0])
    lo = [min(p[d] for p in pts) for d in range(dims)]
    hi = [max(p[d] for p in pts) for d in range(dims)]
    center = tuple((lo[d] + hi[d]) / 2.0 for d in range(dims))
    extent = max(hi[d] - lo[d] for d in range(dims))
    scale = (1.0 / extent) if extent > 0 else 1.0
    return center, scale


def apply_frame(points, frame):
    """Map ``points`` through a ``(center, scale)`` frame."""
    center, scale = frame
    return [tuple((float(p[d]) - center[d]) * scale for d in range(len(center)))
            for p in points]


# ===========================================================================
# The three RLCAD terms
# ===========================================================================
def iou_term(candidate: ShapeSample, target: ShapeSample, *,
             spacing: float = DEFAULT_VOXEL_SPACING,
             origin: Sequence[float] = (0.0, 0.0, 0.0)) -> float:
    """Volumetric IoU on a shared world-space occupancy grid, in ``[0, 1]``."""
    a = candidate.voxels if candidate.voxels is not None else voxelize_points(
        candidate.points, origin=origin, spacing=spacing)
    b = target.voxels if target.voxels is not None else voxelize_points(
        target.points, origin=origin, spacing=spacing)
    return voxel_iou(a, b)


def mmd_term(candidate: ShapeSample, target: ShapeSample, *,
             emd_max_points: int = DEFAULT_EMD_MAX_POINTS):
    """RLCAD's ``R_MMD``, mapped into ``[0, 1]``; see the module docstring.

    Returns ``(r_mmd, chamfer, emd, notes)`` -- the raw distances are returned
    alongside the reward so the breakdown can show what drove it.
    """
    frame = shared_unit_cube_frame(target.points)
    a = apply_frame(candidate.points, frame)
    b = apply_frame(target.points, frame)

    cd = chamfer_distance(a, b, scale=1.0)

    notes = []
    k = min(len(a), len(b), max(1, int(emd_max_points)))
    if k < max(len(a), len(b)):
        notes.append(
            "emd: clouds stride-subsampled to %d points (candidate %d, target "
            "%d) -- Hungarian assignment is O(n^3) and needs equal cardinality"
            % (k, len(a), len(b)))
    emd = mean_emd(_stride_subsample(a, k), _stride_subsample(b, k))

    r_mmd = _clip01(1.0 - ((cd + emd) / 2.0) / UNIT_CUBE_DIAGONAL)
    return r_mmd, cd, emd, notes


def nc_term(candidate: ShapeSample, target: ShapeSample) -> Optional[float]:
    """Normal consistency in ``[0, 1]``, or None when normals are absent."""
    ref = target.oriented()
    cand = candidate.oriented()
    if ref is None or cand is None:
        return None
    return normal_consistency(ref, cand, unoriented=True, correspondence="nearest")


# ===========================================================================
# CAD-RL's gate, and the sound process term
# ===========================================================================
def _severity_name(diagnostic) -> str:
    """Upper-case severity of a Diagnostic, tolerating dicts and bare strings.

    ``Severity`` is a ``str`` Enum, so a diagnostic may reach here as the enum,
    as its ``"warning"`` value, or as a ``to_dict()`` mapping; all three must
    gate identically or the gate would be trivially bypassable by serialising.
    """
    sev = diagnostic.get("severity") if isinstance(diagnostic, dict) else getattr(
        diagnostic, "severity", None)
    name = getattr(sev, "name", None)
    if name is None:
        name = sev if isinstance(sev, str) else ""
    return str(name).upper()


def exec_gate(result) -> float:
    """CAD-RL's binary ``R_exec``: 1.0 if the op stream built, else 0.0.

    Multiplicative by design -- an unbuildable candidate scores exactly zero, so
    no geometric partial credit can leak from a malformed sample
    (arXiv:2508.10118).  Note the contrast with ``reward_from_apply``, which
    returns -1.0 on failure: a gate must be 0, not negative, or it would flip
    the sign of the geometry terms instead of erasing them.
    """
    if not bool(getattr(result, "ok", False)):
        return 0.0
    for d in getattr(result, "diagnostics", ()) or ():
        if _severity_name(d) == "ERROR":
            return 0.0
    return 1.0


def process_score(result) -> float:
    """Verifier-derived process score in ``[0, 1]``: 1.0 clean, -0.1 per WARNING.

    The same signal ``reward_from_apply`` computes, rescaled to ``[0, 1]`` so it
    can sit in a weighted sum.  Deterministic and model-free -- this is the
    honest alternative to CAD-RL's GPT-4o ``R_eval``, and it is off by default.
    """
    if not bool(getattr(result, "ok", False)):
        return 0.0
    warns = 0
    for d in getattr(result, "diagnostics", ()) or ():
        if _severity_name(d) == "WARNING":
            warns += 1
    return max(0.0, 1.0 - 0.1 * warns)


# ===========================================================================
# Breakdown
# ===========================================================================
@dataclass(frozen=True)
class RewardBreakdown:
    """Every number behind a composite reward -- the saturation guard's eyes.

    ``total`` is what an RL loop consumes; everything else exists so a stalled
    or dominated reward is visible instead of silent.
    """

    total: float
    exec_gate: float
    geometric: float           # R_geom, raw weighted sum, pre-shaping
    shaped: float              # R_geom ** detail_sensitivity
    process: float
    iou: Optional[float]
    mmd: Optional[float]
    nc: Optional[float]
    chamfer: Optional[float]
    emd: Optional[float]
    weights: Dict[str, float] = field(default_factory=dict)
    contributions: Dict[str, float] = field(default_factory=dict)
    dominant: Optional[str] = None
    dominance: float = 0.0
    headroom: float = 1.0
    saturated: bool = False
    detail_sensitivity: float = 1.0
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "exec_gate": self.exec_gate,
            "geometric": self.geometric,
            "shaped": self.shaped,
            "process": self.process,
            "terms": {"iou": self.iou, "mmd": self.mmd, "nc": self.nc},
            "distances": {"chamfer": self.chamfer, "emd": self.emd},
            "weights": dict(self.weights),
            "contributions": dict(self.contributions),
            "saturation": {
                "saturated": self.saturated,
                "headroom": self.headroom,
                "dominant": self.dominant,
                "dominance": self.dominance,
                "detail_sensitivity": self.detail_sensitivity,
            },
            "notes": list(self.notes),
        }


# ===========================================================================
# The composite
# ===========================================================================
def composite_reward(
    result,
    target: Optional[ShapeSample],
    candidate: Optional[ShapeSample] = None,
    *,
    weights: Optional[Dict[str, float]] = None,
    voxel_spacing: float = DEFAULT_VOXEL_SPACING,
    voxel_origin: Sequence[float] = (0.0, 0.0, 0.0),
    emd_max_points: int = DEFAULT_EMD_MAX_POINTS,
    detail_sensitivity: float = 1.0,
    saturation_threshold: float = DEFAULT_SATURATION_THRESHOLD,
    dominance_threshold: float = DEFAULT_DOMINANCE_THRESHOLD,
    w_geom: float = 1.0,
    w_process: float = 0.0,
) -> RewardBreakdown:
    """``R = R_exec * [w_geom * shape(R_geom) + w_process * R_process]``.

    ``result`` is a CISP ``ApplyOpsResult`` (anything with ``ok`` and
    ``diagnostics``); ``target`` and ``candidate`` are :class:`ShapeSample`.
    When the exec gate is 0 the geometry is never computed at all -- that is the
    point of a multiplicative gate, and it also means ``candidate`` may be None
    for a failed build.

    When either sample lacks normals the NC term is unavailable; its weight is
    redistributed proportionally over the surviving terms and a note is emitted,
    so the reward stays on the same ``[0, 1]`` scale instead of quietly losing
    0.3 of its range.

    See the module docstring for the chamfer variant, the normalisation choice
    and its consequence, the refused GPT-4o judge, and the saturation guard.
    """
    w = dict(RLCAD_WEIGHTS if weights is None else weights)
    for key in ("iou", "mmd", "nc"):
        w.setdefault(key, 0.0)
    if detail_sensitivity <= 0.0:
        raise ValueError("detail_sensitivity must be positive")

    gate = exec_gate(result)
    proc = process_score(result)
    notes = []

    if gate == 0.0:
        notes.append(
            "exec gate 0 (CAD-RL arXiv:2508.10118): the op stream did not "
            "build, so the reward is exactly 0 and no geometry was scored")
        return RewardBreakdown(
            total=0.0, exec_gate=0.0, geometric=0.0, shaped=0.0, process=proc,
            iou=None, mmd=None, nc=None, chamfer=None, emd=None,
            weights=w, contributions={}, dominant=None, dominance=0.0,
            headroom=1.0, saturated=False,
            detail_sensitivity=detail_sensitivity, notes=tuple(notes))

    if target is None or candidate is None:
        raise ValueError(
            "a built result needs both a candidate and a target ShapeSample; "
            "only a gated-out (unbuildable) result may omit them")

    iou = iou_term(candidate, target, spacing=voxel_spacing, origin=voxel_origin)
    mmd, cd, emd, mmd_notes = mmd_term(
        candidate, target, emd_max_points=emd_max_points)
    notes.extend(mmd_notes)
    nc = nc_term(candidate, target)

    terms: Dict[str, Optional[float]] = {"iou": iou, "mmd": mmd, "nc": nc}
    if nc is None:
        notes.append(
            "nc: no normals on one or both samples -- the normal-consistency "
            "term was dropped and its weight redistributed. RLCAD's ablation "
            "puts NC at +0.0496 quality, so this reward is measurably weaker")

    live = {k: v for k, v in terms.items() if v is not None and w.get(k, 0.0) > 0.0}
    live_weight = sum(w[k] for k in live)
    if live_weight <= 0.0:
        raise ValueError("no weighted geometric term could be computed")
    contributions = {k: (w[k] / live_weight) * live[k] for k in live}
    geometric = sum(contributions.values())

    dominant = max(contributions, key=lambda k: contributions[k])
    dominance = (contributions[dominant] / geometric) if geometric > 0.0 else 0.0
    headroom = 1.0 - geometric
    saturated = geometric >= saturation_threshold

    if saturated:
        notes.append(
            "SATURATED: R_geom=%.4f >= %.4f, headroom %.4f (RLCAD failure case "
            "7c) -- further detail ops will move the reward by less than the "
            "noise floor; raise detail_sensitivity or refine the target sampling"
            % (geometric, saturation_threshold, headroom))
    if dominance >= dominance_threshold:
        notes.append(
            "DOMINATED: term '%s' supplies %.1f%% of R_geom; the remaining "
            "terms are contributing almost no gradient"
            % (dominant, 100.0 * dominance))

    shaped = geometric ** detail_sensitivity
    if detail_sensitivity != 1.0:
        notes.append(
            "detail_sensitivity gamma=%.3f applied (R_geom**gamma); ordering is "
            "unchanged, slope near R_geom=1 is gamma" % detail_sensitivity)

    total = gate * (w_geom * shaped + w_process * proc)

    return RewardBreakdown(
        total=total, exec_gate=gate, geometric=geometric, shaped=shaped,
        process=proc, iou=iou, mmd=mmd, nc=nc, chamfer=cd, emd=emd,
        weights=w, contributions=contributions, dominant=dominant,
        dominance=dominance, headroom=headroom, saturated=saturated,
        detail_sensitivity=detail_sensitivity, notes=tuple(notes))


def composite_reward_value(result, target, candidate=None, **kwargs) -> float:
    """``composite_reward(...).total`` -- the scalar an RL loop consumes.

    Prefer :func:`composite_reward` in training code: the breakdown is how a
    saturated or single-term-dominated reward becomes visible instead of
    silently stalling the run.
    """
    return composite_reward(result, target, candidate, **kwargs).total
