"""CME-CAD gated multi-objective reward with a work-plane term (CME-CAD, 2025).

CME-CAD (Heterogeneous Collaborative Multi-Expert RL) scores a generated
CADQuery program against four objectives and combines them with a *gating*
mechanism so that geometric credit is only awarded once the output is both
well-formed and executable::

    R = lambda_format * R_format
      * lambda_exec   * R_exec
      * (lambda_IoU   * R_IoU + lambda_plane * R_plane)

Because ``R_format`` and ``R_exec`` are binary (0/1) and enter multiplicatively,
the total reward can only be positive when *both* core conditions are satisfied.

The distinctive term is the **Work Plane Reward** ``R_plane`` (Eq. 4-6), which
guards coordinate-system consistency -- a geometrically correct solid placed on
the wrong reference frame yields IoU 0, so the plane reward supplies a smooth
signal instead::

    Dis_ori = || O_gen - O_gt ||_2                          (origin deviation)
    Dis_vec = 1/2 * ( 2 - sim(x_gen, x_gt) - sim(y_gen, y_gt) )  (axis deviation)
    R_plane = clamp( 1 - beta * Dis_ori - gamma * Dis_vec, 0, 1 )

where ``sim`` is cosine similarity of the axis vectors and the two frame axes
(x, y) fix the third (z) by orthogonality.

This is not the repository's ``cadrille_reward`` (additive IoU + validity
penalty) nor ``export`` GRPO shaping: it is a *gated multiplicative* composition
with a dedicated origin/orientation reward. Pure-stdlib, deterministic; the IoU
and executability signals are supplied by the caller.
"""

from __future__ import annotations

import math
import re

# Default reward weights (Eq. 7). Format/Exec are pure gates (weight 1.0); the
# geometric terms are convex-combined.
DEFAULT_LAMBDA_FORMAT = 1.0
DEFAULT_LAMBDA_EXEC = 1.0
DEFAULT_LAMBDA_IOU = 1.0
DEFAULT_LAMBDA_PLANE = 1.0

# Work-plane penalty coefficients (Eq. 6).
DEFAULT_BETA = 1.0
DEFAULT_GAMMA = 1.0

# The canonical CME-CAD structured output: a reasoning block followed by a
# fenced CADQuery code block. ``R_format`` checks this ordering.
_FORMAT_RE = re.compile(
    r"<think>.*?</think>\s*```(?:python|cadquery)?\s*.*?```",
    re.DOTALL,
)


def format_reward(text: str) -> float:
    """R_format (Eq. 'Format Reward'): 1.0 iff ``<think>`` reasoning precedes a
    fenced code block, else 0.0."""
    return 1.0 if _FORMAT_RE.search(text or "") else 0.0


def exec_reward(executable: bool) -> float:
    """R_exec: 1.0 if the CADQuery code parses and runs without error, else 0.0."""
    return 1.0 if executable else 0.0


def iou_reward(iou: float) -> float:
    """R_IoU = Jaccard index J(M_gen, M_gt) (Eq. 3). Must lie in [0, 1]."""
    value = float(iou)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"iou must be in [0, 1], got {value!r}")
    return value


def jaccard_iou(inter: float, union: float) -> float:
    """Volumetric Jaccard index |M_gen n M_gt| / |M_gen u M_gt| (Eq. 3).

    ``union`` of 0 (two empty solids) is defined as a perfect overlap (1.0).
    """
    inter = float(inter)
    union = float(union)
    if inter < 0.0 or union < 0.0:
        raise ValueError("intersection and union must be non-negative")
    if inter > union:
        raise ValueError("intersection cannot exceed union")
    if union == 0.0:
        return 1.0
    return inter / union


def origin_deviation(origin_gen, origin_gt) -> float:
    """Dis_ori = Euclidean distance between generated and ground-truth origins
    (Eq. 4)."""
    g = [float(x) for x in origin_gen]
    t = [float(x) for x in origin_gt]
    if len(g) != len(t):
        raise ValueError("origins must have equal dimension")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(g, t)))


def _cosine_sim(u, v) -> float:
    u = [float(x) for x in u]
    v = [float(x) for x in v]
    if len(u) != len(v):
        raise ValueError("vectors must have equal dimension")
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(x * x for x in v))
    if nu == 0.0 or nv == 0.0:
        raise ValueError("axis vectors must be non-zero")
    dot = sum(a * b for a, b in zip(u, v))
    cos = dot / (nu * nv)
    # numerical guard
    return max(-1.0, min(1.0, cos))


def axis_deviation(x_gen, x_gt, y_gen, y_gt) -> float:
    """Dis_vec = 1/2 * (2 - sim(x_gen,x_gt) - sim(y_gen,y_gt)) (Eq. 5).

    Ranges in [0, 2]: 0 when both axes align, 2 when both are anti-parallel.
    """
    sx = _cosine_sim(x_gen, x_gt)
    sy = _cosine_sim(y_gen, y_gt)
    return 0.5 * (2.0 - sx - sy)


def work_plane_reward(origin_gen, origin_gt, x_gen, x_gt, y_gen, y_gt,
                      beta: float = DEFAULT_BETA,
                      gamma: float = DEFAULT_GAMMA) -> float:
    """R_plane = clamp(1 - beta*Dis_ori - gamma*Dis_vec, 0, 1) (Eq. 6)."""
    dis_ori = origin_deviation(origin_gen, origin_gt)
    dis_vec = axis_deviation(x_gen, x_gt, y_gen, y_gt)
    value = 1.0 - float(beta) * dis_ori - float(gamma) * dis_vec
    return max(0.0, min(1.0, value))


def total_reward(r_format: float, r_exec: float, r_iou: float, r_plane: float,
                 lambda_format: float = DEFAULT_LAMBDA_FORMAT,
                 lambda_exec: float = DEFAULT_LAMBDA_EXEC,
                 lambda_iou: float = DEFAULT_LAMBDA_IOU,
                 lambda_plane: float = DEFAULT_LAMBDA_PLANE) -> float:
    """Gated multiplicative total reward (Eq. 7).

    ``R_format`` and ``R_exec`` act as multiplicative gates: if either is 0 the
    total is 0 regardless of the geometric terms.
    """
    gate = (lambda_format * r_format) * (lambda_exec * r_exec)
    geom = lambda_iou * r_iou + lambda_plane * r_plane
    return gate * geom


def reward_components(text: str, executable: bool, iou: float,
                      origin_gen=None, origin_gt=None,
                      x_gen=None, x_gt=None, y_gen=None, y_gt=None,
                      beta: float = DEFAULT_BETA,
                      gamma: float = DEFAULT_GAMMA,
                      lambda_format: float = DEFAULT_LAMBDA_FORMAT,
                      lambda_exec: float = DEFAULT_LAMBDA_EXEC,
                      lambda_iou: float = DEFAULT_LAMBDA_IOU,
                      lambda_plane: float = DEFAULT_LAMBDA_PLANE) -> dict:
    """Compute all four objectives and the gated total for one sample.

    When the pose vectors are omitted the work-plane reward defaults to 0.0
    (no coordinate-frame information available).
    """
    rf = format_reward(text)
    re_ = exec_reward(executable)
    ri = iou_reward(iou) if executable else 0.0
    if None in (origin_gen, origin_gt, x_gen, x_gt, y_gen, y_gt) or not executable:
        rp = 0.0
    else:
        rp = work_plane_reward(origin_gen, origin_gt, x_gen, x_gt, y_gen, y_gt,
                               beta=beta, gamma=gamma)
    total = total_reward(rf, re_, ri, rp,
                         lambda_format=lambda_format, lambda_exec=lambda_exec,
                         lambda_iou=lambda_iou, lambda_plane=lambda_plane)
    return {
        "r_format": rf,
        "r_exec": re_,
        "r_iou": ri,
        "r_plane": rp,
        "total": total,
    }
