"""ReCAD unified reward with min-combined geometry/semantics (ReCAD, AAAI 2026).

ReCAD ("Reinforcement Learning Enhanced Parametric CAD Model Generation with
Vision-Language Models") trains a VLM with GRPO under a *unified* reward
function (Eq. 13) that couples geometric accuracy with semantic (visual)
fidelity through a **minimum**, not a sum or product::

    R(y_pi, Omega) = lambda_1 * min{ IOU_best(Omega_hat, Omega),
                                     phi(sim(I_hat, I), tau) }
                   + lambda_2 * R_f(y_pi)

The distinctive design choices, all genuinely new relative to the repository's
existing rewards, are:

  * **min-combination** of a geometric term and a semantic term.  Neither
    ``cadrille_reward`` (additive ``IOU_SCALE*IoU + penalty``),
    ``cmecad_reward`` (gated multiplicative ``format*exec*(IoU+plane)``) nor
    ``intent2exec_reward`` (gated ``exec*(geom+eval)``) takes a *minimum*.  The
    min makes the reward a *conjunction*: a model is only rewarded when it is
    BOTH geometrically accurate AND visually faithful -- the weaker of the two
    caps the credit, discouraging gaming one term against the other.

  * **thresholded linear scaling** ``phi(s, tau) = max(0, (s - tau)/(1 - tau))``
    (paper: ``tau = 0.55``).  Cosine image similarities are typically high even
    for wrong shapes; ``phi`` rescales ``[tau, 1] -> [0, 1]`` and zeroes anything
    at or below the threshold, so only clearly-similar renders contribute.

  * **additive format reward** ``R_f`` (weight ``lambda_2``): 1.0 iff the output
    *begins* with a well-formed ``<think>...</think>`` block, 0.0 otherwise.
    This differs from ``cmecad_reward.format_reward`` which is a multiplicative
    *gate* additionally requiring a fenced code block; here the think-block
    reward is an independent additive term (paper weights ``lambda_1 = 0.1``,
    ``lambda_2 = 0.9``).

``IOU_best`` and the image similarity are supplied by the caller (the geometry
kernel / DINOv2 encoder are external); this module defines only the
deterministic reward composition.  Pure stdlib, deterministic.
"""

from __future__ import annotations

import re

# Paper hyper-parameters (Implementation Details / Reward Design).
DEFAULT_TAU = 0.55        # phi threshold for image similarity
DEFAULT_LAMBDA_1 = 0.1    # weight on the geometry/semantics conjunction
DEFAULT_LAMBDA_2 = 0.9    # weight on the format reward

# R_f: the output must OPEN with a complete <think>...</think> block.  Leading
# whitespace is tolerated; anything (the answer) may follow the block.
_THINK_RE = re.compile(r"\A\s*<think>.*?</think>", re.DOTALL)


def phi(similarity: float, tau: float = DEFAULT_TAU) -> float:
    """Thresholded linear scaling ``phi(s, tau) = max(0, (s - tau)/(1 - tau))``.

    Maps a similarity ``s`` (typically a cosine in ``[-1, 1]`` but commonly
    ``[0, 1]``) so that values at or below ``tau`` yield 0 and ``s = 1`` yields
    1, with a linear ramp between.  ``tau`` must be in ``[0, 1)``.
    """
    s = float(similarity)
    t = float(tau)
    if not 0.0 <= t < 1.0:
        raise ValueError("tau must be in [0, 1)")
    scaled = (s - t) / (1.0 - t)
    if scaled <= 0.0:
        return 0.0
    if scaled >= 1.0:
        return 1.0
    return scaled


def format_reward(text: str) -> float:
    """R_f (Reward Design): 1.0 iff ``text`` opens with a ``<think>...</think>``.

    Unlike ``cmecad_reward.format_reward`` (which also demands a fenced code
    block and acts as a gate), this is the ReCAD additive think-block reward.
    """
    return 1.0 if _THINK_RE.match(text or "") else 0.0


def geometry_semantics_term(
    iou_best: float,
    similarity: float,
    *,
    tau: float = DEFAULT_TAU,
) -> float:
    """The conjunctive term ``min{ IOU_best, phi(sim, tau) }`` (Eq. 13).

    ``iou_best`` is the intersection-over-union under optimal alignment (e.g.
    from :func:`bench.solid_iou.best_solid_iou`) in ``[0, 1]``; ``similarity``
    is the render cosine similarity passed through :func:`phi`.
    """
    iou = float(iou_best)
    if not 0.0 <= iou <= 1.0 + 1e-9:
        raise ValueError("iou_best must lie in [0, 1]")
    return min(max(iou, 0.0), phi(similarity, tau))


def unified_reward(
    iou_best: float,
    similarity: float,
    text: str,
    *,
    tau: float = DEFAULT_TAU,
    lambda_1: float = DEFAULT_LAMBDA_1,
    lambda_2: float = DEFAULT_LAMBDA_2,
) -> float:
    """ReCAD unified reward (Eq. 13).

    ``R = lambda_1 * min{IOU_best, phi(sim, tau)} + lambda_2 * R_f(text)``.

    ``iou_best`` and ``similarity`` are the geometric and semantic signals; the
    format reward is derived from ``text``.  Weights must be non-negative.
    """
    if lambda_1 < 0.0 or lambda_2 < 0.0:
        raise ValueError("reward weights must be non-negative")
    conj = geometry_semantics_term(iou_best, similarity, tau=tau)
    return lambda_1 * conj + lambda_2 * format_reward(text)
