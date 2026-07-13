"""Deterministic Dual-Teacher DMD distillation schedule (Turbo3D, Hu et al. 2024).

Turbo3D distils a slow many-step multi-view (MV) diffusion teacher into a fast
4-step, 4-view student using Distribution Matching Distillation (DMD, Yin et al.
2024) extended with a **dual teacher**: an MV teacher (view consistency) and a
single-view (SV) teacher (photo-realism). The learned generator / score networks
are out of scope, but several pieces of the distillation recipe are pure,
deterministic bookkeeping that this module implements:

  * ``few_step_timesteps`` -- the fixed set of teacher timesteps the few-step
    student is evaluated at (the step-reduction schedule; the paper uses 4 steps
    over a 1000-step teacher schedule). This is the deterministic timestep
    mapping a distilled few-step generator applies at inference.

  * ``dmd_gradient`` -- the DMD update direction is the difference of the two
    score functions ``s_real - s_fake`` (Yin et al. Eq. 6). Given the (caller
    supplied) real/teacher and fake/student scores, the per-element gradient is a
    closed-form subtraction, optionally scaled by a normalising weight.

  * ``dual_teacher_gradient`` -- the dual-teacher objective (Turbo3D Eq. 7):
    the MV-DMD gradient plus ``lambda`` times the mean of the K per-view SV-DMD
    gradients. ``lambda = 1`` in the paper. This module computes that weighted
    combination deterministically from the supplied MV and per-view SV gradients.

  * ``compounding_collapse_indicator`` -- a scalar diversity monitor: the paper
    observes "compounding mode collapse" when distilling the MV teacher alone
    (outputs collapse toward the synthetic Objaverse mode). Given per-view score
    magnitudes, this reports the fraction of diversity retained -- a deterministic
    proxy used to justify adding the SV teacher.

Pure stdlib, deterministic. Vectors are plain float sequences.
"""

from __future__ import annotations

from math import sqrt
from typing import List, Sequence


# --------------------------------------------------------------------------- #
# Step-reduction schedule
# --------------------------------------------------------------------------- #
def few_step_timesteps(total_steps: int, num_steps: int) -> List[int]:
    """Descending teacher timesteps a ``num_steps`` distilled student evaluates.

    A few-step DMD student is trained to jump along the teacher's schedule at a
    small fixed set of timesteps. This returns ``num_steps`` evenly spaced,
    distinct timesteps of ``1 .. total_steps`` in *decreasing* order, always
    starting at the terminal step ``total_steps`` (pure noise). Turbo3D uses
    ``num_steps = 4`` against a ``total_steps = 1000`` teacher.
    """
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")
    if not 1 <= num_steps <= total_steps:
        raise ValueError("num_steps must be in [1, total_steps]")
    if num_steps == 1:
        return [total_steps]
    steps = []
    for i in range(num_steps):
        # i = 0 -> total_steps (start from noise); i = num_steps-1 -> 1
        t = round(total_steps - i * (total_steps - 1) / (num_steps - 1))
        steps.append(int(t))
    # descending, distinct
    return sorted(set(steps), reverse=True)


def step_reduction_factor(teacher_steps: int, student_steps: int) -> float:
    """Multiplicative denoiser-evaluation speedup ``teacher / student``.

    The dominant inference cost is the number of denoiser evaluations, so the
    step-count ratio is the first-order speedup of distillation (Turbo3D reports
    the distilled model is ~50x faster than the many-step MV teacher).
    """
    if teacher_steps <= 0 or student_steps <= 0:
        raise ValueError("step counts must be positive")
    return teacher_steps / student_steps


# --------------------------------------------------------------------------- #
# DMD gradient: difference of real and fake scores
# --------------------------------------------------------------------------- #
def dmd_gradient(
    score_real: Sequence[float],
    score_fake: Sequence[float],
    weight: float = 1.0,
) -> List[float]:
    """DMD update direction ``weight * (s_real - s_fake)`` (Yin et al. Eq. 6).

    ``score_real`` is the frozen teacher score and ``score_fake`` the trained
    student-distribution score, evaluated at the same noised sample. The DMD
    gradient minimising reverse KL is proportional to their difference; ``weight``
    absorbs the schedule-dependent normalising constant (defaults to 1).
    """
    if len(score_real) != len(score_fake):
        raise ValueError("score vectors must have equal length")
    return [weight * (float(r) - float(f)) for r, f in zip(score_real, score_fake)]


# --------------------------------------------------------------------------- #
# Dual-teacher combination (Turbo3D Eq. 7)
# --------------------------------------------------------------------------- #
def dual_teacher_gradient(
    mv_gradient: Sequence[float],
    sv_gradients: Sequence[Sequence[float]],
    lam: float = 1.0,
) -> List[float]:
    """Combine MV and per-view SV DMD gradients (Turbo3D Eq. 7).

    ``mv_gradient`` is the multi-view DMD gradient (from :func:`dmd_gradient` on
    the MV scores, treating all K views jointly). ``sv_gradients`` is the list of
    K per-view single-view DMD gradients. The dual-teacher gradient is

        g_mv  +  lam * (1/K) * sum_i g_sv_i

    with ``lam = 1`` in the paper. All gradient vectors must share a length.
    """
    if lam < 0.0:
        raise ValueError("lam must be non-negative")
    k = len(sv_gradients)
    if k == 0:
        raise ValueError("need at least one single-view gradient")
    n = len(mv_gradient)
    for g in sv_gradients:
        if len(g) != n:
            raise ValueError("all gradients must share a length")
    out = [float(v) for v in mv_gradient]
    scale = lam / k
    for g in sv_gradients:
        for j in range(n):
            out[j] += scale * float(g[j])
    return out


# --------------------------------------------------------------------------- #
# Compounding mode-collapse diversity indicator
# --------------------------------------------------------------------------- #
def compounding_collapse_indicator(
    per_view_magnitudes: Sequence[float],
) -> float:
    """Diversity-retention proxy in ``[0, 1]`` from per-view score magnitudes.

    Turbo3D reports "compounding mode collapse" when distilling the MV teacher
    alone: outputs collapse to a narrow synthetic mode, so per-view variation
    shrinks. This returns the coefficient-of-variation-style ratio
    ``min / max`` of the supplied non-negative per-view magnitudes: ``1`` means
    perfectly balanced (no collapse), values near ``0`` indicate a collapsed,
    dominated distribution. Deterministic and monotone.
    """
    mags = [float(m) for m in per_view_magnitudes]
    if len(mags) == 0:
        raise ValueError("need at least one magnitude")
    if any(m < 0.0 for m in mags):
        raise ValueError("magnitudes must be non-negative")
    hi = max(mags)
    if hi == 0.0:
        return 0.0
    return min(mags) / hi


def rms(values: Sequence[float]) -> float:
    """Root-mean-square magnitude of a vector (helper for score magnitudes)."""
    vals = [float(v) for v in values]
    if len(vals) == 0:
        raise ValueError("need at least one value")
    return sqrt(sum(v * v for v in vals) / len(vals))
