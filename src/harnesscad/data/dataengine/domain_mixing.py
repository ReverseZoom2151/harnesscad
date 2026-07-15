"""Curriculum domain-mixing schedule (Khoche et al., 2026, "BlendCLIP: Bridging
Synthetic and Real Domains for Zero-Shot 3D Object Classification with Multimodal
Pretraining").

BlendCLIP's core contribution is a *curriculum-based data mixing strategy*: the
model is first grounded in semantically rich synthetic CAD data, then
progressively adapted to real-world scans by increasing the fraction of
real-domain samples per batch over training. The paper reports that introducing
as few as ~1.5% real samples per batch is already highly label-efficient.

The training itself is out of scope, but the *schedule* -- how many real vs.
synthetic samples each batch draws as training progresses -- is a deterministic,
seed-free data-engine computation:

* :func:`real_fraction` gives the target real-domain fraction at a given step for
  a linear warmup-then-hold schedule between ``start`` and ``end`` fractions.
* :func:`batch_composition` splits a batch of fixed size into integer
  real/synthetic counts at a step (largest-remainder rounding, so counts always
  sum to the batch size).
* :func:`schedule_table` materialises the composition across all steps and totals
  the real/synthetic samples seen -- the label-efficiency accounting.

Deterministic, stdlib-only.
"""

from __future__ import annotations

from typing import Dict, List

__all__ = ["real_fraction", "batch_composition", "schedule_table"]


def real_fraction(
    step: int,
    total_steps: int,
    start: float = 0.0,
    end: float = 0.5,
    warmup: float = 1.0,
) -> float:
    """Target real-domain fraction at ``step`` (0-based) of ``total_steps``.

    Linearly ramps from ``start`` to ``end`` over the first ``warmup`` fraction of
    training, then holds at ``end``. ``warmup`` in ``(0, 1]``. Fractions are
    clamped to ``[0, 1]``.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 < warmup <= 1.0:
        raise ValueError("warmup must be in (0, 1]")
    if not (0.0 <= start <= 1.0 and 0.0 <= end <= 1.0):
        raise ValueError("start/end fractions must be in [0, 1]")
    if step < 0 or step >= total_steps:
        raise ValueError("step must be in [0, total_steps)")
    warm_steps = max(1, int(round(warmup * total_steps)))
    if step >= warm_steps:
        return end
    t = step / warm_steps  # in [0, 1)
    return start + (end - start) * t


def batch_composition(batch_size: int, fraction: float) -> Dict[str, int]:
    """Split a batch into ``{"real": r, "synthetic": s}`` with ``r + s = batch_size``.

    ``r = round(batch_size * fraction)`` with the remainder going to synthetic, so
    a small positive fraction yields at least the rounded integer count.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    real = int(round(batch_size * fraction))
    real = max(0, min(batch_size, real))
    return {"real": real, "synthetic": batch_size - real}


def schedule_table(
    total_steps: int,
    batch_size: int,
    start: float = 0.0,
    end: float = 0.5,
    warmup: float = 1.0,
) -> Dict[str, object]:
    """Materialise the per-step batch composition and cumulative sample totals.

    Returns ``{"rows": [{step, fraction, real, synthetic}], "total_real",
    "total_synthetic"}``. The totals give the label-efficiency accounting the
    paper reports.
    """
    rows: List[Dict[str, object]] = []
    total_real = 0
    total_syn = 0
    for step in range(total_steps):
        frac = real_fraction(step, total_steps, start, end, warmup)
        comp = batch_composition(batch_size, frac)
        rows.append({"step": step, "fraction": frac, **comp})
        total_real += comp["real"]
        total_syn += comp["synthetic"]
    return {"rows": rows, "total_real": total_real, "total_synthetic": total_syn}
