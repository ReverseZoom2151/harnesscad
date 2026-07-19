"""VoxHammer inversion-preservation trajectory: caching + reinjection.

During inversion the source latents and attention K/V tokens are cached at every
timestep; during denoising the preserved region is reinjected from the cache at
each step. This deterministically traces the generation path backward and pins
the unedited region to its source features. This module provides:

- ``InversionCache``: per-timestep store/lookup of latents and K/V tokens;
- ``reinject``: overwrite preserved coordinates with the cached inverted latents
  at a given step, built on ``editing.voxhammer_preserve.hard_replace``;
- ``taylor_flow_step``: the second-order Taylor-improved Euler rectified-flow
  update (eqs. 1-2) used for high-fidelity inversion/denoising;
- ``late_cfg``: the late-time classifier-free guidance schedule, active
  only for ``t`` inside ``[t_lo, t_hi]``;
- ``linear_schedule``: the discrete time schedule ``0 = s0 < ... < sT = 1``.

All operations are deterministic and stdlib-only.
"""
from __future__ import annotations

from harnesscad.domain.editing.latent_preserve import hard_replace


def _key(t):
    # Quantise time keys so float schedules round-trip exactly.
    return round(float(t), 12)


class InversionCache:
    """Per-timestep cache of inverted latents and K/V tokens."""

    def __init__(self):
        self._latents = {}
        self._kv = {}

    def store(self, t, latents, kv=None):
        """Cache the latent map (and optional K/V map) at timestep ``t``."""
        k = _key(t)
        self._latents[k] = {c: tuple(float(x) for x in v) for c, v in latents.items()}
        if kv is not None:
            self._kv[k] = {tok: tuple(float(x) for x in v) for tok, v in kv.items()}
        return self

    def has(self, t):
        return _key(t) in self._latents

    def latents_at(self, t):
        k = _key(t)
        if k not in self._latents:
            raise KeyError("no latents cached at t=%r" % (t,))
        return self._latents[k]

    def kv_at(self, t):
        k = _key(t)
        if k not in self._kv:
            raise KeyError("no K/V cached at t=%r" % (t,))
        return self._kv[k]

    def timesteps(self):
        """Cached latent timesteps in ascending order."""
        return tuple(sorted(self._latents))


def reinject(current_latents, cache, t, keep_set):
    """Reinject cached inverted latents into the preserved region at step ``t``."""
    return hard_replace(current_latents, cache.latents_at(t), keep_set)


def linear_schedule(n_steps):
    """Discrete schedule 0 = s0 < s1 < ... < sT = 1 with ``n_steps`` intervals."""
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    return tuple(i / n_steps for i in range(n_steps + 1))


def taylor_flow_step(x, t, dt, f):
    """Second-order Taylor-improved Euler rectified-flow update (eqs. 1-2).

    ``f(x, t)`` is the velocity/noise-prediction field returning a vector the
    same length as ``x``. Uses a half-step finite difference to approximate the
    time derivative, yielding local truncation error O(dt^3). ``dt`` may be
    negative (inversion) or positive (denoising).
    """
    x = tuple(float(v) for v in x)
    f_t = tuple(float(v) for v in f(x, t))
    if len(f_t) != len(x):
        raise ValueError("f must return a vector of the same length as x")
    half = dt / 2.0
    x_mid = tuple(xi + half * fi for xi, fi in zip(x, f_t))
    f_mid = tuple(float(v) for v in f(x_mid, t + half))
    if half == 0.0:
        raise ValueError("dt must be non-zero")
    dfdt = tuple((fm - ft) / half for fm, ft in zip(f_mid, f_t))
    return tuple(
        xi + dt * ft + 0.5 * dt * dt * d
        for xi, ft, d in zip(x, f_t, dfdt)
    )


def late_cfg(f_cond, f_neg, omega, t, t_lo=0.5, t_hi=1.0):
    """Late-time classifier-free guidance.

    For ``t`` in ``[t_lo, t_hi]`` returns ``(1+omega)*f_cond - omega*f_neg``;
    otherwise returns ``f_cond`` unchanged. Gating guidance to late steps keeps
    early inversion steps invertible while sharpening preserved features.
    """
    f_cond = tuple(float(x) for x in f_cond)
    f_neg = tuple(float(x) for x in f_neg)
    if len(f_cond) != len(f_neg):
        raise ValueError("f_cond and f_neg must have equal length")
    if t_lo <= float(t) <= t_hi:
        w = float(omega)
        return tuple((1.0 + w) * c - w * n for c, n in zip(f_cond, f_neg))
    return f_cond
