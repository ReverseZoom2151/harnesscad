"""Vitruvion primitive noise: seeded truncated-normal perturbation of sketch primitives.

Vitruvion (Seff et al., ICLR 2022 -- ``img2cad/noise_models.py``,
``img2cad/constraint_data.py``) trains its *constraint* model on primitives that have
been jittered, so that at inference time it can constrain the imperfect primitives its
own primitive model produced (or that were lifted from an image).  The jitter is applied
in the *parameter* space of ``geometry.vitruvion_sketch_norm.parameterize_entity`` -- so
an arc is perturbed by moving its start/mid/end points and is then re-fit through the
circumcentre, not by nudging its centre and radius.

Faithful details that matter
----------------------------
* **The truncation bounds are in units of the standard deviation, not coordinates.**
  The reference calls ``scipy.stats.truncnorm.rvs(a=-max_diff, b=max_diff, scale=std)``,
  and scipy defines ``a``/``b`` on the *standardised* variable.  With the paper's
  defaults (``std = max_diff = 0.15``) the actual perturbation is therefore bounded by
  ``max_diff * std = 0.0225`` of the sketch's long axis -- 6.6x tighter than the naive
  reading of "max_difference = 0.15".  This module reproduces that exactly and exposes
  the interpretation via ``bounds_in_std_units`` for callers who want the literal one.
* **Arcs get a rejection loop with a halving scale.**  Re-fitting a circle through three
  jittered points can move the centre arbitrarily far (nearly-collinear points), so an
  arc redraws with ``scale = std / 2**trial`` until both the re-derived parameters *and*
  the centre have moved by no more than ``max_diff`` (in coordinate units this time).
  A degenerate (collinear) re-fit also redraws.  Non-arc entities take the first draw.
* **The ``isConstruction`` flag survives** the round-trip through the parameter vector.
* Vitruvion swallows any failure and falls back to the un-noised sketch; here the
  fallback is explicit (``max_trials``) and the original entity is returned unchanged.

Determinism: all randomness comes from a ``random.Random(seed)``; the truncated normal is
sampled by rejection (draw ``gauss(0, 1)``, keep it if it lies in ``[a, b]``), so a given
seed and entity order always yield the same sketch.

Pure stdlib.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

from harnesscad.domain.geometry.sketch.vitruvion_sketch_norm import (
    VArc,
    entity_from_params,
    parameterize_entity,
)

__all__ = [
    "PrimitiveNoiseConfig",
    "truncated_normal",
    "noisify_entity",
    "noisify_sketch",
]


@dataclass(frozen=True)
class PrimitiveNoiseConfig:
    """Vitruvion's ``PrimitiveNoiseConfig`` (constraint-model defaults)."""

    enabled: bool = True
    std: float = 0.15
    max_difference: float = 0.15
    max_trials: int = 24
    bounds_in_std_units: bool = True


def truncated_normal(
    rng: random.Random, a: float, b: float, scale: float, max_draws: int = 1000
) -> float:
    """One draw from a normal truncated to ``[a, b]`` *standard units*, times ``scale``.

    Matches ``scipy.stats.truncnorm.rvs(a, b, scale=scale)``: the support of the result is
    ``[a * scale, b * scale]``.  Sampled by rejection, so it is exactly reproducible from
    ``rng``'s state.  ``ValueError`` if ``a >= b`` or if the region is so improbable that
    ``max_draws`` rejections occur.
    """
    if a >= b:
        raise ValueError("need a < b")
    for _ in range(max_draws):
        z = rng.gauss(0.0, 1.0)
        if a <= z <= b:
            return z * scale
    raise ValueError("rejection sampling failed to hit [a, b]")


def _draw(rng: random.Random, size: int, a: float, b: float, scale: float) -> List[float]:
    return [truncated_normal(rng, a, b, scale) for _ in range(size)]


def noisify_entity(
    entity,
    rng: random.Random,
    std: float = 0.2,
    max_diff: float = 0.1,
    max_trials: int = 24,
    bounds_in_std_units: bool = True,
):
    """Return a jittered copy of ``entity`` (the input is never mutated).

    Arcs redraw with a halving scale until the re-fit stays within ``max_diff`` of the
    original; if ``max_trials`` draws all fail, the original entity is returned unchanged
    (Vitruvion's implicit fallback).
    """
    if std <= 0.0 or max_diff <= 0.0:
        raise ValueError("std and max_diff must be positive")

    params = parameterize_entity(entity)
    if params is None:
        return copy.deepcopy(entity)

    if bounds_in_std_units:
        low, high = -max_diff, max_diff
    else:
        # Literal reading: clip at +/- max_diff in coordinate units.
        low, high = -max_diff / std, max_diff / std

    is_arc = isinstance(entity, VArc)

    for trial in range(max_trials):
        scale = std / (2 ** trial) if is_arc else std
        noise = _draw(rng, len(params), low, high, scale)
        new_params = [p + n for p, n in zip(params, noise)]
        new_entity = entity_from_params(new_params, getattr(entity, "entity_id", None))

        if new_entity is None:
            continue  # degenerate arc re-fit: redraw

        if not is_arc:
            new_entity.is_construction = entity.is_construction
            return new_entity

        actual = parameterize_entity(new_entity)
        if any(abs(x - p) > max_diff for x, p in zip(actual, params)):
            continue
        old_center = entity.center_point
        new_center = new_entity.center_point
        if any(abs(n - o) > max_diff for n, o in zip(new_center, old_center)):
            continue

        new_entity.is_construction = entity.is_construction
        return new_entity

    return copy.deepcopy(entity)


def noisify_sketch(
    entities: Sequence[object],
    seed: int = 0,
    config: Optional[PrimitiveNoiseConfig] = None,
) -> List[object]:
    """Jitter every primitive of a *normalised* sketch; returns a new list.

    With ``config.enabled = False`` the sketch is deep-copied unchanged, so a caller can
    keep one code path.  Deterministic in ``seed``.
    """
    if config is None:
        config = PrimitiveNoiseConfig()
    if not config.enabled:
        return [copy.deepcopy(e) for e in entities]

    rng = random.Random(seed)
    return [
        noisify_entity(
            entity,
            rng,
            std=config.std,
            max_diff=config.max_difference,
            max_trials=config.max_trials,
            bounds_in_std_units=config.bounds_in_std_units,
        )
        for entity in entities
    ]
