"""picasso_self_supervision -- label-free render-compare training scheme.

PICASSO's headline contribution is *rendering self-supervision* (Sec. 4.3): a
CAD-sketch parameterisation network is pre-trained using only sketch **images**,
with no parameter-level annotations.  The learning signal is produced by
rendering the network's predicted primitives and comparing that rendering to the
target sketch image with an image-level loss.  Because the objective is computed
purely from rasters, "the need for corresponding CAD parameterization" is
eliminated (Abstract).

This module builds the deterministic scaffolding of that scheme on top of the
explicit rasteriser (:mod:`drawings.picasso_rasterizer`) and the image losses
(:mod:`drawings.picasso_render_loss`):

* :func:`generate_synthetic_sketch` -- randomly sample a set of parametric
  primitives, matching the paper's synthetic SRN training where "we train it
  synthetically by randomly generating primitives and their explicit renderings"
  (Sec. 4.3).  Seeded for full determinism.
* :func:`srn_training_pair` -- a ``(primitives, image)`` supervised pair for the
  renderer, where the parameters *are* known (synthetic SRN training).
* :class:`SelfSupervisionPair` -- a render-compare training example that exposes
  only the **target image**, never the parameters, and scores any candidate
  parameterisation by rendering it and taking an image loss.  This encodes the
  "no parameter labels" contract of PICASSO pre-training.
* :func:`make_self_supervision_dataset` -- a deterministic dataset of such pairs.
* :func:`render_consistency_loss` -- the label-free objective (multiscale l2 by
  default) between a candidate parameterisation and a target image.
* :func:`refine_by_rendering` -- deterministic *test-time optimisation*: improve
  a candidate parameterisation using only the rendering loss (a discrete,
  gradient-free stand-in for the SRN-based refinement of Sec. 5.2), needing no
  ground-truth parameters.

Pure stdlib; any randomness flows through an explicit :class:`random.Random`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from harnesscad.domain.drawings.picasso_rasterizer import (
    Arc,
    Circle,
    Dot,
    Line,
    Primitive,
    rasterize,
)
from harnesscad.domain.drawings.picasso_render_loss import Image, multiscale_l2_loss


# ---------------------------------------------------------------------------
# Synthetic primitive generation (deterministic, seeded).
# ---------------------------------------------------------------------------


def _rand_point(rng: random.Random, margin: float) -> tuple[float, float]:
    lo, hi = margin, 1.0 - margin
    return rng.uniform(lo, hi), rng.uniform(lo, hi)


def generate_synthetic_sketch(
    rng: random.Random,
    n_primitives: int = 4,
    types: tuple[str, ...] = ("line", "circle", "arc", "point"),
    margin: float = 0.1,
) -> list[Primitive]:
    """Sample ``n_primitives`` random primitives from ``types``.

    All coordinates lie within ``[margin, 1 - margin]`` of the canvas so nothing
    is clipped at the border.  Deterministic given ``rng``.
    """

    if n_primitives < 0:
        raise ValueError("n_primitives must be >= 0")
    if not types:
        raise ValueError("types must be non-empty")
    prims: list[Primitive] = []
    for _ in range(n_primitives):
        kind = rng.choice(types)
        if kind == "line":
            prims.append(Line(_rand_point(rng, margin), _rand_point(rng, margin)))
        elif kind == "circle":
            center = _rand_point(rng, margin)
            max_r = min(center[0], center[1], 1.0 - center[0], 1.0 - center[1])
            radius = rng.uniform(0.05, max(0.05, max_r))
            prims.append(Circle(center, radius))
        elif kind == "arc":
            prims.append(
                Arc(
                    _rand_point(rng, margin),
                    _rand_point(rng, margin),
                    _rand_point(rng, margin),
                )
            )
        elif kind == "point":
            prims.append(Dot(_rand_point(rng, margin)))
        else:
            raise ValueError(f"unknown primitive type: {kind!r}")
    return prims


# ---------------------------------------------------------------------------
# SRN synthetic pair (renderer supervision -- params known).
# ---------------------------------------------------------------------------


def srn_training_pair(
    primitives: list[Primitive],
    width: int = 128,
    height: int = 128,
    stroke_width: float = 1.5,
) -> tuple[list[Primitive], Image]:
    """A ``(primitives, rendered_image)`` pair for training the renderer (SRN).

    In synthetic SRN training the parameters are known, so the pair carries both.
    """

    img = rasterize(primitives, width, height, stroke_width=stroke_width)
    return primitives, img


# ---------------------------------------------------------------------------
# Render-compare self-supervision pair (no parameter labels).
# ---------------------------------------------------------------------------


@dataclass
class SelfSupervisionPair:
    """A label-free training example: only the target image is observable.

    The generating primitives are stored under ``_hidden_truth`` purely for
    evaluation/inspection; the self-supervision contract is that training reads
    only :attr:`target` and never the parameters.
    """

    target: Image
    width: int
    height: int
    stroke_width: float = 1.5
    levels: int = 5
    _hidden_truth: list[Primitive] = field(default_factory=list, repr=False)

    def loss(self, candidate: list[Primitive]) -> float:
        """Rendering-consistency loss of ``candidate`` against the target image.

        Renders the candidate and takes the multiscale l2 image loss -- computed
        without ever consulting the ground-truth parameters.
        """

        return render_consistency_loss(
            candidate,
            self.target,
            width=self.width,
            height=self.height,
            stroke_width=self.stroke_width,
            levels=self.levels,
        )


def make_self_supervision_pair(
    primitives: list[Primitive],
    width: int = 128,
    height: int = 128,
    stroke_width: float = 1.5,
    levels: int = 5,
) -> SelfSupervisionPair:
    """Render ``primitives`` and wrap the raster as a label-free training pair."""

    img = rasterize(primitives, width, height, stroke_width=stroke_width)
    return SelfSupervisionPair(
        target=img,
        width=width,
        height=height,
        stroke_width=stroke_width,
        levels=levels,
        _hidden_truth=list(primitives),
    )


def make_self_supervision_dataset(
    rng: random.Random,
    count: int,
    width: int = 64,
    height: int = 64,
    n_primitives: int = 4,
    stroke_width: float = 1.5,
    levels: int = 5,
) -> list[SelfSupervisionPair]:
    """Build ``count`` deterministic label-free render-compare training pairs."""

    if count < 0:
        raise ValueError("count must be >= 0")
    out: list[SelfSupervisionPair] = []
    for _ in range(count):
        prims = generate_synthetic_sketch(rng, n_primitives)
        out.append(
            make_self_supervision_pair(
                prims, width, height, stroke_width=stroke_width, levels=levels
            )
        )
    return out


# ---------------------------------------------------------------------------
# Label-free objective + test-time optimisation.
# ---------------------------------------------------------------------------


def render_consistency_loss(
    candidate: list[Primitive],
    target: Image,
    width: int = 128,
    height: int = 128,
    stroke_width: float = 1.5,
    levels: int = 5,
) -> float:
    """Multiscale-l2 rendering loss of ``candidate`` vs a target raster.

    The self-supervision objective: no parameter labels are involved, only the
    rendered candidate and the target image.
    """

    rendered = rasterize(candidate, width, height, stroke_width=stroke_width)
    return multiscale_l2_loss(rendered, target, levels=levels)


def _translate(prim: Primitive, dx: float, dy: float) -> Primitive:
    """Return ``prim`` shifted by ``(dx, dy)`` in canvas coordinates."""

    def s(p: tuple[float, float]) -> tuple[float, float]:
        return p[0] + dx, p[1] + dy

    if isinstance(prim, Line):
        return Line(s(prim.start), s(prim.end))
    if isinstance(prim, Circle):
        return Circle(s(prim.center), prim.radius)
    if isinstance(prim, Arc):
        return Arc(s(prim.start), s(prim.mid), s(prim.end))
    if isinstance(prim, Dot):
        return Dot(s(prim.pos))
    raise TypeError(f"unknown primitive type: {type(prim)!r}")


def refine_by_rendering(
    candidate: list[Primitive],
    target: Image,
    width: int = 64,
    height: int = 64,
    stroke_width: float = 1.5,
    levels: int = 5,
    steps: int = 20,
    step_size: float = 0.05,
    shrink: float = 0.5,
    min_step: float = 1e-3,
) -> tuple[list[Primitive], float]:
    """Deterministic test-time optimisation by rendering self-supervision.

    Coordinate-descent over a global translation of the whole candidate sketch,
    minimising the rendering-consistency loss with **no parameter labels**.  On
    each step the four axis-aligned offsets are tried; if none improves, the step
    size is shrunk.  Returns the refined primitives and their final loss.  This is
    a gradient-free analogue of the SRN test-time refinement (Sec. 5.2).
    """

    def loss_of(prims: list[Primitive]) -> float:
        return render_consistency_loss(
            prims, target, width, height, stroke_width=stroke_width, levels=levels
        )

    best = list(candidate)
    best_loss = loss_of(best)
    step = step_size
    for _ in range(steps):
        improved = False
        for dx, dy in ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)):
            trial = [_translate(p, dx, dy) for p in best]
            trial_loss = loss_of(trial)
            if trial_loss < best_loss - 1e-12:
                best, best_loss = trial, trial_loss
                improved = True
                break
        if not improved:
            step *= shrink
            if step < min_step:
                break
    return best, best_loss
