"""Caption-invariant geometric augmentation for GeoCAD stage-1 (Zhang et al. 2025).

GeoCAD's stage-1 pre-training aligns the CAD-text representation with geometric
instructions by augmenting each local part (paper Sec. 3.2, Fig. 3):

    "for each local part, we apply random data augmentation via translation, scaling,
     rotation, and reflection. Notably, the geometric instructions of augmented
     samples remain unchanged due to the geometric consistency (e.g., the geometric
     instructions of the augmented samples in Fig. 3 are all right trapezoids)."

Sec. 4.1: simple parts get translation + scaling + rotation + reflection, while
complex parts get *only translation and scaling* "to avoid semantic inconsistencies
in captions".

The key deterministic property is **similarity invariance**: translation, uniform
scaling, rotation and reflection are similarity transforms, so any caption that
depends only on side-length ratios and angles (i.e. the vertex-based captions of
:mod:`geometry.geocad_vertex_caption`) is invariant under them. This module applies
those transforms to a local part's vertices with a seeded ``random.Random`` (no wall
clock) and exposes the augmentation policy per part branch.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from reconstruction.geocad_part_classifier import BRANCH_VERTEX, BRANCH_VLLM

Point = tuple[float, float]


def translate(vertices: list[Point], dx: float, dy: float) -> list[Point]:
    return [(x + dx, y + dy) for (x, y) in vertices]


def scale(vertices: list[Point], factor: float,
          about: Point = (0.0, 0.0)) -> list[Point]:
    """Uniform scaling about a pivot (preserves angles and length ratios)."""
    if factor == 0:
        raise ValueError("scale factor must be non-zero")
    cx, cy = about
    return [(cx + (x - cx) * factor, cy + (y - cy) * factor) for (x, y) in vertices]


def rotate(vertices: list[Point], angle_deg: float,
           about: Point = (0.0, 0.0)) -> list[Point]:
    """Rotate about a pivot by ``angle_deg`` degrees (counter-clockwise)."""
    cx, cy = about
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    out: list[Point] = []
    for x, y in vertices:
        dx, dy = x - cx, y - cy
        out.append((cx + dx * ca - dy * sa, cy + dx * sa + dy * ca))
    return out


def reflect(vertices: list[Point], axis: str = "x") -> list[Point]:
    """Reflect across the x-axis (``"x"``), y-axis (``"y"``), or line y=x (``"diag"``)."""
    if axis == "x":
        return [(x, -y) for (x, y) in vertices]
    if axis == "y":
        return [(-x, y) for (x, y) in vertices]
    if axis == "diag":
        return [(y, x) for (x, y) in vertices]
    raise ValueError(f"unknown reflection axis: {axis!r}")


@dataclass(frozen=True)
class AugmentPolicy:
    """Which transforms are permitted for a part branch (paper Sec. 4.1)."""

    translation: bool
    scaling: bool
    rotation: bool
    reflection: bool


# Simple parts: all four transforms. Complex parts: translation + scaling only.
POLICY_SIMPLE = AugmentPolicy(True, True, True, True)
POLICY_COMPLEX = AugmentPolicy(True, True, False, False)


def policy_for_branch(branch: str) -> AugmentPolicy:
    if branch == BRANCH_VERTEX:
        return POLICY_SIMPLE
    if branch == BRANCH_VLLM:
        return POLICY_COMPLEX
    raise ValueError(f"unknown branch: {branch!r}")


def augment_once(vertices: list[Point], rng: random.Random,
                 policy: AugmentPolicy = POLICY_SIMPLE) -> list[Point]:
    """Apply one random similarity augmentation permitted by ``policy``.

    Deterministic given ``rng``. The order is translation -> scaling -> rotation ->
    reflection; each is applied only if enabled by the policy. All four are
    similarity transforms, so the vertex-based caption is preserved.
    """
    out = list(vertices)
    if policy.translation:
        out = translate(out, rng.uniform(-50.0, 50.0), rng.uniform(-50.0, 50.0))
    if policy.scaling:
        out = scale(out, rng.uniform(0.25, 4.0))
    if policy.rotation:
        out = rotate(out, rng.uniform(0.0, 360.0))
    if policy.reflection and rng.random() < 0.5:
        out = reflect(out, rng.choice(("x", "y", "diag")))
    return out


def augment_batch(vertices: list[Point], rng: random.Random, count: int,
                  policy: AugmentPolicy = POLICY_SIMPLE) -> list[list[Point]]:
    """Produce ``count`` augmented copies of a local part (deterministic)."""
    return [augment_once(vertices, rng, policy) for _ in range(count)]
