"""Keyframe interpolation and looping animation for SHAPE-IT height fields.

SHAPE-IT's *Animation* element covers "transformations (shape or state changes
such as morphing and revealing)" and "Repetition and Speed ... repetitive or
looping animations, dictating the rhythm and periodicity of the motion"
(Section 3.3), and its heart example uses a *pulsing* animation that
"periodically changes [heartScale]" (Section 5.1).

This module provides the deterministic timeline primitives behind those
behaviours, operating on :class:`geometry.shapeit_heightfield.HeightField`:

* :func:`lerp_field` -- per-pin linear blend of two same-shaped fields.
* easing functions (``ease_linear``, ``ease_in``, ``ease_out``,
  ``ease_in_out``) mapping a normalized time ``t in [0, 1]`` to an eased ``t``.
* :func:`tween` -- morph one field into another over ``n`` frames.
* :func:`keyframe_sequence` -- interpolate through an ordered list of keyframes.
* :func:`pulse` -- an amplitude oscillation (the beating-heart animation).
* :func:`loop` / :func:`ping_pong` -- turn a frame list into a repeating cycle.

Every function is pure and deterministic; frames are always returned as fresh
fields.  Stdlib-only, no wall clock, no randomness.
"""

from __future__ import annotations

from math import cos, pi
from typing import Callable, List, Sequence

from harnesscad.domain.geometry.features.shapeit_heightfield import HeightField
from harnesscad.domain.geometry.features.shapeit_transforms import scale_amplitude

TAU = 2.0 * pi
Easing = Callable[[float], float]


# -- easing curves ----------------------------------------------------------


def ease_linear(t: float) -> float:
    return t


def ease_in(t: float) -> float:
    """Quadratic ease-in (slow start)."""
    return t * t


def ease_out(t: float) -> float:
    """Quadratic ease-out (slow finish)."""
    return 1.0 - (1.0 - t) * (1.0 - t)


def ease_in_out(t: float) -> float:
    """Cubic smoothstep ease-in-out."""
    return t * t * (3.0 - 2.0 * t)


# -- blending ---------------------------------------------------------------


def _check_pair(a: HeightField, b: HeightField) -> None:
    if a.rows != b.rows or a.cols != b.cols:
        raise ValueError("keyframes must share the same grid resolution")


def lerp_field(a: HeightField, b: HeightField, t: float) -> HeightField:
    """Per-pin linear interpolation ``(1 - t) * a + t * b``.

    ``t = 0`` returns a copy of ``a``, ``t = 1`` a copy of ``b``.  The result
    inherits ``a``'s stroke range (and clamps to it).
    """
    _check_pair(a, b)
    out = HeightField(a.rows, a.cols, a.min_height, a.max_height)
    for i in range(len(a.heights)):
        out.heights[i] = a.heights[i] + t * (b.heights[i] - a.heights[i])
    # re-clamp in case a and b used a wider range than a's stroke
    out.apply(lambda h: h)
    return out


def tween(
    a: HeightField,
    b: HeightField,
    frames: int,
    easing: Easing = ease_linear,
) -> List[HeightField]:
    """Morph field ``a`` into field ``b`` over ``frames`` frames (inclusive of
    both endpoints).  ``frames`` must be >= 2.  ``easing`` reshapes the time
    curve (default linear).
    """
    _check_pair(a, b)
    if frames < 2:
        raise ValueError("frames must be >= 2")
    out: List[HeightField] = []
    for i in range(frames):
        t = i / (frames - 1)
        out.append(lerp_field(a, b, easing(t)))
    return out


def keyframe_sequence(
    keyframes: Sequence[HeightField],
    frames_per_segment: int,
    easing: Easing = ease_linear,
) -> List[HeightField]:
    """Interpolate through an ordered list of keyframes.

    Produces ``frames_per_segment`` frames for each adjacent keyframe pair,
    sharing the boundary frame between segments so the returned timeline reads
    smoothly.  Needs >= 2 keyframes and ``frames_per_segment`` >= 2.
    """
    if len(keyframes) < 2:
        raise ValueError("need at least two keyframes")
    if frames_per_segment < 2:
        raise ValueError("frames_per_segment must be >= 2")
    timeline: List[HeightField] = []
    for k in range(len(keyframes) - 1):
        seg = tween(keyframes[k], keyframes[k + 1], frames_per_segment, easing)
        if k > 0:
            seg = seg[1:]  # drop duplicated boundary frame
        timeline.extend(seg)
    return timeline


def pulse(
    base: HeightField,
    frames: int,
    min_gain: float = 0.0,
    max_gain: float = 1.0,
    cycles: float = 1.0,
) -> List[HeightField]:
    """A pulsing amplitude animation (the beating-heart behaviour).

    Each frame is ``base`` with its relief scaled about the floor by a gain
    that oscillates cosinusoidally between ``min_gain`` and ``max_gain`` over
    ``cycles`` full oscillations across ``frames`` frames.  Frame 0 sits at
    ``max_gain`` (fully raised).
    """
    if frames < 1:
        raise ValueError("frames must be >= 1")
    if min_gain < 0.0 or max_gain < 0.0:
        raise ValueError("gains must be >= 0")
    if max_gain < min_gain:
        raise ValueError("max_gain must be >= min_gain")
    mid = (max_gain + min_gain) / 2.0
    amp = (max_gain - min_gain) / 2.0
    out: List[HeightField] = []
    denom = frames if frames > 1 else 1
    for i in range(frames):
        phase = TAU * cycles * (i / denom)
        gain = mid + amp * cos(phase)
        out.append(scale_amplitude(base, gain))
    return out


def loop(frames: Sequence[HeightField], repeats: int) -> List[HeightField]:
    """Concatenate ``frames`` ``repeats`` times (a looping animation)."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    out: List[HeightField] = []
    for _ in range(repeats):
        out.extend(f.copy() for f in frames)
    return out


def ping_pong(frames: Sequence[HeightField]) -> List[HeightField]:
    """Return ``frames`` followed by their reverse (minus the shared ends),
    yielding a there-and-back cycle for seamless looping.
    """
    if len(frames) < 2:
        return [f.copy() for f in frames]
    forward = [f.copy() for f in frames]
    backward = [f.copy() for f in frames[-2:0:-1]]
    return forward + backward
