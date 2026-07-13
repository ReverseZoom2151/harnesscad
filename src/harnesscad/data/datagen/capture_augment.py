"""Deterministic sensor-artifact and sketch-skill augmentation.

Geometry and rendering style are transformed by separate functions so training
pipelines cannot accidentally treat a cosmetic style change as a geometric
label change.  All randomness comes from a caller-provided seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any, Mapping, Sequence

Point = tuple[float, float]
Stroke = tuple[Point, ...]
Sketch = tuple[Stroke, ...]


@dataclass(frozen=True)
class CaptureProfile:
    occlusion: float = 0.0
    missing_segments: float = 0.0
    jitter: float = 0.0
    quantization: float = 0.0
    outlier_rate: float = 0.0
    partial_view: float = 1.0

    def __post_init__(self) -> None:
        for name in ("occlusion", "missing_segments", "outlier_rate"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.jitter < 0 or self.quantization < 0:
            raise ValueError("jitter and quantization must be non-negative")
        if not 0.0 < self.partial_view <= 1.0:
            raise ValueError("partial_view must be in (0, 1]")


CURRICULUM: Mapping[str, CaptureProfile] = {
    "clean": CaptureProfile(),
    "mild": CaptureProfile(0.05, 0.05, 0.01, 0.005, 0.01, 0.9),
    "moderate": CaptureProfile(0.15, 0.12, 0.03, 0.02, 0.04, 0.75),
    "severe": CaptureProfile(0.3, 0.25, 0.07, 0.05, 0.1, 0.55),
}


@dataclass(frozen=True)
class AugmentationEvent:
    family: str
    operation: str
    seed: int
    parameters: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "operation": self.operation,
            "seed": self.seed,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class AugmentedCapture:
    strokes: Sketch
    style: Mapping[str, Any] = field(default_factory=dict)
    provenance: tuple[AugmentationEvent, ...] = ()


@dataclass(frozen=True)
class SkillProfile:
    closure_gap: float
    overshoot: float
    point_dropout: float
    stroke_width_variation: float
    opacity_variation: float


SKILL_LEVELS: Mapping[str, SkillProfile] = {
    "expert": SkillProfile(0.0, 0.0, 0.0, 0.03, 0.02),
    "intermediate": SkillProfile(0.01, 0.015, 0.03, 0.12, 0.08),
    "novice": SkillProfile(0.04, 0.05, 0.12, 0.3, 0.2),
}


def augment_capture(
    strokes: Sequence[Sequence[Point]],
    *,
    seed: int,
    stage: str = "mild",
    profile: CaptureProfile | None = None,
) -> AugmentedCapture:
    """Apply staged capture artifacts while retaining lineage."""

    if profile is None:
        try:
            profile = CURRICULUM[stage]
        except KeyError as exc:
            raise ValueError(f"unknown curriculum stage: {stage}") from exc
    source = _normalise(strokes)
    rng = random.Random(seed)
    working = _partial(source, profile.partial_view)
    working = _occlude(working, profile.occlusion, rng)
    working = _drop_segments(working, profile.missing_segments, rng)
    working = _jitter(working, profile.jitter, rng)
    working = _quantize(working, profile.quantization)
    working = _outliers(working, profile.outlier_rate, profile.jitter, rng)
    event = AugmentationEvent(
        "capture",
        stage,
        seed,
        {
            "profile": {
                "occlusion": profile.occlusion,
                "missing_segments": profile.missing_segments,
                "jitter": profile.jitter,
                "quantization": profile.quantization,
                "outlier_rate": profile.outlier_rate,
                "partial_view": profile.partial_view,
            },
            "source_strokes": len(source),
            "source_points": sum(map(len, source)),
        },
    )
    return AugmentedCapture(working, {}, (event,))


def augment_skill_geometry(
    strokes: Sequence[Sequence[Point]], *, seed: int, level: str
) -> AugmentedCapture:
    """Mutate geometric stroke execution only; style remains untouched."""

    profile = _skill(level)
    rng = random.Random(seed)
    output: list[Stroke] = []
    for stroke in _normalise(strokes):
        kept = tuple(p for p in stroke if rng.random() >= profile.point_dropout)
        if not kept and stroke:
            kept = (stroke[0],)
        points = list(kept)
        if len(points) >= 2 and profile.overshoot:
            x0, y0 = points[-2]
            x1, y1 = points[-1]
            points[-1] = (
                x1 + (x1 - x0) * profile.overshoot,
                y1 + (y1 - y0) * profile.overshoot,
            )
        if (
            len(points) >= 2
            and points[0] == points[-1]
            and profile.closure_gap
        ):
            x, y = points[-1]
            points[-1] = (x + profile.closure_gap, y)
        output.append(tuple(points))
    event = AugmentationEvent(
        "geometry_skill",
        level,
        seed,
        {
            "closure_gap": profile.closure_gap,
            "overshoot": profile.overshoot,
            "point_dropout": profile.point_dropout,
        },
    )
    return AugmentedCapture(tuple(output), {}, (event,))


def augment_render_style(
    strokes: Sequence[Sequence[Point]], *, seed: int, level: str
) -> AugmentedCapture:
    """Produce rendering attributes without changing any coordinates."""

    profile = _skill(level)
    rng = random.Random(seed)
    style = {
        "stroke_widths": tuple(
            round(1.0 + rng.uniform(-profile.stroke_width_variation,
                                    profile.stroke_width_variation), 6)
            for _ in strokes
        ),
        "opacities": tuple(
            round(1.0 - rng.random() * profile.opacity_variation, 6)
            for _ in strokes
        ),
        "cap": rng.choice(("round", "square")),
    }
    event = AugmentationEvent(
        "render_style",
        level,
        seed,
        {
            "stroke_width_variation": profile.stroke_width_variation,
            "opacity_variation": profile.opacity_variation,
        },
    )
    return AugmentedCapture(_normalise(strokes), style, (event,))


def _skill(level: str) -> SkillProfile:
    try:
        return SKILL_LEVELS[level]
    except KeyError as exc:
        raise ValueError(f"unknown skill level: {level}") from exc


def _normalise(strokes: Sequence[Sequence[Point]]) -> Sketch:
    return tuple(tuple((float(x), float(y)) for x, y in stroke) for stroke in strokes)


def _partial(strokes: Sketch, fraction: float) -> Sketch:
    return tuple(stroke[: max(1, round(len(stroke) * fraction))] for stroke in strokes)


def _occlude(strokes: Sketch, rate: float, rng: random.Random) -> Sketch:
    if not rate:
        return strokes
    return tuple(tuple(p for p in stroke if rng.random() >= rate) for stroke in strokes)


def _drop_segments(strokes: Sketch, rate: float, rng: random.Random) -> Sketch:
    if not rate:
        return strokes
    result = []
    for stroke in strokes:
        result.append(tuple(
            point for index, point in enumerate(stroke)
            if index == 0 or rng.random() >= rate
        ))
    return tuple(result)


def _jitter(strokes: Sketch, amount: float, rng: random.Random) -> Sketch:
    if not amount:
        return strokes
    return tuple(tuple(
        (x + rng.uniform(-amount, amount), y + rng.uniform(-amount, amount))
        for x, y in stroke
    ) for stroke in strokes)


def _quantize(strokes: Sketch, step: float) -> Sketch:
    if not step:
        return strokes
    return tuple(tuple(
        (round(x / step) * step, round(y / step) * step) for x, y in stroke
    ) for stroke in strokes)


def _outliers(
    strokes: Sketch, rate: float, jitter: float, rng: random.Random
) -> Sketch:
    if not rate:
        return strokes
    magnitude = max(jitter * 10, 1.0)
    return tuple(tuple(
        (
            (x + rng.uniform(-magnitude, magnitude),
             y + rng.uniform(-magnitude, magnitude))
            if rng.random() < rate else (x, y)
        )
        for x, y in stroke
    ) for stroke in strokes)
