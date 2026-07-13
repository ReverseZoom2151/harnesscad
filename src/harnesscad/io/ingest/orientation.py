"""Deterministic orientation hypotheses, fusion, diagnostics, and sampling.

The module deliberately uses only the standard library.  Rotations are unit
quaternions in ``(w, x, y, z)`` order; antipodal quaternions represent the
same physical rotation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from typing import Callable, Iterable, Mapping, Sequence


_EPS = 1e-12


def _finite(values: Iterable[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


@dataclass(frozen=True)
class Rotation:
    """Validated unit quaternion in ``(w, x, y, z)`` order."""

    w: float
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        values = (self.w, self.x, self.y, self.z)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("quaternion components must be finite")
        norm = math.sqrt(sum(value * value for value in values))
        if norm <= _EPS:
            raise ValueError("quaternion must have non-zero norm")
        normalized = tuple(value / norm for value in values)
        # Canonicalize the antipodal representation for stable equality/replay.
        for value in normalized:
            if abs(value) > _EPS:
                if value < 0:
                    normalized = tuple(-item for item in normalized)
                break
        object.__setattr__(self, "w", normalized[0])
        object.__setattr__(self, "x", normalized[1])
        object.__setattr__(self, "y", normalized[2])
        object.__setattr__(self, "z", normalized[3])

    @classmethod
    def identity(cls) -> "Rotation":
        return cls(1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_axis_angle(
        cls, axis: Sequence[float], angle_radians: float
    ) -> "Rotation":
        components = _finite(axis, "axis")
        if len(components) != 3:
            raise ValueError("axis must have exactly three components")
        if not math.isfinite(angle_radians):
            raise ValueError("angle must be finite")
        length = math.sqrt(sum(value * value for value in components))
        if length <= _EPS:
            raise ValueError("axis must have non-zero length")
        half = angle_radians / 2.0
        scale = math.sin(half) / length
        return cls(
            math.cos(half),
            components[0] * scale,
            components[1] * scale,
            components[2] * scale,
        )

    def compose(self, other: "Rotation") -> "Rotation":
        if not isinstance(other, Rotation):
            raise TypeError("other must be a Rotation")
        a, b = self, other
        return Rotation(
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        )

    def inverse(self) -> "Rotation":
        return Rotation(self.w, -self.x, -self.y, -self.z)

    def apply(self, point: Sequence[float]) -> tuple[float, float, float]:
        values = _finite(point, "point")
        if len(values) != 3:
            raise ValueError("point must have exactly three components")
        qvec = (self.x, self.y, self.z)
        cross = (
            qvec[1] * values[2] - qvec[2] * values[1],
            qvec[2] * values[0] - qvec[0] * values[2],
            qvec[0] * values[1] - qvec[1] * values[0],
        )
        second = (
            qvec[1] * cross[2] - qvec[2] * cross[1],
            qvec[2] * cross[0] - qvec[0] * cross[2],
            qvec[0] * cross[1] - qvec[1] * cross[0],
        )
        return tuple(
            values[index] + 2.0 * (self.w * cross[index] + second[index])
            for index in range(3)
        )  # type: ignore[return-value]


def angular_distance(left: Rotation, right: Rotation) -> float:
    """Shortest physical angular distance in radians, in ``[0, pi]``."""

    if not isinstance(left, Rotation) or not isinstance(right, Rotation):
        raise TypeError("angular_distance expects Rotation instances")
    dot = abs(
        left.w * right.w
        + left.x * right.x
        + left.y * right.y
        + left.z * right.z
    )
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def reference_cube_corners(rotation: Rotation) -> tuple[tuple[float, float, float], ...]:
    """Encode a rotation by the eight transformed corners of a unit cube."""

    if not isinstance(rotation, Rotation):
        raise TypeError("rotation must be a Rotation")
    return tuple(
        rotation.apply((x, y, z))
        for x in (-0.5, 0.5)
        for y in (-0.5, 0.5)
        for z in (-0.5, 0.5)
    )


def symmetry_equivalents(
    orientation: Rotation, symmetries: Iterable[Rotation], tolerance: float = 1e-9
) -> tuple[Rotation, ...]:
    """Return unique explicit modes obtained by right-applying symmetries."""

    if tolerance <= 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    candidates = [orientation]
    candidates.extend(orientation.compose(symmetry) for symmetry in symmetries)
    unique: list[Rotation] = []
    for candidate in candidates:
        if not any(angular_distance(candidate, prior) <= tolerance for prior in unique):
            unique.append(candidate)
    return tuple(unique)


@dataclass(frozen=True)
class OrientationMode:
    rotation: Rotation
    weight: float
    label: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.rotation, Rotation):
            raise TypeError("rotation must be a Rotation")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("mode weight must be finite and non-negative")


@dataclass(frozen=True)
class OrientationDistribution:
    """Normalized, potentially multimodal orientation distribution."""

    modes: tuple[OrientationMode, ...]

    def __post_init__(self) -> None:
        modes = tuple(self.modes)
        if not modes:
            raise ValueError("orientation distribution requires at least one mode")
        total = sum(mode.weight for mode in modes)
        if total <= _EPS:
            raise ValueError("orientation distribution requires positive total weight")
        object.__setattr__(
            self,
            "modes",
            tuple(
                OrientationMode(mode.rotation, mode.weight / total, mode.label)
                for mode in modes
            ),
        )

    @property
    def entropy(self) -> float:
        return -sum(
            mode.weight * math.log(mode.weight)
            for mode in self.modes
            if mode.weight > _EPS
        )

    @property
    def normalized_entropy(self) -> float:
        if len(self.modes) == 1:
            return 0.0
        return self.entropy / math.log(len(self.modes))

    @property
    def confidence(self) -> float:
        return max(mode.weight for mode in self.modes)

    @property
    def ambiguity(self) -> float:
        return self.normalized_entropy

    @property
    def best(self) -> OrientationMode:
        return max(self.modes, key=lambda mode: mode.weight)


ScoreFunction = Callable[[Rotation], float]


def product_of_experts(
    candidates: Sequence[OrientationMode],
    experts: Mapping[str, ScoreFunction],
    *,
    expert_weights: Mapping[str, float] | None = None,
) -> OrientationDistribution:
    """Fuse named non-negative expert scores by a weighted product."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    if not experts:
        raise ValueError("at least one expert is required")
    weights = expert_weights or {}
    unknown = set(weights) - set(experts)
    if unknown:
        raise ValueError(f"weights provided for unknown experts: {sorted(unknown)}")
    fused: list[OrientationMode] = []
    for candidate in candidates:
        score = candidate.weight
        for name, expert in experts.items():
            exponent = float(weights.get(name, 1.0))
            if not math.isfinite(exponent) or exponent < 0:
                raise ValueError("expert weights must be finite and non-negative")
            evidence = float(expert(candidate.rotation))
            if not math.isfinite(evidence) or evidence < 0:
                raise ValueError(f"expert {name!r} returned an invalid score")
            score *= evidence**exponent
        fused.append(OrientationMode(candidate.rotation, score, candidate.label))
    if not any(mode.weight > _EPS for mode in fused):
        raise ValueError("expert fusion assigned zero mass to every candidate")
    return OrientationDistribution(tuple(fused))


@dataclass(frozen=True)
class SampleProvenance:
    seed: int
    coarse_count: int
    fine_per_mode: int
    fine_radius_radians: float
    distribution_digest: str


@dataclass(frozen=True)
class OrientationSamples:
    rotations: tuple[Rotation, ...]
    provenance: SampleProvenance


def _random_rotation(rng: random.Random) -> Rotation:
    # Shoemake's uniform unit-quaternion method.
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    a = math.sqrt(1.0 - u1)
    b = math.sqrt(u1)
    return Rotation(
        b * math.cos(2 * math.pi * u3),
        a * math.sin(2 * math.pi * u2),
        a * math.cos(2 * math.pi * u2),
        b * math.sin(2 * math.pi * u3),
    )


def coarse_to_fine_samples(
    distribution: OrientationDistribution,
    *,
    seed: int,
    coarse_count: int = 16,
    fine_per_mode: int = 4,
    fine_radius_radians: float = math.radians(10),
) -> OrientationSamples:
    """Generate replayable global samples followed by mode-local refinements."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if coarse_count < 0 or fine_per_mode < 0:
        raise ValueError("sample counts must be non-negative")
    if not math.isfinite(fine_radius_radians) or not 0 <= fine_radius_radians <= math.pi:
        raise ValueError("fine radius must be finite and in [0, pi]")
    rng = random.Random(seed)
    rotations = [_random_rotation(rng) for _ in range(coarse_count)]
    for mode in distribution.modes:
        rotations.append(mode.rotation)
        for _ in range(fine_per_mode):
            axis = [rng.gauss(0.0, 1.0) for _ in range(3)]
            angle = rng.uniform(0.0, fine_radius_radians)
            perturbation = Rotation.from_axis_angle(axis, angle)
            rotations.append(mode.rotation.compose(perturbation))
    digest_payload = ";".join(
        f"{mode.rotation.w:.12g},{mode.rotation.x:.12g},"
        f"{mode.rotation.y:.12g},{mode.rotation.z:.12g}:{mode.weight:.12g}"
        for mode in distribution.modes
    )
    provenance = SampleProvenance(
        seed,
        coarse_count,
        fine_per_mode,
        fine_radius_radians,
        hashlib.sha256(digest_payload.encode("ascii")).hexdigest(),
    )
    return OrientationSamples(tuple(rotations), provenance)
