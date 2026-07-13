"""Deterministic geometric invariance and equivariance checks.

The checker is deliberately independent of a geometry kernel.  Callers inject
the transformation and measurement functions, which makes the same contract
usable for CISP programs, backend solids, meshes, or simple test fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isclose, sin
from typing import Any, Callable, Generic, Iterable, Mapping, Sequence, TypeVar

Subject = TypeVar("Subject")
Parameter = TypeVar("Parameter")
Measurement = TypeVar("Measurement")
Point = tuple[float, ...]


@dataclass(frozen=True)
class ContractMetadata:
    """Machine-readable semantics for one geometric consistency contract."""

    name: str
    transformation: str
    relation: str
    observable: str
    description: str = ""
    scale_exponent: float | None = None

    def __post_init__(self) -> None:
        if self.relation not in {"invariant", "equivariant"}:
            raise ValueError("relation must be 'invariant' or 'equivariant'")
        if self.transformation not in {"translation", "rotation", "scale", "custom"}:
            raise ValueError("unsupported transformation metadata")
        if self.relation == "invariant" and self.scale_exponent is not None:
            raise ValueError("invariant contracts cannot have a scale exponent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "transformation": self.transformation,
            "relation": self.relation,
            "observable": self.observable,
            "description": self.description,
            "scale_exponent": self.scale_exponent,
        }


@dataclass(frozen=True)
class PerturbationCase(Generic[Parameter]):
    """A named, deterministic transformation parameter."""

    name: str
    parameter: Parameter


@dataclass(frozen=True)
class ConsistencyResult(Generic[Measurement]):
    case: str
    actual: Measurement
    expected: Measurement
    passed: bool
    absolute_error: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "actual": self.actual,
            "expected": self.expected,
            "passed": self.passed,
            "absolute_error": self.absolute_error,
        }


@dataclass(frozen=True)
class ConsistencyReport(Generic[Measurement]):
    metadata: ContractMetadata
    baseline: Measurement
    results: tuple[ConsistencyResult[Measurement], ...]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "baseline": self.baseline,
            "passed": self.passed,
            "results": [result.to_dict() for result in self.results],
        }


Comparator = Callable[[Measurement, Measurement], bool]
Expected = Callable[[Measurement, Parameter], Measurement]


class InvarianceContract(Generic[Subject, Parameter, Measurement]):
    """Apply perturbations and compare observations with declared semantics."""

    def __init__(
        self,
        metadata: ContractMetadata,
        *,
        transform: Callable[[Subject, Parameter], Subject],
        measure: Callable[[Subject], Measurement],
        expected: Expected[Measurement, Parameter] | None = None,
        comparator: Comparator[Measurement] | None = None,
        rel_tol: float = 1e-9,
        abs_tol: float = 1e-9,
    ) -> None:
        self.metadata = metadata
        self.transform = transform
        self.measure = measure
        self.expected = expected or (lambda baseline, _parameter: baseline)
        self.comparator = comparator
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

    def evaluate(
        self,
        subject: Subject,
        cases: Iterable[PerturbationCase[Parameter]],
    ) -> ConsistencyReport[Measurement]:
        baseline = self.measure(subject)
        results: list[ConsistencyResult[Measurement]] = []
        for case in cases:
            actual = self.measure(self.transform(subject, case.parameter))
            expected = self.expected(baseline, case.parameter)
            passed = self._equal(actual, expected)
            error = _absolute_error(actual, expected)
            results.append(
                ConsistencyResult(case.name, actual, expected, passed, error)
            )
        return ConsistencyReport(self.metadata, baseline, tuple(results))

    def _equal(self, actual: Measurement, expected: Measurement) -> bool:
        if self.comparator is not None:
            return self.comparator(actual, expected)
        return _close(actual, expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol)


def scale_expected(exponent: float) -> Callable[[float, float], float]:
    """Return the expected covariance for a length^``exponent`` observable."""

    return lambda baseline, factor: baseline * factor**exponent


def translate_points(points: Sequence[Point], offset: Sequence[float]) -> tuple[Point, ...]:
    if not points:
        return ()
    if len(offset) != len(points[0]) or any(len(point) != len(offset) for point in points):
        raise ValueError("point and translation dimensions must match")
    return tuple(
        tuple(coordinate + delta for coordinate, delta in zip(point, offset))
        for point in points
    )


def rotate_points_2d(points: Sequence[Point], angle_radians: float) -> tuple[Point, ...]:
    """Rotate 2D points counter-clockwise around the origin."""

    if any(len(point) != 2 for point in points):
        raise ValueError("2D rotation requires two-dimensional points")
    c, s = cos(angle_radians), sin(angle_radians)
    return tuple((x * c - y * s, x * s + y * c) for x, y in points)


def scale_points(points: Sequence[Point], factor: float) -> tuple[Point, ...]:
    if factor <= 0:
        raise ValueError("scale factor must be positive")
    return tuple(tuple(coordinate * factor for coordinate in point) for point in points)


def _close(actual: Any, expected: Any, *, rel_tol: float, abs_tol: float) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return isclose(float(actual), float(expected), rel_tol=rel_tol, abs_tol=abs_tol)
    if (
        isinstance(actual, Sequence)
        and not isinstance(actual, (str, bytes))
        and isinstance(expected, Sequence)
        and not isinstance(expected, (str, bytes))
    ):
        return len(actual) == len(expected) and all(
            _close(a, e, rel_tol=rel_tol, abs_tol=abs_tol)
            for a, e in zip(actual, expected)
        )
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return actual.keys() == expected.keys() and all(
            _close(actual[key], expected[key], rel_tol=rel_tol, abs_tol=abs_tol)
            for key in actual
        )
    return actual == expected


def _absolute_error(actual: Any, expected: Any) -> float | None:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected))
    return None
