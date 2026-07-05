"""Validated parameter-sweep generation for reusable CAD part families."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from itertools import product
import json
import re
from typing import Any, Callable, Iterable, Mapping


_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_.-]+")


@dataclass(frozen=True)
class ParameterAxis:
    name: str
    values: tuple[float | int | str, ...]
    unit: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.values:
            raise ValueError("parameter axes require a name and at least one value")
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"duplicate values in axis {self.name}")


@dataclass(frozen=True)
class FamilySpec:
    family: str
    axes: tuple[ParameterAxis, ...]
    filename_template: str = "{family}_{index:04d}"
    maximum_variants: int = 1000
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.family.strip():
            raise ValueError("family name cannot be empty")
        names = [axis.name for axis in self.axes]
        if len(names) != len(set(names)):
            raise ValueError("axis names must be unique")
        if self.maximum_variants < 1:
            raise ValueError("maximum_variants must be positive")


@dataclass(frozen=True)
class Validation:
    accepted: bool
    checks: Mapping[str, bool]
    message: str = ""


@dataclass(frozen=True)
class FamilyEntry:
    name: str
    parameters: Mapping[str, float | int | str]
    units: Mapping[str, str]
    digest: str
    accepted: bool
    checks: Mapping[str, bool]
    error: str = ""


@dataclass(frozen=True)
class FamilyManifest:
    family: str
    entries: tuple[FamilyEntry, ...]
    metadata: Mapping[str, str]

    @property
    def accepted(self) -> tuple[FamilyEntry, ...]:
        return tuple(entry for entry in self.entries if entry.accepted)

    def to_json(self) -> str:
        return json.dumps(
            {
                "family": self.family,
                "metadata": dict(sorted(self.metadata.items())),
                "entries": [
                    {
                        "name": entry.name,
                        "parameters": dict(sorted(entry.parameters.items())),
                        "units": dict(sorted(entry.units.items())),
                        "digest": entry.digest,
                        "accepted": entry.accepted,
                        "checks": dict(sorted(entry.checks.items())),
                        "error": entry.error,
                    }
                    for entry in self.entries
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )


Builder = Callable[[Mapping[str, float | int | str]], Any]
Validator = Callable[[Any, Mapping[str, float | int | str]], Validation]
Serializer = Callable[[Any], bytes]


def parameter_grid(spec: FamilySpec) -> tuple[Mapping[str, float | int | str], ...]:
    count = 1
    for axis in spec.axes:
        count *= len(axis.values)
    if count > spec.maximum_variants:
        raise ValueError(
            f"family expands to {count} variants; limit is {spec.maximum_variants}"
        )
    return tuple(
        dict(zip((axis.name for axis in spec.axes), values))
        for values in product(*(axis.values for axis in spec.axes))
    )


def _safe_filename(value: str) -> str:
    cleaned = _SAFE_NAME.sub("-", value).strip("-._")
    if not cleaned:
        raise ValueError("filename template produced an empty/unsafe name")
    return cleaned


def generate_family(
    spec: FamilySpec,
    builder: Builder,
    validator: Validator,
    *,
    serializer: Serializer = lambda artifact: repr(artifact).encode("utf-8"),
    continue_on_error: bool = True,
) -> FamilyManifest:
    """Build every parameter combination and retain validation provenance."""
    entries: list[FamilyEntry] = []
    units = {axis.name: axis.unit for axis in spec.axes if axis.unit}
    for index, parameters in enumerate(parameter_grid(spec), start=1):
        context = {"family": spec.family, "index": index, **parameters}
        name = _safe_filename(spec.filename_template.format(**context))
        try:
            artifact = builder(parameters)
            payload = serializer(artifact)
            if not isinstance(payload, bytes):
                raise TypeError("serializer must return bytes")
            validation = validator(artifact, parameters)
            entries.append(
                FamilyEntry(
                    name=name,
                    parameters=dict(parameters),
                    units=units,
                    digest=sha256(payload).hexdigest(),
                    accepted=validation.accepted and all(validation.checks.values()),
                    checks=dict(validation.checks),
                    error=validation.message if not validation.accepted else "",
                )
            )
        except Exception as exc:  # each family member remains independently auditable
            if not continue_on_error:
                raise
            entries.append(
                FamilyEntry(
                    name=name,
                    parameters=dict(parameters),
                    units=units,
                    digest="",
                    accepted=False,
                    checks={},
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return FamilyManifest(spec.family, tuple(entries), dict(spec.metadata))


def range_axis(
    name: str, start: int, stop: int, step: int, *, unit: str = ""
) -> ParameterAxis:
    if step <= 0 or stop < start:
        raise ValueError("range requires positive step and stop >= start")
    return ParameterAxis(name, tuple(range(start, stop + 1, step)), unit)
