"""Graph-CAD iterative-arrangement patterns: grid and polar instantiation.

When many siblings share geometry, material and constraints, the Graph-CAD
graph format collapses them into a single template node with a ``pattern=``
clause instead of one line per instance::

    pattern=grid(rows:2, cols:3, x_spacing:0.1, y_spacing:0.2,
                 start_offset:(0.05, 0.05))
    pattern=polar(count:6, radius:0.08, start_angle:0, angle_step:60)

The engine then instantiates ``<id>_<r>_<c>`` for every grid cell (offset by
``start_offset + (c * dx, r * dy, 0)`` from the template's anchor) and
``<id>_<i>`` for every polar slot (at angle ``start_angle + i * angle_step`` on
a circle of the given radius). That expansion is a closed-form deterministic
rule, so this module implements it exactly: a tolerant ``key:value`` clause
parser, both instantiation rules, and the id-naming scheme the aggregate
feature tokens (``B[*]`` / ``B[k]`` / ``B[i,j]``) rely on.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Tuple

__all__ = [
    "GridPattern",
    "PolarPattern",
    "Instance",
    "parse_pattern",
    "expand_pattern",
    "expand_grid",
    "expand_polar",
]

Vec3 = Tuple[float, float, float]

_CLAUSE = re.compile(r"pattern\s*=\s*(?P<kind>grid|polar)\s*\((?P<body>[^)]*\([^)]*\)[^)]*|[^)]*)\)",
                     re.IGNORECASE)
_PAIR = re.compile(r"(?P<key>[a-zA-Z_]+)\s*:\s*(?P<value>\([^)]*\)|[^,]+)")


@dataclass(frozen=True)
class Instance:
    """One expanded pattern member: its id, its grid/polar index, its offset."""

    instance_id: str
    index: Tuple[int, ...]
    offset: Vec3


@dataclass(frozen=True)
class GridPattern:
    rows: int
    cols: int
    x_spacing: float
    y_spacing: float
    start_offset: Tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if self.rows < 1 or self.cols < 1:
            raise ValueError("grid pattern needs rows >= 1 and cols >= 1")

    @property
    def count(self) -> int:
        return self.rows * self.cols


@dataclass(frozen=True)
class PolarPattern:
    count: int
    radius: float
    start_angle: float = 0.0
    angle_step: float | None = None

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("polar pattern needs count >= 1")
        if self.radius < 0:
            raise ValueError("polar pattern radius must be non-negative")

    def step(self) -> float:
        """The angular step; defaults to a full even distribution."""
        if self.angle_step is not None:
            return self.angle_step
        return 360.0 / self.count


def _fields(body: str) -> Dict[str, str]:
    return {
        match.group("key").lower(): match.group("value").strip()
        for match in _PAIR.finditer(body)
    }


def _pair_value(text: str) -> Tuple[float, float]:
    numbers = [float(item) for item in re.findall(r"[-+0-9.eE]+", text)]
    if len(numbers) != 2:
        raise ValueError(f"expected a 2-tuple, got {text!r}")
    return numbers[0], numbers[1]


def _angle(text: str) -> float:
    cleaned = text.replace("deg", "").replace("°", "").strip()
    return float(cleaned)


def parse_pattern(text: str) -> GridPattern | PolarPattern:
    """Parse a ``pattern=grid(...)`` or ``pattern=polar(...)`` clause."""
    match = _CLAUSE.search(text)
    if not match:
        raise ValueError(f"not a pattern clause: {text!r}")
    kind = match.group("kind").lower()
    fields = _fields(match.group("body"))

    if kind == "grid":
        missing = {"rows", "cols"} - set(fields)
        if missing:
            raise ValueError(f"grid pattern is missing {sorted(missing)}")
        start = (
            _pair_value(fields["start_offset"]) if "start_offset" in fields else (0.0, 0.0)
        )
        return GridPattern(
            rows=int(float(fields["rows"])),
            cols=int(float(fields["cols"])),
            x_spacing=float(fields.get("x_spacing", 0.0)),
            y_spacing=float(fields.get("y_spacing", 0.0)),
            start_offset=start,
        )

    if "count" not in fields:
        raise ValueError("polar pattern is missing 'count'")
    step = _angle(fields["angle_step"]) if "angle_step" in fields else None
    return PolarPattern(
        count=int(float(fields["count"])),
        radius=float(fields.get("radius", 0.0)),
        start_angle=_angle(fields.get("start_angle", "0")),
        angle_step=step,
    )


def expand_grid(node_id: str, pattern: GridPattern) -> Tuple[Instance, ...]:
    """Instantiate ``<id>_<r>_<c>`` in row-major order."""
    x0, y0 = pattern.start_offset
    instances = []
    for row in range(pattern.rows):
        for col in range(pattern.cols):
            offset = (
                x0 + col * pattern.x_spacing,
                y0 + row * pattern.y_spacing,
                0.0,
            )
            instances.append(
                Instance(f"{node_id}_{row}_{col}", (row, col), offset)
            )
    return tuple(instances)


def expand_polar(node_id: str, pattern: PolarPattern) -> Tuple[Instance, ...]:
    """Instantiate ``<id>_<i>`` around a circle of ``pattern.radius``."""
    step = pattern.step()
    instances = []
    for index in range(pattern.count):
        theta = math.radians(pattern.start_angle + index * step)
        offset = (
            pattern.radius * math.cos(theta),
            pattern.radius * math.sin(theta),
            0.0,
        )
        instances.append(Instance(f"{node_id}_{index}", (index,), offset))
    return tuple(instances)


def expand_pattern(node_id: str, pattern: GridPattern | PolarPattern) -> Tuple[Instance, ...]:
    """Expand either pattern kind into its ordered instances."""
    if isinstance(pattern, GridPattern):
        return expand_grid(node_id, pattern)
    if isinstance(pattern, PolarPattern):
        return expand_polar(node_id, pattern)
    raise TypeError(f"unsupported pattern: {type(pattern).__name__}")
