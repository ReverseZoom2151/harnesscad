"""Cardinal-direction landmarks: named anchor points on an entity's bounding box.

CodeToCAD's most transferable idea is that every entity carries a *bounding box*
and that geometry can be addressed symbolically -- ``TOP_FRONT_LEFT``,
``LEFT_CENTER``, ``offset(TOP_LEFT, Point("5mm", 0, 0))`` -- instead of by raw
coordinates.  Landmarks are then stable, human-readable handles that a text-to-CAD
program can reference ("put the hole at the top-front-left corner, 5mm in").

This module reimplements that layer with pure stdlib Python:

* :class:`BoundaryAxis` / :class:`BoundaryBox` -- min/max/center/length per axis,
  construction from points, union, expansion, containment.
* :func:`resolve_cardinal` -- a *compositional* parser for direction names.  The
  upstream project enumerates 45 constants; here any combination of the six
  face tokens (``left/right/front/back/top/bottom``) plus ``center`` is parsed
  greedily, so ``"top_front_left"``, ``"TopFrontLeft"``, ``"TOP-FRONT-LEFT"`` and
  ``"left front top"`` all resolve to the same ``(min, min, max)`` axis selector.
* :func:`cardinal_point` -- the anchor position of a cardinal on a box.
* :class:`Landmark` / :class:`LandmarkRegistry` -- named anchors that are
  *recomputed* from the (possibly updated) bounding box, with an optional offset
  written as a unit expression.
* :func:`nearest` -- the deterministic core of ``find_vertex`` / ``find_edge`` /
  ``find_face``: sort candidates by distance to the anchor, filter by search
  radius, break ties by index so the result never depends on iteration order.

Convention (matches CodeToCAD): +X right, +Y back, +Z up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from numeric.codetocad_length_expression import parse_length

__all__ = [
    "BoundaryAxis",
    "BoundaryBox",
    "CARDINAL_TOKENS",
    "MIN",
    "MAX",
    "CENTER",
    "resolve_cardinal",
    "cardinal_point",
    "Landmark",
    "LandmarkRegistry",
    "nearest",
    "CardinalError",
]

MIN = "min"
MAX = "max"
CENTER = "center"

# token -> (axis index, selector)
CARDINAL_TOKENS: dict[str, tuple[int, str]] = {
    "left": (0, MIN),
    "right": (0, MAX),
    "front": (1, MIN),
    "back": (1, MAX),
    "bottom": (2, MIN),
    "top": (2, MAX),
}

_TOKEN_ORDER = sorted(
    list(CARDINAL_TOKENS) + ["center"], key=lambda t: (-len(t), t)
)


class CardinalError(ValueError):
    """Raised for an unparseable or contradictory cardinal direction."""


@dataclass(frozen=True)
class BoundaryAxis:
    """A 1-D interval with min/max/center/length."""

    min: float
    max: float

    def __post_init__(self) -> None:
        if self.max < self.min:
            raise CardinalError("max must be >= min")

    @property
    def center(self) -> float:
        return (self.min + self.max) / 2.0

    @property
    def length(self) -> float:
        return self.max - self.min

    def select(self, selector: str) -> float:
        if selector == MIN:
            return self.min
        if selector == MAX:
            return self.max
        if selector == CENTER:
            return self.center
        raise CardinalError("unknown selector: " + str(selector))


@dataclass(frozen=True)
class BoundaryBox:
    """An axis-aligned bounding box made of three :class:`BoundaryAxis`."""

    x: BoundaryAxis
    y: BoundaryAxis
    z: BoundaryAxis

    @staticmethod
    def from_points(points) -> "BoundaryBox":
        pts = [tuple(float(c) for c in p) for p in points]
        if not pts:
            raise CardinalError("cannot build a bounding box from zero points")
        if any(len(p) != 3 for p in pts):
            raise CardinalError("points must be 3-dimensional")
        axes = []
        for i in range(3):
            values = [p[i] for p in pts]
            axes.append(BoundaryAxis(min(values), max(values)))
        return BoundaryBox(axes[0], axes[1], axes[2])

    @staticmethod
    def from_extents(
        center: tuple[float, float, float],
        size: tuple[float, float, float],
    ) -> "BoundaryBox":
        axes = []
        for c, s in zip(center, size):
            if s < 0:
                raise CardinalError("size must be non-negative")
            axes.append(BoundaryAxis(c - s / 2.0, c + s / 2.0))
        return BoundaryBox(axes[0], axes[1], axes[2])

    @property
    def axes(self) -> tuple[BoundaryAxis, BoundaryAxis, BoundaryAxis]:
        return (self.x, self.y, self.z)

    @property
    def center(self) -> tuple[float, float, float]:
        return (self.x.center, self.y.center, self.z.center)

    @property
    def size(self) -> tuple[float, float, float]:
        return (self.x.length, self.y.length, self.z.length)

    @property
    def diagonal(self) -> float:
        sx, sy, sz = self.size
        return (sx * sx + sy * sy + sz * sz) ** 0.5

    def contains(self, point, tolerance: float = 0.0) -> bool:
        for axis, value in zip(self.axes, point):
            if value < axis.min - tolerance or value > axis.max + tolerance:
                return False
        return True

    def expand(self, amount: float) -> "BoundaryBox":
        return BoundaryBox(
            *[BoundaryAxis(a.min - amount, a.max + amount) for a in self.axes]
        )

    def union(self, other: "BoundaryBox") -> "BoundaryBox":
        return BoundaryBox(
            *[
                BoundaryAxis(min(a.min, b.min), max(a.max, b.max))
                for a, b in zip(self.axes, other.axes)
            ]
        )

    def translated(self, offset) -> "BoundaryBox":
        return BoundaryBox(
            *[
                BoundaryAxis(a.min + d, a.max + d)
                for a, d in zip(self.axes, offset)
            ]
        )


def resolve_cardinal(name: str) -> tuple[str, str, str]:
    """Parse a cardinal-direction name into a per-axis selector triple.

    Returns ``(x_selector, y_selector, z_selector)`` where each selector is one of
    ``"min"``, ``"max"``, ``"center"``.  Unmentioned axes default to ``"center"``.

    >>> resolve_cardinal("top_front_left")
    ('min', 'min', 'max')
    >>> resolve_cardinal("center")
    ('center', 'center', 'center')
    """
    if not isinstance(name, str):
        raise CardinalError("cardinal must be a string")
    normalized = "".join(
        ch for ch in name.lower() if ch not in "_- \t"
    )
    if not normalized:
        raise CardinalError("empty cardinal direction")

    selectors: list[str | None] = [None, None, None]
    saw_center = False
    i = 0
    while i < len(normalized):
        for token in _TOKEN_ORDER:
            if normalized.startswith(token, i):
                if token == "center":
                    saw_center = True
                else:
                    axis, selector = CARDINAL_TOKENS[token]
                    if selectors[axis] is not None and selectors[axis] != selector:
                        raise CardinalError(
                            "contradictory cardinal direction: " + name
                        )
                    selectors[axis] = selector
                i += len(token)
                break
        else:
            raise CardinalError(
                "cannot parse cardinal direction {0!r} at {1!r}".format(
                    name, normalized[i:]
                )
            )
    if saw_center and all(s is None for s in selectors):
        return (CENTER, CENTER, CENTER)
    return tuple(s or CENTER for s in selectors)  # type: ignore[return-value]


def cardinal_point(
    box: BoundaryBox,
    cardinal: str,
    offset=(0.0, 0.0, 0.0),
) -> tuple[float, float, float]:
    """The anchor point of ``cardinal`` on ``box``, plus an optional offset.

    ``offset`` components may be numbers (metres) or unit expressions ("5mm").
    """
    selectors = resolve_cardinal(cardinal)
    deltas = [parse_length(component) for component in offset]
    return tuple(
        axis.select(selector) + delta
        for axis, selector, delta in zip(box.axes, selectors, deltas)
    )  # type: ignore[return-value]


@dataclass(frozen=True)
class Landmark:
    """A named anchor: a cardinal direction plus an offset expression."""

    name: str
    cardinal: str = "center"
    offset: tuple = (0.0, 0.0, 0.0)

    def position(self, box: BoundaryBox) -> tuple[float, float, float]:
        return cardinal_point(box, self.cardinal, self.offset)


@dataclass
class LandmarkRegistry:
    """Landmarks attached to one entity, recomputed from its current box."""

    box: BoundaryBox
    landmarks: dict = field(default_factory=dict)

    def add(self, name: str, cardinal: str = "center", offset=(0.0, 0.0, 0.0)) -> Landmark:
        if name in self.landmarks:
            raise CardinalError("duplicate landmark: " + name)
        resolve_cardinal(cardinal)  # validate eagerly
        for component in offset:
            parse_length(component)  # validate eagerly
        landmark = Landmark(name=name, cardinal=cardinal, offset=tuple(offset))
        self.landmarks[name] = landmark
        return landmark

    def get(self, name: str) -> Landmark:
        if name not in self.landmarks:
            raise CardinalError("no such landmark: " + name)
        return self.landmarks[name]

    def position(self, name: str) -> tuple[float, float, float]:
        return self.get(name).position(self.box)

    def set_box(self, box: BoundaryBox) -> None:
        """Landmarks follow the entity: positions are always derived, never stored."""
        self.box = box

    def positions(self) -> dict:
        return {
            name: landmark.position(self.box)
            for name, landmark in sorted(self.landmarks.items())
        }


def nearest(
    candidates,
    target,
    search_radius: float | None = None,
):
    """Sort ``candidates`` by distance to ``target``; filter by ``search_radius``.

    ``candidates`` is a sequence of ``(key, point)`` pairs.  Returns a list of
    ``(key, distance)`` sorted by ``(distance, original_index)`` so ties resolve
    deterministically.  This is the backend-free core of the ``find_*`` selectors.
    """
    scored = []
    for index, (key, point) in enumerate(candidates):
        dx = point[0] - target[0]
        dy = point[1] - target[1]
        dz = point[2] - target[2]
        distance = (dx * dx + dy * dy + dz * dz) ** 0.5
        if search_radius is not None and distance > search_radius:
            continue
        scored.append((distance, index, key))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [(key, distance) for distance, _index, key in scored]
