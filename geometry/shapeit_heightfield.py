"""Pin-grid height field: the deterministic 2.5D representation behind SHAPE-IT.

SHAPE-IT (Qian et al., UIST '24) authors *pin-based shape displays* -- a grid
of vertically actuated pins whose per-pin heights form a 2.5D surface (the
paper's 24x24 / 30x30 displays).  Stripped of the LLM code-generation layer,
the underlying object is a plain rectangular grid of clamped pin heights.

This module provides that core representation, :class:`HeightField`, and a
couple of scalar *field metrics* used to compare two displayed shapes (the
paper compares generated vs. intended outputs).  Everything is stdlib-only,
deterministic, and free of any hardware/actuation concern:

* Pin heights are stored row-major as a flat list of floats.
* Heights are clamped to a hardware ``stroke`` range ``[min_height,
  max_height]`` -- pins cannot travel past their mechanical limits (the paper's
  100 mm stroke).  ``set``/``fill``/``add`` all clamp.
* Grid indexing is ``(row, col)`` with ``row`` increasing downward, matching a
  raster / display convention.

The primitive drawers (:mod:`geometry.shapeit_primitives`), spatial transforms
(:mod:`geometry.shapeit_transforms`), and keyframe animation
(:mod:`geometry.shapeit_keyframe`) all operate on this class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


@dataclass
class HeightField:
    """A ``rows`` x ``cols`` grid of clamped pin heights (a 2.5D display).

    Parameters
    ----------
    rows, cols:
        Grid resolution (number of pins along each axis).  Both must be >= 1.
    min_height, max_height:
        Mechanical stroke bounds; every stored height is clamped here.
    fill:
        Initial height for every pin (default the floor ``min_height``),
        clamped into range.
    """

    rows: int
    cols: int
    min_height: float = 0.0
    max_height: float = 1.0
    heights: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.rows < 1 or self.cols < 1:
            raise ValueError("rows and cols must both be >= 1")
        if self.max_height < self.min_height:
            raise ValueError("max_height must be >= min_height")
        n = self.rows * self.cols
        if not self.heights:
            floor = self.min_height
            self.heights = [floor] * n
        else:
            if len(self.heights) != n:
                raise ValueError(
                    "heights length %d != rows*cols %d" % (len(self.heights), n)
                )
            self.heights = [
                _clamp(float(h), self.min_height, self.max_height)
                for h in self.heights
            ]

    # -- construction helpers ------------------------------------------------

    @classmethod
    def filled(
        cls,
        rows: int,
        cols: int,
        value: float,
        min_height: float = 0.0,
        max_height: float = 1.0,
    ) -> "HeightField":
        """A grid whose every pin sits at ``value`` (clamped)."""
        hf = cls(rows, cols, min_height, max_height)
        hf.fill(value)
        return hf

    @classmethod
    def from_rows(
        cls,
        rows_2d: Sequence[Sequence[float]],
        min_height: float = 0.0,
        max_height: float = 1.0,
    ) -> "HeightField":
        """Build from a 2D nested sequence (list of rows)."""
        rows = len(rows_2d)
        if rows == 0:
            raise ValueError("need at least one row")
        cols = len(rows_2d[0])
        if cols == 0:
            raise ValueError("need at least one column")
        flat: List[float] = []
        for r in rows_2d:
            if len(r) != cols:
                raise ValueError("all rows must have equal length")
            flat.extend(float(v) for v in r)
        return cls(rows, cols, min_height, max_height, flat)

    # -- access --------------------------------------------------------------

    def _index(self, row: int, col: int) -> int:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError("(%d, %d) out of bounds" % (row, col))
        return row * self.cols + col

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def get(self, row: int, col: int) -> float:
        return self.heights[self._index(row, col)]

    def set(self, row: int, col: int, value: float) -> None:
        """Set a pin height, clamped to the stroke range."""
        self.heights[self._index(row, col)] = _clamp(
            float(value), self.min_height, self.max_height
        )

    def add(self, row: int, col: int, delta: float) -> None:
        """Add ``delta`` to a pin height, clamped."""
        i = self._index(row, col)
        self.heights[i] = _clamp(
            self.heights[i] + float(delta), self.min_height, self.max_height
        )

    def fill(self, value: float) -> None:
        """Set every pin to ``value`` (clamped)."""
        v = _clamp(float(value), self.min_height, self.max_height)
        for i in range(len(self.heights)):
            self.heights[i] = v

    def to_rows(self) -> List[List[float]]:
        """Return heights as a fresh list-of-rows 2D grid."""
        return [
            self.heights[r * self.cols : (r + 1) * self.cols]
            for r in range(self.rows)
        ]

    def copy(self) -> "HeightField":
        return HeightField(
            self.rows,
            self.cols,
            self.min_height,
            self.max_height,
            list(self.heights),
        )

    # -- whole-field operations ---------------------------------------------

    def apply(self, fn: Callable[[float], float]) -> None:
        """Map ``fn`` over every pin height, clamping the result."""
        for i in range(len(self.heights)):
            self.heights[i] = _clamp(
                float(fn(self.heights[i])), self.min_height, self.max_height
            )

    def normalized(self) -> List[float]:
        """Heights rescaled to ``[0, 1]`` over the stroke range (flat)."""
        span = self.max_height - self.min_height
        if span <= 0.0:
            return [0.0] * len(self.heights)
        return [(h - self.min_height) / span for h in self.heights]

    # -- scalar summaries ----------------------------------------------------

    def max(self) -> float:
        return max(self.heights)

    def min(self) -> float:
        return min(self.heights)

    def mean(self) -> float:
        return sum(self.heights) / len(self.heights)

    def total_travel(self) -> float:
        """Sum over all pins of height above the floor -- 'material' raised."""
        return sum(h - self.min_height for h in self.heights)

    def raised_cells(self, threshold: Optional[float] = None) -> int:
        """Count pins standing above ``threshold`` (default: above floor)."""
        thr = self.min_height if threshold is None else float(threshold)
        return sum(1 for h in self.heights if h > thr)

    def bounding_box(
        self, threshold: Optional[float] = None
    ) -> Optional[Tuple[int, int, int, int]]:
        """Axis-aligned box ``(min_row, min_col, max_row, max_col)`` of pins
        raised above ``threshold`` (inclusive), or ``None`` if none are.
        """
        thr = self.min_height if threshold is None else float(threshold)
        min_r = min_c = None
        max_r = max_c = None
        for r in range(self.rows):
            base = r * self.cols
            for c in range(self.cols):
                if self.heights[base + c] > thr:
                    if min_r is None or r < min_r:
                        min_r = r
                    if max_r is None or r > max_r:
                        max_r = r
                    if min_c is None or c < min_c:
                        min_c = c
                    if max_c is None or c > max_c:
                        max_c = c
        if min_r is None:
            return None
        return (min_r, min_c, max_r, max_c)


# -- field-comparison metrics -----------------------------------------------


def _check_same_shape(a: HeightField, b: HeightField) -> None:
    if a.rows != b.rows or a.cols != b.cols:
        raise ValueError("height fields must share the same grid resolution")


def mean_absolute_error(a: HeightField, b: HeightField) -> float:
    """Mean over pins of ``|a - b|`` -- average per-pin height discrepancy."""
    _check_same_shape(a, b)
    n = len(a.heights)
    return sum(abs(x - y) for x, y in zip(a.heights, b.heights)) / n


def root_mean_square_error(a: HeightField, b: HeightField) -> float:
    """Root-mean-square per-pin height error between two fields."""
    _check_same_shape(a, b)
    n = len(a.heights)
    return (sum((x - y) ** 2 for x, y in zip(a.heights, b.heights)) / n) ** 0.5


def match_ratio(a: HeightField, b: HeightField, tolerance: float = 0.0) -> float:
    """Fraction of pins whose heights agree within ``tolerance``.

    A deterministic analogue of the paper's binary per-segment success score:
    a pin 'matches' when ``|a - b| <= tolerance``.  Returns a value in
    ``[0, 1]``.
    """
    _check_same_shape(a, b)
    if tolerance < 0.0:
        raise ValueError("tolerance must be >= 0")
    n = len(a.heights)
    hits = sum(1 for x, y in zip(a.heights, b.heights) if abs(x - y) <= tolerance)
    return hits / n
