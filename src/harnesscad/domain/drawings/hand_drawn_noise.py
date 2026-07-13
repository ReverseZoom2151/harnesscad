"""Vitruvion hand-drawn stroke noise: a Matern-GP wobble applied along each primitive.

Vitruvion (Seff et al., ICLR 2022 -- ``img2cad/noise_models.py``, class ``RenderNoise``)
trains its image-conditioned model on *hand-drawn-looking* renders.  The wobble is not
i.i.d. pixel noise: it is a **Gaussian process displacement applied along the arclength of
each primitive**, so the stroke wanders smoothly (as a human hand does) instead of
jittering.  This module reimplements it deterministically, in stdlib.

The construction
----------------
1. A Matern kernel is built once per sketch over ``resolution`` (default 500) equally
   spaced arclength stations in ``[0, 1]``::

       d_ij = |x_i - x_j| / length_scale                     (length_scale = 0.05)
       nu=3: K = amp^2 * (1 + sqrt(3) d) exp(-sqrt(3) d)     (amplitude = 0.002)
       nu=5: K = amp^2 * (1 + sqrt(5) d + 5 d^2 / 3) exp(-sqrt(5) d)

   with a ``1e-6`` nugget on the diagonal, and its lower Cholesky factor ``cK`` is cached.
2. A sketch-wide ``scale = 10 * diagonal of the sketch's extent`` fixes how many stations
   a primitive of a given length consumes: a primitive of length ``L`` uses the first
   ``m = floor(L / scale * resolution)`` stations, and its displacement is
   ``scale * cK[:m, :m] @ z`` with ``z`` standard normal.  Because the leading principal
   block of a Cholesky factor is itself the Cholesky factor of the leading block of the
   kernel, this is an exact GP draw on those stations -- and short primitives use a
   correspondingly short (i.e. smoother-looking) piece of the process.
3. The displacement is applied **perpendicular to the primitive**: normal to the direction
   for a line, radially (``radius + y``) for an arc or a circle.  A circle is drawn as an
   arc of 359 degrees starting at a random angle, so the stroke has a visible gap and
   does not close on itself -- exactly what a hand-drawn circle looks like.  Points are
   simply displaced by an isotropic ``sigma = 1e-2`` normal.

Reproduced quirk: the reference's extent computation updates the running x/y ranges with
``-radius`` and ``+radius`` *and separately* with the centre, rather than with
``centre +/- radius``.  The resulting extent is therefore the union of the radius interval
**about the origin** with the set of centres -- not the sketch's bounding box.  Since it
only feeds ``scale`` (a smoothness knob), it is a cosmetic quirk, but reproducing it keeps
stroke statistics identical to the paper's training data; ``bbox_extent=True`` opts into
the corrected version.

Determinism: every draw comes from a ``random.Random(seed)``.

Pure stdlib.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

from harnesscad.domain.geometry.sketch.normalization import VArc, VCircle, VLine, VPoint

Vec2 = Tuple[float, float]

__all__ = [
    "matern_kernel",
    "cholesky",
    "HandDrawnNoise",
]

DEFAULT_RESOLUTION = 500
DEFAULT_LENGTH_SCALE = 0.05
DEFAULT_AMPLITUDE = 0.002
NUGGET = 1e-6
POINT_NOISE_STD = 1e-2


def _linspace(lo: float, hi: float, count: int) -> List[float]:
    if count < 1:
        raise ValueError("count must be positive")
    if count == 1:
        return [lo]
    step = (hi - lo) / (count - 1)
    return [lo + step * i for i in range(count - 1)] + [hi]


def matern_kernel(
    stations: Sequence[float],
    length_scale: float = DEFAULT_LENGTH_SCALE,
    amplitude: float = DEFAULT_AMPLITUDE,
    nu: int = 3,
) -> List[List[float]]:
    """The Matern covariance matrix (``nu`` in ``{3, 5}``) with a ``1e-6`` nugget."""
    if nu not in (3, 5):
        raise ValueError("nu must be 3 or 5")

    root = math.sqrt(nu)
    n = len(stations)
    k: List[List[float]] = [[0.0] * n for _ in range(n)]
    amp2 = amplitude ** 2

    for i in range(n):
        for j in range(n):
            d = abs(stations[i] - stations[j]) / length_scale
            if nu == 3:
                value = (1.0 + root * d) * math.exp(-root * d)
            else:
                value = (1.0 + root * d + nu * d * d / 3.0) * math.exp(-root * d)
            if i == j:
                value += NUGGET
            k[i][j] = amp2 * value
    return k


def cholesky(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    """Lower-triangular Cholesky factor ``L`` with ``L L^T == matrix``."""
    n = len(matrix)
    lower: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            total = sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                value = matrix[i][i] - total
                if value <= 0.0:
                    raise ValueError("matrix is not positive definite")
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = (matrix[i][j] - total) / lower[j][j]
    return lower


class HandDrawnNoise:
    """A seeded hand-drawn renderer for one sketch."""

    def __init__(
        self,
        entities: Sequence[object],
        seed: int = 0,
        resolution: int = DEFAULT_RESOLUTION,
        length_scale: float = DEFAULT_LENGTH_SCALE,
        amplitude: float = DEFAULT_AMPLITUDE,
        nu: int = 3,
        bbox_extent: bool = False,
    ):
        if resolution < 2:
            raise ValueError("resolution must be at least 2")

        self.entities = list(entities)
        self.resolution = resolution
        self.rng = random.Random(seed)

        self.min_x, self.min_y, self.max_x, self.max_y = self._extent(bbox_extent)
        self.scale = 10.0 * math.hypot(self.max_x - self.min_x, self.max_y - self.min_y)

        self.stations = _linspace(0.0, 1.0, resolution)
        self.chol = cholesky(matern_kernel(self.stations, length_scale, amplitude, nu))

    # -- extent -------------------------------------------------------------
    def _extent(self, bbox_extent: bool) -> Tuple[float, float, float, float]:
        min_x = min_y = math.inf
        max_x = max_y = -math.inf

        def update(x: Optional[float] = None, y: Optional[float] = None) -> None:
            nonlocal min_x, max_x, min_y, max_y
            if x is not None:
                min_x, max_x = min(min_x, x), max(max_x, x)
            if y is not None:
                min_y, max_y = min(min_y, y), max(max_y, y)

        for entity in self.entities:
            if isinstance(entity, (VArc, VCircle)):
                if bbox_extent:
                    update(entity.xCenter - entity.radius, entity.yCenter - entity.radius)
                    update(entity.xCenter + entity.radius, entity.yCenter + entity.radius)
                else:
                    # Reference quirk: radius about the origin, centre separately.
                    update(-entity.radius, -entity.radius)
                    update(entity.radius, entity.radius)
                    update(entity.xCenter, entity.yCenter)
            elif isinstance(entity, VLine):
                (sx, sy), (ex, ey) = entity.start_point, entity.end_point
                update(sx, sy)
                update(ex, ey)
            elif isinstance(entity, VPoint):
                update(entity.x, entity.y)

        if min_x is math.inf:
            return (0.0, 0.0, 0.0, 0.0)
        return (min_x, min_y, max_x, max_y)

    # -- GP displacement ----------------------------------------------------
    def _stations_for(self, length: float, minimum: int = 0) -> int:
        if self.scale == 0.0:
            return max(minimum, 0)
        count = int(math.floor(length / self.scale * self.resolution))
        return max(count, minimum)

    def _displacement(self, count: int) -> List[float]:
        """``scale * cK[:count, :count] @ z`` -- an exact GP draw on the first stations."""
        z = [self.rng.gauss(0.0, 1.0) for _ in range(count)]
        return [
            self.scale * sum(self.chol[i][j] * z[j] for j in range(i + 1))
            for i in range(count)
        ]

    # -- primitives ---------------------------------------------------------
    def line(self, start: Vec2, end: Vec2) -> List[Vec2]:
        """A wobbly polyline from ``start`` to ``end`` (displacement normal to it)."""
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        count = self._stations_for(length)
        if count == 0:
            return [start]

        offsets = self._displacement(count)
        theta = math.atan2(end[1] - start[1], end[0] - start[0])
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        points = []
        for i in range(count):
            x = self.stations[i] * self.scale
            y = offsets[i]
            points.append(
                (start[0] + x * cos_t - y * sin_t, start[1] + y * cos_t + x * sin_t)
            )
        return points

    def arc(
        self, center: Vec2, radius: float, start_deg: float, end_deg: float
    ) -> List[Vec2]:
        """A wobbly arc polyline; the displacement is applied radially."""
        start = math.radians(start_deg)
        end = math.radians(end_deg)
        if end < start:
            end += 2 * math.pi

        length = abs(radius * (end - start))
        count = self._stations_for(length, minimum=1)
        offsets = self._displacement(count)
        thetas = _linspace(start, end, count)

        return [
            (
                center[0] + (radius + offsets[i]) * math.cos(thetas[i]),
                center[1] + (radius + offsets[i]) * math.sin(thetas[i]),
            )
            for i in range(count)
        ]

    def circle(self, center: Vec2, radius: float) -> List[Vec2]:
        """A wobbly circle: an arc of 359 degrees starting at a random angle."""
        gap = self.rng.random() * 360.0
        return self.arc(center, radius, gap, gap + 359.0)

    def point(self, center: Vec2) -> Vec2:
        """A point displaced by an isotropic ``sigma = 1e-2`` normal."""
        return (
            center[0] + POINT_NOISE_STD * self.rng.gauss(0.0, 1.0),
            center[1] + POINT_NOISE_STD * self.rng.gauss(0.0, 1.0),
        )

    # -- sketch -------------------------------------------------------------
    def render(self) -> List[List[Vec2]]:
        """One wobbly polyline per entity, in sketch order."""
        out: List[List[Vec2]] = []
        for entity in self.entities:
            if isinstance(entity, VLine):
                out.append(self.line(entity.start_point, entity.end_point))
            elif isinstance(entity, VArc):
                start_deg = math.degrees(entity.startParam)
                end_deg = math.degrees(entity.endParam)
                if entity.clockwise:
                    start_deg, end_deg = -end_deg, -start_deg
                out.append(
                    self.arc(entity.center_point, entity.radius, start_deg, end_deg)
                )
            elif isinstance(entity, VCircle):
                out.append(self.circle(entity.center_point, entity.radius))
            elif isinstance(entity, VPoint):
                out.append([self.point((entity.x, entity.y))])
            else:
                raise ValueError("unsupported entity type: {!r}".format(type(entity)))
        return out
