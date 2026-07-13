"""UV-Net edge U-grids: parameter-domain sampling of B-rep curves.

The second half of UV-Net's input encoding (Jayaraman et al., CVPR 2021) is the
*edge* U-grid: on every B-rep edge, ``num_u`` samples of the curve's parameter
domain, each carrying the 3D point and the unit tangent, concatenated into a
``num_u x 6`` edge feature tensor (``process/solid_to_graph.py``:
``ugrid(edge, method="point")`` + ``ugrid(edge, method="tangent")``).  Edges
without a curve -- e.g. the seam/apex degeneracies of a cone -- are *skipped*;
this module reproduces that filter deterministically with
:func:`is_degenerate`.

Curves (each with ``domain()``, ``point(u)``, ``tangent(u)``):

* :class:`Line`      -- ``P(u) = origin + u * direction``.
* :class:`Circle`    -- ``P(u) = c + r*(cos u * x + sin u * y)``.
* :class:`Ellipse`   -- ``P(u) = c + a cos u * x + b sin u * y``.
* :class:`Polyline`  -- chord-length-parameterised sample of a point list.
* :class:`BSplineCurve` -- delegates to :mod:`geometry.nurbgen_curve`.

Plus :func:`u_grid` (one channel), :func:`edge_feature_grid` (the 6-channel
tensor), :func:`grid_length` (polyline length of the sampled grid, a cheap
deterministic arc-length estimate) and :func:`reverse_grid` (edge orientation
flip: reverse the sample order and negate the tangents, which is what a coedge
with ``orientation == REVERSED`` sees).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

from harnesscad.domain.geometry import nurbgen_curve
from harnesscad.domain.geometry.uvnet_uv_grid import (_add, _cross, _dot, _norm, _normalize,
                                    _orthonormal_frame, _scale, _sub, linspace)

Point = Tuple[float, float, float]
Vec = Tuple[float, float, float]

_EPS = 1e-12

POINT = "point"
TANGENT = "tangent"
PARAMETER = "parameter"


# --------------------------------------------------------------------------- #
# curves
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Line:
    origin: Point
    direction: Vec
    u_range: Tuple[float, float] = (0.0, 1.0)

    def domain(self):
        return self.u_range

    def point(self, u: float) -> Point:
        return _add(self.origin, _scale(self.direction, u))

    def tangent(self, u: float) -> Vec:
        return _normalize(self.direction)


@dataclass(frozen=True)
class Circle:
    centre: Point
    axis: Vec
    radius: float
    ref_dir: Vec | None = None
    u_range: Tuple[float, float] = (0.0, 2.0 * math.pi)

    def frame(self):
        return _orthonormal_frame(self.axis, self.ref_dir)

    def domain(self):
        return self.u_range

    def point(self, u: float) -> Point:
        x, y, _ = self.frame()
        return _add(self.centre,
                    _scale(_add(_scale(x, math.cos(u)), _scale(y, math.sin(u))),
                           self.radius))

    def tangent(self, u: float) -> Vec:
        x, y, _ = self.frame()
        return _normalize(_add(_scale(x, -math.sin(u)), _scale(y, math.cos(u))))


@dataclass(frozen=True)
class Ellipse:
    centre: Point
    axis: Vec
    major_radius: float
    minor_radius: float
    ref_dir: Vec | None = None
    u_range: Tuple[float, float] = (0.0, 2.0 * math.pi)

    def frame(self):
        return _orthonormal_frame(self.axis, self.ref_dir)

    def domain(self):
        return self.u_range

    def point(self, u: float) -> Point:
        x, y, _ = self.frame()
        return _add(self.centre,
                    _add(_scale(x, self.major_radius * math.cos(u)),
                         _scale(y, self.minor_radius * math.sin(u))))

    def tangent(self, u: float) -> Vec:
        x, y, _ = self.frame()
        return _normalize(_add(_scale(x, -self.major_radius * math.sin(u)),
                               _scale(y, self.minor_radius * math.cos(u))))


@dataclass(frozen=True)
class Polyline:
    """Chord-length parameterised polyline; ``u`` runs over ``[0, total_length]``."""

    points: Sequence[Point]

    def _cumulative(self):
        acc = [0.0]
        for a, b in zip(self.points, self.points[1:]):
            acc.append(acc[-1] + _norm(_sub(b, a)))
        return acc

    def domain(self):
        acc = self._cumulative()
        return (0.0, acc[-1])

    def _locate(self, u: float):
        acc = self._cumulative()
        total = acc[-1]
        u = max(0.0, min(total, u))
        for i in range(len(acc) - 1):
            if u <= acc[i + 1] or i == len(acc) - 2:
                seg = acc[i + 1] - acc[i]
                t = 0.0 if seg < _EPS else (u - acc[i]) / seg
                return i, t
        raise ValueError("empty polyline")  # pragma: no cover

    def point(self, u: float) -> Point:
        i, t = self._locate(u)
        a, b = self.points[i], self.points[i + 1]
        return _add(a, _scale(_sub(b, a), t))

    def tangent(self, u: float) -> Vec:
        i, _ = self._locate(u)
        return _normalize(_sub(self.points[i + 1], self.points[i]))


@dataclass(frozen=True)
class BSplineCurve:
    control_points: Sequence[Point]
    weights: Sequence[float]
    degree: int
    knots: Sequence[float]

    def domain(self):
        return (self.knots[self.degree],
                self.knots[len(self.knots) - self.degree - 1])

    def point(self, u: float) -> Point:
        return nurbgen_curve.curve_point(self.control_points, self.weights,
                                         self.degree, self.knots, u)

    def tangent(self, u: float) -> Vec:
        return nurbgen_curve.curve_tangent(self.control_points, self.weights,
                                           self.degree, self.knots, u)


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def curve_parameters(domain, num_u: int) -> list:
    u0, u1 = domain
    return linspace(u0, u1, num_u)


def is_degenerate(curve, num_u: int = 5, tol: float = 1e-9) -> bool:
    """True when the edge carries no usable curve (UV-Net's ``has_curve`` filter).

    Degenerate when the parameter range collapses, when every sample coincides
    (zero-length edge, e.g. the apex of a cone) or when a tangent cannot be
    evaluated.
    """
    try:
        u0, u1 = curve.domain()
    except Exception:
        return True
    if not (u1 - u0) > tol:
        return True
    try:
        pts = [curve.point(u) for u in curve_parameters((u0, u1), num_u)]
        for u in curve_parameters((u0, u1), num_u):
            t = curve.tangent(u)
            if _norm(t) < 0.5:           # tangents are unit by contract
                return True
    except Exception:
        return True
    p0 = pts[0]
    return all(_norm(_sub(p, p0)) <= tol for p in pts)


def u_grid(curve, num_u: int = 10, method: str = POINT) -> list:
    """One channel of the edge U-grid (``point`` / ``tangent`` / ``parameter``)."""
    out = []
    for u in curve_parameters(curve.domain(), num_u):
        if method == POINT:
            out.append(tuple(curve.point(u)))
        elif method == TANGENT:
            out.append(tuple(curve.tangent(u)))
        elif method == PARAMETER:
            out.append(u)
        else:
            raise ValueError("unknown u_grid method: %r" % (method,))
    return out


def edge_feature_grid(curve, num_u: int = 10) -> list:
    """The 6-channel UV-Net edge tensor ``(x, y, z, tx, ty, tz)``, ``num_u x 6``."""
    grid = []
    for u in curve_parameters(curve.domain(), num_u):
        p = curve.point(u)
        t = curve.tangent(u)
        grid.append((p[0], p[1], p[2], t[0], t[1], t[2]))
    return grid


def reverse_grid(grid) -> list:
    """Edge grid as seen from a REVERSED coedge: samples flipped, tangents negated."""
    return [(c[0], c[1], c[2], -c[3], -c[4], -c[5]) for c in reversed(grid)]


def grid_points(grid) -> list:
    return [(c[0], c[1], c[2]) for c in grid]


def grid_length(grid) -> float:
    """Polyline length of a sampled edge grid (arc-length estimate)."""
    pts = grid_points(grid)
    return sum(_norm(_sub(b, a)) for a, b in zip(pts, pts[1:]))


def tangent_turning(grid) -> float:
    """Total turning angle (radians) between consecutive unit tangents.

    ~0 for a straight edge; ``|u1 - u0|`` for a circular arc sampled finely --
    a cheap deterministic curvature signal for the edge feature.
    """
    total = 0.0
    for a, b in zip(grid, grid[1:]):
        ta = (a[3], a[4], a[5])
        tb = (b[3], b[4], b[5])
        c = max(-1.0, min(1.0, _dot(ta, tb)))
        total += math.acos(c)
    return total
