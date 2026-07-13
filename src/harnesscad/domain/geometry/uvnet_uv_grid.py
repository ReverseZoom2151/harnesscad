"""UV-Net face UV-grids: parameter-domain sampling of B-rep surfaces.

UV-Net (Jayaraman et al., "UV-Net: Learning from Boundary Representations",
CVPR 2021, Autodesk) represents a solid by sampling, on every B-rep *face*, a
regular grid of points in the surface's own parametric domain and storing, at
every grid node, the 3D point, the surface normal and a *trimming mask* telling
whether that parameter lies inside the face's trimmed region.  The paper's
``process/solid_to_graph.py`` calls ``occwl.uvgrid(face, method=...)`` for
``point`` / ``normal`` / ``visibility_status`` and concatenates the three into a
``num_u x num_v x 7`` face feature tensor.  The CNN that consumes the grid is
external; the *sampling* is a purely deterministic geometric routine, and that
is what this module rebuilds in stdlib Python.

Contents
--------
* Analytic surfaces with an explicit parametric domain, each exposing
  ``domain()``, ``point(u, v)`` and ``normal(u, v)``:
  :class:`Plane`, :class:`Cylinder`, :class:`Cone`, :class:`Sphere`,
  :class:`Torus`, and :class:`BSplineSurface` (a thin wrapper delegating to
  :mod:`geometry.nurbgen_surface`).
* :func:`grid_parameters` -- the inclusive uniform parameter samples
  (``num_u x num_v`` nodes spanning the closed domain), matching occwl's
  ``uvgrid`` linspace.
* :func:`visibility_status` -- OCC's ``TopAbs_State`` codes for a parameter
  point against the face's trimming loops in the *parameter* plane:
  ``0 = IN``, ``1 = OUT``, ``2 = ON`` (even-odd rule, so inner loops are holes).
  The UV-Net mask is ``status in (0, 2)`` exactly as in the paper's
  ``np.logical_or(visibility_status == 0, visibility_status == 2)``.
* :func:`uv_grid` -- a grid of one channel (``point`` / ``normal`` /
  ``parameter`` / ``mask`` / ``visibility_status``).
* :func:`face_feature_grid` -- the 7-channel UV-Net face tensor
  ``(x, y, z, nx, ny, nz, mask)``.
* :func:`surface_from_fit` -- compose with :mod:`geometry.complexgen_surface_fit`:
  turn a ``(kind, params)`` fit result into a sampleable surface.

All surfaces are frozen dataclasses with orthonormalised frames, so sampling is
bit-for-bit reproducible.  Nothing here is trained, random, or OCC-dependent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

from harnesscad.domain.geometry import nurbgen_surface

Point = Tuple[float, float, float]
Vec = Tuple[float, float, float]

_EPS = 1e-12

IN = 0
OUT = 1
ON = 2

POINT = "point"
NORMAL = "normal"
PARAMETER = "parameter"
MASK = "mask"
VISIBILITY = "visibility_status"


# --------------------------------------------------------------------------- #
# small vector helpers (stdlib, no numpy)
# --------------------------------------------------------------------------- #
def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a):
    return math.sqrt(_dot(a, a))


def _normalize(a):
    n = _norm(a)
    if n < _EPS:
        raise ValueError("cannot normalise a zero-length vector")
    return (a[0] / n, a[1] / n, a[2] / n)


def _orthonormal_frame(axis: Vec, ref: Vec | None) -> Tuple[Vec, Vec, Vec]:
    """Right-handed frame ``(x, y, axis)`` with ``x`` derived from ``ref``."""
    w = _normalize(axis)
    if ref is None:
        helper = (0.0, 0.0, 1.0) if abs(w[2]) < 0.9 else (1.0, 0.0, 0.0)
        ref = _cross(helper, w)
        if _norm(ref) < _EPS:            # pragma: no cover - guarded by helper
            ref = _cross((0.0, 1.0, 0.0), w)
    x = _sub(ref, _scale(w, _dot(ref, w)))
    x = _normalize(x)
    y = _cross(w, x)
    return x, y, w


# --------------------------------------------------------------------------- #
# analytic surfaces (parametric domain + point + normal)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Plane:
    """``P(u, v) = origin + u * x + v * y`` over ``[u0, u1] x [v0, v1]``."""

    origin: Point
    axis: Vec = (0.0, 0.0, 1.0)
    ref_dir: Vec | None = None
    u_range: Tuple[float, float] = (0.0, 1.0)
    v_range: Tuple[float, float] = (0.0, 1.0)
    reverse: bool = False

    def frame(self):
        return _orthonormal_frame(self.axis, self.ref_dir)

    def domain(self):
        return self.u_range, self.v_range

    def point(self, u: float, v: float) -> Point:
        x, y, _ = self.frame()
        return _add(self.origin, _add(_scale(x, u), _scale(y, v)))

    def normal(self, u: float, v: float) -> Vec:
        _, _, w = self.frame()
        return _scale(w, -1.0) if self.reverse else w


@dataclass(frozen=True)
class Cylinder:
    """``P(u, v) = origin + r*(cos u * x + sin u * y) + v * axis``."""

    origin: Point
    axis: Vec
    radius: float
    ref_dir: Vec | None = None
    u_range: Tuple[float, float] = (0.0, 2.0 * math.pi)
    v_range: Tuple[float, float] = (0.0, 1.0)
    reverse: bool = False

    def frame(self):
        return _orthonormal_frame(self.axis, self.ref_dir)

    def domain(self):
        return self.u_range, self.v_range

    def _radial(self, u: float) -> Vec:
        x, y, _ = self.frame()
        return _add(_scale(x, math.cos(u)), _scale(y, math.sin(u)))

    def point(self, u: float, v: float) -> Point:
        _, _, w = self.frame()
        return _add(self.origin, _add(_scale(self._radial(u), self.radius),
                                      _scale(w, v)))

    def normal(self, u: float, v: float) -> Vec:
        n = self._radial(u)
        return _scale(n, -1.0) if self.reverse else n


@dataclass(frozen=True)
class Cone:
    """OCC-style cone: radius ``r`` at ``v = 0``, half-angle ``alpha``.

    ``P(u, v) = origin + (r + v*sin a) * radial(u) + v*cos a * axis``.
    """

    origin: Point
    axis: Vec
    radius: float
    half_angle: float
    ref_dir: Vec | None = None
    u_range: Tuple[float, float] = (0.0, 2.0 * math.pi)
    v_range: Tuple[float, float] = (0.0, 1.0)
    reverse: bool = False

    def frame(self):
        return _orthonormal_frame(self.axis, self.ref_dir)

    def domain(self):
        return self.u_range, self.v_range

    def _radial(self, u: float) -> Vec:
        x, y, _ = self.frame()
        return _add(_scale(x, math.cos(u)), _scale(y, math.sin(u)))

    def point(self, u: float, v: float) -> Point:
        _, _, w = self.frame()
        r = self.radius + v * math.sin(self.half_angle)
        return _add(self.origin,
                    _add(_scale(self._radial(u), r),
                         _scale(w, v * math.cos(self.half_angle))))

    def normal(self, u: float, v: float) -> Vec:
        _, _, w = self.frame()
        rad = self._radial(u)
        n = _normalize(_sub(_scale(rad, math.cos(self.half_angle)),
                            _scale(w, math.sin(self.half_angle))))
        return _scale(n, -1.0) if self.reverse else n


@dataclass(frozen=True)
class Sphere:
    """``P(u, v) = c + r*cos v * radial(u) + r*sin v * axis``, ``v in [-pi/2, pi/2]``."""

    centre: Point
    radius: float
    axis: Vec = (0.0, 0.0, 1.0)
    ref_dir: Vec | None = None
    u_range: Tuple[float, float] = (0.0, 2.0 * math.pi)
    v_range: Tuple[float, float] = (-math.pi / 2.0, math.pi / 2.0)
    reverse: bool = False

    def frame(self):
        return _orthonormal_frame(self.axis, self.ref_dir)

    def domain(self):
        return self.u_range, self.v_range

    def _dir(self, u: float, v: float) -> Vec:
        x, y, w = self.frame()
        rad = _add(_scale(x, math.cos(u)), _scale(y, math.sin(u)))
        return _add(_scale(rad, math.cos(v)), _scale(w, math.sin(v)))

    def point(self, u: float, v: float) -> Point:
        return _add(self.centre, _scale(self._dir(u, v), self.radius))

    def normal(self, u: float, v: float) -> Vec:
        n = self._dir(u, v)
        return _scale(n, -1.0) if self.reverse else n


@dataclass(frozen=True)
class Torus:
    """``P(u, v) = c + (R + r cos v) * radial(u) + r sin v * axis``."""

    centre: Point
    axis: Vec
    major_radius: float
    minor_radius: float
    ref_dir: Vec | None = None
    u_range: Tuple[float, float] = (0.0, 2.0 * math.pi)
    v_range: Tuple[float, float] = (0.0, 2.0 * math.pi)
    reverse: bool = False

    def frame(self):
        return _orthonormal_frame(self.axis, self.ref_dir)

    def domain(self):
        return self.u_range, self.v_range

    def _radial(self, u: float) -> Vec:
        x, y, _ = self.frame()
        return _add(_scale(x, math.cos(u)), _scale(y, math.sin(u)))

    def point(self, u: float, v: float) -> Point:
        _, _, w = self.frame()
        rad = self._radial(u)
        r = self.major_radius + self.minor_radius * math.cos(v)
        return _add(self.centre,
                    _add(_scale(rad, r),
                         _scale(w, self.minor_radius * math.sin(v))))

    def normal(self, u: float, v: float) -> Vec:
        _, _, w = self.frame()
        n = _add(_scale(self._radial(u), math.cos(v)),
                 _scale(w, math.sin(v)))
        n = _normalize(n)
        return _scale(n, -1.0) if self.reverse else n


@dataclass(frozen=True)
class BSplineSurface:
    """NURBS patch sampled through :mod:`geometry.nurbgen_surface`."""

    poles: Sequence[Sequence[Point]]
    weights: Sequence[Sequence[float]]
    deg_u: int
    deg_v: int
    knots_u: Sequence[float]
    knots_v: Sequence[float]
    reverse: bool = False

    def domain(self):
        u0 = self.knots_u[self.deg_u]
        u1 = self.knots_u[len(self.knots_u) - self.deg_u - 1]
        v0 = self.knots_v[self.deg_v]
        v1 = self.knots_v[len(self.knots_v) - self.deg_v - 1]
        return (u0, u1), (v0, v1)

    def point(self, u: float, v: float) -> Point:
        return nurbgen_surface.surface_point(
            self.poles, self.weights, self.deg_u, self.deg_v,
            self.knots_u, self.knots_v, u, v)

    def normal(self, u: float, v: float) -> Vec:
        n = nurbgen_surface.surface_normal(
            self.poles, self.weights, self.deg_u, self.deg_v,
            self.knots_u, self.knots_v, u, v)
        return _scale(n, -1.0) if self.reverse else tuple(n)


def surface_from_fit(kind: str, params, u_range=None, v_range=None):
    """Build a sampleable surface from a ``complexgen_surface_fit`` result.

    ``kind`` / ``params`` follow :mod:`geometry.complexgen_surface_fit`:
    ``plane -> (normal, offset)``, ``sphere -> (centre, radius)``,
    ``cylinder -> (axis_point, axis_dir, radius)``,
    ``cone -> (apex, axis_dir, half_angle)``,
    ``torus -> (centre, axis_dir, major, minor)``.
    """
    if kind == "plane":
        normal, offset = params
        normal = _normalize(tuple(normal))
        origin = _scale(normal, offset)
        kw = {}
        if u_range is not None:
            kw["u_range"] = tuple(u_range)
        if v_range is not None:
            kw["v_range"] = tuple(v_range)
        return Plane(origin=origin, axis=normal, **kw)
    if kind == "sphere":
        centre, radius = params
        return Sphere(centre=tuple(centre), radius=float(radius))
    if kind == "cylinder":
        axis_point, axis_dir, radius = params
        kw = {"v_range": tuple(v_range)} if v_range is not None else {}
        return Cylinder(origin=tuple(axis_point), axis=tuple(axis_dir),
                        radius=float(radius), **kw)
    if kind == "cone":
        apex, axis_dir, half_angle = params
        kw = {"v_range": tuple(v_range)} if v_range is not None else {}
        return Cone(origin=tuple(apex), axis=tuple(axis_dir), radius=0.0,
                    half_angle=float(half_angle), **kw)
    if kind == "torus":
        centre, axis_dir, major, minor = params
        return Torus(centre=tuple(centre), axis=tuple(axis_dir),
                     major_radius=float(major), minor_radius=float(minor))
    raise ValueError("unknown surface kind: %r" % (kind,))


# --------------------------------------------------------------------------- #
# parameter-domain sampling
# --------------------------------------------------------------------------- #
def linspace(a: float, b: float, n: int) -> list:
    """``n`` inclusive uniform samples of ``[a, b]`` (occwl's uvgrid spacing)."""
    if n < 1:
        raise ValueError("need at least one sample")
    if n == 1:
        return [0.5 * (a + b)]
    step = (b - a) / (n - 1)
    return [a + step * i for i in range(n)]


def grid_parameters(domain, num_u: int, num_v: int) -> list:
    """``num_u x num_v`` grid of ``(u, v)`` parameters over the closed domain."""
    (u0, u1), (v0, v1) = domain
    us = linspace(u0, u1, num_u)
    vs = linspace(v0, v1, num_v)
    return [[(u, v) for v in vs] for u in us]


# --------------------------------------------------------------------------- #
# trimming mask in the parameter plane
# --------------------------------------------------------------------------- #
def _on_segment(p, a, b, tol: float) -> bool:
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    if seg2 < _EPS:
        return math.hypot(px - ax, py - ay) <= tol
    t = ((px - ax) * dx + (py - ay) * dy) / seg2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy) <= tol


def visibility_status(uv, loops, tol: float = 1e-9) -> int:
    """OCC ``TopAbs_State`` of parameter ``uv`` w.r.t. trimming ``loops``.

    ``loops`` is a sequence of closed polygons in the ``(u, v)`` plane (the
    outer wire first, inner wires -- holes -- after; the even-odd rule makes
    nesting order irrelevant).  Returns :data:`IN` (0), :data:`OUT` (1) or
    :data:`ON` (2).  ``loops`` empty/None means an untrimmed face: everything
    is :data:`IN`.
    """
    if not loops:
        return IN
    u, v = uv
    inside = False
    for loop in loops:
        n = len(loop)
        if n < 2:
            continue
        for i in range(n):
            a = loop[i]
            b = loop[(i + 1) % n]
            if _on_segment((u, v), a, b, tol):
                return ON
            ay, by = a[1], b[1]
            if (ay > v) != (by > v):
                x = a[0] + (v - ay) * (b[0] - a[0]) / (by - ay)
                if x > u:
                    inside = not inside
    return IN if inside else OUT


def trimming_mask(uv, loops, tol: float = 1e-9) -> int:
    """UV-Net's binary mask: 1 when the parameter is inside or on the boundary."""
    return 1 if visibility_status(uv, loops, tol) in (IN, ON) else 0


# --------------------------------------------------------------------------- #
# the grids
# --------------------------------------------------------------------------- #
def uv_grid(surface, num_u: int, num_v: int, method: str = POINT,
            trim_loops=None, tol: float = 1e-9) -> list:
    """One channel of the UV-grid: a ``num_u x num_v`` nested list.

    ``method`` is ``point`` (3-tuples), ``normal`` (3-tuples), ``parameter``
    (2-tuples), ``visibility_status`` (int codes) or ``mask`` (0/1), mirroring
    ``occwl.uvgrid(face, method=...)``.
    """
    params = grid_parameters(surface.domain(), num_u, num_v)
    out = []
    for row in params:
        line = []
        for (u, v) in row:
            if method == POINT:
                line.append(tuple(surface.point(u, v)))
            elif method == NORMAL:
                line.append(tuple(surface.normal(u, v)))
            elif method == PARAMETER:
                line.append((u, v))
            elif method == VISIBILITY:
                line.append(visibility_status((u, v), trim_loops, tol))
            elif method == MASK:
                line.append(trimming_mask((u, v), trim_loops, tol))
            else:
                raise ValueError("unknown uv_grid method: %r" % (method,))
        out.append(line)
    return out


def face_feature_grid(surface, num_u: int = 10, num_v: int = 10,
                      trim_loops=None, tol: float = 1e-9) -> list:
    """The 7-channel UV-Net face tensor ``(x, y, z, nx, ny, nz, mask)``.

    Shape: ``num_u x num_v x 7`` as nested tuples/lists, matching the
    ``np.concatenate((points, normals, mask), axis=-1)`` of the paper.
    """
    grid = []
    for row in grid_parameters(surface.domain(), num_u, num_v):
        line = []
        for (u, v) in row:
            p = surface.point(u, v)
            n = surface.normal(u, v)
            m = trimming_mask((u, v), trim_loops, tol)
            line.append((p[0], p[1], p[2], n[0], n[1], n[2], float(m)))
        grid.append(line)
    return grid


def grid_shape(grid) -> Tuple[int, int, int]:
    """``(num_u, num_v, channels)`` of a feature grid."""
    if not grid or not grid[0]:
        return (len(grid), 0, 0)
    return (len(grid), len(grid[0]), len(grid[0][0]))


def masked_points(grid) -> list:
    """The 3D points of a 7-channel face grid whose mask channel is 1."""
    return [(c[0], c[1], c[2]) for row in grid for c in row if c[6] >= 0.5]


def mask_ratio(grid) -> float:
    """Fraction of grid nodes inside the trimmed region."""
    total = sum(len(row) for row in grid)
    if total == 0:
        return 0.0
    hit = sum(1 for row in grid for c in row if c[6] >= 0.5)
    return hit / total
