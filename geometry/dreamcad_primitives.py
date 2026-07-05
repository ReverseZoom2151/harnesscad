"""Analytic parametric CAD surfaces: plane, cylinder, cone, sphere, torus.

DreamCAD's filtering pipeline enumerates the standard CAD surface families
(planes, cylinders, spheres, B-splines, tori, cones, revolutions).  This
module provides their deterministic *forward* parameterisations over the unit
(u, v) square [0, 1]^2, together with analytic unit normals and a uniform
sampler.  These are the closed-form primitives a patch-based representation
approximates; they are useful as ground truth for fitting/consistency tests
and as exact references for the rational-Bezier tessellator.

Every surface exposes ``point(u, v)`` and ``normal(u, v)`` and can be handed
to :func:`sample_surface` for a deterministic point cloud.  All maths is
stdlib-only; no randomness, no wall clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt

TAU = 2.0 * pi


def _unit(vec):
    length = sqrt(sum(c * c for c in vec))
    if length <= 0.0:
        raise ValueError("cannot normalise a zero-length vector")
    return tuple(c / length for c in vec)


def _check_uv(u, v):
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        raise ValueError("(u, v) must lie in the unit square")


@dataclass(frozen=True)
class Plane:
    """Finite planar patch: origin + u*e_u + v*e_v over [0, 1]^2."""

    origin: tuple = (0.0, 0.0, 0.0)
    edge_u: tuple = (1.0, 0.0, 0.0)
    edge_v: tuple = (0.0, 1.0, 0.0)

    def point(self, u, v):
        _check_uv(u, v)
        return tuple(self.origin[d] + u * self.edge_u[d] + v * self.edge_v[d]
                     for d in range(3))

    def normal(self, u, v):
        _check_uv(u, v)
        eu, ev = self.edge_u, self.edge_v
        cross = (eu[1] * ev[2] - eu[2] * ev[1],
                 eu[2] * ev[0] - eu[0] * ev[2],
                 eu[0] * ev[1] - eu[1] * ev[0])
        return _unit(cross)


@dataclass(frozen=True)
class Cylinder:
    """Right circular cylinder about +z; u -> angle, v -> height."""

    radius: float = 1.0
    height: float = 1.0
    center: tuple = (0.0, 0.0, 0.0)

    def point(self, u, v):
        _check_uv(u, v)
        a = TAU * u
        return (self.center[0] + self.radius * cos(a),
                self.center[1] + self.radius * sin(a),
                self.center[2] + self.height * v)

    def normal(self, u, v):
        _check_uv(u, v)
        a = TAU * u
        return (cos(a), sin(a), 0.0)


@dataclass(frozen=True)
class Cone:
    """Truncated cone about +z; radius varies linearly from base to top."""

    base_radius: float = 1.0
    top_radius: float = 0.0
    height: float = 1.0
    center: tuple = (0.0, 0.0, 0.0)

    def _radius(self, v):
        return self.base_radius + (self.top_radius - self.base_radius) * v

    def point(self, u, v):
        _check_uv(u, v)
        a = TAU * u
        r = self._radius(v)
        return (self.center[0] + r * cos(a),
                self.center[1] + r * sin(a),
                self.center[2] + self.height * v)

    def normal(self, u, v):
        _check_uv(u, v)
        a = TAU * u
        dr = self.top_radius - self.base_radius
        # S_u = r*(-sin, cos, 0)*TAU ; S_v = (dr*cos, dr*sin, height)
        s_u = (-sin(a), cos(a), 0.0)
        s_v = (dr * cos(a), dr * sin(a), self.height)
        cross = (s_u[1] * s_v[2] - s_u[2] * s_v[1],
                 s_u[2] * s_v[0] - s_u[0] * s_v[2],
                 s_u[0] * s_v[1] - s_u[1] * s_v[0])
        return _unit(cross)


@dataclass(frozen=True)
class Sphere:
    """Sphere; u -> azimuth in [0, 2pi], v -> polar angle in [0, pi]."""

    radius: float = 1.0
    center: tuple = (0.0, 0.0, 0.0)

    def point(self, u, v):
        _check_uv(u, v)
        azimuth = TAU * u
        polar = pi * v
        return (self.center[0] + self.radius * sin(polar) * cos(azimuth),
                self.center[1] + self.radius * sin(polar) * sin(azimuth),
                self.center[2] + self.radius * cos(polar))

    def normal(self, u, v):
        p = self.point(u, v)
        return _unit(tuple(p[d] - self.center[d] for d in range(3)))


@dataclass(frozen=True)
class Torus:
    """Torus with major radius R and minor radius r; u, v -> two angles."""

    major_radius: float = 2.0
    minor_radius: float = 0.5
    center: tuple = (0.0, 0.0, 0.0)

    def point(self, u, v):
        _check_uv(u, v)
        a = TAU * u
        b = TAU * v
        rr = self.major_radius + self.minor_radius * cos(b)
        return (self.center[0] + rr * cos(a),
                self.center[1] + rr * sin(a),
                self.center[2] + self.minor_radius * sin(b))

    def normal(self, u, v):
        _check_uv(u, v)
        a = TAU * u
        b = TAU * v
        # Outward normal points radially from the tube centre circle.
        return (cos(b) * cos(a), cos(b) * sin(a), sin(b))


def sample_surface(surface, resolution=8):
    """Uniformly sample ``surface.point`` on an r x r (u, v) grid."""
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    coords = [i / (resolution - 1) for i in range(resolution)]
    return [surface.point(u, v) for u in coords for v in coords]
