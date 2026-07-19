"""Equal-area sphere-to-square parametrization (CFD, Appx. ).

Consistent Flow Distillation warps points on the object surface to a 2D
reference space through a mapping designed so that points uniformly scattered
on the sphere remain uniform after being mapped to the square. The
motivation is noise-map fairness, but the mapping itself is a purely
deterministic, verifiable geometric primitive: an *equal-area* projection
between a spherical lune and a planar triangle.

For a mechanical-CAD harness this is directly useful for distributing inspection
/ render camera viewpoints uniformly over a sphere around a part, and for
building area-fair spherical parametrizations without a learned model.

Mapping):

    r  = sqrt(1 - cos(theta))
    xr = r
    yr = r * (2 * phi / (pi/2) - 1)

Key property (proven in this approach and checked numerically in the tests):

    |d(xr, yr) / d(theta, phi)| = (2 / pi) * sin(theta)

so the map's Jacobian cancels the sphere's sin(theta) area element, making the
push-forward of a uniform spherical measure uniform on the plane.

Everything here is stdlib-only and deterministic (a seeded PRNG is used only for
the optional uniform-viewpoint generator).
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

_HALF_PI = math.pi / 2.0
_SQRT2 = math.sqrt(2.0)


def sphere_to_square(theta: float, phi: float) -> Tuple[float, float]:
    """Map a spherical direction to the planar reference space.

    theta is the polar angle in [0, pi]; phi is the azimuth in [0, pi/2).
    Returns (xr, yr) with 0 <= xr <= sqrt(2) and -xr <= yr <= xr.
    """
    if theta < 0.0 or theta > math.pi:
        raise ValueError("theta must lie in [0, pi]")
    if phi < 0.0 or phi >= _HALF_PI:
        raise ValueError("phi must lie in [0, pi/2)")
    r = math.sqrt(1.0 - math.cos(theta))
    xr = r
    yr = r * (2.0 * phi / _HALF_PI - 1.0)
    return xr, yr


def square_to_sphere(xr: float, yr: float) -> Tuple[float, float]:
    """Inverse of :func:`sphere_to_square`.

    Returns (theta, phi). Requires 0 <= xr <= sqrt(2) and |yr| <= xr.
    """
    if xr < 0.0 or xr > _SQRT2 + 1e-12:
        raise ValueError("xr must lie in [0, sqrt(2)]")
    if abs(yr) > xr + 1e-12:
        raise ValueError("must satisfy |yr| <= xr")
    cos_theta = 1.0 - xr * xr
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = math.acos(cos_theta)
    if xr == 0.0:
        return theta, 0.0
    frac = (yr / xr + 1.0) / 2.0
    phi = frac * _HALF_PI
    # Guard tiny numerical overshoot so phi stays in [0, pi/2).
    phi = min(max(phi, 0.0), _HALF_PI - 1e-15)
    return theta, phi


def area_jacobian(theta: float) -> float:
    """Analytic area element |d(xr,yr)/d(theta,phi)| = (2/pi) sin(theta)."""
    return (2.0 / math.pi) * math.sin(theta)


def numerical_jacobian(theta: float, phi: float, eps: float = 1e-6) -> float:
    """Finite-difference determinant of the map's Jacobian at (theta, phi)."""
    x0, y0 = sphere_to_square(theta, phi)
    xt, yt = sphere_to_square(theta + eps, phi)
    xp, yp = sphere_to_square(theta, phi + eps)
    dx_dtheta = (xt - x0) / eps
    dy_dtheta = (yt - y0) / eps
    dx_dphi = (xp - x0) / eps
    dy_dphi = (yp - y0) / eps
    return abs(dx_dtheta * dy_dphi - dx_dphi * dy_dtheta)


def _direction(theta: float, phi: float) -> Tuple[float, float, float]:
    st = math.sin(theta)
    return (st * math.cos(phi), st * math.sin(phi), math.cos(theta))


def uniform_octant_directions(
    n: int, seed: int = 0
) -> List[Tuple[float, float, float]]:
    """Deterministically sample ``n`` unit directions in the phi-lune.

    Sampling is performed *uniformly in the planar triangle* covered by the map
    and inverted back onto the sphere. Because the map is equal-area, the
    resulting directions are uniform over the spherical lune
    (cos(theta) ~ U[-1, 1], phi ~ U[0, pi/2)).
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = random.Random(seed)
    out: List[Tuple[float, float, float]] = []
    while len(out) < n:
        # Rejection-sample a point uniformly in {0<=xr<=sqrt2, |yr|<=xr}.
        xr = _SQRT2 * math.sqrt(rng.random())  # density proportional to xr
        yr = rng.uniform(-xr, xr)
        theta, phi = square_to_sphere(xr, yr)
        out.append(_direction(theta, phi))
    return out


def mean_vector(vectors: Sequence[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """Component-wise mean of a set of vectors (small helper for tests)."""
    if not vectors:
        raise ValueError("need at least one vector")
    sx = sum(v[0] for v in vectors)
    sy = sum(v[1] for v in vectors)
    sz = sum(v[2] for v in vectors)
    m = float(len(vectors))
    return sx / m, sy / m, sz / m
