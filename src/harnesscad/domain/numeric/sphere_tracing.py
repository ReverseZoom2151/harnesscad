"""Sphere tracing (ray marching) of a signed distance field.

An implicit-geometry renderer renders every shape on the GPU by *sphere tracing*: to find where a ray
``o + t*dir`` (``dir`` a unit vector) first meets the surface ``f = 0``, march
forward in steps equal to the current field value.  Because a valid SDF is
1-Lipschitz (``|grad f| <= 1``), ``|f(p)|`` is a guaranteed lower bound on the
Euclidean distance to the surface, so a step of ``f(p)`` can never overshoot the
boundary -- the ray converges monotonically onto the first hit (Hart 1996,
"Sphere Tracing").

This module implements that renderer core in stdlib Python:

* ``sphere_trace`` -- march a ray to the first surface crossing; returns a hit
  distance ``t`` (within ``epsilon``) or ``None`` on a miss (exceeded
  ``max_dist`` or ``max_steps``).
* ``estimate_normal`` -- surface normal by central differences of the field
  (the standard 6-tap tetrahedron/central estimator), normalised.
* ``ray_direction`` -- normalise a direction vector.

For a *correct* SDF the analytic first-hit ``t`` for e.g. a sphere is recovered
to within ``epsilon``.  If the field underestimates distance by a factor > 1
(a bad Lipschitz constant), tracing still converges but takes more steps; if it
*overestimates* (Lipschitz > 1) the trace can tunnel through the surface -- hence
the ``lipschitz`` scale argument, mirroring the ``lipschitz k`` fix.

stdlib-only, deterministic, no randomness, no wall clock.
"""

from __future__ import annotations

from math import sqrt
from typing import Callable, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Field = Callable[[Vec3], float]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add_scaled(o, d, t):
    return (o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t)


def _norm(v):
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def ray_direction(v: Sequence[float]) -> Vec3:
    """Return the unit vector along ``v``."""
    n = _norm(v)
    if n == 0.0:
        raise ValueError("zero-length ray direction")
    return (v[0] / n, v[1] / n, v[2] / n)


def sphere_trace(
    field: Field,
    origin: Sequence[float],
    direction: Sequence[float],
    epsilon: float = 1e-6,
    max_dist: float = 1e4,
    max_steps: int = 512,
    lipschitz: float = 1.0,
) -> Optional[float]:
    """March a ray ``origin + t*direction`` onto the first surface hit.

    ``direction`` must be a unit vector.  Steps forward by ``field(p)/lipschitz``
    (a safe lower bound on distance-to-surface for a 1-Lipschitz field).  Returns
    the hit parameter ``t`` when ``|field| <= epsilon``, or ``None`` if the ray
    travels past ``max_dist`` or exhausts ``max_steps`` (a miss).

    ``lipschitz`` (>= 1) rescales the step to compensate a field whose gradient
    magnitude exceeds 1 (the ``lipschitz k``); use 1.0 for an exact field.
    """
    if lipschitz <= 0.0:
        raise ValueError("lipschitz scale must be positive")
    o = (float(origin[0]), float(origin[1]), float(origin[2]))
    d = (float(direction[0]), float(direction[1]), float(direction[2]))
    t = 0.0
    for _ in range(max_steps):
        p = _add_scaled(o, d, t)
        dist = field(p)
        if abs(dist) <= epsilon:
            return t
        t += dist / lipschitz
        if t > max_dist or t < 0.0:
            return None
    return None


def estimate_normal(field: Field, p: Sequence[float], h: float = 1e-5) -> Vec3:
    """Unit surface normal at ``p`` via central differences of ``field``.

    ``n_i = (f(p + h e_i) - f(p - h e_i)) / 2h``, then normalised.  Points in
    the direction of increasing distance (outward from a solid).
    """
    p = (float(p[0]), float(p[1]), float(p[2]))
    nx = field((p[0] + h, p[1], p[2])) - field((p[0] - h, p[1], p[2]))
    ny = field((p[0], p[1] + h, p[2])) - field((p[0], p[1] - h, p[2]))
    nz = field((p[0], p[1], p[2] + h)) - field((p[0], p[1], p[2] - h))
    n = _norm((nx, ny, nz))
    if n == 0.0:
        raise ValueError("degenerate gradient; cannot estimate normal")
    return (nx / n, ny / n, nz / n)
