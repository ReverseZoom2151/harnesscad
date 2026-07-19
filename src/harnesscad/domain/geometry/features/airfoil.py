"""NACA 4-digit airfoil coordinate generation (deterministic, stdlib-only).

This module implements the closed-form parametrisation of the classic NACA
4-digit series.

The NACA 4-digit designation ``MPTT`` encodes three normalised parameters over
a unit chord ``x in [0, 1]``:

    m = M/100    maximum camber (fraction of chord)
    p = P/10     chordwise position of maximum camber (fraction of chord)
    t = TT/100   maximum thickness (fraction of chord)

e.g. NACA 2412 -> m=0.02, p=0.40, t=0.12.

Thickness distribution (the standard NACA half-thickness polynomial)::

    yt = 5*t*(0.2969*sqrt(x) - 0.1260*x - 0.3516*x^2
                              + 0.2843*x^3 - 0.1015*x^4)

Mean camber line and its slope are piecewise about ``p``. Surface points are
obtained by offsetting the camber line normal to the local camber slope.

Points are generated with cosine spacing (denser near the sharp leading edge),
matching ``x = 0.5*(1 - cos(theta))``. ``airfoil_polygon`` returns a single
closed loop (upper trailing->leading, then lower leading->trailing) with no
duplicated endpoints, ready to hand to a polygon/extrude routine.

All trig is in RADIANS here (Python ``math``); the OpenSCAD source used degrees
but the numeric results are identical.
"""

from __future__ import annotations

import math
from typing import List, Tuple

Point = Tuple[float, float]


def thickness(x: float, t: float) -> float:
    """NACA 4-digit half-thickness ``yt`` at chord fraction ``x`` (0..1).

    ``x`` is clamped at 0 under the square root to stay defined at/just below
    the leading edge (mirrors ``max(0, x)`` in the OpenSCAD source).
    """
    xc = max(0.0, x)
    return 5.0 * t * (
        0.2969 * math.sqrt(xc)
        - 0.1260 * xc
        - 0.3516 * xc ** 2
        + 0.2843 * xc ** 3
        - 0.1015 * xc ** 4
    )


def camber(x: float, m: float, p: float) -> float:
    """Mean camber-line ordinate ``yc`` at chord fraction ``x``."""
    if m == 0 or p == 0:
        return 0.0
    if x < p:
        return (m / p ** 2) * (2 * p * x - x ** 2)
    return (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * x - x ** 2)


def camber_slope(x: float, m: float, p: float) -> float:
    """Slope ``dyc/dx`` of the mean camber line at chord fraction ``x``."""
    if m == 0 or p == 0:
        return 0.0
    if x < p:
        return (2 * m / p ** 2) * (p - x)
    return (2 * m / (1 - p) ** 2) * (p - x)


def surface_point(x: float, m: float, p: float, t: float, upper: bool) -> Point:
    """Return the (x, y) coordinate on the upper or lower surface.

    The half-thickness is applied normal to the camber line, so the returned
    x differs slightly from the input chord fraction (standard NACA construction).
    """
    yt = thickness(x, t)
    yc = camber(x, m, p)
    theta = math.atan(camber_slope(x, m, p))
    if upper:
        return (x - yt * math.sin(theta), yc + yt * math.cos(theta))
    return (x + yt * math.sin(theta), yc - yt * math.cos(theta))


def cosine_spacing(n: int) -> List[float]:
    """``n+1`` chord fractions in ``[0, 1]`` with cosine (leading-edge) clustering."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return [0.5 * (1 - math.cos(math.pi * i / n)) for i in range(n + 1)]


def airfoil_polygon(m: float, p: float, t: float, n: int = 80) -> List[Point]:
    """Closed airfoil outline as one loop of (x, y) points.

    Upper surface runs trailing-edge -> leading-edge (i = n..0); the lower
    surface then runs leading-edge -> trailing-edge (i = 1..n-1), skipping the
    shared leading/trailing endpoints so the loop closes without duplicates.

    Matches the ordering of ``naca_points`` in the CADAM benchmark (upper built
    from i=0..N giving LE..TE, then lower i=N-1..1). Here we keep the same set
    of points; the polygon is a single non-self-intersecting closed ring.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    xs = cosine_spacing(n)
    upper = [surface_point(xs[i], m, p, t, upper=True) for i in range(n + 1)]
    lower = [surface_point(xs[i], m, p, t, upper=False) for i in range(n - 1, 0, -1)]
    return upper + lower


def scale_polygon(points: List[Point], chord: float) -> List[Point]:
    """Scale a unit-chord outline to a physical chord length."""
    return [(px * chord, py * chord) for (px, py) in points]


def max_thickness_fraction(t: float, n: int = 200) -> Tuple[float, float]:
    """Numerically locate ``(x, yt)`` of maximum half-thickness on a unit chord.

    For the NACA 4-digit polynomial the maximum full thickness equals ``t`` and
    occurs near x = 0.30; returns the sampled maximum of the half-thickness.
    """
    xs = cosine_spacing(n)
    best_x, best_y = xs[0], thickness(xs[0], t)
    for x in xs[1:]:
        y = thickness(x, t)
        if y > best_y:
            best_x, best_y = x, y
    return best_x, best_y
