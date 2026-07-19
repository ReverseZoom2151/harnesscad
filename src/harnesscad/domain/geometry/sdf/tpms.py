"""Triply-periodic minimal surfaces (TPMS) as approximate SDFs.

The ``gyroid`` is a trigonometric implicit surface:
``cos x sin y + cos y sin z + cos z sin x``.  Its zero level set is the gyroid,
an infinite labyrinthine minimal surface popular in 3D-printed lattices.  This
module adds the other classic TPMS from the same family (Schwarz-P, Schwarz-D /
diamond, and Neovius) using their standard trigonometric implicit forms.

**These are implicit fields, not exact distance fields.**  The raw implicit
value ``F(p)`` is not Euclidean distance: ``|grad F|`` is not 1.  A common
recommendation is ``lipschitz 1.33`` (dividing the gyroid field by ``4/3``) as a light
practical correction for meshing -- but the *true* supremum of ``|grad F|`` for
the gyroid, Schwarz-P and Schwarz-D families is ``sqrt(3) ~ 1.732`` (and ``7``
for Neovius), so ``4/3`` does not fully bound the gradient.  Each function here
takes a ``lipschitz`` flag: when true the raw implicit is divided by the true
gradient-magnitude bound (verified numerically in the tests) to yield a
*conservative* 1-Lipschitz approximate SDF safe for sphere tracing.

A ``period`` parameter scales the spatial frequency: with ``w = 2*pi/period``
the trig arguments become ``w*x`` etc., and the field is divided by ``w`` so the
gradient scale is period-independent.

All fields are periodic and every function's zero set passes through the origin
(``F(0) = 0``) for the surfaces whose implicit form vanishes there.
stdlib-only, deterministic.
"""

from __future__ import annotations

from math import cos, pi, sin, sqrt
from typing import Sequence

# Gradient-magnitude bounds used for Lipschitz normalisation.  These are the
# true suprema of |grad F| over a period (confirmed numerically in the tests):
# sqrt(3) for the gyroid / Schwarz-P / Schwarz-D families, 7 for Neovius.
# (The commonly cited 4/3 for the gyroid only bounds the field loosely -- see
# the module docstring.)
GYROID_LIPSCHITZ = sqrt(3.0)
SCHWARZ_P_LIPSCHITZ = sqrt(3.0)
SCHWARZ_D_LIPSCHITZ = sqrt(3.0)
NEOVIUS_LIPSCHITZ = 7.0


def _freq(period: float):
    if period <= 0.0:
        raise ValueError("period must be positive")
    return 2.0 * pi / period


def gyroid(p: Sequence[float], period: float = 2.0 * pi, lipschitz: bool = False) -> float:
    """Gyroid implicit field ``cos x sin y + cos y sin z + cos z sin x``.

    With ``lipschitz=True`` the value is divided by ``sqrt(3)`` (the true
    gradient bound) and by the frequency, giving a conservative approximate SDF.
    """
    w = _freq(period)
    x, y, z = w * p[0], w * p[1], w * p[2]
    f = cos(x) * sin(y) + cos(y) * sin(z) + cos(z) * sin(x)
    if lipschitz:
        return f / (GYROID_LIPSCHITZ * w)
    return f


def schwarz_p(p: Sequence[float], period: float = 2.0 * pi, lipschitz: bool = False) -> float:
    """Schwarz Primitive (P) surface: ``cos x + cos y + cos z``."""
    w = _freq(period)
    x, y, z = w * p[0], w * p[1], w * p[2]
    f = cos(x) + cos(y) + cos(z)
    if lipschitz:
        return f / (SCHWARZ_P_LIPSCHITZ * w)
    return f


def schwarz_d(p: Sequence[float], period: float = 2.0 * pi, lipschitz: bool = False) -> float:
    """Schwarz Diamond (D) surface.

    ``sin x sin y sin z + sin x cos y cos z + cos x sin y cos z
       + cos x cos y sin z``.
    """
    w = _freq(period)
    x, y, z = w * p[0], w * p[1], w * p[2]
    f = (sin(x) * sin(y) * sin(z)
         + sin(x) * cos(y) * cos(z)
         + cos(x) * sin(y) * cos(z)
         + cos(x) * cos(y) * sin(z))
    if lipschitz:
        return f / (SCHWARZ_D_LIPSCHITZ * w)
    return f


def neovius(p: Sequence[float], period: float = 2.0 * pi, lipschitz: bool = False) -> float:
    """Neovius surface: ``3(cos x + cos y + cos z) + 4 cos x cos y cos z``."""
    w = _freq(period)
    x, y, z = w * p[0], w * p[1], w * p[2]
    f = 3.0 * (cos(x) + cos(y) + cos(z)) + 4.0 * cos(x) * cos(y) * cos(z)
    if lipschitz:
        return f / (NEOVIUS_LIPSCHITZ * w)
    return f
