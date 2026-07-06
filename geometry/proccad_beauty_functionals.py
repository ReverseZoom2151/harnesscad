"""Discrete "beauty functionals" for aesthetic shape refinement.

From Séquin, *Interactive Procedural Computer-Aided Design*, Section 4.3
("Beauty Functionals"). To refine a shape for aesthetics the paper optimises a
surface against a chosen energy functional and envisions "a whole arsenal of such
beauty functionals ... similar to the style sheets found in desktop publishing",
so a designer can "apply the desired style to a region with a click". Named
functionals mentioned:

* **minimal-surface / arc-length** energy -- soap-film-like, penalises length;
* **bending energy** -- the integral of curvature squared (fair-surface design);
* **Minimum Variation** (MVS) -- "integrates the square of the *change* in
  curvature", built on the premise that "curvature should not be penalized a
  priori": the most perfect closed shape (a circle/sphere, constant curvature)
  gets an overall penalty of *zero*.

This module implements those three functionals for a discretised planar curve
(polyline), plus a :class:`StyleSheet` mapping style names to functionals and a
:func:`apply_to_region` that evaluates a style on a sub-range of a curve.

Pure stdlib, deterministic.
"""

from __future__ import annotations

from math import atan2, pi
from typing import Callable, Dict, List, Sequence, Tuple

Point = Tuple[float, float]
Functional = Callable[[Sequence[Point], bool], float]


def _seg_lengths(curve: Sequence[Point], closed: bool) -> List[float]:
    pts = list(curve)
    n = len(pts)
    out: List[float] = []
    last = n if closed else n - 1
    for i in range(last):
        a = pts[i]
        b = pts[(i + 1) % n]
        out.append(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5)
    return out


def arc_length(curve: Sequence[Point], closed: bool = False) -> float:
    """Total length -- the discrete minimal-surface / soap-film functional."""
    return sum(_seg_lengths(curve, closed))


def _turning_angles(curve: Sequence[Point], closed: bool) -> List[Tuple[int, float]]:
    """Signed exterior turning angle at each interior (or all, if closed) vertex.

    Returns ``(vertex_index, angle)`` pairs.
    """
    pts = list(curve)
    n = len(pts)
    result: List[Tuple[int, float]] = []
    indices = range(n) if closed else range(1, n - 1)
    for i in indices:
        prev = pts[(i - 1) % n]
        cur = pts[i]
        nxt = pts[(i + 1) % n]
        v1 = (cur[0] - prev[0], cur[1] - prev[1])
        v2 = (nxt[0] - cur[0], nxt[1] - cur[1])
        a1 = atan2(v1[1], v1[0])
        a2 = atan2(v2[1], v2[0])
        d = a2 - a1
        # wrap into (-pi, pi]
        while d <= -pi:
            d += 2 * pi
        while d > pi:
            d -= 2 * pi
        result.append((i, d))
    return result


def discrete_curvature(curve: Sequence[Point], closed: bool = False) -> List[float]:
    """Discrete curvature at each vertex: turning angle divided by the mean of
    the two adjacent segment lengths (units of 1/length)."""
    pts = list(curve)
    n = len(pts)
    lengths = _seg_lengths(curve, closed=True)  # need wrap-around for adjacency
    out: List[float] = []
    for i, angle in _turning_angles(curve, closed):
        len_prev = lengths[(i - 1) % n]
        len_next = lengths[i % n]
        ds = 0.5 * (len_prev + len_next)
        out.append(angle / ds if ds > 0 else 0.0)
    return out


def bending_energy(curve: Sequence[Point], closed: bool = False) -> float:
    """Integral of curvature squared (fair-surface functional).

    Zero for a straight polyline; positive whenever the curve bends.
    """
    kappa = discrete_curvature(curve, closed)
    lengths = _seg_lengths(curve, closed=True)
    n = len(curve)
    total = 0.0
    interior = list(range(n)) if closed else list(range(1, n - 1))
    for k, i in zip(kappa, interior):
        ds = 0.5 * (lengths[(i - 1) % n] + lengths[i % n])
        total += k * k * ds
    return total


def minimum_variation(curve: Sequence[Point], closed: bool = False) -> float:
    """Minimum-Variation functional: integral of (change in curvature) squared.

    Following the paper's premise, *constant* curvature is not penalised, so a
    regular polygon (discrete circle) yields ~0, while any variation in curvature
    along the curve raises the cost.
    """
    kappa = discrete_curvature(curve, closed)
    if len(kappa) < 2:
        return 0.0
    lengths = _seg_lengths(curve, closed=True)
    n = len(curve)
    interior = list(range(n)) if closed else list(range(1, n - 1))
    total = 0.0
    m = len(kappa)
    pairs = range(m) if closed else range(m - 1)
    for j in pairs:
        k0 = kappa[j]
        k1 = kappa[(j + 1) % m]
        i0 = interior[j]
        i1 = interior[(j + 1) % m]
        # arc distance between the two curvature samples ~ segment between them
        seg = lengths[i0 % n] if not closed else lengths[i0 % n]
        ds = seg if seg > 0 else 1.0
        dk = (k1 - k0) / ds
        total += dk * dk * ds
    return total


class StyleSheet:
    """A named catalogue of beauty functionals (the "style sheet")."""

    _BUILTINS: Dict[str, Functional] = {
        "minimal": lambda c, cl: arc_length(c, cl),
        "bending": lambda c, cl: bending_energy(c, cl),
        "min_variation": lambda c, cl: minimum_variation(c, cl),
    }

    def __init__(self) -> None:
        self._styles: Dict[str, Functional] = dict(self._BUILTINS)

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._styles))

    def register(self, name: str, functional: Functional) -> None:
        self._styles[name] = functional

    def evaluate(self, name: str, curve: Sequence[Point], closed: bool = False) -> float:
        if name not in self._styles:
            raise KeyError(f"unknown style '{name}'")
        return self._styles[name](curve, closed)


def apply_to_region(
    style_sheet: StyleSheet,
    name: str,
    curve: Sequence[Point],
    start: int,
    end: int,
) -> float:
    """Evaluate a style on the sub-range ``curve[start:end]`` (a picked region).

    Mirrors the paper's "apply the desired style to a region with a click".
    """
    if not (0 <= start < end <= len(curve)):
        raise ValueError("invalid region range")
    return style_sheet.evaluate(name, curve[start:end], closed=False)
