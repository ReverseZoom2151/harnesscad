"""Procedural symmetry / repetition operators and parameter reduction.

From Séquin, *Interactive Procedural Computer-Aided Design*, Section 2. The paper
argues that imposing symmetry on a procedural layout is doubly valuable:

* geometrically it "cancels out to first order the effects of processing
  variations", so a designer wants to *replicate* a single motif with bilateral
  or 4-fold symmetry rather than let each leg vary independently; and
* it "reduces by a factor of four the number of parameters that need to be
  adjusted and thus the dimension of the search space" -- an ``order``-fold
  symmetric layout has only ``1/order`` as many independent parameters.

This module implements both deterministic pieces:

* geometric replication operators -- :func:`rotate_point`, :func:`mirror_point`,
  :func:`nfold`, :func:`bilateral`, :func:`dihedral`;
* a :class:`SymmetryReducer` that maps between the small vector of *independent*
  parameters and the full replicated vector, and a
  :func:`symmetry_consistency` check that verifies a full parameter vector (or a
  set of placed motifs) actually obeys the claimed symmetry.

Pure stdlib, deterministic.
"""

from __future__ import annotations

from math import cos, isclose, sin, tau
from typing import List, Sequence, Tuple

Point = Tuple[float, float]


# -- geometric replication --------------------------------------------------


def rotate_point(p: Point, angle: float, center: Point = (0.0, 0.0)) -> Point:
    """Rotate ``p`` by ``angle`` radians about ``center`` (counter-clockwise)."""
    cx, cy = center
    dx, dy = p[0] - cx, p[1] - cy
    ca, sa = cos(angle), sin(angle)
    return (cx + dx * ca - dy * sa, cy + dx * sa + dy * ca)


def mirror_point(p: Point, axis: str = "y", center: Point = (0.0, 0.0)) -> Point:
    """Reflect ``p`` across the ``x`` or ``y`` axis through ``center``.

    ``axis='y'`` mirrors left/right (negates x offset); ``axis='x'`` mirrors
    up/down (negates y offset).
    """
    cx, cy = center
    if axis == "y":
        return (2 * cx - p[0], p[1])
    if axis == "x":
        return (p[0], 2 * cy - p[1])
    raise ValueError("axis must be 'x' or 'y'")


def nfold(
    motif: Sequence[Point], order: int, center: Point = (0.0, 0.0)
) -> Tuple[Tuple[Point, ...], ...]:
    """Replicate ``motif`` with ``order``-fold rotational symmetry.

    Returns a tuple of ``order`` copies, copy ``k`` rotated by ``k*tau/order``.
    Copy 0 is the original motif.
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    copies: List[Tuple[Point, ...]] = []
    for k in range(order):
        angle = tau * k / order
        copies.append(tuple(rotate_point(p, angle, center) for p in motif))
    return tuple(copies)


def bilateral(
    motif: Sequence[Point], axis: str = "y", center: Point = (0.0, 0.0)
) -> Tuple[Tuple[Point, ...], Tuple[Point, ...]]:
    """Original motif plus its mirror across ``axis`` (2-fold reflection)."""
    return tuple(motif), tuple(mirror_point(p, axis, center) for p in motif)


def dihedral(
    motif: Sequence[Point], order: int, center: Point = (0.0, 0.0)
) -> Tuple[Tuple[Point, ...], ...]:
    """Dihedral symmetry: ``order`` rotations, each also mirrored (2*order copies).

    This is the "4-fold symmetry" a MEMS designer imposes -- rotational plus
    reflective -- giving the tightest constraint on the layout.
    """
    out: List[Tuple[Point, ...]] = []
    for rot in nfold(motif, order, center):
        out.append(rot)
        out.append(tuple(mirror_point(p, "y", center) for p in rot))
    return tuple(out)


# -- parameter-space reduction ----------------------------------------------


class SymmetryReducer:
    """Map between independent parameters and a full ``order``-replicated vector.

    An ``order``-fold symmetric layout repeats one base block of ``base_size``
    parameters ``order`` times, so the full vector has ``order * base_size``
    entries but only ``base_size`` are independent. This is exactly the
    "factor of ``order``" search-space reduction the paper describes.
    """

    def __init__(self, order: int, base_size: int) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        if base_size < 1:
            raise ValueError("base_size must be >= 1")
        self.order = order
        self.base_size = base_size

    @property
    def full_size(self) -> int:
        return self.order * self.base_size

    def reduced_count(self, total: int) -> int:
        """Number of independent parameters for a full vector of ``total`` entries."""
        if total % self.order != 0:
            raise ValueError("total must be divisible by order")
        return total // self.order

    def expand(self, base: Sequence[float]) -> Tuple[float, ...]:
        """Replicate the base block ``order`` times into the full vector."""
        if len(base) != self.base_size:
            raise ValueError(f"base must have {self.base_size} entries")
        return tuple(base) * self.order

    def reduce(self, full: Sequence[float], *, rel_tol: float = 1e-9) -> Tuple[float, ...]:
        """Recover the base block from a full symmetric vector.

        Raises ``ValueError`` if ``full`` is not actually ``order`` repetitions
        of a common base block (a symmetry-consistency failure).
        """
        if len(full) != self.full_size:
            raise ValueError(f"full must have {self.full_size} entries")
        if not self.is_symmetric(full, rel_tol=rel_tol):
            raise ValueError("full vector does not obey the claimed symmetry")
        return tuple(full[: self.base_size])

    def is_symmetric(self, full: Sequence[float], *, rel_tol: float = 1e-9) -> bool:
        """True iff ``full`` is ``order`` identical copies of a base block."""
        if len(full) != self.full_size:
            return False
        base = full[: self.base_size]
        for k in range(1, self.order):
            block = full[k * self.base_size : (k + 1) * self.base_size]
            for a, b in zip(base, block):
                if not isclose(a, b, rel_tol=rel_tol, abs_tol=rel_tol):
                    return False
        return True


def symmetry_consistency(
    copies: Sequence[Sequence[Point]], *, abs_tol: float = 1e-9
) -> bool:
    """Check that every replicated motif has the same shape (edge lengths).

    Rotations/reflections preserve pairwise distances, so a set of genuine
    symmetric copies must share an identical sequence of consecutive edge
    lengths. Returns True when they match within ``abs_tol``.
    """
    if not copies:
        return True
    ref = _edge_lengths(copies[0])
    for c in copies[1:]:
        lengths = _edge_lengths(c)
        if len(lengths) != len(ref):
            return False
        for a, b in zip(ref, lengths):
            if abs(a - b) > abs_tol:
                return False
    return True


def _edge_lengths(motif: Sequence[Point]) -> Tuple[float, ...]:
    return tuple(
        ((motif[i + 1][0] - motif[i][0]) ** 2 + (motif[i + 1][1] - motif[i][1]) ** 2)
        ** 0.5
        for i in range(len(motif) - 1)
    )
