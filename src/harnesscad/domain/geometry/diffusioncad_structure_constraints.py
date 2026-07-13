"""Structural-constraint equations from Diffusion-CAD Table II.

Diffusion-CAD achieves *structure control* by applying deterministic constraint
equations to the predicted CAD sketch sequence after each denoising step
(Section III-C.4, Fig. 4, Table II). Because the CAD sequence representation is
discretised to integers, the paper deliberately uses *simplified* axis-aligned
equations so the enforced coordinates stay integral: a general "make these two
lines perpendicular" equation would produce non-integer results and yield an
invalid (non-extractable) CAD model.

Four points ``A(x1,y1), B(x2,y2), C(x3,y3), D(x4,y4)`` can be paired into line
segments. Table II gives, verbatim:

    Point-point coincidence  A, B                 x1 = x2, y1 = y2
    Line-line parallel       AB || CD             x1 = x2, x3 = x4
    Symmetry                 AB, CD               y1 = y2, x4 = (x1+x2)/2,
                                                  x3 = x4, (x1+x2) mod 2 = 0
    Line-line perpendicular  AB _|_ CD            x1 = x2, y3 = y4

This module implements each equation as a deterministic *projection/repair* that
takes predicted integer coordinates and returns coordinates that satisfy the
constraint, plus a validity flag. The symmetry constraint additionally reports
whether the integer-parity precondition ``(x1 + x2) mod 2 == 0`` holds — the
paper's key observation that non-integer midpoints break the discretised
representation.

Pure stdlib, no floats leak into the enforced coordinates (they stay ``int``),
fully deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

Point = Tuple[int, int]


@dataclass(frozen=True)
class ConstraintResult:
    """Outcome of enforcing one Table-II constraint.

    ``points`` is the repaired ``(A, B, C, D)`` tuple (each an ``(x, y)`` int
    pair; unused points are echoed unchanged). ``satisfied`` is whether the
    input already satisfied the equation. ``valid`` is whether the *repaired*
    result is representable as integers without loss (always True except when a
    symmetry midpoint would be fractional). ``note`` explains any invalidity.
    """

    points: Tuple[Point, Point, Point, Point]
    satisfied: bool
    valid: bool
    note: str = ""


def _as_int_point(p: Point) -> Point:
    x, y = p
    return (int(x), int(y))


def point_point_coincidence(a: Point, b: Point) -> ConstraintResult:
    """``x1 = x2, y1 = y2`` — collapse B onto A.

    The repair snaps B to A (A is treated as the anchor). Both coordinates are
    already integral so the result is always valid.
    """
    a = _as_int_point(a)
    b = _as_int_point(b)
    satisfied = a == b
    repaired = a
    return ConstraintResult((a, repaired, a, repaired), satisfied, True)


def line_line_parallel(a: Point, b: Point, c: Point, d: Point) -> ConstraintResult:
    """``x1 = x2, x3 = x4`` — make AB and CD both vertical (hence parallel).

    The simplified Table-II equation enforces parallelism by making both
    segments axis-aligned (vertical). B snaps to A's x; D snaps to C's x.
    """
    a = _as_int_point(a)
    b = _as_int_point(b)
    c = _as_int_point(c)
    d = _as_int_point(d)
    satisfied = a[0] == b[0] and c[0] == d[0]
    nb = (a[0], b[1])
    nd = (c[0], d[1])
    return ConstraintResult((a, nb, c, nd), satisfied, True)


def line_line_perpendicular(a: Point, b: Point, c: Point, d: Point) -> ConstraintResult:
    """``x1 = x2, y3 = y4`` — AB vertical, CD horizontal (hence perpendicular).

    B snaps to A's x (AB vertical); D snaps to C's y (CD horizontal).
    """
    a = _as_int_point(a)
    b = _as_int_point(b)
    c = _as_int_point(c)
    d = _as_int_point(d)
    satisfied = a[0] == b[0] and c[1] == d[1]
    nb = (a[0], b[1])
    nd = (d[0], c[1])
    return ConstraintResult((a, nb, c, nd), satisfied, True)


def symmetry(a: Point, b: Point, c: Point, d: Point) -> ConstraintResult:
    """Symmetry constraint from Table II.

    ``y1 = y2`` (AB horizontal), ``x3 = x4`` (CD vertical axis), and the axis
    passes through the midpoint of AB: ``x4 = (x1 + x2) / 2``. The integer
    representation requires ``(x1 + x2) mod 2 == 0`` for the midpoint to be
    integral — otherwise the enforced axis coordinate would be fractional and
    the discretised CAD model becomes invalid.

    The repair makes B share A's y (horizontal AB) and places CD on the integer
    midpoint axis. When the parity precondition fails, ``valid`` is False and the
    axis is rounded to the nearest integer (the paper flags such cases as a
    source of invalid models).
    """
    a = _as_int_point(a)
    b = _as_int_point(b)
    c = _as_int_point(c)
    d = _as_int_point(d)
    ny = a[1]
    nb = (b[0], ny)  # y1 = y2
    parity_ok = (a[0] + b[0]) % 2 == 0
    if parity_ok:
        axis = (a[0] + b[0]) // 2
        valid = True
        note = ""
    else:
        # Fractional midpoint -> not representable; round (paper: invalid model).
        axis = round((a[0] + b[0]) / 2)
        valid = False
        note = "symmetry axis midpoint non-integer ((x1+x2) odd)"
    nc = (axis, c[1])
    nd = (axis, d[1])  # x3 = x4 = axis
    satisfied = (
        parity_ok
        and a[1] == b[1]
        and c[0] == d[0]
        and c[0] == (a[0] + b[0]) // 2
    )
    return ConstraintResult((a, nb, nc, nd), satisfied, valid, note)


# Dispatch table so callers can drive control by constraint name (as the paper's
# structure-control loop does after each denoising step).
_CONSTRAINTS = {
    "coincidence": point_point_coincidence,
    "parallel": line_line_parallel,
    "perpendicular": line_line_perpendicular,
    "symmetry": symmetry,
}


def enforce(constraint: str, points) -> ConstraintResult:
    """Apply a named Table-II constraint.

    ``points`` is a sequence of ``(x, y)`` pairs. Coincidence uses the first two;
    the others use all four. Raises ``KeyError`` for unknown constraint names and
    ``ValueError`` for too few points.
    """
    fn = _CONSTRAINTS[constraint]
    pts = list(points)
    if constraint == "coincidence":
        if len(pts) < 2:
            raise ValueError("coincidence needs 2 points")
        return fn(pts[0], pts[1])
    if len(pts) < 4:
        raise ValueError(f"{constraint} needs 4 points")
    return fn(pts[0], pts[1], pts[2], pts[3])


def available_constraints() -> tuple[str, ...]:
    return tuple(sorted(_CONSTRAINTS))
