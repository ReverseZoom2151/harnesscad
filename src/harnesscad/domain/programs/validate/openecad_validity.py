"""Loop-closure and profile validity for OpenECAD sketches (Yuan et al., 2024).

The paper's analysis of generated code (Sec. 6.4.1) singles out two structural
properties a valid OpenECAD sketch must satisfy:

* consecutive curves in a loop are *connected end-to-end*, and the first and last
  curves are joined, so the loop is **closed**;
* the positions of points keep each **profile valid** (every loop closed and
  non-empty).

This module checks those properties deterministically over the curve calls of
:mod:`programs.openecad_script`. A ``add_circle`` is a self-closed single-curve
loop; ``add_line`` / ``add_arc`` contribute ``start``/``end`` points that must
chain up. Nothing here is learned -- it is the geometric validity test the
paper's scoring and rendering rely on.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from harnesscad.domain.programs.ast import openecad as oe

DEFAULT_TOL = 1e-6

# Curve target names encode ``<Kind><step>_<loop>_<index>`` (Algorithm 1).
_CURVE_NAME = re.compile(r"^(Line|Arc|Circle)(\d+)_(\d+)_(\d+)$")


def curve_endpoints(call: oe.Call) -> tuple[tuple, tuple] | None:
    """Return ``(start, end)`` for a line/arc, or ``None`` for a circle.

    A circle is self-closed and has no chaining endpoints.
    """
    if call.func == oe.ADD_CIRCLE:
        return None
    if call.func in (oe.ADD_LINE, oe.ADD_ARC):
        start = call.keyword("start")
        end = call.keyword("end")
        if start is None or end is None:
            raise ValueError(f"{call.func} missing start/end")
        return tuple(start), tuple(end)
    raise ValueError(f"not a curve command: {call.func!r}")


def _dist(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def loop_gaps(curves: list[oe.Call]) -> list[float]:
    """Chaining gaps between consecutive curves plus the closing gap.

    For ``n`` non-circle curves this returns ``n`` distances: gap ``i`` is between
    ``end(curve_i)`` and ``start(curve_{i+1 mod n})``. An empty list means the
    loop is a single self-closed circle (or has no chainable curves).
    """
    pts = [curve_endpoints(c) for c in curves]
    chain = [p for p in pts if p is not None]
    if not chain:
        return []
    gaps = []
    n = len(chain)
    for i in range(n):
        _, end = chain[i]
        start_next = chain[(i + 1) % n][0]
        gaps.append(_dist(end, start_next))
    return gaps


def is_closed_loop(curves: list[oe.Call], tol: float = DEFAULT_TOL) -> bool:
    """True when *curves* form a single closed loop.

    A lone circle is closed. Otherwise every chaining gap (including first<->last)
    must be within *tol*.
    """
    if not curves:
        return False
    circles = [c for c in curves if c.func == oe.ADD_CIRCLE]
    if circles:
        # A circle only forms a valid loop on its own.
        return len(curves) == 1
    return all(g <= tol for g in loop_gaps(curves))


def profile_is_valid(loops: list[list[oe.Call]], tol: float = DEFAULT_TOL) -> bool:
    """True when a profile has at least one loop and every loop is closed."""
    if not loops:
        return False
    return all(loop and is_closed_loop(loop, tol) for loop in loops)


@dataclass(frozen=True)
class LoopKey:
    step: int
    loop: int


def loops_from_program(program: oe.Program) -> dict[LoopKey, list[oe.Call]]:
    """Group a program's curve calls into loops using the naming convention.

    Curve targets ``Line0_0_2`` etc. are grouped by ``(step, loop)`` and ordered
    by their curve index. Returns an insertion-ordered mapping.
    """
    groups: dict[LoopKey, list[tuple[int, oe.Call]]] = {}
    for name, call in program.calls():
        if call.func not in oe.CURVE_FUNCS:
            continue
        m = _CURVE_NAME.match(name)
        if not m:
            continue
        _, step, loop, idx = m.groups()
        key = LoopKey(int(step), int(loop))
        groups.setdefault(key, []).append((int(idx), call))
    return {k: [c for _, c in sorted(v)] for k, v in groups.items()}


def program_profiles_valid(program: oe.Program, tol: float = DEFAULT_TOL) -> bool:
    """True when every sketch step of *program* has a valid profile."""
    grouped = loops_from_program(program)
    if not grouped:
        return False
    by_step: dict[int, list[list[oe.Call]]] = {}
    for key, curves in grouped.items():
        by_step.setdefault(key.step, []).append(curves)
    return all(profile_is_valid(loops, tol) for loops in by_step.values())
