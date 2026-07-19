"""Deterministic CAD command-sequence -> CadQuery-code translator.

A text-to-CadQuery corpus is built by translating each command-sequence sample
(a minimal JSON of sketch loops + an extrusion) into equivalent CadQuery code,
typically with an LLM in a feedback loop. The LLM step is out of scope here, but
the underlying mapping is a small, *fully deterministic* rewrite -- which this
module implements.

It consumes the compact CAD-operation command representation from
:mod:`reconstruction.deepcad_command_spec` (the six command types ``SOL / Line /
Arc / Circle / Ext / EOS`` with their 16-slot parameter vector) and emits a
:class:`programs.t2cq_ast.CqProgram` (which then serializes to runnable CadQuery
Python). The rewrite follows these worked examples:

  * a **circle** loop  -> ``.circle(r)`` (centred with ``.moveTo`` when off-origin),
  * a **polyline** loop -> ``.moveTo(x0, y0).lineTo(...)....close()``,
  * an **arc** segment  -> ``.threePointArc((mx, my), (ex, ey))`` (the mid-point is
    derived from the sweep angle ``alpha`` and ccw flag ``f``),
  * the **extrusion**   -> ``.extrude(depth)`` where ``depth = e1 + e2`` (the two
    extrude distances),
  * the **coordinate system** (Ext angles ``theta/phi/gamma`` + origin
    ``px/py/pz``) -> trailing ``.rotate(...)`` / ``.translate(...)`` calls, e.g.
    ``part_1.rotate((0,0,0),(0,0,1),-90)`` then ``.translate((0, 0.5625, 0))``.

Each ``SOL ... Ext`` group becomes one ``part_k`` assignment. The boolean
type ``b`` on the extrusion selects how ``part_k`` combines into the running result
(new body / union / cut / intersect), yielding a final ``result`` assignment.

Pure stdlib (``math`` only) and deterministic (no wall clock, no RNG). It imports
the command spec and the CadQuery AST by full path and executes no CAD kernel.
"""

from __future__ import annotations

import math

from harnesscad.domain.reconstruction.tokens.deepcad_commands import (
    ARC, CIRCLE, EOS, EXT, LINE, SOL, Command,
)
from harnesscad.domain.programs.ast.cadquery import (
    Assign, Call, Chain, CqProgram, VarRef, Workplane,
)

# Boolean-operation codes on the extrusion:
# NewBody / Join(union) / Cut / Intersect.
NEW_BODY = 0
JOIN = 1
CUT = 2
INTERSECT = 3

_BOOL_METHOD = {JOIN: "union", CUT: "cut", INTERSECT: "intersect"}

# CadQuery plane name for an axis-aligned sketch. The default sketch plane is
# XY; general orientations are applied afterwards via explicit rotate/translate so
# the sketch itself is always authored on "XY".
_DEFAULT_PLANE = "XY"


def _round(value: float, ndigits: int = 6) -> float:
    """Deterministic rounding that never returns ``-0.0``."""
    r = round(float(value), ndigits)
    return 0.0 if r == 0.0 else r


def _arc_midpoint(start, end, alpha: float, ccw: bool):
    """Mid-point of a circular arc from ``start`` to ``end`` sweeping angle ``alpha``.

    An arc is encoded by its end-point plus the sweep angle ``alpha`` and a
    counter-clockwise flag ``f``. CadQuery's ``threePointArc``
    needs an intermediate point, so we reconstruct the arc's circle and take the
    point at the sweep mid-angle. Degenerate cases (zero sweep / coincident points)
    fall back to the chord mid-point, which serializes to a valid straight segment.
    """
    (x0, y0), (x1, y1) = start, end
    chord = math.hypot(x1 - x0, y1 - y0)
    if chord == 0.0 or alpha == 0.0:
        return (_round((x0 + x1) / 2.0), _round((y0 + y1) / 2.0))
    # Radius from chord and sweep: chord = 2 R sin(alpha/2).
    half = alpha / 2.0
    sin_half = math.sin(half)
    if abs(sin_half) < 1e-12:
        return (_round((x0 + x1) / 2.0), _round((y0 + y1) / 2.0))
    radius = chord / (2.0 * abs(sin_half))
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    # Perpendicular distance from chord mid-point to the arc mid-point.
    sagitta = radius - math.sqrt(max(0.0, radius * radius - (chord / 2.0) ** 2))
    # Unit normal to the chord; +normal is the left side for a ccw sweep.
    dx, dy = (x1 - x0) / chord, (y1 - y0) / chord
    nx, ny = dy, -dx
    sign = 1.0 if ccw else -1.0
    return (_round(mx + sign * sagitta * nx), _round(my + sign * sagitta * ny))


def _loop_calls(loop: list[Command]) -> list[Call]:
    """Translate one closed sketch loop (curve commands) into CadQuery chain calls."""
    if not loop:
        return []
    # A single circle loop -> centred .circle(r).
    if len(loop) == 1 and loop[0].type == CIRCLE:
        c = loop[0]
        cx, cy, r = _round(c.get("x")), _round(c.get("y")), _round(c.get("r"))
        calls: list[Call] = []
        if cx != 0.0 or cy != 0.0:
            calls.append(Call("moveTo", (cx, cy)))
        calls.append(Call("circle", (r,)))
        return calls
    # Polyline / arc loop. Curve commands give end-points; the start of the
    # first command is the end-point of the last (closed loop), so we seed moveTo
    # from the last vertex.
    verts: list[tuple[float, float]] = [
        (_round(c.get("x")), _round(c.get("y"))) for c in loop
        if c.type in (LINE, ARC)
    ]
    if not verts:
        return []
    start = verts[-1]
    calls = [Call("moveTo", (start[0], start[1]))]
    cur = start
    for c in loop:
        if c.type == LINE:
            end = (_round(c.get("x")), _round(c.get("y")))
            calls.append(Call("lineTo", (end[0], end[1])))
            cur = end
        elif c.type == ARC:
            end = (_round(c.get("x")), _round(c.get("y")))
            alpha = c.get("alpha")
            ccw = c.get("f") >= 0.5
            mid = _arc_midpoint(cur, end, alpha if alpha != -1.0 else 0.0, ccw)
            calls.append(Call("threePointArc", (mid, end)))
            cur = end
    calls.append(Call("close"))
    return calls


def _split_groups(commands: list[Command]) -> list[tuple[list[list[Command]], Command]]:
    """Split a command list into ``(loops, ext)`` groups.

    Each group is the loops accumulated since the previous extrusion, terminated by
    an ``Ext`` command. Leading ``SOL`` open a new loop; ``EOS`` ends the sequence.
    """
    groups: list[tuple[list[list[Command]], Command]] = []
    loops: list[list[Command]] = []
    current: list[Command] | None = None
    for cmd in commands:
        if cmd.type == EOS:
            break
        if cmd.type == SOL:
            current = []
            loops.append(current)
        elif cmd.type in (LINE, ARC, CIRCLE):
            if current is None:
                current = []
                loops.append(current)
            current.append(cmd)
        elif cmd.type == EXT:
            groups.append((loops, cmd))
            loops = []
            current = None
    return groups


def _coordinate_calls(ext: Command) -> list[Call]:
    """Trailing ``.rotate`` / ``.translate`` calls for the extrusion's pose.

    Rotations about the ZYZ axes for ``(theta, phi, gamma)``
    (in radians -> degrees) followed by a translation to ``(px, py, pz)``. Only
    non-identity transforms are emitted, so axis-aligned parts stay clean.
    """
    calls: list[Call] = []
    gamma, theta, phi = ext.get("gamma"), ext.get("theta"), ext.get("phi")
    origin = (0.0, 0.0, 0.0)
    # ZYZ order: apply gamma about z, then theta about y, then phi about z.
    for angle, axis in ((gamma, (0, 0, 1)), (theta, (0, 1, 0)), (phi, (0, 0, 1))):
        if angle not in (-1.0, 0.0) and angle != 0.0:
            deg = _round(math.degrees(angle))
            if deg != 0.0:
                calls.append(Call("rotate", (origin, axis, deg)))
    px, py, pz = _round(ext.get("px")), _round(ext.get("py")), _round(ext.get("pz"))
    px = 0.0 if px == -1.0 else px
    py = 0.0 if py == -1.0 else py
    pz = 0.0 if pz == -1.0 else pz
    if (px, py, pz) != (0.0, 0.0, 0.0):
        calls.append(Call("translate", ((px, py, pz),)))
    return calls


def _extrude_depth(ext: Command) -> float:
    """Total extrusion depth = e1 + e2 (the two extrude distances)."""
    e1 = ext.get("e1")
    e2 = ext.get("e2")
    e1 = 0.0 if e1 == -1.0 else e1
    e2 = 0.0 if e2 == -1.0 else e2
    depth = _round(e1 + e2)
    if depth == 0.0:
        depth = _round(e1) if e1 != 0.0 else 1.0
    return depth


def translate_to_program(commands: list[Command]) -> CqProgram:
    """Translate a CAD command list into a CadQuery :class:`CqProgram`.

    Produces one ``part_k`` assignment per extrusion group and a final ``result``
    that combines the parts per each group's boolean type (new body / union / cut /
    intersect). Raises :class:`ValueError` if no extrusion group is present.
    """
    groups = _split_groups(commands)
    if not groups:
        raise ValueError("command sequence contains no extrusion group")

    statements: list[Assign] = []
    result_var: str | None = None
    for index, (loops, ext) in enumerate(groups, start=1):
        calls: list[Call] = []
        for loop in loops:
            calls.extend(_loop_calls(loop))
        calls.append(Call("extrude", (_extrude_depth(ext),)))
        calls.extend(_coordinate_calls(ext))
        part = f"part_{index}"
        statements.append(Assign(part, Chain(Workplane(_DEFAULT_PLANE), tuple(calls))))

        b = ext.get("b")
        b = NEW_BODY if b == -1.0 else int(round(b))
        if result_var is None or b == NEW_BODY:
            if result_var is None:
                result_var = part
            else:
                # A fresh new-body extrusion keeps the latest solid as the result.
                result_var = part
        else:
            method = _BOOL_METHOD.get(b, "union")
            combined = f"result_{index}"
            statements.append(
                Assign(combined, Chain(VarRef(result_var), (Call(method, (VarRef(part),)),))))
            result_var = combined

    return CqProgram(tuple(statements), result_var)


def translate_to_code(commands: list[Command]) -> str:
    """Convenience: translate a CAD command list straight to CadQuery source."""
    from harnesscad.domain.programs.ast.cadquery import serialize
    return serialize(translate_to_program(commands))
