"""Deterministic state-transition validator for the CadQuery ``Workplane``.

CadQuery's ``Workplane`` is a fluent *state machine* built
around a shared ``CQContext`` that tracks **pending edges** and **pending
wires**: 2D drawing verbs (``lineTo``, ``threePointArc``, ``spline`` ...) queue
*edges*; ``close`` / ``wire`` fuse queued edges into a *wire*; closed-profile
verbs (``circle``, ``rect``, ``polygon``) queue wires directly; and 3D verbs
(``extrude``, ``revolve``, ``loft`` ...) *consume* pending wires to make solids.
Getting this ordering wrong -- extruding with nothing pending, ``close``-ing an
empty path, lofting a single profile -- is a whole class of errors that arity
checking cannot see.

The harness's :mod:`programs.t2cq_ast` validates CadQuery programs by *arity*
(and workplane-name / variable use).  This module is the complementary
*semantic* layer: it models what each verb **consumes** and **produces** and
replays a call sequence through an abstract :class:`State`, reporting the
pending-model violations arity checking misses:

* ``extrude``/``revolve``/... with no pending wire;
* pending edges never fused into a wire before a 3D op (missing
  ``close()``/``wire()``);
* ``close()`` with no open path;
* ``loft`` with fewer than two profiles;
* a boolean (``cut``/``union``/``intersect``) with no base solid;
* combining profiles drawn on incompatible planes before a single 3D op.

Input is either an explicit ``[(method, arg_count), ...]`` sequence or a
CadQuery source string parsed with the stdlib :mod:`ast` (:func:`calls_from_code`).
Everything is deterministic and stdlib-only.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "Diagnostic",
    "State",
    "calls_from_code",
    "validate_calls",
    "validate_code",
    "is_valid_code",
    "EDGE_OPS",
    "CLOSED_PROFILE_OPS",
    "CONSUME_WIRE_OPS",
]

# 2D verbs that queue one or more pending *edges* and open a path.
EDGE_OPS = frozenset(
    {
        "lineTo",
        "line",
        "vLine",
        "hLine",
        "vLineTo",
        "hLineTo",
        "polarLine",
        "polarLineTo",
        "threePointArc",
        "sagittaArc",
        "radiusArc",
        "tangentArcPoint",
        "spline",
        "splineApprox",
        "bezier",
        "ellipseArc",
    }
)

# verbs that queue a closed *wire* directly (no open path).
CLOSED_PROFILE_OPS = frozenset(
    {"circle", "rect", "ellipse", "polygon", "polyline", "slot2D"}
)

# 3D verbs that consume pending wires to produce a solid.
CONSUME_WIRE_OPS = frozenset(
    {
        "extrude",
        "revolve",
        "cutBlind",
        "cutThruAll",
        "twistExtrude",
        "sweep",
        "loft",
    }
)

# verbs that produce a solid directly, independent of pending state.
PRIMITIVE_SOLID_OPS = frozenset({"box", "sphere", "cylinder", "wedge", "text"})

# boolean ops requiring an existing base solid.
BOOLEAN_OPS = frozenset({"cut", "union", "intersect", "__add__", "__sub__", "__mul__"})

# verbs that establish / switch the active plane.
PLANE_OPS = frozenset({"workplane", "faces", "workplaneFromTagged"})


@dataclass
class Diagnostic:
    """A single validation finding."""

    index: int
    method: str
    message: str
    severity: str = "error"  # "error" | "warning"

    def __str__(self) -> str:
        return f"[{self.severity}] call {self.index} {self.method!r}: {self.message}"


@dataclass
class State:
    """Abstract pending-model state of a Workplane chain."""

    pending_edges: int = 0
    pending_wires: int = 0
    open_path: bool = False
    has_solid: bool = False
    plane_serial: int = 0  # increments on each plane switch
    # distinct planes on which the current pending wires were created
    profile_planes: set = field(default_factory=set)

    def snapshot(self) -> Tuple:
        return (
            self.pending_edges,
            self.pending_wires,
            self.open_path,
            self.has_solid,
            self.plane_serial,
            frozenset(self.profile_planes),
        )


def _fuse_edges(state: State) -> None:
    """close()/wire(): turn queued edges into a pending wire on the active plane."""
    if state.pending_edges > 0:
        state.pending_wires += 1
        state.profile_planes.add(state.plane_serial)
    state.pending_edges = 0
    state.open_path = False


def validate_calls(
    calls: Sequence[Tuple[str, int]], root_plane: Optional[str] = "XY"
) -> List[Diagnostic]:
    """Replay a ``[(method, arg_count), ...]`` sequence and report violations.

    ``arg_count`` is accepted for interface parity with arity checkers but only a
    few verbs use it (``loft`` inspects nothing here; profile counting is by
    call).  Returns diagnostics in call order.
    """
    state = State()
    diags: List[Diagnostic] = []

    for i, (method, _argc) in enumerate(calls):
        if method in EDGE_OPS:
            state.pending_edges += 1
            state.open_path = True

        elif method == "moveTo" or method == "move":
            # relocating the pen: if a path was open, the current run of edges is
            # left as-is; a fresh sub-path may begin. No violation, but a lone
            # moveTo does not itself create geometry.
            pass

        elif method in CLOSED_PROFILE_OPS:
            state.pending_wires += 1
            state.profile_planes.add(state.plane_serial)

        elif method == "close":
            if not state.open_path and state.pending_edges == 0:
                diags.append(
                    Diagnostic(i, method, "close() with no open path (no edges drawn)")
                )
            _fuse_edges(state)

        elif method == "wire":
            _fuse_edges(state)

        elif method == "consolidateWires":
            _fuse_edges(state)
            if state.pending_wires > 1:
                state.pending_wires = 1

        elif method in PLANE_OPS:
            # switching planes: pending edges do not carry to the new plane
            # cleanly. Flag dangling open edges.
            if state.pending_edges > 0:
                diags.append(
                    Diagnostic(
                        i,
                        method,
                        "plane switch with unfused pending edges; call "
                        "close()/wire() before changing planes",
                        severity="warning",
                    )
                )
            state.plane_serial += 1

        elif method in CONSUME_WIRE_OPS:
            # auto-note edges that were never fused into a wire
            if state.pending_edges > 0 and state.pending_wires == 0:
                diags.append(
                    Diagnostic(
                        i,
                        method,
                        "pending edges were never fused into a wire; a 3D op "
                        "needs close() or wire() first",
                    )
                )
            elif state.pending_wires == 0:
                diags.append(
                    Diagnostic(
                        i, method, f"{method} with no pending wire to build from"
                    )
                )
            elif method == "loft" and state.pending_wires < 2:
                diags.append(
                    Diagnostic(
                        i,
                        method,
                        f"loft needs at least 2 profiles, found {state.pending_wires}",
                    )
                )
            elif len(state.profile_planes) > 1:
                diags.append(
                    Diagnostic(
                        i,
                        method,
                        "pending wires span "
                        f"{len(state.profile_planes)} incompatible planes",
                        severity="warning",
                    )
                )
            # consume
            state.pending_edges = 0
            state.pending_wires = 0
            state.open_path = False
            state.profile_planes.clear()
            state.has_solid = True

        elif method in PRIMITIVE_SOLID_OPS:
            state.has_solid = True

        elif method in BOOLEAN_OPS:
            if not state.has_solid:
                diags.append(
                    Diagnostic(
                        i, method, f"{method} with no base solid on the stack"
                    )
                )
            state.has_solid = True

        # unknown / neutral verbs (tag, val, vertices, translate, fillet ...) are
        # ignored: they do not change the pending model at this granularity.

    # end-of-chain: dangling pending geometry that is never consumed is a smell
    if state.pending_edges > 0:
        diags.append(
            Diagnostic(
                len(calls),
                "<end>",
                f"{state.pending_edges} pending edge op(s) never fused into a wire",
                severity="warning",
            )
        )
    if state.pending_wires > 0:
        diags.append(
            Diagnostic(
                len(calls),
                "<end>",
                f"{state.pending_wires} pending wire(s) never consumed by a 3D op",
                severity="warning",
            )
        )
    return diags


def calls_from_code(code: str) -> List[Tuple[str, int]]:
    """Extract the ordered ``(method, arg_count)`` chain from CadQuery source.

    Parses a single fluent expression such as
    ``cq.Workplane("XY").lineTo(1, 0).close().extrude(1)`` with the stdlib
    :mod:`ast` and returns methods in call (left-to-right) order.  The
    ``Workplane(...)`` root itself is not returned as a call.
    """
    tree = ast.parse(code.strip(), mode="eval") if _is_expr(code) else ast.parse(code)
    call_expr = _find_last_call(tree)
    if call_expr is None:
        return []
    chain: List[Tuple[str, int]] = []
    node: ast.AST = call_expr
    while isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Call):
            # a method call in the fluent chain; descend
            argc = len(node.args) + len(node.keywords)
            chain.append((func.attr, argc))
            node = func.value
        else:
            # reached the root constructor: cq.Workplane(...) (Attribute over a
            # Name) or bare Workplane(...) (Name). Not a chain method.
            break
    chain.reverse()
    return chain


def _is_expr(code: str) -> bool:
    try:
        ast.parse(code.strip(), mode="eval")
        return True
    except SyntaxError:
        return False


def _find_last_call(tree: ast.AST) -> Optional[ast.Call]:
    if isinstance(tree, ast.Expression):
        return tree.body if isinstance(tree.body, ast.Call) else None
    last: Optional[ast.Call] = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if last is None or getattr(node, "col_offset", 0) >= getattr(
                last, "col_offset", 0
            ):
                # prefer the outermost (largest) call expression
                pass
    # simplest robust approach: find an Assign/Expr whose value is a Call
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.Expr)) and isinstance(
            node.value, ast.Call
        ):
            last = node.value
    return last


def validate_code(code: str, root_plane: Optional[str] = "XY") -> List[Diagnostic]:
    """Parse CadQuery source and validate its pending-model transitions."""
    return validate_calls(calls_from_code(code), root_plane)


def is_valid_code(code: str) -> bool:
    """True when the code raises no *error*-severity diagnostics."""
    return not any(d.severity == "error" for d in validate_code(code))
