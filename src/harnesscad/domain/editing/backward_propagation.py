"""Backward propagation ``put``: a 3D-view edit -> an updated CSG program.

This is the genuinely new half of the bidirectional transformation -- the rest of
the codebase only does forward CSG evaluation. Here a *direct-manipulation edit
on the output* is propagated *back to the source program* while keeping code and
view coherent (the "put" direction of the lens formulation of bidirectional
programming).

The edit rules:

  * **Translation** -- a translate element is added to the CSG tree and the code,
    unless an existing one only affects the translated element (then that one is
    modified instead). The gizmo applies the accumulated translation and rotation
    from the root down to the selected object, so a *world-space* drag is
    converted to a *local* parameter using the selected node's parent frame.
  * **Rotation** -- a rotate element is added only if necessary; otherwise an
    existing one is modified.
  * **Scaling** -- *Scale*: if the selection is the only child of a scale element,
    that element's parameters are updated; otherwise a new scale element is added.
    *Scale primitive*: if the selected part is a primitive, its instantiating
    parameters are updated directly.

Plus the two lens laws that make the transformation well-behaved:

  * **GetPut** -- ``put`` of an unchanged view returns the original program.
  * **PutGet** -- after ``put`` of an edit, re-evaluating ``get`` reflects that
    exact edit on the selected element.

Deterministic, stdlib-only. Builds on :mod:`programs.bidircsg_ast` and
:mod:`programs.bidircsg_forward`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from harnesscad.domain.programs.ast.bidirectional_csg import (
    Node,
    Path,
    Primitive,
    Rotate,
    Scale,
    Translate,
    Vec3,
    iter_nodes,
    node_at,
    parent_path,
    replace_at,
    wrap_at,
)
from harnesscad.domain.programs.runtime.csg_forward_eval import GeomNode, find_instance, get

_EPS = 1e-9


@dataclass
class PutResult:
    """Outcome of a backward edit."""

    program: Node
    edited_path: Path   # where the edited element now lives in the new program
    reused: bool        # True if an existing transform was modified in place


def _is_zero(v: Vec3, eps: float = _EPS) -> bool:
    return all(abs(c) <= eps for c in v)


def _instance(program: Node, source_path: Path, call_stack: Tuple[int, ...]) -> GeomNode:
    inst = find_instance(get(program), source_path, call_stack)
    if inst is None:
        raise ValueError("no output instance for %r / %r" % (source_path, call_stack))
    return inst


# --------------------------------------------------------------------------
# Translation.
# --------------------------------------------------------------------------

def put_translate(
    program: Node,
    source_path: Path,
    world_delta: Vec3,
    call_stack: Tuple[int, ...] = (),
) -> PutResult:
    """Propagate a world-space translation of the selected element back to code.

    Reuses the parent ``Translate`` if there is one (when it only affects the
    translated element); otherwise inserts a new ``translate`` above the element.
    """
    if _is_zero(world_delta):
        return PutResult(program, source_path, False)  # GetPut

    if source_path:
        ppath = parent_path(source_path)
        parent = node_at(program, ppath)
        if isinstance(parent, Translate):
            # Reuse: modify the existing translate's offset in its own frame.
            frame = _instance(program, ppath, call_stack).parent_transform
            local = frame.apply_inverse_linear(world_delta)
            new_offset = tuple(o + d for o, d in zip(parent.offset, local))
            new_prog = replace_at(program, ppath, Translate(new_offset, parent.child))
            return PutResult(new_prog, source_path, True)

    # Insert a new translate above the selected node.
    frame = _instance(program, source_path, call_stack).parent_transform
    local = frame.apply_inverse_linear(world_delta)
    new_prog = wrap_at(program, source_path, lambda old: Translate(tuple(local), old))
    return PutResult(new_prog, source_path + (0,), False)


# --------------------------------------------------------------------------
# Rotation (local euler delta, degrees, about the object's own axes).
# --------------------------------------------------------------------------

def put_rotate(
    program: Node,
    source_path: Path,
    angle_delta: Vec3,
    call_stack: Tuple[int, ...] = (),
) -> PutResult:
    """Propagate a rotation of the selected element back to code."""
    if _is_zero(angle_delta):
        return PutResult(program, source_path, False)

    if source_path:
        ppath = parent_path(source_path)
        parent = node_at(program, ppath)
        if isinstance(parent, Rotate):
            new_angles = tuple(a + d for a, d in zip(parent.angles, angle_delta))
            new_prog = replace_at(program, ppath, Rotate(new_angles, parent.child))
            return PutResult(new_prog, source_path, True)

    new_prog = wrap_at(
        program, source_path, lambda old: Rotate(tuple(angle_delta), old)
    )
    return PutResult(new_prog, source_path + (0,), False)


# --------------------------------------------------------------------------
# Scaling.
# --------------------------------------------------------------------------

def put_scale(
    program: Node,
    source_path: Path,
    factor_delta: Vec3,
    call_stack: Tuple[int, ...] = (),
) -> PutResult:
    """Propagate a whole-subtree *Scale* edit back to code.

    Reuses a parent ``Scale`` (multiplying its factors) if present; otherwise
    inserts a new ``scale``.
    """
    if all(abs(f - 1.0) <= _EPS for f in factor_delta):
        return PutResult(program, source_path, False)

    if source_path:
        ppath = parent_path(source_path)
        parent = node_at(program, ppath)
        if isinstance(parent, Scale):
            new_factors = tuple(f * d for f, d in zip(parent.factors, factor_delta))
            new_prog = replace_at(program, ppath, Scale(new_factors, parent.child))
            return PutResult(new_prog, source_path, True)

    new_prog = wrap_at(
        program, source_path, lambda old: Scale(tuple(factor_delta), old)
    )
    return PutResult(new_prog, source_path + (0,), False)


def put_scale_primitive(
    program: Node,
    source_path: Path,
    factor_delta: Vec3,
) -> PutResult:
    """Propagate a *Scale primitive* edit: update the primitive's params.

    Only valid when the selected node is a primitive; the instantiating
    parameters are scaled component-wise (extra params beyond 3 are left as-is,
    e.g. a cylinder's height/radius pair is scaled by the first factors).
    """
    node = node_at(program, source_path)
    if not isinstance(node, Primitive):
        raise ValueError("scale primitive requires a primitive node")
    if all(abs(f - 1.0) <= _EPS for f in factor_delta):
        return PutResult(program, source_path, False)
    new_params = tuple(
        p * factor_delta[i] if i < len(factor_delta) else p
        for i, p in enumerate(node.params)
    )
    new_prog = replace_at(program, source_path, Primitive(node.kind, new_params))
    return PutResult(new_prog, source_path, True)


# --------------------------------------------------------------------------
# Lens laws / round-trip consistency.
# --------------------------------------------------------------------------

def get_put_holds(program: Node) -> bool:
    """GetPut: an unchanged view yields the identical program at every element."""
    for path, _ in iter_nodes(program):
        if put_translate(program, path, (0.0, 0.0, 0.0)).program != program:
            return False
        if put_rotate(program, path, (0.0, 0.0, 0.0)).program != program:
            return False
        if put_scale(program, path, (1.0, 1.0, 1.0)).program != program:
            return False
    return True


def put_get_translate_holds(
    program: Node,
    source_path: Path,
    world_delta: Vec3,
    call_stack: Tuple[int, ...] = (),
    tol: float = 1e-6,
) -> bool:
    """PutGet for translation: after ``put``, the element's anchor moved by delta."""
    before = _instance(program, source_path, call_stack).anchor
    res = put_translate(program, source_path, world_delta, call_stack)
    after = _instance(res.program, res.edited_path, call_stack).anchor
    return all(
        abs((a - b) - d) <= tol
        for a, b, d in zip(after, before, world_delta)
    )


def world_point(node: GeomNode, local_point: Vec3) -> Vec3:
    """World position of a local point on an output node (probe for rotate/scale)."""
    return node.world_transform.apply(local_point)


def put_get_probe(
    program: Node,
    source_path: Path,
    put_result: PutResult,
    local_point: Vec3,
    expected_world: Vec3,
    call_stack: Tuple[int, ...] = (),
    tol: float = 1e-6,
) -> bool:
    """Generic PutGet probe: a local point maps to ``expected_world`` after edit."""
    after = _instance(put_result.program, put_result.edited_path, call_stack)
    got = world_point(after, local_point)
    return all(abs(g - e) <= tol for g, e in zip(got, expected_world))


def roundtrip_anchor_neutral(
    program: Node,
    source_path: Path,
    world_delta: Vec3,
    call_stack: Tuple[int, ...] = (),
    tol: float = 1e-6,
) -> bool:
    """Translate by delta then by -delta -> the element's anchor is unchanged."""
    start = _instance(program, source_path, call_stack).anchor
    r1 = put_translate(program, source_path, world_delta, call_stack)
    back = tuple(-c for c in world_delta)
    r2 = put_translate(r1.program, r1.edited_path, back, call_stack)
    end = _instance(r2.program, r2.edited_path, call_stack).anchor
    return all(abs(a - b) <= tol for a, b in zip(start, end))
