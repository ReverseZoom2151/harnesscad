"""Static analysis of BlenderLLM ``bpy`` scripts (Du et al., 2024).

BlenderLLM emits a Python script that drives Blender's ``bpy`` API to build a
model. Two deterministic things can be checked about such a script *without*
running Blender:

* **Syntax validity** -- the paper reports a syntax-error rate ``E_syntax`` for
  every model; a script either parses as Python or it does not. We decide that
  with :func:`compile`, so the result is exactly Blender's own "does it parse"
  gate, minus the runtime.
* **Operation vocabulary** -- the model's job is to call the mesh-primitive and
  transform operators (``bpy.ops.mesh.primitive_cube_add``,
  ``bpy.ops.transform.resize`` ...). Extracting that call sequence from the AST
  gives a runtime-free view of what the script *does*, which downstream metrics
  (see :mod:`bench.blenderllm_complexity`) build on.

Everything here is pure ``ast`` inspection: nothing is imported, evaluated, or
sent to Blender.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Sequence

# Recognised bpy.ops sub-modules that create or edit geometry. This is the
# deterministic "geometry-op vocabulary" the model is expected to speak.
PRIMITIVE_ADDERS = frozenset(
    {
        "primitive_cube_add",
        "primitive_uv_sphere_add",
        "primitive_ico_sphere_add",
        "primitive_cylinder_add",
        "primitive_cone_add",
        "primitive_torus_add",
        "primitive_plane_add",
        "primitive_circle_add",
        "primitive_grid_add",
        "primitive_monkey_add",
    }
)

TRANSFORM_OPS = frozenset({"translate", "resize", "rotate", "mirror"})

MODIFIER_OPS = frozenset({"modifier_add", "modifier_apply", "shade_smooth", "shade_flat"})


@dataclass(frozen=True)
class BpyCall:
    """A single ``bpy.ops.<group>.<op>(...)`` call extracted from a script."""

    group: str  # e.g. "mesh", "transform", "object"
    op: str  # e.g. "primitive_cube_add"
    num_args: int  # positional argument count
    keywords: tuple[str, ...]  # keyword names, in source order

    @property
    def dotted(self) -> str:
        return f"bpy.ops.{self.group}.{self.op}"

    @property
    def is_primitive(self) -> bool:
        return self.op in PRIMITIVE_ADDERS

    @property
    def is_transform(self) -> bool:
        return self.group == "transform" or self.op in TRANSFORM_OPS


@dataclass(frozen=True)
class SyntaxCheck:
    ok: bool
    error: str | None
    lineno: int | None
    offset: int | None


def check_syntax(script: str) -> SyntaxCheck:
    """Whether ``script`` parses as Python (the ``E_syntax`` gate)."""
    try:
        compile(script, "<bpy>", "exec")
    except SyntaxError as exc:
        return SyntaxCheck(False, exc.msg, exc.lineno, exc.offset)
    return SyntaxCheck(True, None, None, None)


def _dotted_chain(node: ast.AST) -> list[str] | None:
    """Return the attribute chain of a ``a.b.c`` expression, else ``None``."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        parts.reverse()
        return parts
    return None


def extract_calls(script: str) -> list[BpyCall]:
    """Ordered ``bpy.ops.<group>.<op>(...)`` calls in ``script``.

    Raises ``SyntaxError`` if the script does not parse -- call
    :func:`check_syntax` first when the input may be malformed.
    """
    tree = ast.parse(script)
    calls: list[BpyCall] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        chain = _dotted_chain(node.func)
        if chain is None or len(chain) != 4:
            continue
        if chain[0] != "bpy" or chain[1] != "ops":
            continue
        keywords = tuple(kw.arg for kw in node.keywords if kw.arg is not None)
        calls.append(BpyCall(chain[2], chain[3], len(node.args), keywords))
    return calls


def extract_primitives(script: str) -> list[BpyCall]:
    """Only the geometry-creating primitive-adder calls."""
    return [call for call in extract_calls(script) if call.is_primitive]


def is_recognized_vocabulary(call: BpyCall) -> bool:
    """Whether ``call`` is in the known primitive/transform/modifier vocabulary."""
    return (
        call.is_primitive
        or call.is_transform
        or call.op in MODIFIER_OPS
        or call.group in {"mesh", "object"}
    )


def vocabulary_coverage(script: str) -> float:
    """Fraction of bpy.ops calls that use recognised geometry vocabulary.

    Returns ``1.0`` for a script with no bpy.ops calls (vacuously covered).
    """
    calls = extract_calls(script)
    if not calls:
        return 1.0
    known = sum(1 for call in calls if is_recognized_vocabulary(call))
    return known / len(calls)
