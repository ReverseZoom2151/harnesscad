"""FreeCAD Part-macro representation for Query2CAD (Badagabettu et al., 2024).

Query2CAD's distinctive output is not a generic code string but an *executable
FreeCAD macro*: a Python program that drives the FreeCAD ``Part`` workbench to
build a solid (sec. 4, "generates a Python macro that can be executed on the
FreeCAD software"). This module gives that macro a structured, deterministic
representation so the rest of the pipeline (VQAScore gating, refinement,
metrics) can reason about it without any FreeCAD install.

The representation models the operations the paper's dataset exercises
(sec. 5.1): primitive solids (box, sphere, cylinder, cone, torus), rigid
placement of each primitive, and boolean combination (fuse / cut / common).
A :class:`FreeCADMacro` serialises to a deterministic ``.FCMacro`` Python source
string that mirrors the FreeCAD ``Part`` scripting API, and reports a coarse
operation taxonomy used to bucket a query's difficulty.

Stdlib only, deterministic, no FreeCAD, no LLM, no wall clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Primitive kinds and their required dimension parameters (FreeCAD Part API).
PRIMITIVE_PARAMS: Dict[str, Tuple[str, ...]] = {
    "box": ("length", "width", "height"),
    "sphere": ("radius",),
    "cylinder": ("radius", "height"),
    "cone": ("radius1", "radius2", "height"),
    "torus": ("radius1", "radius2"),
}

# Boolean operations and the FreeCAD Part function that realises each.
BOOLEAN_FUNCS: Dict[str, str] = {
    "fuse": "fuse",
    "cut": "cut",
    "common": "common",
}


def _fmt_num(value: float) -> str:
    """Deterministic numeric formatting: integers stay integral, else trimmed."""
    f = float(value)
    if f == int(f):
        return str(int(f))
    return repr(round(f, 6))


@dataclass(frozen=True)
class Primitive:
    """One FreeCAD ``Part`` primitive placed at a position.

    ``name`` is the FreeCAD object label (unique within a macro). ``params`` maps
    the primitive's required dimension names to numeric values. ``position`` is
    the (x, y, z) placement of the primitive's local origin (default origin).
    """

    name: str
    kind: str
    params: Dict[str, float]
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.kind not in PRIMITIVE_PARAMS:
            raise ValueError("unknown primitive kind: %r" % (self.kind,))
        required = set(PRIMITIVE_PARAMS[self.kind])
        got = set(self.params)
        if got != required:
            raise ValueError(
                "%s requires params %s, got %s"
                % (self.kind, sorted(required), sorted(got)))
        for k, v in self.params.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError("param %r must be numeric" % (k,))
            if float(v) < 0:
                raise ValueError("param %r must be non-negative" % (k,))
        if len(self.position) != 3:
            raise ValueError("position must be a 3-tuple")

    def to_lines(self) -> List[str]:
        """FreeCAD Part source lines constructing and placing this primitive."""
        cls = self.kind.capitalize()
        # FreeCAD exposes primitives both as Part.makeBox(...) helpers and as
        # Part.Box() objects; Query2CAD's macros use the object form so a
        # Placement can be attached. Emit that form deterministically.
        lines = [
            "%s = Part.%s()" % (self.name, cls),
        ]
        for p in PRIMITIVE_PARAMS[self.kind]:
            attr = "".join(part.capitalize() for part in p.split("_")) \
                if "_" in p else p.capitalize()
            lines.append("%s.%s = %s" % (self.name, attr, _fmt_num(self.params[p])))
        x, y, z = self.position
        if (x, y, z) != (0.0, 0.0, 0.0):
            lines.append(
                "%s.Placement = FreeCAD.Placement("
                "FreeCAD.Vector(%s, %s, %s), FreeCAD.Rotation())"
                % (self.name, _fmt_num(x), _fmt_num(y), _fmt_num(z)))
        return lines


@dataclass(frozen=True)
class BooleanOp:
    """A boolean combination of two named shapes producing a new named shape."""

    name: str
    op: str
    left: str
    right: str

    def __post_init__(self) -> None:
        if self.op not in BOOLEAN_FUNCS:
            raise ValueError("unknown boolean op: %r" % (self.op,))

    def to_lines(self) -> List[str]:
        return ["%s = %s.%s(%s)"
                % (self.name, self.left, BOOLEAN_FUNCS[self.op], self.right)]


@dataclass
class FreeCADMacro:
    """An ordered FreeCAD Part macro: primitives, then boolean combinations.

    ``result`` names the shape that is shown at the end (the final CAD model). If
    omitted it defaults to the last-defined shape.
    """

    primitives: List[Primitive] = field(default_factory=list)
    booleans: List[BooleanOp] = field(default_factory=list)
    result: Optional[str] = None

    def _defined_names(self) -> List[str]:
        return [p.name for p in self.primitives] + [b.name for b in self.booleans]

    def validate(self) -> None:
        """Check name uniqueness, boolean operand definition order, and result."""
        seen: set = set()
        available: set = set()
        for p in self.primitives:
            if p.name in seen:
                raise ValueError("duplicate shape name: %r" % (p.name,))
            seen.add(p.name)
            available.add(p.name)
        for b in self.booleans:
            if b.name in seen:
                raise ValueError("duplicate shape name: %r" % (b.name,))
            for operand in (b.left, b.right):
                if operand not in available:
                    raise ValueError(
                        "boolean %r references undefined shape %r"
                        % (b.name, operand))
            seen.add(b.name)
            available.add(b.name)
        if self.result is not None and self.result not in available:
            raise ValueError("result %r is undefined" % (self.result,))
        if not available:
            raise ValueError("macro defines no shapes")

    def final_shape(self) -> str:
        self.validate()
        if self.result is not None:
            return self.result
        return self._defined_names()[-1]

    def to_source(self) -> str:
        """Serialise to a deterministic FreeCAD ``.FCMacro`` Python source string.

        The header always imports FreeCAD/Part and creates a document; the footer
        shows the final shape and recomputes, matching the scaffolding a runnable
        FreeCAD macro requires.
        """
        final = self.final_shape()
        lines = [
            "import FreeCAD",
            "import Part",
            'doc = FreeCAD.newDocument("Query2CAD")',
        ]
        for p in self.primitives:
            lines.extend(p.to_lines())
        for b in self.booleans:
            lines.extend(b.to_lines())
        lines.append("Part.show(%s)" % final)
        lines.append("doc.recompute()")
        return "\n".join(lines) + "\n"

    def operation_summary(self) -> Dict[str, object]:
        """Coarse taxonomy of the macro: primitive count, kinds, boolean count.

        ``num_primitives`` and ``num_booleans`` drive the difficulty heuristic in
        :func:`estimate_difficulty` (easy = one primitive, no booleans; harder as
        primitives/booleans accumulate, mirroring the dataset in sec. 5.1).
        """
        self.validate()
        kinds: Dict[str, int] = {}
        for p in self.primitives:
            kinds[p.kind] = kinds.get(p.kind, 0) + 1
        ops: Dict[str, int] = {}
        for b in self.booleans:
            ops[b.op] = ops.get(b.op, 0) + 1
        return {
            "num_primitives": len(self.primitives),
            "num_booleans": len(self.booleans),
            "primitive_kinds": kinds,
            "boolean_ops": ops,
        }


def estimate_difficulty(macro: FreeCADMacro) -> str:
    """Bucket a macro into easy / medium / hard (sec. 5.1 dataset taxonomy).

    * easy   -- a single primitive shape, no boolean combination.
    * medium -- a few primitives and/or a boolean op or explicit placement.
    * hard   -- many primitives or several boolean operations (intricate design).
    """
    summary = macro.operation_summary()
    n_prim = summary["num_primitives"]
    n_bool = summary["num_booleans"]
    placed = any(p.position != (0.0, 0.0, 0.0) for p in macro.primitives)
    if n_prim <= 1 and n_bool == 0 and not placed:
        return "easy"
    if n_prim >= 4 or n_bool >= 2:
        return "hard"
    return "medium"
